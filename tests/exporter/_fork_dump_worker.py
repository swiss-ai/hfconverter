"""Build a REAL fork GPTModel on CPU and dump its on-disk torch_dist key namespace + fused tensors.

Runs as a SUBPROCESS with PYTHONPATH=<fork>, because the fork's `megatron.core` and the venv's pip
`megatron.core` cannot coexist in one interpreter (first import wins). Invoked by
test_fork_real_keys.py; not a test itself.

Usage:  python _fork_dump_worker.py <out.pt> [--moe-layer-freq-int]

What it proves (see the test for the assertions):
  * the on-disk ShardedTensor.key namespace the exporter must match (incl. the DOUBLED
    'experts.experts.' segment) -- this is what torch_dist persists, not the in-memory dict keys;
  * the fused-QKV interleave and the SwiGLU gate/up order, by handing back the fork's OWN fused
    weights alongside the q/k/v the fork's OWN attention code extracts from them;
  * that an int moe_layer_freq yields HOMOGENEOUS keys (layer index folded into a tensor axis).

Carve-outs (structural, not laziness):
  * LatentMoE is skipped: moe_layer.py:244-266 hard-codes TELinear behind `assert HAVE_TE`, so the
    latent projections cannot be instantiated without a GPU/TE build.
  * The layer spec is the LOCAL one (TE is unavailable on CPU). Its sharded_state_dict_keys_map
    (gpt_layer_specs.py:502-505) renames `pre_mlp_layernorm.` -> `mlp.linear_fc1.layer_norm_`
    UNCONDITIONALLY -- even on MoE layers, where the TE spec keeps `pre_mlp_layernorm.weight`.
    The test undoes exactly that one rename; the real production checkpoint (TE spec) is the
    evidence, and test_fork_real_keys.py::TestRealCheckpointKeys pins it.
"""

import argparse
import os
import socket
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

# Block apex BEFORE any megatron import (a None sys.modules entry makes `import apex`
# raise ImportError). gpt_layer_specs.py:57-70 sets LNImpl = FusedLayerNorm when apex
# imports, and that wrapper asserts normalization == "LayerNorm" (fused_layer_norm.py:67)
# -- it cannot build this model's RMSNorm, so on the NGC image (which ships apex) every
# fork-build test here SKIPPED (jobs 2784199/2784678: 287 passed, 14 skipped). Blocking
# it pins LNImpl = WrappedTorchNorm in every environment, matching the apex-less venv
# where these tests always passed. CPU-test-only: training and stage-1 build the TE spec
# on GPU (TENorm) and never reach mcore's apex fallback.
sys.modules["apex"] = None

import torch
import torch.distributed as dist

HIDDEN = 32
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 8
FFN = 64
EXPERTS = 8
MOE_FFN = 16
VOCAB = 128


def _cpu_shim():
    # The fork's router allocates its buffers on torch.cuda.current_device() (router.py:179/192/226)
    # and mcore's async save calls torch.cuda.synchronize() unconditionally.
    if not torch.cuda.is_available():
        torch.cuda.synchronize = lambda *a, **k: None
        torch.cuda.current_device = lambda: torch.device("cpu")


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def build_model(moe_layer_freq):
    from megatron.core import parallel_state
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.transformer_config import TransformerConfig

    parallel_state.initialize_model_parallel(1, 1)
    config = TransformerConfig(
        num_layers=3,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_query_groups=KV_HEADS,
        kv_channels=HEAD_DIM,
        ffn_hidden_size=FFN,
        use_cpu_initialization=True,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        add_bias_linear=False,
        qk_layernorm=True,
        sandwich_norm=True,
        residual_output_scaling=True,
        num_moe_experts=EXPERTS,
        moe_router_topk=2,
        moe_ffn_hidden_size=MOE_FFN,
        moe_shared_expert_intermediate_size=MOE_FFN,
        moe_layer_freq=moe_layer_freq,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_load_balancing_type="quantile_balancing",
        moe_router_topk_scaling_factor=2.5,
        moe_grouped_gemm=False,  # SequentialMLP: fork pins its keys identical to TEGroupedMLP
        moe_token_dispatcher_type="allgather",  # alltoall needs a cuda Stream
        pipeline_dtype=torch.float32,
    )
    spec = get_gpt_decoder_block_spec(config, use_transformer_engine=False)
    model = GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=VOCAB,
        max_sequence_length=64,
        position_embedding_type="rope",
        rotary_base=500000,
        share_embeddings_and_output_weights=False,
        pre_process=True,
        post_process=True,
    )
    return model


def dump(out_path, moe_layer_freq):
    from megatron.core import dist_checkpointing

    model = build_model(moe_layer_freq)
    payload = {"keys": {}, "tensors": {}, "fork_ref": {}}

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "ck")
        os.makedirs(ckpt)
        dist_checkpointing.save(model.sharded_state_dict(), ckpt)
        for key, sharded in dist_checkpointing.load_tensors_metadata(ckpt).items():
            payload["keys"][key] = {
                "shape": tuple(int(d) for d in sharded.global_shape),
                "dtype": str(sharded.dtype),
            }

    layer = model.decoder.layers[0]
    attn = layer.self_attention
    payload["tensors"]["linear_qkv.weight"] = attn.linear_qkv.weight.detach().clone()

    # Only the list-freq build has a dense layer 0; the int-freq build (homogeneous keys) is
    # all-MoE and only needs the key namespace.
    has_dense_mlp = hasattr(layer.mlp, "linear_fc1")
    if has_dense_mlp:
        payload["tensors"]["dense_fc1.weight"] = layer.mlp.linear_fc1.weight.detach().clone()
        payload["tensors"]["dense_fc2.weight"] = layer.mlp.linear_fc2.weight.detach().clone()

    # The fork's OWN q/k/v extraction from its OWN fused weight: this is the de-interleave
    # ground truth (attention.py get_query_key_value_tensors), captured on a fixed input.
    #
    # That method applies q_layernorm/k_layernorm AFTER the split (attention.py:1487-1491), so we
    # capture it TWICE: once with the norms bypassed (isolates the pure de-interleave, which is what
    # exporter/transforms.split_qkv implements) and once as-is (lets the test also check that our
    # HF-side qk-norm placement reproduces the fork's normalized q/k).
    torch.manual_seed(0)
    hidden = torch.randn(5, 1, HIDDEN)  # Megatron is sequence-first: [S, B, H]
    q_norm, k_norm = attn.q_layernorm, attn.k_layernorm
    try:
        attn.q_layernorm, attn.k_layernorm = None, None
        with torch.no_grad():
            q_raw, k_raw, v_raw = attn.get_query_key_value_tensors(hidden)
    finally:
        attn.q_layernorm, attn.k_layernorm = q_norm, k_norm
    with torch.no_grad():
        q_normed, k_normed, _ = attn.get_query_key_value_tensors(hidden)

    payload["fork_ref"]["hidden"] = hidden
    payload["fork_ref"]["q"] = q_raw.detach().clone()
    payload["fork_ref"]["k"] = k_raw.detach().clone()
    payload["fork_ref"]["v"] = v_raw.detach().clone()
    payload["fork_ref"]["q_normed"] = q_normed.detach().clone()
    payload["fork_ref"]["k_normed"] = k_normed.detach().clone()
    payload["tensors"]["q_layernorm.weight"] = attn.q_layernorm.weight.detach().clone()
    payload["tensors"]["k_layernorm.weight"] = attn.k_layernorm.weight.detach().clone()

    # The fork's OWN SwiGLU consumption of its OWN fused fc1: gate/up order ground truth.
    if has_dense_mlp:
        with torch.no_grad():
            mlp_out, _ = layer.mlp(hidden)
        payload["fork_ref"]["mlp_in"] = hidden
        payload["fork_ref"]["mlp_out"] = mlp_out.detach().clone()

    # Per-expert modules (SequentialMLP) so the test can prove the stacking axis/order of the
    # on-disk [E, ...] tensors against the fork's own per-expert parameters.
    moe_layer = None
    for cand in model.decoder.layers:
        if hasattr(cand.mlp, "experts"):
            moe_layer = cand
            break
    if moe_layer is not None:
        experts = moe_layer.mlp.experts
        for idx, expert in enumerate(experts.local_experts):
            payload["tensors"][f"expert{idx}.fc1.weight"] = expert.linear_fc1.weight.detach().clone()
            payload["tensors"][f"expert{idx}.fc2.weight"] = expert.linear_fc2.weight.detach().clone()
        payload["tensors"]["router.weight"] = moe_layer.mlp.router.weight.detach().clone()

    payload["geometry"] = {
        "hidden": HIDDEN, "heads": HEADS, "kv_heads": KV_HEADS, "head_dim": HEAD_DIM,
        "ffn": FFN, "experts": EXPERTS, "moe_ffn": MOE_FFN, "vocab": VOCAB,
    }
    torch.save(payload, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--moe-layer-freq-int", action="store_true",
                        help="use int moe_layer_freq (=1) -> homogeneous, layer-index-free keys")
    args = parser.parse_args()

    _cpu_shim()
    dist.init_process_group(backend="gloo", world_size=1, rank=0,
                            init_method=f"tcp://127.0.0.1:{_free_port()}")
    try:
        dump(args.out, 1 if args.moe_layer_freq_int else [0, 1, 1])
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    # Mandatory: mcore's async-save machinery spawns a multiprocessing Manager whose child
    # re-imports __main__.
    main()
