"""Independent HF-to-Megatron mapping used to build synthetic exporter fixtures.

This module intentionally does not import ``exporter``. It creates fork-named tensors,
saved argument namespaces, and `torch_dist` checkpoints that the production mapping must
convert back to the original Hugging Face model.

The inverse mapping independently implements grouped QKV fusion, gate/up fusion, and
expert-axis stacking. Tiny configs use the only embedding and residual multipliers that
the checkpoint argument booleans can express for their geometry.
"""

import argparse
import math
import os
import socket
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(_HERE)
REPO_ROOT = os.path.dirname(TESTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
# Tiny geometry used by the synthetic checkpoints.
# ---------------------------------------------------------------------------
TINY_VOCAB = 128
TINY_HIDDEN = 32
TINY_INTERMEDIATE = 64
TINY_LAYERS = 3
TINY_FIRST_K_DENSE = 1
TINY_HEADS = 4
TINY_KV_HEADS = 2
TINY_HEAD_DIM = 8
# >= 11 ON PURPOSE. With 8 experts every index is a single digit, so lexicographic and numeric
# ordering coincide and NO test can detect an expert-ordering bug (e.g. emitting/iterating experts
# as 0,1,10,11,...,2). The production model has 128 experts, where such a bug would silently
# scramble every expert. 12 makes 10 and 11 sort before 2. Found by mutation testing 2026-07-14.
TINY_N_EXPERTS = 12
TINY_TOPK = 2
TINY_MOE_INTERMEDIATE = 16
TINY_SHARED_INTERMEDIATE = 16  # n_shared_experts == 1
TINY_LATENT = 24
TINY_MAX_POS = 64
TINY_ROPE_THETA = 500000.0
TINY_EPS = 1e-5

# Only multipliers the checkpoint args can express (see module docstring).
TINY_EMBEDDING_MULTIPLIER = math.sqrt(TINY_HIDDEN)  # 5.656854249492381
TINY_RESIDUAL_MULTIPLIER = 1.0 / math.sqrt(2 * TINY_LAYERS)  # 0.4082482904638631

# Round-trip flag matrix: (id, sandwich, latent, qb, expert_bias_present).
# QB + expert_bias on for (F,F) and (T,T); QB off elsewhere; plus one expert_bias-absent case.
ROUNDTRIP_COMBOS = [
    ("plain-qb-bias", False, None, True, True),
    ("sandwich", True, None, False, True),
    ("latent", False, TINY_LATENT, False, True),
    ("sandwich-latent-qb-bias", True, TINY_LATENT, True, True),
    ("plain-no-expert-bias", False, None, False, False),
]


def tiny_export_config(
    sandwich_norm=False,
    moe_latent_size=None,
    use_quantile_balancing=False,
    **overrides,
):
    """Apertus2Config at the frozen tiny geometry, exporter-consistent multipliers."""
    from configuration_apertus2 import Apertus2Config

    kwargs = dict(
        vocab_size=TINY_VOCAB,
        hidden_size=TINY_HIDDEN,
        intermediate_size=TINY_INTERMEDIATE,
        num_hidden_layers=TINY_LAYERS,
        num_attention_heads=TINY_HEADS,
        num_key_value_heads=TINY_KV_HEADS,
        head_dim=TINY_HEAD_DIM,
        max_position_embeddings=TINY_MAX_POS,
        rms_norm_eps=TINY_EPS,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": TINY_ROPE_THETA,
            "partial_rotary_factor": 1.0,
        },
        use_qk_norm=True,
        n_routed_experts=TINY_N_EXPERTS,
        num_experts_per_tok=TINY_TOPK,
        moe_intermediate_size=TINY_MOE_INTERMEDIATE,
        n_shared_experts=1,
        first_k_dense_replace=TINY_FIRST_K_DENSE,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        sandwich_norm=sandwich_norm,
        moe_latent_size=moe_latent_size,
        use_quantile_balancing=use_quantile_balancing,
        embedding_multiplier=TINY_EMBEDDING_MULTIPLIER,
        residual_multiplier=TINY_RESIDUAL_MULTIPLIER,
        initializer_range=0.02,
        use_cache=True,
    )
    kwargs.update(overrides)
    return Apertus2Config(**kwargs)


def build_tiny_model(
    sandwich_norm=False,
    moe_latent_size=None,
    use_quantile_balancing=False,
    seed=0,
    randomize_router_buffers=True,
    zero_expert_bias=False,
    **overrides,
):
    """Seeded tiny eval-mode Apertus2ForCausalLM.

    randomize_router_buffers: fills the fp32 router buffers with non-zero seeded values so a
    fidelity test actually distinguishes "copied" from "synthesized zeros" (the class inits
    them to zeros, which would mask that bug class). zero_expert_bias keeps
    e_score_correction_bias at zeros — required by the expert_bias-absent roundtrip combo,
    where the exporter synthesizes fp32 zeros and the comparison target must equal them.
    """
    from modeling_apertus2 import Apertus2ForCausalLM

    torch.manual_seed(seed)
    config = tiny_export_config(
        sandwich_norm,
        moe_latent_size,
        use_quantile_balancing,
        **overrides,
    )
    model = Apertus2ForCausalLM(config)
    model.eval()
    if randomize_router_buffers:
        generator = torch.Generator().manual_seed(seed + 999)
        with torch.no_grad():
            for layer_idx, layer in enumerate(model.model.layers):
                if not config.is_moe_layer(layer_idx):
                    continue
                gate = layer.mlp.gate
                if not zero_expert_bias:
                    gate.e_score_correction_bias.copy_(
                        torch.randn(
                            gate.e_score_correction_bias.shape,
                            generator=generator,
                            dtype=torch.float32,
                        )
                    )
                if getattr(gate, "qb_beta", None) is not None:
                    gate.qb_beta.copy_(
                        0.1
                        * torch.randn(
                            gate.qb_beta.shape, generator=generator, dtype=torch.float32
                        )
                    )
    return model


# ---------------------------------------------------------------------------
# CPU shim + ws=1 gloo process group (probe.py recipe)
# ---------------------------------------------------------------------------


def install_cpu_shim():
    """No-op the two unconditional CUDA calls in mcore 0.18's save path (probe.py:25-33)."""
    if not torch.cuda.is_available():
        torch.cuda.synchronize = lambda *a, **kw: None
        torch.cuda.current_device = lambda: torch.device("cpu")


def find_free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def ensure_default_pg():
    """Idempotent: init a world_size=1 gloo default pg via tcp init if none is alive.

    Safe to call before every dist_checkpointing use — tolerates the exporter having created
    and/or destroyed its own group in between.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            world_size=1,
            rank=0,
            init_method=f"tcp://127.0.0.1:{find_free_port()}",
        )


# ---------------------------------------------------------------------------
# Inverse transforms (merge; the exporter owns the splits)
# ---------------------------------------------------------------------------


def merge_qkv(q, k, v, num_q_heads, num_kv_heads, head_dim):
    """HF q/k/v projections -> fork fused qkv, per-KV-group [q...q, k, v] blocks.

    Explicit per-group torch.cat, implemented independently of exporter.transforms.split_qkv.
    Works for 2D weights ([rows, hidden]) and 1D biases alike (row == leading dim).
    """
    assert num_q_heads % num_kv_heads == 0, (num_q_heads, num_kv_heads)
    heads_per_group = num_q_heads // num_kv_heads
    assert q.shape[0] == num_q_heads * head_dim, (q.shape, num_q_heads, head_dim)
    assert k.shape[0] == num_kv_heads * head_dim, (k.shape, num_kv_heads, head_dim)
    assert v.shape[0] == num_kv_heads * head_dim, (v.shape, num_kv_heads, head_dim)
    blocks = []
    for group in range(num_kv_heads):
        q_start = group * heads_per_group * head_dim
        blocks.append(q[q_start : q_start + heads_per_group * head_dim])
        blocks.append(k[group * head_dim : (group + 1) * head_dim])
        blocks.append(v[group * head_dim : (group + 1) * head_dim])
    return torch.cat(blocks, dim=0)


def merge_qkv_gate(q, g, k, v, num_q_heads, num_kv_heads, head_dim):
    """HF q/g/k/v projections -> fork fused qkv with attention_output_gate.

    Per KV group the fork stores [q...q, g...g, k, v] blocks: one gate block per query head,
    ordered like the query blocks, between the queries and the key (attention.py's
    "(2 * np/ng + 2) * hn" layout). Implemented independently of exporter.transforms.
    """
    assert num_q_heads % num_kv_heads == 0, (num_q_heads, num_kv_heads)
    heads_per_group = num_q_heads // num_kv_heads
    assert q.shape == g.shape, (q.shape, g.shape)
    assert q.shape[0] == num_q_heads * head_dim, (q.shape, num_q_heads, head_dim)
    assert k.shape[0] == num_kv_heads * head_dim, (k.shape, num_kv_heads, head_dim)
    assert v.shape[0] == num_kv_heads * head_dim, (v.shape, num_kv_heads, head_dim)
    blocks = []
    for group in range(num_kv_heads):
        q_start = group * heads_per_group * head_dim
        blocks.append(q[q_start : q_start + heads_per_group * head_dim])
        blocks.append(g[q_start : q_start + heads_per_group * head_dim])
        blocks.append(k[group * head_dim : (group + 1) * head_dim])
        blocks.append(v[group * head_dim : (group + 1) * head_dim])
    return torch.cat(blocks, dim=0)


def merge_gate_up(gate, up):
    """HF gate/up projections -> fork fused fc1 rows [gate; up], gate first."""
    assert gate.shape == up.shape, (gate.shape, up.shape)
    return torch.cat([gate, up], dim=0)


def _grab(state_dict, key):
    tensor = state_dict[key]
    return tensor.detach().clone().contiguous()


def to_megatron_tensors(model, config=None, expert_bias_present=True):
    """Apertus2ForCausalLM (or its runtime state dict) -> fork-keyed {str: Tensor}.

    Key renames follow the production mapping in reverse. FP32 router buffers stay
    fp32. e_score_correction_bias maps to router.expert_bias only when expert_bias_present;
    qb_beta maps to router.qb_beta only when config.use_quantile_balancing.
    """
    if isinstance(model, torch.nn.Module):
        state_dict = model.state_dict()
        config = config if config is not None else model.config
    else:
        state_dict = dict(model)
        assert config is not None, "config is required when passing a raw state dict"

    num_layers = config.num_hidden_layers
    moe_intermediate = config.moe_intermediate_size
    num_experts = config.n_routed_experts

    out = {}
    out["embedding.word_embeddings.weight"] = _grab(state_dict, "model.embed_tokens.weight")
    out["output_layer.weight"] = _grab(state_dict, "lm_head.weight")
    out["decoder.final_layernorm.weight"] = _grab(state_dict, "model.norm.weight")

    for layer in range(num_layers):
        hf = f"model.layers.{layer}."
        mg = f"decoder.layers.{layer}."

        # attention (all layers)
        out[mg + "self_attention.linear_qkv.layer_norm_weight"] = _grab(
            state_dict, hf + "attention_layernorm.weight"
        )
        if getattr(config, "attention_output_gate", False):
            out[mg + "self_attention.linear_qkv.weight"] = merge_qkv_gate(
                _grab(state_dict, hf + "self_attn.q_proj.weight"),
                _grab(state_dict, hf + "self_attn.g_proj.weight"),
                _grab(state_dict, hf + "self_attn.k_proj.weight"),
                _grab(state_dict, hf + "self_attn.v_proj.weight"),
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
            )
        else:
            out[mg + "self_attention.linear_qkv.weight"] = merge_qkv(
                _grab(state_dict, hf + "self_attn.q_proj.weight"),
                _grab(state_dict, hf + "self_attn.k_proj.weight"),
                _grab(state_dict, hf + "self_attn.v_proj.weight"),
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
            )
        out[mg + "self_attention.q_layernorm.weight"] = _grab(
            state_dict, hf + "self_attn.q_norm.weight"
        )
        out[mg + "self_attention.k_layernorm.weight"] = _grab(
            state_dict, hf + "self_attn.k_norm.weight"
        )
        out[mg + "self_attention.linear_proj.weight"] = _grab(
            state_dict, hf + "self_attn.o_proj.weight"
        )
        if config.sandwich_norm:
            out[mg + "post_self_attn_layernorm.weight"] = _grab(
                state_dict, hf + "post_attention_layernorm.weight"
            )
        if config.sandwich_norm:
            out[mg + "post_mlp_layernorm.weight"] = _grab(
                state_dict, hf + "post_feedforward_layernorm.weight"
            )

        if not config.is_moe_layer(layer):
            # dense layer: TE-fused pre-norm rides on linear_fc1
            out[mg + "mlp.linear_fc1.layer_norm_weight"] = _grab(
                state_dict, hf + "feedforward_layernorm.weight"
            )
            out[mg + "mlp.linear_fc1.weight"] = merge_gate_up(
                _grab(state_dict, hf + "mlp.gate_proj.weight"),
                _grab(state_dict, hf + "mlp.up_proj.weight"),
            )
            out[mg + "mlp.linear_fc2.weight"] = _grab(state_dict, hf + "mlp.down_proj.weight")
            continue

        # MoE layer: standalone TENorm pre-norm
        out[mg + "pre_mlp_layernorm.weight"] = _grab(
            state_dict, hf + "feedforward_layernorm.weight"
        )
        out[mg + "mlp.router.weight"] = _grab(state_dict, hf + "mlp.gate.weight")
        if expert_bias_present:
            bias = _grab(state_dict, hf + "mlp.gate.e_score_correction_bias")
            assert bias.dtype == torch.float32, bias.dtype
            out[mg + "mlp.router.expert_bias"] = bias
        if config.use_quantile_balancing:
            qb_beta = _grab(state_dict, hf + "mlp.gate.qb_beta")
            assert qb_beta.dtype == torch.float32, qb_beta.dtype
            out[mg + "mlp.router.qb_beta"] = qb_beta
        if config.moe_latent_size is not None:
            out[mg + "mlp.fc1_latent_proj.weight"] = _grab(
                state_dict, hf + "mlp.latent_down_proj.weight"
            )
            out[mg + "mlp.fc2_latent_proj.weight"] = _grab(
                state_dict, hf + "mlp.latent_up_proj.weight"
            )

        # routed experts: rebuild the stacked fork tensors from PER-EXPERT pieces, stacking
        # on a NEW axis 0. The per-expert gate/up slices of the runtime fused container are
        # exactly the on-disk per-expert tensors (pinned by the class test
        # test_on_disk_keys_are_per_expert_and_contract_exact: gate first, then up, dim 0).
        fused_gate_up = state_dict[hf + "mlp.experts.gate_up_proj"]  # [E, 2F, in_e]
        fused_down = state_dict[hf + "mlp.experts.down_proj"]  # [E, in_e, F]
        assert fused_gate_up.shape[0] == num_experts, fused_gate_up.shape
        assert fused_gate_up.shape[1] == 2 * moe_intermediate, fused_gate_up.shape
        per_expert_fc1 = []
        per_expert_fc2 = []
        for expert in range(num_experts):
            gate = fused_gate_up[expert, :moe_intermediate].detach().clone()
            up = fused_gate_up[expert, moe_intermediate:].detach().clone()
            per_expert_fc1.append(merge_gate_up(gate, up))
            per_expert_fc2.append(fused_down[expert].detach().clone())
        # 'experts.experts.' — the on-disk ShardedTensor.key doubles the segment (fork
        # experts.py:533-534 / :950). Do NOT "simplify" this to a single 'experts.': that spelling
        # is the fork's in-memory dict key, not what torch_dist persists. Cross-checked against a
        # real fork-built model in test_fork_real_keys.py.
        out[mg + "mlp.experts.experts.linear_fc1.weight"] = torch.stack(
            per_expert_fc1, dim=0
        ).contiguous()
        out[mg + "mlp.experts.experts.linear_fc2.weight"] = torch.stack(
            per_expert_fc2, dim=0
        ).contiguous()

        # shared expert
        out[mg + "mlp.shared_experts.linear_fc1.weight"] = merge_gate_up(
            _grab(state_dict, hf + "mlp.shared_experts.gate_proj.weight"),
            _grab(state_dict, hf + "mlp.shared_experts.up_proj.weight"),
        )
        out[mg + "mlp.shared_experts.linear_fc2.weight"] = _grab(
            state_dict, hf + "mlp.shared_experts.down_proj.weight"
        )

    return out


# ---------------------------------------------------------------------------
# Args Namespace with the fork spellings consumed by config derivation.
# ---------------------------------------------------------------------------


def _rope_theta(config):
    rope = getattr(config, "rope_parameters", None)
    if rope is not None:
        try:
            return float(rope["rope_theta"])
        except TypeError:
            return float(rope.rope_theta)
    return float(config.rope_theta)


def make_args_namespace(config, expert_bias_present=True, **overrides):
    """argparse.Namespace mirroring the fork's checkpoint args for `config`.

    scale_embeddings_by_sqrt_hidden / residual_output_scaling are derived from the config's
    multipliers and cross-checked: the args can only express {1.0, sqrt(hidden)} and
    {1.0, 1/sqrt(2*num_layers)} — passing a config with any other multiplier is a test-
    authoring bug and raises here.
    """
    embedding_multiplier = float(config.embedding_multiplier)
    scale_embeddings = not math.isclose(embedding_multiplier, 1.0)
    if scale_embeddings and not math.isclose(
        embedding_multiplier, math.sqrt(config.hidden_size)
    ):
        raise AssertionError(
            f"embedding_multiplier {embedding_multiplier} is neither 1.0 nor "
            f"sqrt(hidden)={math.sqrt(config.hidden_size)}; not expressible as checkpoint args"
        )
    residual_multiplier = float(config.residual_multiplier)
    residual_scaling = not math.isclose(residual_multiplier, 1.0)
    if residual_scaling and not math.isclose(
        residual_multiplier, 1.0 / math.sqrt(2 * config.num_hidden_layers)
    ):
        raise AssertionError(
            f"residual_multiplier {residual_multiplier} is neither 1.0 nor "
            f"1/sqrt(2*num_layers)={1.0 / math.sqrt(2 * config.num_hidden_layers)}"
        )

    if config.use_quantile_balancing:
        # fork stores a LIST when several balancing losses are stacked
        load_balancing = ["seq_aux_loss", "quantile_balancing"]
    else:
        load_balancing = "aux_loss"  # scalar string spelling

    if config.moe_layer_freq is None:
        first_k_dense = config.first_k_dense_replace
        moe_layer_freq = [0] * first_k_dense + [1] * (
            config.num_hidden_layers - first_k_dense
        )
    else:
        moe_layer_freq = list(config.moe_layer_freq)

    args = argparse.Namespace(
        # geometry
        padded_vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        group_query_attention=True,
        num_query_groups=config.num_key_value_heads,
        kv_channels=config.head_dim,
        ffn_hidden_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        seq_length=config.max_position_embeddings,
        # MoE
        num_experts=config.n_routed_experts,  # fork arg name (NOT num_moe_experts)
        moe_router_topk=config.num_experts_per_tok,
        moe_ffn_hidden_size=config.moe_intermediate_size,
        moe_shared_expert_intermediate_size=config.moe_intermediate_size
        * config.n_shared_experts,
        moe_layer_freq=moe_layer_freq,
        moe_latent_size=config.moe_latent_size,
        moe_router_load_balancing_type=load_balancing,
        # A real fork persists its estimator name, not the HF score-space value; translating
        # back makes every synthetic checkpoint exercise the exporter's alias normalization.
        moe_router_quantile_balancing_method={
            "sigmoid": "histogram",
            "legacy": "legacy_average",
        }[config.moe_router_quantile_balancing_method],
        moe_router_topk_scaling_factor=config.routed_scaling_factor,
        moe_router_score_function="sigmoid",
        # as in the real 1.5b run: mcore's DEFAULT is None (router computes in the activation
        # dtype), while the HF router is fp32 -- the exporter refuses anything but 'fp32'
        moe_router_dtype="fp32",
        moe_router_enable_expert_bias=expert_bias_present,
        # HF uses (1, 1) for ordinary routing. Megatron represents the same inactive path with
        # both arguments unset; only carry explicit values when more than one group is requested.
        moe_router_num_groups=(config.n_group if config.n_group != 1 else None),
        moe_router_group_topk=(config.topk_group if config.n_group != 1 else None),
        moe_router_topk_limited_devices=None,
        moe_shared_expert_gate=False,
        moe_use_offloading_experts=False,
        # norms / activations
        normalization="RMSNorm",
        layernorm_epsilon=config.rms_norm_eps,  # attr name; CLI flag is --norm-epsilon
        swiglu=True,
        sssglu=False,
        glu_linear_offset=0.0,
        activation_func_clamp_value=None,
        qk_layernorm=config.use_qk_norm,
        attention_output_gate=bool(getattr(config, "attention_output_gate", False)),
        sandwich_norm=config.sandwich_norm,
        # biases
        add_bias_linear=False,
        add_qkv_bias=False,
        # rope
        position_embedding_type="rope",
        rotary_base=int(_rope_theta(config)),
        rotary_percent=1.0,
        rotary_interleaved=False,
        use_rope_scaling=False,
        rope_scaling_factor=1.0,
        # multipliers
        scale_embeddings_by_sqrt_hidden=scale_embeddings,
        residual_output_scaling=residual_scaling,
        fp32_residual_connection=False,
        # excluded fork features (must all be off/None for the exporter to proceed)
        multi_latent_attention=False,
        mtp_num_layers=None,
        pnglu=False,
        use_mup=False,
        softmax_scale=None,
        apply_query_key_layer_scaling=False,
        window_size=None,
        window_attn_skip_freq=None,
        no_rope_freq=None,
        softmax_type="vanilla",
        # embeddings / misc
        untie_embeddings_and_output_weights=True,
        attention_dropout=config.attention_dropout,
        init_method_std=config.initializer_range,
        bf16=False,  # tiny test models are fp32; override for bf16 runs
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


# ---------------------------------------------------------------------------
# synthetic torch_dist checkpoint (probe.py recipe)
# ---------------------------------------------------------------------------


def save_synthetic_checkpoint(tensors, args, ckpt_dir, extra_tensors=None, iteration=100):
    """dist_checkpointing.save of `tensors` (+ optional extras) with the common part
    {'args': args, 'iteration': iteration, 'checkpoint_version': 3.0}.

    `extra_tensors` lets tests inject optimizer.* dummies / unknown keys / forbidden keys.
    Requires (and lazily ensures) a live default pg; installs the CPU shim.
    """
    import megatron.core.dist_checkpointing as dist_checkpointing
    from megatron.core.dist_checkpointing import ShardedTensor

    install_cpu_shim()
    ensure_default_pg()

    ckpt_dir = str(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    assert not os.listdir(ckpt_dir), f"refusing to save into non-empty dir {ckpt_dir}"

    payload = dict(tensors)
    if extra_tensors:
        overlap = set(payload) & set(extra_tensors)
        assert not overlap, f"extra_tensors would overwrite: {sorted(overlap)}"
        payload.update(extra_tensors)

    sharded_state_dict = {
        key: ShardedTensor.from_rank_offsets(key, tensor, replica_id=0)
        for key, tensor in payload.items()
    }
    # non-ShardedBase entries land in common.pt
    sharded_state_dict["args"] = args
    sharded_state_dict["iteration"] = iteration
    sharded_state_dict["checkpoint_version"] = 3.0

    dist_checkpointing.save(sharded_state_dict, ckpt_dir)
    return ckpt_dir


def load_plain_tensor_dict(ckpt_dir):
    """load_plain_tensors + the probe-documented common.pt filter (tensors only)."""
    import megatron.core.dist_checkpointing as dist_checkpointing

    install_cpu_shim()
    ensure_default_pg()
    plain = dist_checkpointing.load_plain_tensors(str(ckpt_dir))
    return {k: v for k, v in plain.items() if isinstance(v, torch.Tensor)}


# ---------------------------------------------------------------------------
# standalone self-check (runnable before exporter/ exists)
# ---------------------------------------------------------------------------


def _self_check(tmp_root):
    import shutil

    import megatron.core.dist_checkpointing as dist_checkpointing

    sandwich, latent, qb, expert_bias = True, TINY_LATENT, True, True
    model = build_tiny_model(sandwich, latent, qb, seed=0)
    tensors = to_megatron_tensors(model, model.config, expert_bias_present=expert_bias)
    args = make_args_namespace(model.config, expert_bias_present=expert_bias)

    print(f"(T,T,QB+bias) tiny combo: {len(tensors)} megatron tensors")
    for key in sorted(tensors):
        t = tensors[key]
        print(f"  {key}  {tuple(t.shape)}  {t.dtype}")

    ckpt_dir = os.path.join(tmp_root, "ckpt_selfcheck")
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    save_synthetic_checkpoint(tensors, args, ckpt_dir)

    common = dist_checkpointing.load_common_state_dict(ckpt_dir)
    assert vars(common["args"]) == vars(args), "args did not round trip"
    assert common["iteration"] == 100 and common["checkpoint_version"] == 3.0

    loaded = load_plain_tensor_dict(ckpt_dir)
    assert set(loaded) == set(tensors), (
        sorted(set(loaded) ^ set(tensors))
    )
    for key, ref in tensors.items():
        assert loaded[key].dtype == ref.dtype, key
        assert torch.equal(loaded[key], ref), key
    print("save -> load_plain_tensors round trip: bitwise OK, dtypes preserved "
          f"({sum(1 for t in tensors.values() if t.dtype == torch.float32)} fp32 tensors incl. "
          "router.expert_bias / router.qb_beta)")
    return 0


if __name__ == "__main__":
    # Keep everything under __main__: mcore's save spawns a multiprocessing Manager that
    # re-imports __main__ (probe_ep2.py gotcha).
    _scratch = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "_selfcheck_tmp")
    code = _self_check(_scratch)
    print("MEGATRON_MOCK SELF-CHECK PASSED")
    sys.exit(code)
