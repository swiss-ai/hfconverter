"""Validate the exporter against the FORK ITSELF, not against our mirror of it.

Everything else in tests/exporter/ round-trips through megatron_mock.py, which is the hand-written
INVERSE of exporter/mapping.py. Any assumption the two SHARE is invisible to those tests: on
2026-07-14 both hand-wrote `mlp.experts.linear_fc1.weight` while the fork actually writes
`mlp.experts.experts.linear_fc1.weight`, and 97 tests stayed green while a real conversion would
have found zero expert weights. These tests exist so that cannot happen again:

  TestForkKeyNamespace / TestForkLayouts  -- build a REAL fork GPTModel on CPU (subprocess, local
      layer spec), save a REAL torch_dist checkpoint, and compare the on-disk key namespace and the
      fused-tensor layouts against what the exporter expects.
  TestHomogeneousKeys  -- pin the fork behavior that makes an int moe_layer_freq unreadable.
  TestRealCheckpointKeys  -- audit the REAL production checkpoint on scratch (TE spec), which is the
      only evidence for the TE-vs-local key differences and for the real geometry.
  TestRealKdaCheckpointKeys  -- the KDA analogue: audit the real KDA smoke checkpoint (torch_dist,
      TE spec) against derive_config + build_plan. Metadata-only, so it needs no FLA, no GPU, and
      no HF KDA module. The fork-BUILD analogue (constructing a KDA model on CPU) is deferred:
      GatedDeltaNet.__init__ raises without flash-linear-attention, which the pinned container
      does not ship yet.

They skip cleanly when the fork checkout / real checkpoints are unavailable.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("megatron.core")

import torch

from exporter.config_from_args import derive_config
from exporter.mapping import build_plan
from exporter.transforms import split_gated_fc1, split_qkv

# Env-overridable; default is the post-2026-07-15 scratch location. The old
# /users/mvasilev/apertus tree is a stale pre-relocation copy — pointing there
# tests a 5-day-old fork (and skips everything once that leftover is deleted).
FORK_PATH = Path(os.environ.get("FORK", "/iopsstor/scratch/cscs/mvasilev/Megatron-LM-MoE"))
WORKER = Path(__file__).parent / "_fork_dump_worker.py"

REAL_CKPT = Path(
    "/iopsstor/scratch/cscs/mvasilev/Megatron-LM-MoE/_research/results/ckpts/"
    "1.5b-moe-128e-muonmd-lr1e-3-mlr1e-2-fp8-smoke/iter_0001192/mp_rank_00_000/model_optim_rng.pt"
)

# The KDA smoke run the mapping commits were verified against: 10 layers, KDA at
# 0,1,2,4,5,6,8 (freq [1,1,1,0,1,1,1,0,1,0]), 6 heads x 128, safe gate g_min -5,
# latent MoE, torch_dist format. iter_0000010 is latest_checkpointed_iteration.
REAL_KDA_CKPT = Path(
    "/iopsstor/scratch/cscs/mvasilev/Megatron-LM-MoE/_research/results/ckpts/"
    "scaling-ladder/kda-1.5b-moe-256e-latent-kda31-nope-smoke-muonmd-lr2.83e-3-"
    "mlr1.41e-2-latmoe-e256-top8-kda31-nope/iter_0000010"
)

# The fork's LOCAL layer spec renames pre_mlp_layernorm. -> mlp.linear_fc1.layer_norm_
# unconditionally (gpt_layer_specs.py:502-505) -- even on MoE layers, where the TE spec (what real
# runs use) keeps `pre_mlp_layernorm.weight`. TestRealCheckpointKeys pins that TE spelling against
# the production checkpoint. This is the ONLY sanctioned local-vs-TE difference; anything else that
# diverges is a bug in our mapping.
def _localspec_to_te(keys, first_k_dense):
    out = set()
    for key in keys:
        parts = key.split(".")
        is_fused_mlp_norm = (
            len(parts) == 6
            and parts[0:2] == ["decoder", "layers"]
            and parts[3:] == ["mlp", "linear_fc1", "layer_norm_weight"]
        )
        if is_fused_mlp_norm and int(parts[2]) >= first_k_dense:
            out.add(f"decoder.layers.{parts[2]}.pre_mlp_layernorm.weight")
            continue
        out.add(key)
    return out


@pytest.fixture(scope="module")
def fork_dump(tmp_path_factory):
    """Real fork GPTModel -> real torch_dist checkpoint -> on-disk keys + fused tensors."""
    if not FORK_PATH.is_dir():
        pytest.skip(f"fork checkout not present at {FORK_PATH}")
    out = tmp_path_factory.mktemp("forkdump") / "dump.pt"
    env = dict(os.environ, PYTHONPATH=str(FORK_PATH), PYTHONWARNINGS="ignore")
    proc = subprocess.run(
        [sys.executable, str(WORKER), str(out)],
        env=env, capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0:
        pytest.skip(f"fork model could not be built on CPU:\n{proc.stderr[-2000:]}")
    return torch.load(out, weights_only=False)


@pytest.fixture(scope="module")
def exporter_plan(fork_dump):
    """The exporter's expected Megatron key set for the same geometry the worker built."""
    geo = fork_dump["geometry"]
    cfg = dict(
        vocab_size=geo["vocab"], hidden_size=geo["hidden"], intermediate_size=geo["ffn"],
        num_hidden_layers=3, num_attention_heads=geo["heads"],
        num_key_value_heads=geo["kv_heads"], head_dim=geo["head_dim"],
        n_routed_experts=geo["experts"], num_experts_per_tok=2,
        moe_intermediate_size=geo["moe_ffn"], n_shared_experts=1, first_k_dense_replace=1,
        use_qk_norm=True, sandwich_norm=True, moe_latent_size=None,
        use_quantile_balancing=True,
    )
    return build_plan(cfg, expert_bias_present=True)


class TestForkKeyNamespace:
    def test_exporter_expects_exactly_the_keys_the_fork_writes(self, fork_dump, exporter_plan):
        fork_keys = _localspec_to_te(set(fork_dump["keys"]), first_k_dense=1)
        expected = {row.megatron_key for row in exporter_plan.rows}
        missing = sorted(fork_keys - expected)  # fork writes it, we would not consume it
        extra = sorted(expected - fork_keys)  # we demand it, fork does not write it
        assert not missing, f"fork writes keys the exporter does not map: {missing}"
        assert not extra, f"exporter expects keys the fork does not write: {extra}"

    def test_routed_expert_keys_carry_the_doubled_experts_segment(self, fork_dump):
        # The regression guard for the 2026-07-14 bug. TEGroupedMLP/SequentialMLP return dict keys
        # under one prefix but rewrite the on-disk ShardedTensor.key with an EXTRA 'experts.'
        # (experts.py:533-534 / :950). torch_dist persists ShardedTensor.key.
        keys = set(fork_dump["keys"])
        assert "decoder.layers.1.mlp.experts.experts.linear_fc1.weight" in keys
        assert "decoder.layers.1.mlp.experts.experts.linear_fc2.weight" in keys
        assert "decoder.layers.1.mlp.experts.linear_fc1.weight" not in keys

    def test_shared_experts_and_router_do_not_double(self, fork_dump):
        keys = set(fork_dump["keys"])
        assert "decoder.layers.1.mlp.shared_experts.linear_fc1.weight" in keys
        assert "decoder.layers.1.mlp.router.weight" in keys
        assert "decoder.layers.1.mlp.router.expert_bias" in keys
        assert "decoder.layers.1.mlp.router.qb_beta" in keys

    def test_shapes_match_the_mapping_table(self, fork_dump, exporter_plan):
        fork_shapes = {k: v["shape"] for k, v in fork_dump["keys"].items()}
        for row in exporter_plan.rows:
            key = row.megatron_key
            if key not in fork_shapes:  # local-spec renamed pre-norms: shape is [H] either way
                continue
            assert fork_shapes[key] == tuple(row.shape), (
                f"{key}: fork wrote {fork_shapes[key]}, mapping expects {tuple(row.shape)}"
            )

    def test_fp32_router_buffers_stay_fp32_on_disk(self, fork_dump):
        for key, meta in fork_dump["keys"].items():
            if key.endswith("router.expert_bias") or key.endswith("router.qb_beta"):
                assert meta["dtype"] == "torch.float32", (key, meta)

    def test_sandwich_post_norms_exist_on_every_layer_including_the_dense_one(self, fork_dump):
        keys = set(fork_dump["keys"])
        for layer in range(3):
            assert f"decoder.layers.{layer}.post_self_attn_layernorm.weight" in keys
            assert f"decoder.layers.{layer}.post_mlp_layernorm.weight" in keys


class TestForkLayouts:
    """Prove the tensor layouts against the fork's OWN consumption of its OWN weights."""

    def test_split_qkv_recovers_what_fork_attention_extracts(self, fork_dump):
        geo = fork_dump["geometry"]
        fused = fork_dump["tensors"]["linear_qkv.weight"]
        hidden = fork_dump["fork_ref"]["hidden"]  # [S, B, H]

        q_w, k_w, v_w = split_qkv(fused, geo["heads"], geo["kv_heads"], geo["head_dim"])

        # HF-side computation from the split weights, reshaped into the fork's [S, B, heads, D].
        seq, batch = hidden.shape[0], hidden.shape[1]
        ours_q = (hidden @ q_w.T).view(seq, batch, geo["heads"], geo["head_dim"])
        ours_k = (hidden @ k_w.T).view(seq, batch, geo["kv_heads"], geo["head_dim"])
        ours_v = (hidden @ v_w.T).view(seq, batch, geo["kv_heads"], geo["head_dim"])

        torch.testing.assert_close(ours_q, fork_dump["fork_ref"]["q"], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(ours_k, fork_dump["fork_ref"]["k"], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(ours_v, fork_dump["fork_ref"]["v"], rtol=1e-5, atol=1e-5)

    def test_hf_qk_norm_placement_reproduces_fork_normalized_qk(self, fork_dump):
        # The fork normalizes q/k AFTER the de-interleave, on the [S, B, heads, head_dim] view
        # (attention.py:1487-1491). Our HF class normalizes on [B, S, heads, head_dim] before the
        # transpose -- the same reduction axis. Prove the two agree using the fork's own norm gains.
        geo = fork_dump["geometry"]
        fused = fork_dump["tensors"]["linear_qkv.weight"]
        hidden = fork_dump["fork_ref"]["hidden"]
        eps = 1e-5

        def rms_norm(x, weight):
            var = x.float().pow(2).mean(-1, keepdim=True)
            return (x.float() * torch.rsqrt(var + eps)).type_as(x) * weight

        q_w, k_w, v_w = split_qkv(fused, geo["heads"], geo["kv_heads"], geo["head_dim"])
        seq, batch = hidden.shape[0], hidden.shape[1]
        ours_q = rms_norm(
            (hidden @ q_w.T).view(seq, batch, geo["heads"], geo["head_dim"]),
            fork_dump["tensors"]["q_layernorm.weight"],
        )
        ours_k = rms_norm(
            (hidden @ k_w.T).view(seq, batch, geo["kv_heads"], geo["head_dim"]),
            fork_dump["tensors"]["k_layernorm.weight"],
        )
        torch.testing.assert_close(ours_q, fork_dump["fork_ref"]["q_normed"], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(ours_k, fork_dump["fork_ref"]["k_normed"], rtol=1e-5, atol=1e-5)

    def test_split_gated_fc1_matches_fork_swiglu_consumption(self, fork_dump):
        fc1 = fork_dump["tensors"]["dense_fc1.weight"]
        fc2 = fork_dump["tensors"]["dense_fc2.weight"]
        x = fork_dump["fork_ref"]["mlp_in"]

        gate_w, up_w = split_gated_fc1(fc1)
        ours = (torch.nn.functional.silu(x @ gate_w.T) * (x @ up_w.T)) @ fc2.T

        # If gate/up were swapped, silu would land on the wrong half and this would diverge.
        torch.testing.assert_close(ours, fork_dump["fork_ref"]["mlp_out"], rtol=1e-5, atol=1e-5)

    def test_stacked_expert_axis_is_the_global_expert_index(self, fork_dump):
        # We cannot read the stacked tensor's values from metadata alone, so prove the axis via the
        # fork's per-expert modules: on-disk [E, ...] must stack local_experts in index order.
        geo = fork_dump["geometry"]
        e0_fc1 = fork_dump["tensors"]["expert0.fc1.weight"]
        e1_fc1 = fork_dump["tensors"]["expert1.fc1.weight"]
        assert e0_fc1.shape == (2 * geo["moe_ffn"], geo["hidden"])
        assert not torch.equal(e0_fc1, e1_fc1), "experts must be distinct for this test to bite"

        stacked_shape = fork_dump["keys"][
            "decoder.layers.1.mlp.experts.experts.linear_fc1.weight"
        ]["shape"]
        assert stacked_shape == (geo["experts"], 2 * geo["moe_ffn"], geo["hidden"])

        fc2_shape = fork_dump["keys"][
            "decoder.layers.1.mlp.experts.experts.linear_fc2.weight"
        ]["shape"]
        assert fc2_shape == (geo["experts"], geo["hidden"], geo["moe_ffn"])


class TestHomogeneousKeys:
    """An int moe_layer_freq folds the layer index into a tensor axis -> our mapping cannot read it."""

    def test_int_moe_layer_freq_drops_the_layer_index_from_the_keys(self, tmp_path):
        if not FORK_PATH.is_dir():
            pytest.skip(f"fork checkout not present at {FORK_PATH}")
        out = tmp_path / "homogeneous.pt"
        env = dict(os.environ, PYTHONPATH=str(FORK_PATH), PYTHONWARNINGS="ignore")
        proc = subprocess.run(
            [sys.executable, str(WORKER), str(out), "--moe-layer-freq-int"],
            env=env, capture_output=True, text=True, timeout=900,
        )
        if proc.returncode != 0:
            pytest.skip(f"fork model could not be built on CPU:\n{proc.stderr[-2000:]}")
        keys = torch.load(out, weights_only=False)["keys"]

        assert "decoder.layers.mlp.experts.experts.linear_fc1.weight" in keys
        assert not any(k.startswith("decoder.layers.0.") for k in keys)
        # the layer count becomes the leading tensor axis
        assert keys["decoder.layers.self_attention.linear_qkv.weight"]["shape"][0] == 3


@pytest.fixture(scope="module")
def real():
    sd = torch.load(REAL_CKPT, map_location="cpu", weights_only=False, mmap=True)
    return {"keys": {k: tuple(v.shape) for k, v in sd["model"].items() if hasattr(v, "shape")},
            "args": sd["args"]}


@pytest.mark.skipif(not REAL_CKPT.exists(), reason="real production checkpoint not on this filesystem")
class TestRealCheckpointKeys:
    """Audit the REAL 1.5b MoE checkpoint (TE spec, EP4, legacy torch format).

    This is the only ground truth for the TE-vs-local key differences, and the only place the
    exporter's arg audit meets a real training Namespace.
    """

    def test_moe_layers_use_a_standalone_pre_mlp_layernorm(self, real):
        # The TE spec cannot fuse a norm into a MoE layer's mlp, so it keeps pre_mlp_layernorm --
        # exactly what exporter/mapping.py expects, and what the local-spec dump above renames.
        assert "decoder.layers.1.pre_mlp_layernorm.weight" in real["keys"]
        assert "decoder.layers.1.mlp.linear_fc1.layer_norm_weight" not in real["keys"]

    def test_dense_layer_fuses_its_mlp_pre_norm(self, real):
        assert "decoder.layers.0.mlp.linear_fc1.layer_norm_weight" in real["keys"]
        assert "decoder.layers.0.pre_mlp_layernorm.weight" not in real["keys"]

    def test_fused_qkv_row_count_matches_the_mapping_formula(self, real):
        args = real["args"]
        head_dim = args.kv_channels or args.hidden_size // args.num_attention_heads
        expected = (args.num_attention_heads + 2 * args.num_query_groups) * head_dim
        assert real["keys"]["decoder.layers.0.self_attention.linear_qkv.weight"] == (
            expected, args.hidden_size,
        )

    def test_vocab_is_unpadded(self, real):
        rows = real["keys"]["embedding.word_embeddings.weight"][0]
        assert rows == real["args"].padded_vocab_size == 200064

    def test_derive_config_accepts_the_real_training_args(self, real):
        # The exporter's arg audit, run against a REAL Namespace produced by a real training run.
        derived = derive_config(real["args"])
        kwargs = derived.kwargs
        assert kwargs["hidden_size"] == 768
        assert kwargs["num_hidden_layers"] == 10
        assert kwargs["n_routed_experts"] == 128
        assert kwargs["num_experts_per_tok"] == 4
        assert kwargs["first_k_dense_replace"] == 1
        assert kwargs["moe_intermediate_size"] == 448
        assert kwargs["n_shared_experts"] == 1
        assert kwargs["vocab_size"] == 200064
        assert kwargs["head_dim"] == 128
        assert kwargs["num_key_value_heads"] == 3
        assert kwargs["routed_scaling_factor"] == 2.5
        assert kwargs["rope_parameters"]["rope_theta"] == 500000.0
        # this run: expert-bias router, no sandwich, no latent, no QB
        assert derived.expert_bias_present is True
        assert kwargs["sandwich_norm"] is False
        assert kwargs["moe_latent_size"] is None
        assert kwargs["use_quantile_balancing"] is False
        # both scalar multipliers were ON in this run
        assert kwargs["embedding_multiplier"] == pytest.approx(768 ** 0.5)
        assert kwargs["residual_multiplier"] == pytest.approx(1.0 / (2 * 10) ** 0.5)


@pytest.fixture(scope="module")
def real_kda():
    """Args + tensor metadata of the real KDA smoke checkpoint, via the production reader."""
    from exporter.reader import load_args, load_metadata

    args, _ = load_args(REAL_KDA_CKPT)
    return {"args": args, "meta": load_metadata(REAL_KDA_CKPT)}


@pytest.fixture(scope="module")
def real_kda_plan(real_kda):
    derived = derive_config(real_kda["args"])
    return derived, build_plan(derived.kwargs, expert_bias_present=derived.expert_bias_present)


# Every tensor the fork writes for one KDA layer (TE spec). The in_proj/conv1d entries are the
# pre-split per-component keys from KimiDeltaAttention._in_proj_sharded_split; note there is NO
# unsplit in_proj.weight, NO decay_out_proj.bias, and the output gate DOES carry a bias -- the
# fork's one deviation from Kimi-Linear, which the mapping must forward.
KDA_LAYER_KEYS = {
    "in_proj.layer_norm_weight",
    "in_proj.weight.query",
    "in_proj.weight.key",
    "in_proj.weight.value",
    "in_proj.weight.decay_low_rank",
    "in_proj.weight.gate_low_rank",
    "in_proj.weight.beta",
    "conv1d.weight.query",
    "conv1d.weight.key",
    "conv1d.weight.value",
    "A_log",
    "dt_bias",
    "decay_out_proj.weight",
    "gate_out_proj.weight",
    "gate_out_proj.bias",
    "out_norm.weight",
    "out_proj.weight",
}


@pytest.mark.skipif(not REAL_KDA_CKPT.exists(), reason="real KDA smoke checkpoint not on this filesystem")
class TestRealKdaCheckpointKeys:
    """Audit the REAL KDA smoke checkpoint (TE spec, torch_dist) against the exporter.

    megatron_mock.kda_attention_key_triples and mapping._kda_attention_rows were hand-written in
    the same commits, so a shared wrong assumption is invisible to the mock round-trip tests --
    exactly the 2026-07-14 failure mode this module exists for. This class is the KDA ground
    truth: the on-disk namespace a real fork training run produced.
    """

    def test_derive_config_accepts_the_real_kda_training_args(self, real_kda_plan):
        derived, _ = real_kda_plan
        kwargs = derived.kwargs
        kda, full = "linear_attention", "full_attention"
        assert kwargs["layer_types"] == [kda, kda, kda, full, kda, kda, kda, full, kda, full]
        assert kwargs["linear_num_key_heads"] == 6
        assert kwargs["linear_num_value_heads"] == 6
        assert kwargs["linear_key_head_dim"] == 128
        assert kwargs["linear_value_head_dim"] == 128
        assert kwargs["linear_conv_kernel_dim"] == 4
        assert kwargs["gate_lower_bound"] == -5.0
        assert kwargs["linear_attn_output_gate_bias"] is True  # fork always trains the bias
        # this run: quantile-balancing router without expert bias
        assert derived.expert_bias_present is False

    def test_plan_bijects_the_real_kda_checkpoint(self, real_kda, real_kda_plan):
        # The claim the KDA mapping commit makes: exact bijection of all 241 model tensors.
        _, plan = real_kda_plan
        model_keys = {k for k in real_kda["meta"] if not k.startswith("optimizer.")}
        expected = {row.megatron_key for row in plan.rows}
        missing = sorted(model_keys - expected)  # fork writes it, we would not consume it
        extra = sorted(expected - model_keys)  # we demand it, fork does not write it
        assert not missing, f"real KDA checkpoint has keys the exporter does not map: {missing}"
        assert not extra, f"exporter expects keys the real KDA checkpoint lacks: {extra}"
        assert len(model_keys) == 241

    def test_plan_shapes_match_the_real_kda_checkpoint(self, real_kda, real_kda_plan):
        _, plan = real_kda_plan
        meta = real_kda["meta"]
        for row in plan.rows:
            assert tuple(meta[row.megatron_key].global_shape) == tuple(row.shape), (
                f"{row.megatron_key}: checkpoint wrote "
                f"{tuple(meta[row.megatron_key].global_shape)}, mapping expects {tuple(row.shape)}"
            )

    def test_kda_layer_writes_exactly_the_seventeen_expected_keys(self, real_kda):
        prefix = "decoder.layers.0.self_attention."
        on_disk = {k.removeprefix(prefix) for k in real_kda["meta"] if k.startswith(prefix)}
        assert on_disk == KDA_LAYER_KEYS

    def test_softmax_layers_keep_the_standard_attention_namespace(self, real_kda):
        keys = set(real_kda["meta"])
        assert "decoder.layers.3.self_attention.linear_qkv.weight" in keys
        assert not any(
            k.startswith("decoder.layers.3.self_attention.") and k.split(".", 4)[-1] in KDA_LAYER_KEYS
            for k in keys
        )

    def test_kda_tensors_are_bf16_on_disk(self, real_kda):
        # The fork constructs A_log/dt_bias as fp32 (kimi_delta_attention.py) but Float16Module
        # downcasts the whole module, so on disk every KDA tensor is uniformly bf16 -- the fact
        # that lets the mapping copy KDA rows without dtype pins. The only fp32 survivors are the
        # router's qb_beta buffers, which have an explicit fp32-restore hook.
        non_bf16 = {
            k: str(meta.dtype)
            for k, meta in real_kda["meta"].items()
            if not k.startswith("optimizer.") and meta.dtype != torch.bfloat16
        }
        assert all(k.endswith("mlp.router.qb_beta") for k in non_bf16), non_bf16
        assert set(non_bf16.values()) == {"torch.float32"}
        assert len(non_bf16) == 9
