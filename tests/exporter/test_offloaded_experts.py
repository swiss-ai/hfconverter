"""Layout tests for the offloaded-expert weight variant (chonk-SWA gap 5).

INDEPENDENCE RULE (as in test_transforms.py): the expected outputs here are built BY HAND
from the fork spec, never from the exporter's own helpers.

``OffloadingExpertsMLP`` keeps a fused ``(num_local_experts, out, in)`` master in memory but
persists each expert TRANSPOSED, as ``(in, out)``, under ``experts.weight1`` / ``weight2``
instead of ``experts.linear_fc1.weight`` / ``linear_fc2.weight``. The fork's own ``merge_fn``
states the round trip exactly (``fp8_utils.py``): "list of (in, out) -> fused
(num_local_experts, out, in)", rebuilt with ``s.transpose(0, 1)``.

Why this file exists at all: a forgotten transpose is INVISIBLE to every other check.
``split_gated_fc1`` only rejects an odd dimension 0, and the untransposed source's dimension 0
is the (even) latent size; the resulting wrong output shapes have exactly the same element
count as the right ones, so the shard planner and its byte accounting raise nothing. The
geometry below therefore uses three pairwise-distinct dimensions, and the values are chosen so
that a transpose, a gate/up swap, and an expert mix-up each produce a different answer.
"""

import pytest

pytest.importorskip("megatron.core")

import torch  # noqa: E402

from exporter import mapping  # noqa: E402

import megatron_mock  # noqa: E402

# Pairwise distinct on purpose: with in == out a transpose error survives every shape check.
EXPERTS = 3
LATENT_IN = 4  # expert input width  (moe_latent_size)
EXPERT_FFN = 5  # expert hidden width  (moe_ffn_hidden_size)


def build_config(**overrides):
    """Tiny HF config kwargs with a latent MoE, so expert input width != hidden size."""
    config = megatron_mock.tiny_export_config(False, LATENT_IN, False)
    args = megatron_mock.make_args_namespace(
        config, expert_bias_present=True, moe_ffn_hidden_size=EXPERT_FFN,
        moe_shared_expert_intermediate_size=EXPERT_FFN, num_experts=EXPERTS,
        moe_router_topk=2, moe_latent_size=LATENT_IN, **overrides,
    )
    from exporter.config_from_args import derive_config

    return derive_config(args)


def build_weight1(dtype=torch.float32):
    """On-disk (E, in, 2F). Element (e, i, j) is uniquely decodable from its value."""
    values = torch.arange(EXPERTS * LATENT_IN * 2 * EXPERT_FFN, dtype=dtype)
    return values.reshape(EXPERTS, LATENT_IN, 2 * EXPERT_FFN)


def build_weight2(dtype=torch.float32):
    """On-disk (E, F, in)."""
    values = torch.arange(EXPERTS * EXPERT_FFN * LATENT_IN, dtype=dtype)
    return values.reshape(EXPERTS, EXPERT_FFN, LATENT_IN)


class TestDerivedLayoutSelection:
    def test_offloading_selects_the_fused_transposed_keys(self):
        derived = build_config(moe_use_offloading_experts=True)
        plan = mapping.build_plan(derived.kwargs, True, derived.offloaded_experts)
        sources = {row.megatron_key for row in plan.rows}
        assert any(k.endswith("mlp.experts.experts.weight1") for k in sources)
        assert any(k.endswith("mlp.experts.experts.weight2") for k in sources)
        assert not any(k.endswith("mlp.experts.experts.linear_fc1.weight") for k in sources)

    def test_grouped_layout_is_still_the_default(self):
        derived = build_config()
        plan = mapping.build_plan(derived.kwargs, True, derived.offloaded_experts)
        sources = {row.megatron_key for row in plan.rows}
        assert any(k.endswith("mlp.experts.experts.linear_fc1.weight") for k in sources)
        assert not any(k.endswith("mlp.experts.experts.weight1") for k in sources)

    def test_row_shape_is_the_on_disk_shape(self):
        # validate_metadata compares Row.shape against the checkpoint's global shape, so it must
        # describe the file, not the nn.Linear orientation the outputs end up in.
        derived = build_config(moe_use_offloading_experts=True)
        plan = mapping.build_plan(derived.kwargs, True, derived.offloaded_experts)
        by_key = {row.megatron_key: row for row in plan.rows}
        fc1 = next(r for k, r in by_key.items() if k.endswith("weight1"))
        fc2 = next(r for k, r in by_key.items() if k.endswith("weight2"))
        assert fc1.shape == (EXPERTS, LATENT_IN, 2 * EXPERT_FFN)
        assert fc2.shape == (EXPERTS, EXPERT_FFN, LATENT_IN)

    def test_planned_output_shapes_stay_nn_linear_shaped(self):
        # The regression this guards: plan_hf_tensors unpacks Row.shape positionally. With the
        # transposed row shape and the old unpacking, gate/up would be planned as (in/2, 2F) and
        # down as (F, in) -- both with the SAME element count as the correct shapes, so the
        # shard layout and its byte totals would be identical and nothing would complain until
        # produce_one's final assertion, long after the output directory was claimed.
        derived = build_config(moe_use_offloading_experts=True)
        plan = mapping.build_plan(derived.kwargs, True, derived.offloaded_experts)
        specs = {s.hf_key: s for s in mapping.plan_hf_tensors(plan, torch.float32)}
        layer = derived.kwargs["first_k_dense_replace"]
        for expert in range(EXPERTS):
            base = f"model.layers.{layer}.mlp.experts.{expert}"
            assert specs[f"{base}.gate_proj.weight"].shape == (EXPERT_FFN, LATENT_IN)
            assert specs[f"{base}.up_proj.weight"].shape == (EXPERT_FFN, LATENT_IN)
            assert specs[f"{base}.down_proj.weight"].shape == (LATENT_IN, EXPERT_FFN)


class TestTransposedConversion:
    """Hand-computed reference for every produced tensor."""

    @staticmethod
    def _converted(dtype=torch.float32):
        derived = build_config(moe_use_offloading_experts=True)
        plan = mapping.build_plan(derived.kwargs, True, derived.offloaded_experts)
        tensors = {}
        for row in plan.rows:
            if row.megatron_key.endswith("weight1"):
                tensors[row.megatron_key] = build_weight1(dtype)
            elif row.megatron_key.endswith("weight2"):
                tensors[row.megatron_key] = build_weight2(dtype)
            else:
                tensors[row.megatron_key] = torch.zeros(row.shape, dtype=row.dtype or dtype)
        return derived, mapping.convert(plan, tensors)

    def test_gate_and_up_match_a_hand_built_reference(self):
        derived, out = self._converted()
        layer = derived.kwargs["first_k_dense_replace"]
        source = build_weight1()
        for expert in range(EXPERTS):
            # The fork computes y = x @ weight1[e] then chunk(y, 2, dim=-1): the FIRST half of
            # the output columns is the gate. Columns become rows under the transpose.
            expert_matrix = source[expert]                      # (in, 2F)
            expected_gate = expert_matrix[:, :EXPERT_FFN].T      # (F, in)
            expected_up = expert_matrix[:, EXPERT_FFN:].T        # (F, in)
            base = f"model.layers.{layer}.mlp.experts.{expert}"
            torch.testing.assert_close(out[f"{base}.gate_proj.weight"], expected_gate,
                                       rtol=0.0, atol=0.0)
            torch.testing.assert_close(out[f"{base}.up_proj.weight"], expected_up,
                                       rtol=0.0, atol=0.0)
            assert not torch.equal(expected_gate, expected_up), "fixture cannot detect a swap"

    def test_down_projection_matches_a_hand_built_reference(self):
        derived, out = self._converted()
        layer = derived.kwargs["first_k_dense_replace"]
        source = build_weight2()
        for expert in range(EXPERTS):
            expected = source[expert].T  # (F, in) -> nn.Linear (in, F)
            key = f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"
            torch.testing.assert_close(out[key], expected, rtol=0.0, atol=0.0)

    def test_a_forgotten_transpose_would_change_the_values(self):
        # Proves the fixture is a real discriminator rather than accidentally symmetric.
        derived, out = self._converted()
        layer = derived.kwargs["first_k_dense_replace"]
        untransposed_gate = build_weight1()[0][:, :EXPERT_FFN]
        produced = out[f"model.layers.{layer}.mlp.experts.0.gate_proj.weight"]
        assert produced.shape != untransposed_gate.shape

    def test_experts_are_not_mixed_up(self):
        derived, out = self._converted()
        layer = derived.kwargs["first_k_dense_replace"]
        keys = [f"model.layers.{layer}.mlp.experts.{e}.down_proj.weight" for e in range(EXPERTS)]
        produced = [out[k] for k in keys]
        for first in range(EXPERTS):
            for second in range(first + 1, EXPERTS):
                assert not torch.equal(produced[first], produced[second])

    def test_dtype_is_preserved(self):
        _, out = self._converted(dtype=torch.bfloat16)
        expert_keys = [k for k in out if ".mlp.experts." in k and k.endswith("_proj.weight")]
        assert expert_keys
        assert all(out[k].dtype is torch.bfloat16 for k in expert_keys)


class TestKeyScannerStaysStrict:
    """Offloaded keys are supported only when the args said to expect them."""

    def test_unexpected_offloaded_keys_are_still_rejected(self):
        with pytest.raises(ValueError, match="OffloadingExpertsMLP"):
            mapping.assert_no_unsupported(
                {"decoder.layers.1.mlp.experts.experts.weight1"}, offloaded_experts=False
            )

    def test_expected_offloaded_keys_are_accepted(self):
        mapping.assert_no_unsupported(
            {"decoder.layers.1.mlp.experts.experts.weight1"}, offloaded_experts=True
        )

    def test_other_absent_features_are_unaffected(self):
        with pytest.raises(ValueError, match="shared expert gate"):
            mapping.assert_no_unsupported(
                {"decoder.layers.1.mlp.shared_experts.gate_weight"}, offloaded_experts=True
            )
