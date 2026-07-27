from __future__ import annotations

from argparse import Namespace

import pytest

from cluster.validate_source_profile import validate


def saved_profile() -> dict:
    return {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "expert_model_parallel_size": 4,
        "expert_tensor_parallel_size": 1,
        "context_parallel_size": 1,
        "bf16": True,
        "fp16": False,
        "moe_router_load_balancing_type": "seq_aux_loss",
        "moe_aux_loss_coeff": 1e-4,
        "init_method_std": 0.0360844,
        "layernorm_epsilon": 1e-5,
        "sandwich_norm": False,
        "moe_router_dtype": "fp32",
        "transformer_impl": "transformer_engine",
    }


def requested_profile(**overrides) -> Namespace:
    values = {
        "src_tp": 1,
        "src_pp": 1,
        "src_ep": 4,
        "src_etp": 1,
        "src_cp": 1,
        "precision": "bf16",
        "routing_type": "seq_aux_loss",
        "moe_aux_loss_coeff": "1e-4",
        "init_method_std": "0.0360844",
        "norm_epsilon": "1e-5",
        "sandwich_norm": "0",
    }
    values.update(overrides)
    return Namespace(**values)


def test_matching_profile_passes():
    validate(saved_profile(), requested_profile())


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"src_ep": 2}, "expert_model_parallel_size"),
        ({"precision": "fp16"}, "precision"),
        ({"routing_type": "aux_loss"}, "moe_router_load_balancing_type"),
        ({"moe_aux_loss_coeff": "1e-3"}, "moe_aux_loss_coeff"),
        ({"norm_epsilon": "1e-6"}, "layernorm_epsilon"),
        ({"sandwich_norm": "1"}, "sandwich_norm"),
    ],
)
def test_mismatch_is_named(override: dict, message: str):
    with pytest.raises(ValueError, match=message):
        validate(saved_profile(), requested_profile(**override))


def test_non_fp32_source_router_is_rejected():
    saved = saved_profile()
    saved["moe_router_dtype"] = None

    with pytest.raises(ValueError, match="fp32"):
        validate(saved, requested_profile())


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("moe_expert_capacity_factor", 1.0),
        ("layernorm_zero_centered_gamma", True),
        ("qk_l2_norm", True),
        ("moe_router_group_topk", 1),
        ("moe_apply_probs_on_input", True),
        ("use_mup", True),
        ("attention_output_gate", True),
        ("softmax_type", "learnable"),
    ],
)
def test_non_restored_active_semantics_are_rejected(name: str, value):
    saved = saved_profile()
    saved[name] = value

    with pytest.raises(ValueError, match=name):
        validate(saved, requested_profile())


def test_features_the_exporter_now_translates_are_not_rejected_here():
    # rope scaling, per-layer NoPE, and offloaded experts became supported translations on
    # 2026-07-24. This preflight compares a LEGACY checkpoint's saved args against Stage-1
    # inputs; keeping gates the exporter has dropped would refuse work it can now do, and would
    # leave two files disagreeing about what is supported.
    saved = saved_profile()
    saved.update(
        use_rope_scaling=True,
        rope_scaling_factor=2.0,
        no_rope_freq=[0, 0, 1],
        moe_use_offloading_experts=True,
    )

    validate(saved, requested_profile())
