#!/usr/bin/env python3
"""Compare user-supplied Stage-1 values with a legacy checkpoint's saved arguments.

This uses ``ckpt_args.py``. Tensor storages are skipped by its restricted unpickler, so this
preflight reads argument metadata without materializing the model in login-node RAM. Skipping
storages is a memory optimization, not a pickle security sandbox: inspect only checkpoints
whose files you trust.
"""

from __future__ import print_function

import argparse
import math
import sys

from ckpt_args import read_checkpoint_args


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _float_list(text):
    return [float(item) for item in text.split()]


def _bool(text):
    lowered = str(text).lower()
    if lowered in ("1", "true", "yes"):
        return True
    if lowered in ("0", "false", "no"):
        return False
    raise ValueError("expected a boolean, got {!r}".format(text))


def _same_number(actual, expected):
    try:
        return math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=0.0)
    except (TypeError, ValueError):
        return False


def validate(saved, requested):
    problems = []

    exact_fields = (
        ("tensor_model_parallel_size", "src_tp"),
        ("pipeline_model_parallel_size", "src_pp"),
        ("expert_model_parallel_size", "src_ep"),
        ("expert_tensor_parallel_size", "src_etp"),
        ("context_parallel_size", "src_cp"),
    )
    for checkpoint_name, request_name in exact_fields:
        actual = saved.get(checkpoint_name)
        expected = getattr(requested, request_name)
        if actual != expected:
            problems.append(
                "{}={!r}, but the command supplied {!r}".format(
                    checkpoint_name, actual, expected
                )
            )

    saved_precision = None
    if saved.get("bf16") is True and not saved.get("fp16"):
        saved_precision = "bf16"
    elif saved.get("fp16") is True and not saved.get("bf16"):
        saved_precision = "fp16"
    if saved_precision != requested.precision:
        problems.append(
            "saved precision is {!r}, but PRECISION={!r}".format(
                saved_precision, requested.precision
            )
        )

    actual_routing = [str(value) for value in _as_list(
        saved.get("moe_router_load_balancing_type")
    )]
    expected_routing = requested.routing_type.split()
    if actual_routing != expected_routing:
        problems.append(
            "moe_router_load_balancing_type={!r}, but ROUTING_TYPE={!r}".format(
                actual_routing, expected_routing
            )
        )

    try:
        actual_aux = _as_list(saved.get("moe_aux_loss_coeff"))
        expected_aux = _float_list(requested.moe_aux_loss_coeff)
    except ValueError as exc:
        problems.append("cannot parse MOE_AUX_LOSS_COEFF: {}".format(exc))
    else:
        if len(actual_aux) != len(expected_aux) or not all(
            _same_number(actual, expected)
            for actual, expected in zip(actual_aux, expected_aux)
        ):
            problems.append(
                "moe_aux_loss_coeff={!r}, but MOE_AUX_LOSS_COEFF={!r}".format(
                    actual_aux, expected_aux
                )
            )

    for checkpoint_name, request_name in (
        ("init_method_std", "init_method_std"),
        ("layernorm_epsilon", "norm_epsilon"),
    ):
        actual = saved.get(checkpoint_name)
        expected = getattr(requested, request_name)
        if not _same_number(actual, expected):
            problems.append(
                "{}={!r}, but the command supplied {!r}".format(
                    checkpoint_name, actual, expected
                )
            )

    try:
        expected_sandwich = _bool(requested.sandwich_norm)
    except ValueError as exc:
        problems.append(str(exc))
    else:
        actual_sandwich = saved.get("sandwich_norm")
        if actual_sandwich is not expected_sandwich:
            problems.append(
                "sandwich_norm={!r}, but SANDWICH_NORM={!r}".format(
                    actual_sandwich, expected_sandwich
                )
            )

    # These are fixed by the current Stage-1/HF implementation, not user overrides.
    if saved.get("moe_router_dtype") != "fp32":
        problems.append(
            "moe_router_dtype={!r}; this exporter supports only source fp32 routers".format(
                saved.get("moe_router_dtype")
            )
        )
    if saved.get("transformer_impl", "transformer_engine") != "transformer_engine":
        problems.append(
            "transformer_impl={!r}; Stage 1 uses transformer_engine".format(
                saved.get("transformer_impl")
            )
        )

    # These forward-semantic values are not all restored by the inspected fork.  Stage 2 would
    # see Stage 1's replacement defaults rather than the legacy values, and a strict tensor load
    # might not notice because several have no parameter footprint.  Reject active values until
    # Stage 1 exposes and propagates them deliberately.
    safe_semantics = (
        (saved.get("moe_expert_capacity_factor") is None,
         "moe_expert_capacity_factor", saved.get("moe_expert_capacity_factor")),
        (not saved.get("moe_pad_expert_input_to_capacity", False),
         "moe_pad_expert_input_to_capacity", saved.get("moe_pad_expert_input_to_capacity")),
        (saved.get("moe_input_jitter_eps") is None,
         "moe_input_jitter_eps", saved.get("moe_input_jitter_eps")),
        (not saved.get("moe_router_force_load_balancing", False),
         "moe_router_force_load_balancing", saved.get("moe_router_force_load_balancing")),
        (saved.get("moe_router_force_biased") is None,
         "moe_router_force_biased", saved.get("moe_router_force_biased")),
        (not saved.get("layernorm_zero_centered_gamma", False),
         "layernorm_zero_centered_gamma", saved.get("layernorm_zero_centered_gamma")),
        (not saved.get("qk_l2_norm", False),
         "qk_l2_norm", saved.get("qk_l2_norm")),
        (not saved.get("moe_router_group_topk"),
         "moe_router_num_groups/moe_router_group_topk",
         (saved.get("moe_router_num_groups"), saved.get("moe_router_group_topk"))),
        (saved.get("moe_router_topk_limited_devices") is None,
         "moe_router_topk_limited_devices", saved.get("moe_router_topk_limited_devices")),
        (not saved.get("moe_shared_expert_gate", False),
         "moe_shared_expert_gate", saved.get("moe_shared_expert_gate")),
        (not saved.get("moe_apply_probs_on_input", False),
         "moe_apply_probs_on_input", saved.get("moe_apply_probs_on_input")),
        (not saved.get("use_mup", False),
         "use_mup", saved.get("use_mup")),
        (not saved.get("pnglu", False),
         "pnglu", saved.get("pnglu")),
        (saved.get("mtp_num_layers") in (None, 0),
         "mtp_num_layers", saved.get("mtp_num_layers")),
        (saved.get("rotary_seq_len_interpolation_factor") is None,
         "rotary_seq_len_interpolation_factor",
         saved.get("rotary_seq_len_interpolation_factor")),
        (not saved.get("apply_residual_connection_post_layernorm", False),
         "apply_residual_connection_post_layernorm",
         saved.get("apply_residual_connection_post_layernorm")),
        (not saved.get("attention_output_gate", False),
         "attention_output_gate", saved.get("attention_output_gate")),
        (saved.get("softmax_type", "vanilla") in ("vanilla", None),
         "softmax_type", saved.get("softmax_type")),
    )
    for is_safe, name, actual in safe_semantics:
        if not is_safe:
            problems.append(
                "{}={!r}; the current Stage 1 does not safely propagate this source option".format(
                    name, actual
                )
            )

    if problems:
        raise ValueError("source profile mismatch:\n  " + "\n  ".join(problems))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--src-tp", type=int, required=True)
    parser.add_argument("--src-pp", type=int, required=True)
    parser.add_argument("--src-ep", type=int, required=True)
    parser.add_argument("--src-etp", type=int, required=True)
    parser.add_argument("--src-cp", type=int, required=True)
    parser.add_argument("--precision", choices=("bf16", "fp16"), required=True)
    parser.add_argument("--routing-type", required=True)
    parser.add_argument("--moe-aux-loss-coeff", required=True)
    parser.add_argument("--init-method-std", required=True)
    parser.add_argument("--norm-epsilon", required=True)
    parser.add_argument("--sandwich-norm", required=True)
    args = parser.parse_args(argv)

    try:
        saved = read_checkpoint_args(args.checkpoint)
        validate(saved, args)
    except (OSError, ValueError) as exc:
        parser.exit(1, "FATAL: {}\n".format(exc))

    print("Source checkpoint profile matches the supplied conversion parameters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
