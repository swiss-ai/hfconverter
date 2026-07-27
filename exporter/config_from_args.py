# Copyright 2026 the Swiss AI Initiative. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Translate saved Megatron training arguments into ``ApertusMoeConfig`` values.

The checkpoint is the only source of model configuration. Before deriving Hugging Face fields, the
module checks named support boundaries: options whose tensor layout or forward-pass math cannot be
represented exactly by the current HF model. Successful checks are recorded in
``conversion_info.json`` so a colleague can audit why the conversion was accepted.
"""

import math
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, NamedTuple


class DerivedConfig(NamedTuple):
    kwargs: dict[str, Any]
    expert_bias_present: bool
    checks: list[str]
    # Training-time storage choices that change checkpoint key names or tensor layout but not the
    # exported model. They pick a mapping-table variant and deliberately never reach config.json.
    offloaded_experts: bool = False


def _req(args: Namespace, name: str) -> Any:
    if not hasattr(args, name):
        raise ValueError(f"checkpoint args missing required attribute 'args.{name}'")
    return getattr(args, name)


@dataclass
class _SupportBoundary:
    """Collect explicit compatibility decisions made before tensor loading."""

    passed: list[str] = field(default_factory=list)

    def require(self, condition: bool, description: str, actual: Any) -> None:
        if not condition:
            raise ValueError(
                f"unsupported checkpoint: expected {description}; got {actual!r}"
            )
        self.passed.append(description)


def _derive_moe_layer_freq(moe_layer_freq: Any, num_layers: int) -> tuple[list[int], int]:
    """Return Megatron's explicit per-layer dense/MoE schedule and its legacy cutoff.

    An integer ``N > 1`` means MoE at 0-indexed layers divisible by ``N`` and produces
    non-homogeneous, per-layer checkpoint keys. Integer ``1`` is deliberately different:
    Megatron writes a homogeneous checkpoint whose layer number becomes a leading tensor axis.
    This exporter maps explicit per-layer keys, so that representation remains unsupported.
    """
    if isinstance(moe_layer_freq, int):
        if moe_layer_freq == 1:
            raise ValueError(
                f"args.moe_layer_freq={moe_layer_freq!r}: makes the fork write HOMOGENEOUS "
                "checkpoint keys (no per-layer index; the layer becomes a leading tensor axis), "
                "which this exporter cannot read. Use an explicit list or an integer > 1."
            )
        if moe_layer_freq <= 0:
            raise ValueError(
                f"args.moe_layer_freq={moe_layer_freq!r}: integer frequency must be positive"
            )
        pattern = [1 if index % moe_layer_freq == 0 else 0 for index in range(num_layers)]
    else:
        try:
            pattern = list(moe_layer_freq)
        except TypeError as exc:
            raise ValueError(
                f"args.moe_layer_freq={moe_layer_freq!r}: expected an integer or a 0/1 list"
            ) from exc

    if len(pattern) != num_layers:
        raise ValueError(
            f"args.moe_layer_freq={pattern!r}: expected exactly {num_layers} entries"
        )
    if any(entry not in (0, 1) for entry in pattern):
        raise ValueError(
            f"args.moe_layer_freq={pattern!r}: entries must be 0 (dense) or 1 (MoE)"
        )

    pattern = [int(entry) for entry in pattern]
    leading_dense = 0
    while leading_dense < num_layers and pattern[leading_dense] == 0:
        leading_dense += 1
    return pattern, leading_dense


# Megatron picks the activation from exactly one of these flags (the --swiglu ... --pn3glu
# dispatch chain in megatron/training/arguments.py). Only the two mapped here have an HF
# activation with identical math; the rest are listed so a rejection can name the flag it found
# instead of failing later as a silent SiLU model.
_GLU_FLAG_TO_HIDDEN_ACT = {"swiglu": "silu", "sssglu": "sssglu"}
_UNMAPPED_ACTIVATION_FLAGS = (
    "squared_relu", "quick_geglu", "ssglu", "reglu", "rlglu", "lglu", "situ", "pnglu",
    "gxpr", "gxpry", "gxprv2", "gxr2", "xpr", "xr2", "xr2glu", "xssglu", "pn3glu",
)


def _derive_hidden_act(args: Namespace, support: _SupportBoundary) -> str:
    """Resolve Megatron's activation flags to one ``hidden_act`` string.

    The flags are mutually exclusive in the fork, which asserts as much while building its
    config. Requiring exactly one here turns a future flag combination into a named error rather
    than a model whose weights and activation disagree.
    """
    enabled = [flag for flag in _GLU_FLAG_TO_HIDDEN_ACT if getattr(args, flag, False)]
    unmapped = [flag for flag in _UNMAPPED_ACTIVATION_FLAGS if getattr(args, flag, False)]
    support.require(
        not unmapped,
        "no unmapped Megatron activation flag is set (only --swiglu and --sssglu have an "
        "exactly equivalent Hugging Face activation)",
        unmapped,
    )
    support.require(
        len(enabled) == 1,
        "exactly one of args.swiglu / args.sssglu is set",
        enabled,
    )
    hidden_act = _GLU_FLAG_TO_HIDDEN_ACT[enabled[0]]
    support.passed.append(f"args.{enabled[0]} -> hidden_act = {hidden_act!r}")
    return hidden_act


def _validate_base_architecture(args: Namespace, support: _SupportBoundary) -> str:
    """Validate choices that determine parameter types and checkpoint key names."""
    normalization = _req(args, "normalization")
    support.require(
        normalization == "RMSNorm", "args.normalization == 'RMSNorm'", normalization
    )
    hidden_act = _derive_hidden_act(args, support)
    glu_linear_offset = getattr(args, "glu_linear_offset", 0.0)
    support.require(
        glu_linear_offset == 0.0, "args.glu_linear_offset == 0.0", glu_linear_offset
    )
    activation_clamp = getattr(args, "activation_func_clamp_value", None)
    support.require(
        activation_clamp is None,
        "args.activation_func_clamp_value is None",
        activation_clamp,
    )
    add_bias_linear = _req(args, "add_bias_linear")
    support.require(not add_bias_linear, "args.add_bias_linear is falsy", add_bias_linear)
    add_qkv_bias = _req(args, "add_qkv_bias")
    support.require(not add_qkv_bias, "args.add_qkv_bias is falsy", add_qkv_bias)

    # The local implementation persists its pre-MLP norm under a different key namespace.
    transformer_impl = getattr(args, "transformer_impl", "transformer_engine")
    support.require(
        transformer_impl == "transformer_engine",
        "args.transformer_impl == 'transformer_engine'",
        transformer_impl,
    )
    multi_latent_attention = _req(args, "multi_latent_attention")
    support.require(
        not multi_latent_attention,
        "args.multi_latent_attention is falsy",
        multi_latent_attention,
    )
    mtp_num_layers = getattr(args, "mtp_num_layers", None)
    support.require(
        mtp_num_layers in (None, 0), "args.mtp_num_layers in (None, 0)", mtp_num_layers
    )
    pnglu = getattr(args, "pnglu", False)
    support.require(not pnglu, "args.pnglu is falsy", pnglu)
    use_mup = getattr(args, "use_mup", False)
    support.require(not use_mup, "args.use_mup is falsy", use_mup)

    untied = _req(args, "untie_embeddings_and_output_weights")
    support.require(
        bool(untied),
        "args.untie_embeddings_and_output_weights is truthy (tie_word_embeddings = False)",
        untied,
    )
    return hidden_act


def _validate_position_and_norm_math(args: Namespace, support: _SupportBoundary) -> None:
    """Validate semantic flags that may leave no tensor-key evidence of changed model math."""
    position_type = _req(args, "position_embedding_type")
    support.require(
        position_type == "rope",
        "args.position_embedding_type == 'rope'",
        position_type,
    )
    rotary_percent = _req(args, "rotary_percent")
    support.require(
        rotary_percent == 1.0, "args.rotary_percent == 1.0", rotary_percent
    )
    rotary_interleaved = _req(args, "rotary_interleaved")
    support.require(
        not rotary_interleaved, "args.rotary_interleaved is falsy", rotary_interleaved
    )

    # These flags are legacy-path-only in the pinned fork. Keeping them disabled prevents a future
    # fork wiring change from silently changing semantics without changing checkpoint keys.
    interpolation = getattr(args, "rotary_seq_len_interpolation_factor", None)
    support.require(
        interpolation is None,
        "args.rotary_seq_len_interpolation_factor is None (defensive: legacy-path-only flag)",
        interpolation,
    )
    post_layernorm_residual = getattr(
        args, "apply_residual_connection_post_layernorm", False
    )
    support.require(
        not post_layernorm_residual,
        "args.apply_residual_connection_post_layernorm is falsy (defensive: legacy-path-only)",
        post_layernorm_residual,
    )

    zero_centered_gamma = getattr(args, "layernorm_zero_centered_gamma", False)
    support.require(
        not zero_centered_gamma,
        "args.layernorm_zero_centered_gamma is falsy",
        zero_centered_gamma,
    )
    qk_l2_norm = getattr(args, "qk_l2_norm", False)
    support.require(not qk_l2_norm, "args.qk_l2_norm is falsy", qk_l2_norm)


# Megatron's llama3 scaling passes only `factor`; the other three parameters keep the defaults
# hardcoded in RotaryEmbedding._apply_scaling. They are NOT arguments, and must not be sourced
# from args: original_max_position_embeddings is 8192 while this run's max_position_embeddings is
# 32768, and reading the wrong one silently moves every frequency band.
_LLAMA3_LOW_FREQ_FACTOR = 1.0
_LLAMA3_HIGH_FREQ_FACTOR = 4.0
_LLAMA3_ORIGINAL_MAX_POSITION_EMBEDDINGS = 8192


def _derive_rope_parameters(args: Namespace, support: _SupportBoundary) -> dict[str, Any]:
    """Build ``rope_parameters``, translating Megatron's optional llama3 frequency scaling.

    Megatron's ``_apply_scaling`` was adapted from the Hugging Face implementation and cites it,
    so an active scaling maps onto ``rope_type='llama3'`` exactly. Without scaling the dictionary
    stays ``'default'``, keeping every previously exported config byte-identical.
    """
    rope_parameters: dict[str, Any] = {
        "rope_type": "default",
        "rope_theta": float(_req(args, "rotary_base")),
        # Apertus rotates the full head; some related HF configs default this to 0.5.
        "partial_rotary_factor": 1.0,
    }
    use_rope_scaling = getattr(args, "use_rope_scaling", False)
    rope_scaling_factor = getattr(args, "rope_scaling_factor", None)
    if not use_rope_scaling or rope_scaling_factor in (None, 1.0):
        support.passed.append(
            "rope scaling off (args.use_rope_scaling falsy or args.rope_scaling_factor == 1.0)"
        )
        return rope_parameters

    support.require(
        isinstance(rope_scaling_factor, (int, float)) and rope_scaling_factor > 0,
        "args.rope_scaling_factor is a positive number when args.use_rope_scaling is set",
        rope_scaling_factor,
    )
    rope_parameters.update(
        rope_type="llama3",
        factor=float(rope_scaling_factor),
        low_freq_factor=_LLAMA3_LOW_FREQ_FACTOR,
        high_freq_factor=_LLAMA3_HIGH_FREQ_FACTOR,
        original_max_position_embeddings=_LLAMA3_ORIGINAL_MAX_POSITION_EMBEDDINGS,
    )
    support.passed.append(
        f"args.use_rope_scaling with factor {float(rope_scaling_factor)} -> rope_type 'llama3' "
        f"(low {_LLAMA3_LOW_FREQ_FACTOR}, high {_LLAMA3_HIGH_FREQ_FACTOR}, original context "
        f"{_LLAMA3_ORIGINAL_MAX_POSITION_EMBEDDINGS}; the fork hardcodes these three)"
    )
    return rope_parameters


def _derive_no_rope_layers(
    args: Namespace, support: _SupportBoundary, num_layers: int
) -> list[int]:
    """Translate Megatron's ``no_rope_freq`` into HF's ``no_rope_layers``, inverting polarity.

    Megatron marks the layers that SKIP the rotation; Hugging Face (SmolLM3, Llama4) marks the
    layers that KEEP it. An un-inverted list loads cleanly and generates fluent text while every
    layer rotates the wrong way, so the inversion happens here, once.
    """
    no_rope_freq = getattr(args, "no_rope_freq", None)
    if not no_rope_freq:
        support.passed.append("args.no_rope_freq is unset: every layer applies RoPE")
        return [1] * num_layers

    if isinstance(no_rope_freq, int):
        # TransformerConfig expands an integer N to ([0]*(N-1) + [1]) repeated, i.e. NoPE on
        # every Nth 1-indexed layer, and asserts the divisibility below.
        support.require(
            no_rope_freq >= 1 and num_layers % no_rope_freq == 0,
            "args.no_rope_freq is a positive integer dividing args.num_layers",
            (no_rope_freq, num_layers),
        )
        skips_rope = ([0] * (no_rope_freq - 1) + [1]) * (num_layers // no_rope_freq)
    else:
        skips_rope = list(no_rope_freq)
        support.require(
            len(skips_rope) == num_layers,
            "len(args.no_rope_freq) == args.num_layers",
            (len(skips_rope), num_layers),
        )

    no_rope_layers = [0 if skip else 1 for skip in skips_rope]
    nope_layer_indices = [index for index, rope in enumerate(no_rope_layers) if not rope]
    support.passed.append(
        f"args.no_rope_freq -> no_rope_layers (inverted): {len(nope_layer_indices)} NoPE layers "
        f"at 0-indexed {nope_layer_indices}"
    )
    return no_rope_layers


def _derive_attention_window(
    args: Namespace, support: _SupportBoundary, num_layers: int
) -> tuple[int | None, list[str]]:
    """Translate Megatron's sliding-window settings into ``sliding_window`` and ``layer_types``.

    Megatron's ``window_size`` is inclusive at both ends: with ``(w, 0)`` a query attends to
    itself plus ``w`` earlier tokens. Hugging Face's ``sliding_window`` counts admitted keys.
    The window is therefore ``w + 1``, and writing ``w`` would silently drop one key per query.
    """
    window_size = getattr(args, "window_size", None)
    if not window_size:
        support.passed.append("args.window_size is unset: every layer uses full attention")
        return None, ["full_attention"] * num_layers

    support.require(
        len(window_size) == 2,
        "args.window_size is a (left, right) pair",
        window_size,
    )
    left_context, right_context = window_size
    support.require(
        right_context == 0,
        "args.window_size right context == 0 (a causal decoder cannot look ahead)",
        window_size,
    )
    support.require(
        isinstance(left_context, int) and left_context >= 0,
        "args.window_size left context is a non-negative integer",
        left_context,
    )
    sliding_window = left_context + 1

    # is_layer_window_attention accepts three forms; the fork treats a missing skip frequency as
    # "every layer slides", so an unset value must not fall through to full attention.
    skip_freq = getattr(args, "window_attn_skip_freq", None)
    if skip_freq is None:
        layer_types = ["sliding_attention"] * num_layers
    elif isinstance(skip_freq, int):
        support.require(
            skip_freq >= 1,
            "args.window_attn_skip_freq is a positive integer when given as an int",
            skip_freq,
        )
        # The fork slides where layer_number % freq != 0, with layer_number 1-indexed.
        layer_types = [
            "full_attention" if (index + 1) % skip_freq == 0 else "sliding_attention"
            for index in range(num_layers)
        ]
    else:
        skip_freq = list(skip_freq)
        support.require(
            len(skip_freq) == num_layers,
            "len(args.window_attn_skip_freq) == args.num_layers",
            (len(skip_freq), num_layers),
        )
        # The list is 0-indexed and 1 means "this layer uses the sliding window".
        layer_types = ["sliding_attention" if slides else "full_attention" for slides in skip_freq]

    full_layer_indices = [i for i, kind in enumerate(layer_types) if kind == "full_attention"]
    support.passed.append(
        f"args.window_size {tuple(window_size)} -> sliding_window = {sliding_window} (inclusive "
        f"of the query's own position); full attention at 0-indexed {full_layer_indices}"
    )
    return sliding_window, layer_types


def _validate_residual_scheme(
    args: Namespace, support: _SupportBoundary
) -> tuple[bool, Any]:
    """Validate supported post-norm/residual schemes and return KEEL settings."""
    keel = bool(_req(args, "keel"))
    keel_alpha = getattr(args, "keel_alpha", None)
    sandwich_norm = bool(_req(args, "sandwich_norm"))
    residual_output_scaling = bool(_req(args, "residual_output_scaling"))
    fp32_residual = getattr(args, "fp32_residual_connection", False)
    support.require(
        not fp32_residual, "args.fp32_residual_connection is falsy", fp32_residual
    )
    support.require(
        not (keel and sandwich_norm),
        "not (args.keel and args.sandwich_norm) (KEEL is mutually exclusive with sandwich_norm)",
        (keel, sandwich_norm),
    )
    support.require(
        not (keel and residual_output_scaling),
        "not (args.keel and args.residual_output_scaling) (the fork forbids KEEL with "
        "--residual-output-scaling; KEEL carries the residual by keel_alpha)",
        (keel, residual_output_scaling),
    )
    return keel, keel_alpha


def _validate_router_and_attention(args: Namespace, support: _SupportBoundary) -> None:
    """Validate routing and attention behavior represented by the current HF implementation."""
    score_function = _req(args, "moe_router_score_function")
    support.require(
        score_function == "sigmoid",
        "args.moe_router_score_function == 'sigmoid'",
        score_function,
    )

    # The HF router computes logits in fp32. A Megatron checkpoint trained with activation-dtype
    # routing can select different experts near score ties without changing any tensor or key.
    router_dtype = getattr(args, "moe_router_dtype", None)
    support.require(
        router_dtype == "fp32",
        "args.moe_router_dtype == 'fp32' (the HF router computes logits in fp32)",
        router_dtype,
    )
    limited_devices = getattr(args, "moe_router_topk_limited_devices", None)
    support.require(
        limited_devices is None,
        "args.moe_router_topk_limited_devices is None",
        limited_devices,
    )
    shared_expert_gate = getattr(args, "moe_shared_expert_gate", False)
    support.require(
        not shared_expert_gate, "args.moe_shared_expert_gate is falsy", shared_expert_gate
    )
    expert_count = _req(args, "num_experts")
    support.require(expert_count is not None, "args.num_experts is not None", expert_count)

    # These switches alter inference values without adding a distinctive checkpoint tensor.
    # Fail closed instead of producing an apparently complete but mathematically different HF
    # model. In particular, apply_probs_on_input is valid in Megatron for top-k 1 but cannot be
    # moved through a nonlinear SwiGLU expert to the HF output side.
    apply_probs_on_input = getattr(args, "moe_apply_probs_on_input", False)
    support.require(
        not apply_probs_on_input,
        "args.moe_apply_probs_on_input is falsy",
        apply_probs_on_input,
    )
    input_jitter = getattr(args, "moe_input_jitter_eps", None)
    support.require(
        input_jitter is None,
        "args.moe_input_jitter_eps is None",
        input_jitter,
    )
    force_load_balancing = getattr(args, "moe_router_force_load_balancing", False)
    support.require(
        not force_load_balancing,
        "args.moe_router_force_load_balancing is falsy",
        force_load_balancing,
    )
    force_biased = getattr(args, "moe_router_force_biased", None)
    support.require(
        force_biased is None,
        "args.moe_router_force_biased is None",
        force_biased,
    )
    capacity_factor = getattr(args, "moe_expert_capacity_factor", None)
    support.require(
        capacity_factor is None,
        "args.moe_expert_capacity_factor is None (the HF router is drop-free)",
        capacity_factor,
    )

    attention_output_gate = getattr(args, "attention_output_gate", False)
    support.require(
        not attention_output_gate,
        "args.attention_output_gate is falsy",
        attention_output_gate,
    )
    softmax_scale = getattr(args, "softmax_scale", None)
    support.require(softmax_scale is None, "args.softmax_scale is None", softmax_scale)
    query_key_layer_scaling = getattr(args, "apply_query_key_layer_scaling", False)
    support.require(
        not query_key_layer_scaling,
        "args.apply_query_key_layer_scaling is falsy",
        query_key_layer_scaling,
    )
    softmax_type = getattr(args, "softmax_type", "vanilla")
    support.require(
        softmax_type in ("vanilla", None),
        "args.softmax_type in ('vanilla', None)",
        softmax_type,
    )


def _derive_group_limits(
    args: Namespace,
    support: _SupportBoundary,
    *,
    expert_count: int,
    topk: int,
    use_quantile_balancing: bool,
) -> tuple[int, int]:
    """Translate Megatron's optional group-limited routing settings.

    Megatron enables this path only when ``moe_router_group_topk`` is truthy. A group count by
    itself therefore has no effect and is normalized to HF's ordinary-routing values ``(1, 1)``.
    When the path is active, the HF router implements the same contiguous expert groups and the
    same ``topk // group_topk`` group score.
    """
    router_groups = _req(args, "moe_router_num_groups")
    group_topk = _req(args, "moe_router_group_topk")
    if use_quantile_balancing:
        support.require(
            router_groups is None and group_topk is None,
            "args.moe_router_num_groups and args.moe_router_group_topk are both None when "
            "quantile balancing is enabled",
            (router_groups, group_topk),
        )
        return 1, 1

    if not group_topk:
        support.passed.append(
            "group-limited routing is disabled (args.moe_router_group_topk is None or 0)"
        )
        return 1, 1

    support.require(
        isinstance(router_groups, int)
        and not isinstance(router_groups, bool)
        and router_groups >= 1,
        "args.moe_router_num_groups is a positive integer when group routing is enabled",
        router_groups,
    )
    support.require(
        isinstance(group_topk, int)
        and not isinstance(group_topk, bool)
        and group_topk >= 1,
        "args.moe_router_group_topk is a positive integer",
        group_topk,
    )
    support.require(
        expert_count % router_groups == 0,
        "args.num_experts is divisible by args.moe_router_num_groups",
        (expert_count, router_groups),
    )
    support.require(
        group_topk <= router_groups,
        "args.moe_router_group_topk <= args.moe_router_num_groups",
        (group_topk, router_groups),
    )
    support.require(
        group_topk <= topk,
        "args.moe_router_group_topk <= args.moe_router_topk",
        (group_topk, topk),
    )
    selected_group_capacity = group_topk * (expert_count // router_groups)
    support.require(
        topk <= selected_group_capacity,
        "args.moe_router_topk fits inside the selected expert groups",
        (topk, selected_group_capacity),
    )
    return router_groups, group_topk


def _derive_expert_storage(args: Namespace, support: _SupportBoundary) -> bool:
    """Decide which expert weight layout the checkpoint uses.

    ``OffloadingExpertsMLP`` streams experts from CPU memory during training and persists them
    as one fused ``weight1``/``weight2`` pair per layer, transposed relative to the grouped
    implementation. That is purely a storage difference, so it selects a mapping-table variant
    rather than being rejected — but the fp8 variants must still be checked, because an inplace
    fp8 parameter without extra storage would put packed fp8 bytes where the bf16 master belongs.
    """
    offloaded = bool(getattr(args, "moe_use_offloading_experts", False))
    if not offloaded:
        support.passed.append("args.moe_use_offloading_experts is falsy (grouped expert layout)")
        return False

    inplace_fp8 = bool(getattr(args, "moe_use_inplace_fp8_param", False))
    extra_fp8_storage = bool(getattr(args, "moe_use_extra_fp8_param_storage", False))
    support.require(
        not inplace_fp8 or extra_fp8_storage,
        "args.moe_use_extra_fp8_param_storage is set whenever args.moe_use_inplace_fp8_param is "
        "(the fork asserts the same before saving: without it weight1/weight2 hold packed fp8 "
        "bytes rather than the bf16 master)",
        (inplace_fp8, extra_fp8_storage),
    )
    support.passed.append(
        "args.moe_use_offloading_experts: expert weights are stored as fused, transposed "
        "weight1/weight2 (in, out) per layer"
    )
    return True


def derive_config(args: Namespace) -> DerivedConfig:
    """Validate checkpoint support and derive the Hugging Face configuration."""
    support = _SupportBoundary()
    num_layers = _req(args, "num_layers")
    hidden_act = _validate_base_architecture(args, support)
    _validate_position_and_norm_math(args, support)
    rope_parameters = _derive_rope_parameters(args, support)
    no_rope_layers = _derive_no_rope_layers(args, support, num_layers)
    sliding_window, layer_types = _derive_attention_window(args, support, num_layers)
    keel, keel_alpha = _validate_residual_scheme(args, support)
    _validate_router_and_attention(args, support)
    offloaded_experts = _derive_expert_storage(args, support)

    # Geometry is derived only after all support boundaries above have passed.
    hidden_size = _req(args, "hidden_size")
    num_heads = _req(args, "num_attention_heads")
    num_kv_heads = (
        _req(args, "num_query_groups")
        if _req(args, "group_query_attention")
        else num_heads
    )
    head_dim = _req(args, "kv_channels") or hidden_size // num_heads
    dense_intermediate_size = _req(args, "ffn_hidden_size")
    moe_intermediate_size = _req(args, "moe_ffn_hidden_size") or dense_intermediate_size

    expert_count = _req(args, "num_experts")
    topk = _req(args, "moe_router_topk")
    support.require(
        topk >= 1,
        "args.moe_router_topk >= 1",
        topk,
    )
    support.require(
        topk <= expert_count,
        "args.moe_router_topk <= args.num_experts",
        (topk, expert_count),
    )

    shared_size = _req(args, "moe_shared_expert_intermediate_size")
    if (
        shared_size is None
        or shared_size % moe_intermediate_size != 0
        or shared_size < moe_intermediate_size
    ):
        raise ValueError(
            f"args.moe_shared_expert_intermediate_size={shared_size!r} must be a positive "
            f"multiple of moe_intermediate_size={moe_intermediate_size} (n_shared_experts >= 1)"
        )
    support.passed.append(
        "args.moe_shared_expert_intermediate_size is a positive multiple of "
        "moe_intermediate_size"
    )

    moe_layer_freq, first_k_dense = _derive_moe_layer_freq(
        _req(args, "moe_layer_freq"), num_layers
    )
    support.passed.append(
        "args.moe_layer_freq -> explicit per-layer dense (0) / MoE (1) schedule"
    )

    topk_scaling = _req(args, "moe_router_topk_scaling_factor")
    balancing = _req(args, "moe_router_load_balancing_type")
    if isinstance(balancing, str):
        balancing_methods = [balancing]
    elif balancing is None:
        balancing_methods = []
    else:
        balancing_methods = list(balancing)
    support.require(
        "sinkhorn" not in balancing_methods,
        "'sinkhorn' not in args.moe_router_load_balancing_type",
        balancing,
    )
    use_quantile_balancing = "quantile_balancing" in balancing_methods
    n_group, topk_group = _derive_group_limits(
        args,
        support,
        expert_count=expert_count,
        topk=topk,
        use_quantile_balancing=use_quantile_balancing,
    )

    scale_embeddings = bool(_req(args, "scale_embeddings_by_sqrt_hidden"))
    scale_residuals = bool(_req(args, "residual_output_scaling"))
    embedding_multiplier = math.sqrt(hidden_size) if scale_embeddings else 1.0
    residual_multiplier = 1.0 / math.sqrt(2 * num_layers) if scale_residuals else 1.0

    kwargs: dict[str, Any] = {
        # Shape validation later checks this against the embedding matrix row count.
        "vocab_size": _req(args, "padded_vocab_size"),
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "intermediate_size": dense_intermediate_size,
        "moe_intermediate_size": moe_intermediate_size,
        "n_routed_experts": expert_count,
        "num_experts_per_tok": topk,
        "n_shared_experts": shared_size // moe_intermediate_size,
        "first_k_dense_replace": first_k_dense,
        "moe_layer_freq": moe_layer_freq,
        # The Megatron fork treats None and 0.0 as no additional router scaling.
        "routed_scaling_factor": float(topk_scaling) if topk_scaling else 1.0,
        "rms_norm_eps": _req(args, "layernorm_epsilon"),
        "hidden_act": hidden_act,
        "rope_parameters": rope_parameters,
        "no_rope_layers": no_rope_layers,
        "sliding_window": sliding_window,
        "layer_types": layer_types,
        "max_position_embeddings": _req(args, "max_position_embeddings"),
        "attention_dropout": _req(args, "attention_dropout"),
        "initializer_range": _req(args, "init_method_std"),
        "use_qk_norm": bool(_req(args, "qk_layernorm")),
        "sandwich_norm": bool(_req(args, "sandwich_norm")),
        "moe_latent_size": _req(args, "moe_latent_size"),
        "keel": keel,
        "keel_alpha": keel_alpha,
        "use_quantile_balancing": use_quantile_balancing,
        "embedding_multiplier": embedding_multiplier,
        "residual_multiplier": residual_multiplier,
        "tie_word_embeddings": False,
        # The fork skips renormalization for top-k 1 and always renormalizes for top-k > 1.
        "norm_topk_prob": topk > 1,
        "n_group": n_group,
        "topk_group": topk_group,
    }

    expert_bias_present = bool(_req(args, "moe_router_enable_expert_bias"))
    support.passed.append(f"args.moe_router_enable_expert_bias = {expert_bias_present}")

    return DerivedConfig(kwargs, expert_bias_present, support.passed, offloaded_experts)
