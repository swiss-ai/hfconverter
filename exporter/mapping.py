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
"""Describe how Megatron checkpoint tensors become Hugging Face tensors.

The central object is :class:`Row`: one Megatron source tensor, the Hugging Face tensor names it
creates, and the transform between them. The same rows are used for three jobs:

* check that every checkpoint model tensor is understood;
* derive output shapes before loading tensor data;
* perform the actual conversion.

Using one table for all three jobs prevents validation and conversion from silently drifting apart.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from .transforms import split_gated_fc1, split_qkv

logger = logging.getLogger(__name__)

COPY = "copy"
SPLIT_QKV = "split_qkv"
SPLIT_GATED_FC1 = "split_gated_fc1"
EXPERTS_FC1 = "experts_slice_split_fc1"  # slice expert axis 0 -> split_gated_fc1 per expert
EXPERTS_FC2 = "experts_slice_fc2"  # slice expert axis 0
# Offloaded experts persist each expert as (in, out) instead of nn.Linear's (out, in), so these
# two variants transpose before doing exactly what their non-transposed counterparts do.
EXPERTS_FC1_T = "experts_slice_transpose_split_fc1"
EXPERTS_FC2_T = "experts_slice_transpose_fc2"
SYNTH = "synth_zeros"  # HF tensor with no source (expert_bias off) -> zeros; produce_one only

_DROP_PREFIXES = ("optimizer.", "opt_param_scheduler")

# Keys that must be absent: predicate -> fork feature to name in the error.
# The offloaded-expert keys are conditional rather than listed here: they are supported, but only
# when args.moe_use_offloading_experts said to expect them (see assert_no_unsupported).
_ABSENT_FEATURES: list[tuple[Callable[[str], bool], str]] = [
    (lambda k: k.endswith("shared_experts.gate_weight"),
     "shared expert gate (moe_shared_expert_gate)"),
    (lambda k: k.endswith("router.bias"), "router bias"),
    (lambda k: "core_attention" in k, "core_attention parameters (unsupported attention variant)"),
    (lambda k: k.startswith("mtp"), "multi-token prediction (mtp_num_layers)"),
    (lambda k: "polynorm_glu" in k, "PNGLU (polynorm_glu)"),
    (lambda k: k.startswith("embedding.position_embeddings."),
     "learned position embeddings (position_embedding_type != 'rope')"),
]


@dataclass(frozen=True)
class Row:
    """One mapping-table row: a Megatron tensor and the HF tensor(s) it becomes."""

    megatron_key: str
    hf_keys: tuple[str, ...]
    transform: str
    shape: tuple[int, ...]  # expected global shape in the checkpoint
    dtype: torch.dtype | None = None  # pinned dtype (fp32 router buffers); None = params dtype


@dataclass(frozen=True)
class SynthesizedTensor:
    """An HF tensor with no Megatron source (expert_bias off -> zeros), for conversion_info."""

    hf_key: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    reason: str


@dataclass(frozen=True)
class HFTensorSpec:
    """One Hugging Face output tensor, planned before its source is loaded.

    A :class:`Row` is one source -> many outputs. This class is the inverse view: one output and
    the recipe needed to make it. ``part`` identifies q/k/v (0/1/2), gate/up (0/1), or an expert
    number. A ``None`` Megatron key means the output is synthesized rather than read.
    """

    hf_key: str
    megatron_key: str | None
    shape: tuple[int, ...]
    dtype: torch.dtype
    transform: str
    part: int


@dataclass
class Plan:
    rows: list[Row]
    synthesized: list[SynthesizedTensor]
    geometry: dict[str, Any]  # the Apertus2Config kwargs the table was built from


@dataclass(frozen=True)
class _Geometry:
    """Shape values used repeatedly while building the mapping table.

    Names here deliberately mirror the model concepts instead of the single-letter notation often
    used in papers. Shape comments below still show that notation where it helps compare formulas.
    """

    hidden_size: int
    vocabulary_size: int
    query_heads: int
    key_value_heads: int
    head_dim: int
    dense_ffn_size: int
    expert_ffn_size: int
    expert_count: int
    shared_ffn_size: int
    latent_size: int | None
    first_dense_layers: int
    moe_layer_freq: tuple[int, ...]
    num_layers: int
    use_sandwich_norm: bool
    use_qk_norm: bool
    use_quantile_balancing: bool

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "_Geometry":
        expert_ffn_size = config["moe_intermediate_size"]
        num_layers = config["num_hidden_layers"]
        schedule = config.get("moe_layer_freq")
        if schedule is None:
            first_dense_layers = config["first_k_dense_replace"]
            schedule = [0] * first_dense_layers + [1] * (num_layers - first_dense_layers)
        else:
            schedule = list(schedule)
            if len(schedule) != num_layers or any(entry not in (0, 1) for entry in schedule):
                raise ValueError(
                    "mapping config moe_layer_freq must contain one 0/1 entry per layer; "
                    f"got {schedule!r} for {num_layers} layers"
                )
            first_dense_layers = 0
            while first_dense_layers < num_layers and schedule[first_dense_layers] == 0:
                first_dense_layers += 1
        return cls(
            hidden_size=config["hidden_size"],
            vocabulary_size=config["vocab_size"],
            query_heads=config["num_attention_heads"],
            key_value_heads=config["num_key_value_heads"],
            head_dim=config["head_dim"],
            dense_ffn_size=config["intermediate_size"],
            expert_ffn_size=expert_ffn_size,
            expert_count=config["n_routed_experts"],
            shared_ffn_size=config["n_shared_experts"] * expert_ffn_size,
            latent_size=config["moe_latent_size"],
            first_dense_layers=first_dense_layers,
            moe_layer_freq=tuple(int(entry) for entry in schedule),
            num_layers=num_layers,
            use_sandwich_norm=config["sandwich_norm"],
            use_qk_norm=config["use_qk_norm"],
            use_quantile_balancing=config["use_quantile_balancing"],
        )

    @property
    def expert_input_size(self) -> int:
        return self.latent_size or self.hidden_size

    def is_moe_layer(self, layer_index: int) -> bool:
        return bool(self.moe_layer_freq[layer_index])


def _model_level_rows(shape: _Geometry) -> list[Row]:
    """Embedding, language-model head, and final normalization outside decoder layers."""
    hidden = shape.hidden_size
    vocabulary = shape.vocabulary_size
    return [
        # [vocabulary, hidden] -> token vectors used at the start of the model.
        Row(
            "embedding.word_embeddings.weight",
            ("model.embed_tokens.weight",),
            COPY,
            (vocabulary, hidden),
        ),
        # [vocabulary, hidden] -> logits over the vocabulary at the end of the model.
        Row("output_layer.weight", ("lm_head.weight",), COPY, (vocabulary, hidden)),
        Row("decoder.final_layernorm.weight", ("model.norm.weight",), COPY, (hidden,)),
    ]


def _attention_rows(layer_index: int, shape: _Geometry) -> list[Row]:
    """Map one decoder layer's attention block, preserving row order in the fused QKV matrix."""
    megatron = f"decoder.layers.{layer_index}"
    hf = f"model.layers.{layer_index}"
    hidden = shape.hidden_size
    q_heads = shape.query_heads
    kv_heads = shape.key_value_heads
    head_dim = shape.head_dim

    rows = [
        # Transformer Engine stores the pre-attention norm beside the fused QKV projection.
        Row(
            f"{megatron}.self_attention.linear_qkv.layer_norm_weight",
            (f"{hf}.attention_layernorm.weight",),
            COPY,
            (hidden,),
        ),
        # Megatron [(Q + 2*K)*D, H] -> HF q [Q*D,H], k [K*D,H], v [K*D,H].
        Row(
            f"{megatron}.self_attention.linear_qkv.weight",
            (
                f"{hf}.self_attn.q_proj.weight",
                f"{hf}.self_attn.k_proj.weight",
                f"{hf}.self_attn.v_proj.weight",
            ),
            SPLIT_QKV,
            ((q_heads + 2 * kv_heads) * head_dim, hidden),
        ),
    ]
    if shape.use_qk_norm:
        rows.extend(
            [
                Row(
                    f"{megatron}.self_attention.q_layernorm.weight",
                    (f"{hf}.self_attn.q_norm.weight",),
                    COPY,
                    (head_dim,),
                ),
                Row(
                    f"{megatron}.self_attention.k_layernorm.weight",
                    (f"{hf}.self_attn.k_norm.weight",),
                    COPY,
                    (head_dim,),
                ),
            ]
        )
    rows.append(
        Row(
            f"{megatron}.self_attention.linear_proj.weight",
            (f"{hf}.self_attn.o_proj.weight",),
            COPY,
            (hidden, q_heads * head_dim),
        )
    )
    return rows


def _post_norm_rows(layer_index: int, shape: _Geometry) -> list[Row]:
    """Map optional norms that run after attention and feed-forward residual blocks."""
    megatron = f"decoder.layers.{layer_index}"
    hf = f"model.layers.{layer_index}"
    rows: list[Row] = []

    if shape.use_sandwich_norm:
        rows.append(
            Row(
                f"{megatron}.post_self_attn_layernorm.weight",
                (f"{hf}.post_attention_layernorm.weight",),
                COPY,
                (shape.hidden_size,),
            )
        )
    if shape.use_sandwich_norm:
        rows.append(
            Row(
                f"{megatron}.post_mlp_layernorm.weight",
                (f"{hf}.post_feedforward_layernorm.weight",),
                COPY,
                (shape.hidden_size,),
            )
        )
    return rows


def _dense_mlp_rows(layer_index: int, shape: _Geometry) -> list[Row]:
    """Map a normal SwiGLU feed-forward block (used before the first MoE layer)."""
    megatron = f"decoder.layers.{layer_index}"
    hf = f"model.layers.{layer_index}"
    hidden = shape.hidden_size
    intermediate = shape.dense_ffn_size
    return [
        Row(
            f"{megatron}.mlp.linear_fc1.layer_norm_weight",
            (f"{hf}.feedforward_layernorm.weight",),
            COPY,
            (hidden,),
        ),
        # Megatron [2*F_dense,H] -> HF gate [F_dense,H] + up [F_dense,H].
        Row(
            f"{megatron}.mlp.linear_fc1.weight",
            (f"{hf}.mlp.gate_proj.weight", f"{hf}.mlp.up_proj.weight"),
            SPLIT_GATED_FC1,
            (2 * intermediate, hidden),
        ),
        Row(
            f"{megatron}.mlp.linear_fc2.weight",
            (f"{hf}.mlp.down_proj.weight",),
            COPY,
            (hidden, intermediate),
        ),
    ]


def _routed_expert_rows(
    layer_index: int, shape: _Geometry, offloaded_experts: bool
) -> list[Row]:
    """Map the stacked routed-expert weights, whose layout depends on the training implementation.

    The doubled "experts.experts" source name is intentional and load-bearing. Megatron's grouped
    and sequential expert implementations both add the second segment to the persisted
    ShardedTensor key. DCP metadata therefore reports this doubled name even though the live
    module hierarchy appears to contain only one "experts" container.

    ``OffloadingExpertsMLP`` stores each expert transposed, as ``(in, out)``, and fuses the pair
    of projections into ``weight1``/``weight2`` instead of ``linear_fc1``/``linear_fc2``
    (``make_fused_experts_sharded_factory`` in the fork's ``fp8_utils``, whose ``merge_fn``
    transposes straight back). Both offloading variants agree on that layout, so one branch
    covers them, and the gate/up split still lands on dimension 0 once transposed.
    """
    megatron = f"decoder.layers.{layer_index}"
    hf = f"model.layers.{layer_index}"
    experts = shape.expert_count
    expert_ffn = shape.expert_ffn_size
    expert_input = shape.expert_input_size

    gate_up_keys = tuple(
        f"{hf}.mlp.experts.{expert}.{projection}_proj.weight"
        for expert in range(experts)
        for projection in ("gate", "up")
    )
    down_keys = tuple(f"{hf}.mlp.experts.{expert}.down_proj.weight" for expert in range(experts))

    if offloaded_experts:
        return [
            # Megatron [E,input,2*F] -> 2E HF matrices: gate/up [F,input] per expert.
            Row(f"{megatron}.mlp.experts.experts.weight1", gate_up_keys, EXPERTS_FC1_T,
                (experts, expert_input, 2 * expert_ffn)),
            # Megatron [E,F,input] -> E HF down-projection matrices [input,F].
            Row(f"{megatron}.mlp.experts.experts.weight2", down_keys, EXPERTS_FC2_T,
                (experts, expert_ffn, expert_input)),
        ]
    return [
        # Megatron [E,2*F,input] -> 2E HF matrices: gate/up [F,input] per expert.
        Row(f"{megatron}.mlp.experts.experts.linear_fc1.weight", gate_up_keys, EXPERTS_FC1,
            (experts, 2 * expert_ffn, expert_input)),
        # Megatron [E,input,F] -> E HF down-projection matrices [input,F].
        Row(f"{megatron}.mlp.experts.experts.linear_fc2.weight", down_keys, EXPERTS_FC2,
            (experts, expert_input, expert_ffn)),
    ]


def _moe_mlp_rows(
    layer_index: int, shape: _Geometry, expert_bias_present: bool, offloaded_experts: bool
) -> tuple[list[Row], list[SynthesizedTensor]]:
    """Map one mixture-of-experts block: router, routed experts, and shared experts."""
    megatron = f"decoder.layers.{layer_index}"
    hf = f"model.layers.{layer_index}"
    hidden = shape.hidden_size
    experts = shape.expert_count

    rows = [
        Row(
            f"{megatron}.pre_mlp_layernorm.weight",
            (f"{hf}.feedforward_layernorm.weight",),
            COPY,
            (hidden,),
        ),
        # [experts, hidden]: one score-producing vector per routed expert.
        Row(
            f"{megatron}.mlp.router.weight",
            (f"{hf}.mlp.gate.weight",),
            COPY,
            (experts, hidden),
        ),
    ]
    synthesized: list[SynthesizedTensor] = []
    if expert_bias_present:
        rows.append(
            Row(
                f"{megatron}.mlp.router.expert_bias",
                (f"{hf}.mlp.gate.e_score_correction_bias",),
                COPY,
                (experts,),
                torch.float32,
            )
        )
    else:
        # The HF router always owns this buffer. When Megatron disabled it, zeros reproduce the
        # absence of a correction while keeping the HF state dict complete.
        synthesized.append(
            SynthesizedTensor(
                f"{hf}.mlp.gate.e_score_correction_bias",
                (experts,),
                torch.float32,
                "args.moe_router_enable_expert_bias is False: synthesized fp32 zeros",
            )
        )
    if shape.use_quantile_balancing:
        rows.append(
            Row(
                f"{megatron}.mlp.router.qb_beta",
                (f"{hf}.mlp.gate.qb_beta",),
                COPY,
                (experts,),
                torch.float32,
            )
        )
    if shape.latent_size:
        rows.extend(
            [
                Row(
                    f"{megatron}.mlp.fc1_latent_proj.weight",
                    (f"{hf}.mlp.latent_down_proj.weight",),
                    COPY,
                    (shape.latent_size, hidden),
                ),
                Row(
                    f"{megatron}.mlp.fc2_latent_proj.weight",
                    (f"{hf}.mlp.latent_up_proj.weight",),
                    COPY,
                    (hidden, shape.latent_size),
                ),
            ]
        )

    rows.extend(_routed_expert_rows(layer_index, shape, offloaded_experts))
    rows.extend(
        [
            # Shared experts are one fused gated block in both frameworks.
            Row(
                f"{megatron}.mlp.shared_experts.linear_fc1.weight",
                (
                    f"{hf}.mlp.shared_experts.gate_proj.weight",
                    f"{hf}.mlp.shared_experts.up_proj.weight",
                ),
                SPLIT_GATED_FC1,
                (2 * shape.shared_ffn_size, hidden),
            ),
            Row(
                f"{megatron}.mlp.shared_experts.linear_fc2.weight",
                (f"{hf}.mlp.shared_experts.down_proj.weight",),
                COPY,
                (hidden, shape.shared_ffn_size),
            ),
        ]
    )
    return rows, synthesized


def build_plan(
    cfg: dict[str, Any], expert_bias_present: bool, offloaded_experts: bool = False
) -> Plan:
    """Build the complete source-to-output mapping in model execution order."""
    shape = _Geometry.from_config(cfg)
    rows = _model_level_rows(shape)
    synthesized: list[SynthesizedTensor] = []
    for layer_index in range(shape.num_layers):
        rows.extend(_attention_rows(layer_index, shape))
        rows.extend(_post_norm_rows(layer_index, shape))
        if shape.is_moe_layer(layer_index):
            moe_rows, moe_synthesized = _moe_mlp_rows(
                layer_index, shape, expert_bias_present, offloaded_experts
            )
            rows.extend(moe_rows)
            synthesized.extend(moe_synthesized)
        else:
            rows.extend(_dense_mlp_rows(layer_index, shape))

    plan = Plan(rows, synthesized, dict(cfg))
    consumed = [row.megatron_key for row in plan.rows]
    produced = expected_produced(plan)
    if len(consumed) != len(set(consumed)) or len(produced) != len(set(produced)):
        raise RuntimeError("mapping table bug: duplicate Megatron or HF key in generated rows")
    return plan


def expected_consumed(plan: Plan) -> set[str]:
    return {row.megatron_key for row in plan.rows}


def expected_produced(plan: Plan) -> list[str]:
    """Return every expected Hugging Face key in deterministic write order."""
    keys = [key for row in plan.rows for key in row.hf_keys]
    keys.extend(s.hf_key for s in plan.synthesized)
    return keys


def partition_universe(
    universe: set[str], strict_optimizer: bool = False
) -> tuple[set[str], set[str]]:
    """Separate model tensors from training-only optimizer tensors."""
    dropped = {k for k in universe if k.startswith(_DROP_PREFIXES)}
    if dropped:
        if strict_optimizer:
            raise ValueError(
                "--strict-optimizer: checkpoint contains "
                f"{len(dropped)} optimizer/opt_param_scheduler keys:\n  "
                + "\n  ".join(sorted(dropped))
            )
        logger.info(
            "dropping %d optimizer/opt_param_scheduler keys (training state, not exported)",
            len(dropped),
        )
    model_keys = universe - dropped
    # `*._extra_state*` are ShardedObjects and never appear in load_tensors_metadata
    # and are excluded defensively as a never-consumed class.
    extra_state = {k for k in model_keys if "._extra_state" in k}
    if extra_state:
        logger.debug("ignoring %d _extra_state entries", len(extra_state))
        model_keys = model_keys - extra_state
    return model_keys, dropped


def assert_no_unsupported(model_keys: set[str], offloaded_experts: bool = False) -> None:
    """Fail with the fork feature name if any must-be-absent key class is present."""
    findings = []
    absent_features = list(_ABSENT_FEATURES)
    if not offloaded_experts:
        # These keys are readable, but only under the transposed layout the args did not request;
        # reading them as if they were grouped-expert tensors would silently transpose the model.
        absent_features.append(
            (lambda k: k.endswith("experts.weight1") or k.endswith("experts.weight2"),
             "OffloadingExpertsMLP weights without args.moe_use_offloading_experts")
        )
    for predicate, feature in absent_features:
        hits = sorted(k for k in model_keys if predicate(k))
        if hits:
            findings.append(f"{feature}:\n  " + "\n  ".join(hits))
    if findings:
        raise ValueError(
            "checkpoint contains keys from fork features this exporter does not map:\n"
            + "\n".join(findings)
        )


def check_consumed(model_keys: set[str], plan: Plan) -> None:
    """Require a one-to-one match between checkpoint model keys and mapping sources."""
    expected = expected_consumed(plan)
    unexpected = sorted(model_keys - expected)
    missing = sorted(expected - model_keys)
    if unexpected or missing:
        parts = ["Megatron->HF bijection failed: checkpoint keys != mapping table for the derived config."]
        if unexpected:
            parts.append("checkpoint keys not consumed by any mapping row (new fork feature?):\n  "
                         + "\n  ".join(unexpected))
        if missing:
            parts.append("mapping rows whose Megatron key is missing from the checkpoint:\n  "
                         + "\n  ".join(missing))
        raise ValueError("\n".join(parts))


def validate_metadata(plan: Plan, metadata: dict[str, Any]) -> list[str]:
    """Validate global shapes and dtypes before loading tensor data.

    The current HF model/config has one parameter dtype plus explicitly fp32 router buffers.
    Rejecting any other mixed-parameter layout also keeps metadata-only shard sizes identical to
    the tensors that will actually be written.
    """
    parameter_dtype: torch.dtype | None = None
    mixed_parameter_dtypes: list[tuple[str, torch.dtype]] = []
    for row in plan.rows:
        meta = metadata[row.megatron_key]  # presence guaranteed by check_consumed
        got_shape = tuple(meta.global_shape)
        if got_shape != row.shape:
            raise ValueError(
                f"shape mismatch for {row.megatron_key}: checkpoint has {got_shape}, derived "
                f"config expects {row.shape}"
            )
        if row.dtype is not None and meta.dtype != row.dtype:
            raise ValueError(
                f"dtype mismatch for {row.megatron_key}: checkpoint has {meta.dtype}, expected "
                f"{row.dtype} (router buffers are kept in fp32)"
            )
        if row.dtype is None:
            if parameter_dtype is None:
                parameter_dtype = meta.dtype
            elif meta.dtype != parameter_dtype:
                mixed_parameter_dtypes.append((row.megatron_key, meta.dtype))
    if mixed_parameter_dtypes:
        details = "\n  ".join(
            f"{key}: {dtype}" for key, dtype in mixed_parameter_dtypes
        )
        raise ValueError(
            "mixed parameter dtypes are not supported: expected every model parameter to use "
            f"{parameter_dtype}, but found:\n  {details}"
        )
    checks = [
        f"validated global shapes of all {len(plan.rows)} mapped checkpoint tensors against "
        "derived config geometry (incl. vocab_size == embedding rows)",
        f"validated one model parameter dtype ({parameter_dtype}) across all unpinned tensors",
    ]
    pinned = sum(1 for r in plan.rows if r.dtype is not None)
    if pinned:
        checks.append(f"validated fp32 dtype of {pinned} router buffer tensors (expert_bias/qb_beta)")
    return checks


def params_dtype(plan: Plan, dtype_of: Callable[[str], torch.dtype]) -> torch.dtype:
    """Read the model parameter dtype from the first normal (non-router-buffer) source.

    ``dtype_of`` can read from lightweight metadata or a loaded tensor. Router buffers are pinned
    to fp32, so they cannot determine the main parameter dtype written to ``config.json``.
    """
    for row in plan.rows:
        if row.dtype is None:
            return dtype_of(row.megatron_key)
    raise RuntimeError("mapping table bug: no non-pinned row to derive the params dtype from")


def plan_hf_tensors(plan: Plan, dtype: torch.dtype) -> list[HFTensorSpec]:
    """Expand source rows into one metadata-only specification per HF output tensor.

    Output order is the same order used by :func:`convert`, which also fixes shard packing order.
    Shapes come only from the mapping table and model geometry; no tensor bytes are loaded here.
    """
    config = plan.geometry
    query_heads = config["num_attention_heads"]
    key_value_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    specs: list[HFTensorSpec] = []
    for row in plan.rows:
        output_dtype = row.dtype or dtype
        source_key = row.megatron_key
        if row.transform == COPY:
            specs.append(
                HFTensorSpec(row.hf_keys[0], source_key, row.shape, output_dtype, COPY, 0)
            )
        elif row.transform == SPLIT_QKV:
            remaining_dimensions = row.shape[1:]  # (hidden,) for weights; empty for biases
            first_dimensions = (
                query_heads * head_dim,
                key_value_heads * head_dim,
                key_value_heads * head_dim,
            )
            for part, first_dimension in enumerate(first_dimensions):
                specs.append(
                    HFTensorSpec(
                        row.hf_keys[part],
                        source_key,
                        (first_dimension, *remaining_dimensions),
                        output_dtype,
                        SPLIT_QKV,
                        part,
                    )
                )
        elif row.transform == SPLIT_GATED_FC1:
            half_shape = (row.shape[0] // 2, *row.shape[1:])
            for part in (0, 1):
                specs.append(
                    HFTensorSpec(
                        row.hf_keys[part],
                        source_key,
                        half_shape,
                        output_dtype,
                        SPLIT_GATED_FC1,
                        part,
                    )
                )
        elif row.transform in (EXPERTS_FC1, EXPERTS_FC1_T):
            # Outputs are always nn.Linear-shaped, so read the source dimensions by name rather
            # than by position: only the stored order differs between the two layouts.
            if row.transform == EXPERTS_FC1:
                _expert_count, doubled_ffn_size, expert_input_size = row.shape
            else:
                _expert_count, expert_input_size, doubled_ffn_size = row.shape
            output_shape = (doubled_ffn_size // 2, expert_input_size)
            for part, hf_key in enumerate(row.hf_keys):
                specs.append(
                    HFTensorSpec(
                        hf_key,
                        source_key,
                        output_shape,
                        output_dtype,
                        row.transform,
                        part,
                    )
                )
        elif row.transform in (EXPERTS_FC2, EXPERTS_FC2_T):
            if row.transform == EXPERTS_FC2:
                _expert_count, expert_input_size, expert_ffn_size = row.shape
            else:
                _expert_count, expert_ffn_size, expert_input_size = row.shape
            output_shape = (expert_input_size, expert_ffn_size)
            for expert_index, hf_key in enumerate(row.hf_keys):
                specs.append(
                    HFTensorSpec(
                        hf_key,
                        source_key,
                        output_shape,
                        output_dtype,
                        row.transform,
                        expert_index,
                    )
                )
        else:
            raise RuntimeError(f"mapping table bug: unknown transform tag {row.transform!r}")
    for synthesized in plan.synthesized:
        specs.append(
            HFTensorSpec(
                synthesized.hf_key,
                None,
                synthesized.shape,
                synthesized.dtype,
                SYNTH,
                0,
            )
        )
    return specs


def produce_one(
    spec: HFTensorSpec,
    source: torch.Tensor | None,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Create one HF tensor from its already-loaded Megatron source.

    Splits may recreate sibling views while processing a later output, but this avoids retaining all
    siblings at once. No dtype conversion occurs; output bytes therefore remain bit-identical.
    """
    if spec.transform == SYNTH:
        out = torch.zeros(spec.shape, dtype=spec.dtype)
    elif source is None:
        raise RuntimeError(f"produce_one: {spec.hf_key} needs a source but none was supplied")
    elif spec.transform == COPY:
        out = source
    elif spec.transform == SPLIT_QKV:
        out = split_qkv(source, num_q_heads, num_kv_heads, head_dim)[spec.part]
    elif spec.transform == SPLIT_GATED_FC1:
        out = split_gated_fc1(source)[spec.part]
    elif spec.transform == EXPERTS_FC1:
        out = split_gated_fc1(source[spec.part // 2])[spec.part % 2]
    elif spec.transform == EXPERTS_FC1_T:
        # (in, 2F) -> (2F, in) puts the gate rows first, exactly where split_gated_fc1 and
        # Megatron's own chunk(y, 2, dim=-1) on the GEMM output both expect them.
        out = split_gated_fc1(source[spec.part // 2].transpose(0, 1))[spec.part % 2]
    elif spec.transform == EXPERTS_FC2:
        out = source[spec.part]
    elif spec.transform == EXPERTS_FC2_T:
        out = source[spec.part].transpose(0, 1)
    else:
        raise RuntimeError(f"mapping table bug: unknown transform tag {spec.transform!r}")
    if tuple(out.shape) != spec.shape:
        raise ValueError(
            f"produced {spec.hf_key} has shape {tuple(out.shape)}, expected {spec.shape}")
    return out


def convert(plan: Plan, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert an already-loaded whole-model tensor dictionary.

    The production exporter streams shards, but this public helper remains useful for small models,
    tests, and callers that already have all tensors in memory. Both paths share the same planning
    and per-output transform functions.
    """
    cfg = plan.geometry
    num_q_heads = cfg["num_attention_heads"]
    num_kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    # Validate each source once; one source can feed many per-output specs.
    for row in plan.rows:
        tensor = tensors[row.megatron_key]
        if tuple(tensor.shape) != row.shape:
            raise ValueError(
                f"loaded tensor {row.megatron_key} has shape {tuple(tensor.shape)}, "
                f"expected {row.shape}"
            )
    out: dict[str, torch.Tensor] = {}
    for spec in plan_hf_tensors(plan, params_dtype(plan, lambda k: tensors[k].dtype)):
        if spec.hf_key in out:
            raise ValueError(f"HF key produced twice: {spec.hf_key}")
        source = tensors[spec.megatron_key] if spec.megatron_key is not None else None
        out[spec.hf_key] = produce_one(spec, source, num_q_heads, num_kv_heads, head_dim)
    for synth in plan.synthesized:
        logger.info("synthesizing %s = zeros(%s, %s): %s",
                    synth.hf_key, list(synth.shape), synth.dtype, synth.reason)
    return out


def check_produced(produced_keys, plan: Plan) -> None:
    """Assert that the produced Hugging Face key set exactly matches the plan."""
    expected = expected_produced(plan)
    got = set(produced_keys)
    extra = sorted(got - set(expected))
    missing = sorted(set(expected) - got)
    if extra or missing or len(got) != len(expected):
        parts = ["Megatron->HF bijection failed on the produced HF key set."]
        if extra:
            parts.append("produced keys outside the API contract (new fork feature?):\n  "
                         + "\n  ".join(extra))
        if missing:
            parts.append("expected HF keys never produced:\n  " + "\n  ".join(missing))
        raise ValueError("\n".join(parts))
