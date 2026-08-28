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
"""Configuration for the Apertus 2 mixture-of-experts language model.

This file describes *which* model to build; tensor operations live in
``modeling_apertus2.py``.  The most important choices are documented on
``Apertus2Config`` so a saved ``config.json`` remains understandable on its own.
"""

from huggingface_hub.dataclasses import strict

from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters


@strict
class Apertus2Config(PreTrainedConfig):
    r"""
    Settings used by :class:`Apertus2Model` and :class:`Apertus2ForCausalLM`.

    A decoder layer always has attention followed by a feed-forward block.  ``moe_layer_freq``
    optionally marks every layer as dense (``0``) or MoE (``1``).  Older configs without that
    list retain the contiguous convention: layers below ``first_k_dense_replace`` are dense and
    every later layer uses routed experts plus a shared expert.  The options below change that
    basic data flow:

    - ``sandwich_norm`` adds a second RMSNorm to each branch.  In tensor shorthand, a branch is
      ``x + residual_multiplier * post_norm(branch(pre_norm(x)))``.
    - ``moe_latent_size`` makes routed experts work in a smaller feature space.  The router and
      shared expert still see the full ``hidden_size`` representation.
    - ``use_quantile_balancing`` subtracts ``qb_beta`` before expert selection.  The
      ``moe_router_quantile_balancing_method`` chooses its score space: ``sigmoid`` (the
      default) selects from sigmoid router scores, ``legacy`` from raw router logits.
      Megatron's training-time estimator names are accepted and normalized on load
      (``average``/``histogram`` -> ``sigmoid``, ``legacy_average`` -> ``legacy``).  QB
      replaces correction-bias selection and cannot be combined with group-limited routing.
    - ``attention_output_gate`` adds a per-channel sigmoid gate to every attention layer:
      ``g_proj`` reads the same normalized input as Q/K/V and its sigmoid multiplies the
      attention output right before ``o_proj``.  The gate skips QK-norm and RoPE.
    - ``embedding_multiplier`` scales token embeddings once, before the decoder stack.
    - ``residual_multiplier`` scales both attention and feed-forward branch outputs before they
      are added back to the residual stream.

    Two independent per-layer schedules describe attention:

    - ``layer_types`` marks each layer ``"full_attention"``, ``"sliding_attention"``, or
      ``"linear_attention"``.  Sliding layers see ``sliding_window`` keys, counting the query's
      own position.  Linear-attention layers are KDA (Kimi Delta Attention) recurrent layers:
      their geometry comes from the ``linear_*`` fields, and ``gate_lower_bound`` selects the
      decay-gate form — a float ``g_min`` in ``[-5, 0)`` for the bounded Kimi-K3 gate
      ``g = g_min * sigmoid(exp(A_log) * (z + dt_bias))``, or ``None`` for the unbounded
      ``g = -exp(A_log) * softplus(z + dt_bias)``.  KDA layers ignore RoPE and
      ``attention_output_gate`` (they carry their own sigmoid gate), and cannot be mixed with
      sliding-window layers in one model.
    - ``no_rope_layers`` marks each layer ``1`` (rotate) or ``0`` (NoPE).  The polarity follows
      SmolLM3, and is therefore the *inverse* of Megatron's ``--no-rope-freq``.

    They are deliberately **not** derived from each other.  Upstream models that carry both
    (SmolLM3, Llama4) compute one from the other; Megatron lets the two patterns differ, so
    coupling them here would bake in an assumption a checkpoint only happens to satisfy.

    Parameter names intentionally match the exported checkpoints. See
    ``modeling_apertus2`` for shape-by-shape docstrings and ``exporter/README.md`` for the
    conversion layout.
    """

    model_type = "apertus2"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 500000.0

    # Tensor parallelism is intentionally unsupported for now, so no base_model_tp_plan is
    # published.  An absent plan makes from_pretrained(tp_plan="auto") silently replicate the
    # whole model on every rank instead of sharding, so the model additionally rejects
    # tp_size > 1 at forward time.  Commit a465e6e holds a working replicated-KV TP
    # implementation and tests (needed because 3 K/V heads make the stock colwise-K/V recipe
    # undivisible) if support is revived.
    #
    # Expert parallelism IS supported (EP-only, the same shape DeepseekV4 ships upstream).
    # Enable with from_pretrained(..., distributed_config=DistributedConfig(
    # enable_expert_parallel=True)): ep_router remaps the gate's (logits, weights, indices)
    # triple to rank-local expert ids with a sentinel, grouped_gemm shards both stacked
    # expert banks on the expert axis, and moe_tp_experts all-reduces the routed output.
    # Everything else — attention, router weights, shared expert, latent projections — stays
    # replicated; the shared expert is added AFTER the all-reduce, so it is counted once.
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",  # must stay right after the experts.* entries (pattern priority)
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }

    vocab_size: int = 200064
    hidden_size: int = 768
    intermediate_size: int = 1920
    num_hidden_layers: int = 10
    num_attention_heads: int = 6
    num_key_value_heads: int = 3
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 8192
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    use_qk_norm: bool = True
    attention_output_gate: bool = False
    sliding_window: int | None = None
    layer_types: list[str] | None = None
    no_rope_layers: list[int] | None = None
    # KDA (Kimi Delta Attention) geometry, active on the layers whose layer_types entry is
    # "linear_attention".  Field names follow Qwen3-Next; gate_lower_bound is the Kimi-K3 knob
    # vLLM consumes.  All six stay None on a pure-softmax model.  The low-rank bottleneck width
    # of the decay and output-gate projections is not a field: it equals linear_value_head_dim
    # by KDA construction.
    linear_num_key_heads: int | None = None
    linear_num_value_heads: int | None = None
    linear_key_head_dim: int | None = None
    linear_value_head_dim: int | None = None
    linear_conv_kernel_dim: int | None = None
    gate_lower_bound: float | None = None
    moe_intermediate_size: int = 448
    num_experts_per_tok: int = 4
    n_shared_experts: int = 1
    n_routed_experts: int = 128
    routed_scaling_factor: float = 2.5
    n_group: int = 1
    topk_group: int = 1
    first_k_dense_replace: int = 1
    moe_layer_freq: list[int] | None = None
    norm_topk_prob: bool = True
    sandwich_norm: bool = False
    moe_latent_size: int | None = None
    use_quantile_balancing: bool = False
    # Score space for QB expert selection; "legacy" keeps the raw-logit selection of the earliest
    # QB exports.  Exports from that era bundle their own modeling code, so the modern sigmoid
    # space is the default here; loading such an old field-less config.json with THIS code
    # requires setting the field to "legacy" explicitly.
    moe_router_quantile_balancing_method: str = "sigmoid"
    embedding_multiplier: float = 27.712812921102035
    residual_multiplier: float = 0.22360679774997896
    pad_token_id: int | None = 3
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2

    def __post_init__(self, **kwargs):
        """Validate choices that would otherwise create a silently different model."""
        # Older exports may carry these inactive fields. Normalize false/null away, but reject
        # active values so a removed residual mode can never degrade silently to the plain path.
        keel = kwargs.pop("keel", False)
        keel_alpha = kwargs.pop("keel_alpha", None)
        if keel or keel_alpha is not None:
            raise ValueError(
                "KEEL residual mode is not supported; re-export the checkpoint without "
                f"keel/keel_alpha (got keel={keel!r}, keel_alpha={keel_alpha!r})."
            )
        self._validate_checkpoint_contract()
        self._validate_router_options()
        self._set_mlp_schedule()
        self._set_attention_schedules()
        self._validate_linear_attention()
        self._set_full_rotary_defaults(kwargs)
        super().__post_init__(**kwargs)

    def _validate_checkpoint_contract(self) -> None:
        """Keep settings that are fixed by the Apertus checkpoint format."""
        if self.tie_word_embeddings:
            raise ValueError(
                "Apertus2 is untied (--untie-embeddings-and-output-weights): "
                "tie_word_embeddings=True is not supported and would silently do nothing."
            )

    def _validate_router_options(self) -> None:
        """Ensure Hugging Face routing performs the same math as the Megatron model."""
        if self.use_quantile_balancing:
            # Megatron names its training-time quantile estimators; inference only needs the
            # selection score space, so the estimator names collapse onto the canonical pair.
            aliases = {
                "average": "sigmoid",
                "histogram": "sigmoid",
                "legacy_average": "legacy",
            }
            method = aliases.get(
                self.moe_router_quantile_balancing_method,
                self.moe_router_quantile_balancing_method,
            )
            if method not in ("sigmoid", "legacy"):
                raise ValueError(
                    "moe_router_quantile_balancing_method must be 'sigmoid' or 'legacy' "
                    "(Megatron spellings 'average', 'histogram', and 'legacy_average' are also "
                    f"accepted); got {self.moe_router_quantile_balancing_method!r}."
                )
            self.moe_router_quantile_balancing_method = method
        # Megatron normalizes selected weights exactly when more than one expert is selected.
        if self.num_experts_per_tok == 1 and self.norm_topk_prob:
            raise ValueError(
                "num_experts_per_tok=1 with norm_topk_prob=True is not supported: the Megatron "
                "fork skips top-k renormalization when topk==1, but HF would renormalize the "
                "single score to 1.0. Set norm_topk_prob=False for topk=1."
            )
        if self.num_experts_per_tok > 1 and not self.norm_topk_prob:
            raise ValueError(
                "num_experts_per_tok>1 with norm_topk_prob=False is not supported: the Megatron "
                "fork always renormalizes top-k weights when topk>1. Set norm_topk_prob=True for "
                "topk>1."
            )
        if self.use_quantile_balancing and (self.n_group != 1 or self.topk_group != 1):
            raise ValueError(
                f"use_quantile_balancing=True is incompatible with group-limited routing "
                f"(got n_group={self.n_group}, topk_group={self.topk_group}); the Megatron fork "
                "forbids QB with num_groups/group_topk set. Use n_group=1 and topk_group=1."
            )

    def _set_mlp_schedule(self) -> None:
        """Validate dense/MoE placement while preserving legacy cutoff-only configs.

        ``moe_layer_freq`` is authoritative when present. ``first_k_dense_replace`` stays in
        the public config for compatibility with existing exports and is normalized to the
        number of leading dense layers for a per-layer schedule. It is only a complete
        architecture description when ``moe_layer_freq`` is absent.
        """
        if self.moe_layer_freq is None:
            if (
                not isinstance(self.first_k_dense_replace, int)
                or isinstance(self.first_k_dense_replace, bool)
                or not 0 <= self.first_k_dense_replace <= self.num_hidden_layers
            ):
                raise ValueError(
                    "first_k_dense_replace must be an integer between 0 and "
                    f"num_hidden_layers={self.num_hidden_layers}; got "
                    f"{self.first_k_dense_replace!r}"
                )
            return

        pattern = list(self.moe_layer_freq)
        if len(pattern) != self.num_hidden_layers:
            raise ValueError(
                f"moe_layer_freq has {len(pattern)} entries but the model has "
                f"{self.num_hidden_layers} layers; one 0/1 entry per layer is required."
            )
        if any(entry not in (0, 1) for entry in pattern):
            raise ValueError(
                "moe_layer_freq entries must be 0 (dense) or 1 (MoE); "
                f"got {pattern}"
            )

        self.moe_layer_freq = [int(entry) for entry in pattern]
        leading_dense = 0
        while (
            leading_dense < self.num_hidden_layers
            and self.moe_layer_freq[leading_dense] == 0
        ):
            leading_dense += 1
        self.first_k_dense_replace = leading_dense

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Return whether ``layer_idx`` owns routed experts."""
        if self.moe_layer_freq is not None:
            return bool(self.moe_layer_freq[layer_idx])
        return layer_idx >= self.first_k_dense_replace

    def _set_attention_schedules(self) -> None:
        """Fill in the per-layer attention and RoPE schedules, then check they are usable.

        Both lists default to the behaviour of a checkpoint that never mentions them: every layer
        full attention, every layer rotated.  ``layer_types`` must end up non-``None`` because
        Transformers branches on the attribute's presence, not on its value, and would then choke
        on ``None``; ``PreTrainedConfig.validate_layer_type`` checks its entries and length for us.
        """
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if self.no_rope_layers is None:
            self.no_rope_layers = [1] * self.num_hidden_layers

        if "sliding_attention" in self.layer_types and not self.sliding_window:
            raise ValueError(
                "layer_types requests sliding attention but sliding_window is "
                f"{self.sliding_window!r}; set a positive window length. The window counts the "
                "query's own position, so a Megatron --window-size of (w, 0) is w + 1 here."
            )
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(f"sliding_window must be a positive number of keys; got {self.sliding_window}")
        if len(self.no_rope_layers) != self.num_hidden_layers:
            raise ValueError(
                f"no_rope_layers has {len(self.no_rope_layers)} entries but the model has "
                f"{self.num_hidden_layers} layers; one entry per layer is required."
            )
        # SmolLM3 polarity: 1 rotates, 0 is NoPE.  Megatron's --no-rope-freq is the opposite way
        # round, so the exporter inverts; a stray 2 here would read as "rotate" and hide a bug.
        if any(entry not in (0, 1) for entry in self.no_rope_layers):
            raise ValueError(
                f"no_rope_layers entries must be 0 (NoPE) or 1 (rotate); got {self.no_rope_layers}"
            )

    def _validate_linear_attention(self) -> None:
        """Keep the KDA field set and the layer_types schedule describing the same model.

        Runs after ``_set_attention_schedules``, so ``layer_types`` is always a full list here.
        A half-described KDA model must never construct: geometry without schedule (or the
        reverse) would build softmax layers where the checkpoint holds recurrent-state weights.
        """
        geometry = {
            "linear_num_key_heads": self.linear_num_key_heads,
            "linear_num_value_heads": self.linear_num_value_heads,
            "linear_key_head_dim": self.linear_key_head_dim,
            "linear_value_head_dim": self.linear_value_head_dim,
            "linear_conv_kernel_dim": self.linear_conv_kernel_dim,
        }
        if "linear_attention" not in self.layer_types:
            stray = {name: value for name, value in geometry.items() if value is not None}
            if self.gate_lower_bound is not None:
                stray["gate_lower_bound"] = self.gate_lower_bound
            if stray:
                raise ValueError(
                    "KDA fields are set but no layer_types entry is 'linear_attention'; the "
                    "schedule and the geometry describe different models: "
                    f"{stray!r}"
                )
            return

        if "sliding_attention" in self.layer_types:
            raise ValueError(
                "layer_types mixes 'linear_attention' (KDA) and 'sliding_attention' layers; "
                "KDA and sliding-window attention cannot coexist in one Apertus2 model: "
                f"{self.layer_types}"
            )
        for name, value in geometry.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(
                    f"layer_types contains 'linear_attention' layers, so {name} must be a "
                    f"positive integer; got {value!r}"
                )
        if self.linear_num_key_heads != self.linear_num_value_heads:
            raise ValueError(
                "linear_num_key_heads must equal linear_num_value_heads (KDA lays out "
                "dt_bias/A_log and the delta-rule state per value head against key channels); "
                f"got {self.linear_num_key_heads} and {self.linear_num_value_heads}"
            )
        if self.gate_lower_bound is not None and not (
            isinstance(self.gate_lower_bound, (int, float))
            and -5.0 <= float(self.gate_lower_bound) < 0.0
        ):
            raise ValueError(
                "gate_lower_bound must be None (unbounded softplus decay gate) or a float in "
                "[-5, 0) (the bounded Kimi-K3 gate; FlashKDA and vLLM assert the same range); "
                f"got {self.gate_lower_bound!r}"
            )

    def _set_full_rotary_defaults(self, kwargs: dict) -> None:
        """Configure RoPE over the entire attention head, never an accidental half-head."""
        explicit_prf = kwargs.get("partial_rotary_factor")
        if explicit_prf is not None and explicit_prf != 1.0:
            raise ValueError(
                f"Apertus2 uses full rotary embeddings; got explicit partial_rotary_factor="
                f"{explicit_prf}. Only 1.0 is supported (omit it to default to 1.0)."
            )
        if isinstance(self.rope_parameters, dict):
            prf_in_dict = self.rope_parameters.get("partial_rotary_factor")
            if prf_in_dict is not None and prf_in_dict != 1.0:
                raise ValueError(
                    f"Apertus2 uses full rotary embeddings; got rope_parameters['partial_rotary"
                    f"_factor']={prf_in_dict}. Only 1.0 is supported (omit it to default to 1.0)."
                )
        if self.rope_parameters is None:
            self.rope_parameters = {
                "rope_type": "default",
                "rope_theta": self.default_theta,
                # Glm4Moe defaults to 0.5; Apertus rotates the full attention head.
                "partial_rotary_factor": 1.0,
            }
        if isinstance(self.rope_parameters, dict):
            self.rope_parameters.setdefault("partial_rotary_factor", 1.0)
        kwargs.setdefault("partial_rotary_factor", 1.0)


# Import-time, load-bearing for trust_remote_code artifacts: sets _auto_class so that
# save_pretrained writes auto_map = {"AutoConfig": "configuration_apertus2.Apertus2Config"}
# into config.json and copies this file into the save directory, making the saved dir
# self-describing for AutoConfig.from_pretrained(..., trust_remote_code=True).
Apertus2Config.register_for_auto_class("AutoConfig")


__all__ = ["Apertus2Config"]
