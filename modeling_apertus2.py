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
"""Readable Hugging Face implementation of the Apertus 2 MoE decoder.

The complete tensor journey is:

1. Token ids become embeddings with shape ``[batch, sequence, hidden]``.
2. Every decoder layer runs self-attention, then a dense MLP or a mixture of experts (MoE).
3. Each block returns the same ``[batch, sequence, hidden]`` shape and updates the residual
   stream.  Plain, sandwich-norm, and KEEL layers differ only in how that update is combined.
4. A final RMSNorm produces hidden states; ``Apertus2ForCausalLM`` projects them to one score
   per vocabulary item.

For MoE layers, the router chooses ``top_k`` experts independently for every token.  Routed
experts may work in a smaller latent dimension, while the shared expert always works in the
full hidden dimension.  Checkpoint-facing module and parameter names intentionally remain
compatible with Transformers. The conversion and checkpoint layouts are described in the
exporter's README.

Shape legend used in comments: ``B`` = batch, ``S`` = sequence length, ``H`` = hidden size,
``Q`` = query heads, ``K`` = key/value heads, ``D`` = head size, and ``E`` = experts.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

import transformers.initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.generation import GenerationMixin
from transformers.integrations import use_experts_implementation, use_kernel_forward_from_hub, use_kernelized_func
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel

# RoPE comes verbatim from Llama, the canonical Transformers implementation.
# Apertus uses exactly Llama's convention: full-head
# rotation (the config pins partial_rotary_factor=1.0) with non-interleaved rotate-half.
# RoPE contributes no state-dict keys (inv_freq buffers are non-persistent), so no
# checkpoint contract is involved; consumers resolve this import against their installed
# transformers at load time.
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

try:
    from .configuration_apertus2 import Apertus2Config
except ImportError:
    from configuration_apertus2 import Apertus2Config


# Checkpoint conversion between per-expert and stacked expert layouts.
# Megatron saves one key per expert, while this runtime stores all experts in two stacked 3-D
# tensors.  This import-time registration teaches Transformers how to convert between layouts.
if get_checkpoint_conversion_mapping("apertus2") is None:
    # Keep a private list because Transformers' registry stores the object it receives.
    register_checkpoint_conversion_mapping("apertus2", list(get_checkpoint_conversion_mapping("qwen2_moe")))


# Activation functions.
class Apertus2SSSGLU(nn.Module):
    """Megatron's ``sssglu`` gate: a shifted, recentred softsign.

    ``gate(x) = softsign(x - 1) + 0.5``, so ``gate(1) = 0.5`` and the output spans
    ``(-0.5, 1.5)`` — unlike SiLU it is bounded, and unlike a sigmoid it is not confined to
    ``(0, 1)``.  It replaces only the gate half of the GLU; the elementwise product with the
    linear half, and therefore every tensor shape and checkpoint key, is unchanged.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.softsign(hidden_states - 1.0) + 0.5


# Activations Transformers does not ship, keyed exactly like ``ACT2FN`` so ``config.hidden_act``
# stays a plain string in config.json.  A local table is used instead of inserting into ACT2FN,
# because a trust_remote_code file must not mutate library globals for the whole process.
_APERTUS_ACT2CLS: dict[str, type[nn.Module]] = {"sssglu": Apertus2SSSGLU}


def resolve_activation(name: str) -> nn.Module:
    """Look up ``name`` locally first, then in Transformers' ``ACT2FN``.

    ``ACT2FN`` is a ``ClassInstantier``: it overrides only ``__getitem__``, which constructs the
    module.  Indexing (never ``.get``, which would hand back the uninstantiated class) keeps both
    branches returning a ready-to-call ``nn.Module``.
    """
    local_cls = _APERTUS_ACT2CLS.get(name)
    return local_cls() if local_cls is not None else ACT2FN[name]


# Grouped-query attention helper.
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Give every query head its corresponding key/value head.

    ``[B, K, S, D] -> [B, K * n_rep, S, D]``.  ``expand`` avoids a copy until the final
    reshape.  Optimized attention backends perform this grouping internally; eager attention
    uses this explicit version.
    """
    batch_size, num_key_value_heads, sequence_length, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    expanded = hidden_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, n_rep, sequence_length, head_dim
    )
    return expanded.reshape(batch_size, num_key_value_heads * n_rep, sequence_length, head_dim)


# Reference attention backend.
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    """Compute scaled dot-product attention with explicit, easy-to-follow operations.

    Queries are ``[B, Q, S, D]`` and keys/values start as ``[B, K, S, D]``.  Attention scores
    are ``[B, Q, query_S, key_S]``; the returned context is ``[B, query_S, Q, D]``.
    """
    repeated_keys = repeat_kv(key, module.num_key_value_groups)
    repeated_values = repeat_kv(value, module.num_key_value_groups)

    attention_scores = torch.matmul(query, repeated_keys.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    # Softmax in float32 prevents overflow, then returns to the model's working dtype.
    attention_weights = F.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query.dtype)
    attention_weights = F.dropout(attention_weights, p=dropout, training=module.training)
    context = torch.matmul(attention_weights, repeated_values)
    context = context.transpose(1, 2).contiguous()

    return context, attention_weights


# Self-attention.
@use_kernelized_func(apply_rotary_pos_emb)
class Apertus2Attention(nn.Module):
    """Grouped-query self-attention with RMSNorm applied separately to every Q/K head.

    A projection first turns ``[B, S, H]`` into query ``[B, S, Q, D]`` and key/value
    ``[B, S, K, D]`` tensors.  RoPE and the attention backend operate after heads move to the
    second axis.  The output projection joins all query heads and restores ``[B, S, H]``.

    Two config schedules make layers differ.  ``layer_types[layer_idx]`` decides whether this
    layer attends to the whole prefix or only the last ``sliding_window`` keys, and
    ``no_rope_layers[layer_idx]`` decides whether its queries and keys are rotated at all.
    ``sliding_window`` is forwarded to the attention backend as well as shaping the mask,
    because the flash-attention path carries the window in that keyword and nowhere else.
    """

    def __init__(self, config: Apertus2Config, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None
        self.use_rope = bool(config.no_rope_layers[layer_idx])
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.rope_parameters = config.rope_parameters
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = Apertus2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Apertus2RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]

        # [B, S, H] -> [B, S, heads, D].
        head_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(head_shape)
        key_states = self.k_proj(hidden_states).view(head_shape)
        value_states = self.v_proj(hidden_states).view(head_shape)

        # RMSNorm reduces over D, so each head is normalized independently before RoPE.
        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        # Attention backends use [B, heads, S, D].
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # NoPE layers skip the rotation entirely; positions then reach them only through the
        # causal mask.  The guard has to wrap the call, not zero the tables, so that a kernelized
        # apply_rotary_pos_emb is skipped too.
        if self.use_rope:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attention_output, attention_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        # [B, S, Q, D] -> [B, S, Q * D] -> [B, S, H].
        attention_output = attention_output.reshape(*input_shape, -1).contiguous()
        attention_output = self.o_proj(attention_output)
        return attention_output, attention_weights


# Dense gated MLP.
class Apertus2MLP(nn.Module):
    """A gated feed-forward block used by dense layers and the shared expert.

    For every token: ``[H] -> gate/up [intermediate] -> elementwise product -> [H]``.
    The two input projections are separate here but correspond to Megatron's fused ``linear_fc1``.
    """

    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = resolve_activation(config.hidden_act)

    def forward(self, x):
        gated_features = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(gated_features)


# Router logits and top-k selection.
class Apertus2TopkRouter(nn.Module):
    """Score every routed expert and choose ``top_k`` experts for every token.

    ``[B, S, H]`` is flattened to ``[B * S, H]`` and projected with an ``[E, H]`` weight in
    float32, producing raw logits ``[B * S, E]``.  Selection lives here (not in
    ``Apertus2MoE``) so ``forward`` returns the triple Transformers' expert parallelism
    expects from a gate: ``(router_logits, top_k_weights, top_k_indices)``.  Under EP the
    ``ep_router`` plan entry rewrites that pair per rank — non-local weights become zero and
    non-local indices become the sentinel ``num_local_experts`` — and reads ``num_experts``
    from this module, so that attribute name is part of the contract.
    """

    def __init__(self, config: Apertus2Config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.use_quantile_balancing = config.use_quantile_balancing

        # The router sees full hidden states even when routed experts use a latent dimension.
        self.weight = nn.Parameter(torch.empty((self.num_experts, config.hidden_size)))
        # from_pretrained keeps selection offsets in fp32 so checkpoint loading does not round a
        # top-k boundary. A later explicit model.to(lower_dtype) follows normal PyTorch semantics.
        self.register_buffer("e_score_correction_bias", torch.zeros((self.num_experts), dtype=torch.float32))
        # QB replaces correction-bias selection.  Its buffer is absent when QB is disabled so
        # checkpoint keys continue to describe the architecture exactly.
        if config.use_quantile_balancing:
            self.register_buffer("qb_beta", torch.zeros((self.num_experts), dtype=torch.float32))

    def _select_experts_with_group_limit(self, selection_scores: torch.Tensor) -> torch.Tensor:
        """Choose top-k experts, optionally restricting each token to its best expert groups."""
        if self.n_group == 1 and self.topk_group == 1:
            return torch.topk(selection_scores, k=self.top_k, dim=-1, sorted=False).indices

        experts_per_group = self.num_experts // self.n_group
        experts_used_to_rank_group = self.top_k // self.topk_group

        # [tokens, E] -> [tokens, groups, experts_per_group].  A group's score is the sum of
        # its strongest candidates, matching Megatron's group_limited_topk.
        scores_by_group = selection_scores.view(-1, self.n_group, experts_per_group)
        strongest_in_each_group = scores_by_group.topk(experts_used_to_rank_group, dim=-1).values
        group_scores = strongest_in_each_group.sum(dim=-1)
        selected_groups = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False).indices

        selected_group_mask = torch.zeros_like(group_scores)
        selected_group_mask.scatter_(1, selected_groups, 1)
        selected_expert_mask = (
            selected_group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, experts_per_group)
            .reshape(-1, self.num_experts)
            .bool()
        )
        scores_in_selected_groups = selection_scores.masked_fill(~selected_expert_mask, float("-inf"))
        return torch.topk(scores_in_selected_groups, k=self.top_k, dim=-1, sorted=False).indices

    def route_tokens_to_experts(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Turn raw ``[tokens, E]`` router logits into expert ids and mixture weights.

        Selection offsets decide *which* experts run, but never alter their mixture weights:

        - standard routing selects from ``sigmoid(logits) + correction_bias``;
        - quantile balancing selects from raw ``logits - qb_beta``;
        - both gather weights from the same bias-free ``sigmoid(logits)`` tensor;
        - gathered weights are normalized first and multiplied by the routing scale last.
        """
        gate_scores = router_logits.sigmoid()
        if self.use_quantile_balancing:
            # QB and correction bias are mutually exclusive selection paths.
            selection_scores = router_logits - self.qb_beta
            selected_experts = torch.topk(selection_scores, k=self.top_k, dim=-1, sorted=False).indices
        else:
            selection_scores = gate_scores + self.e_score_correction_bias
            selected_experts = self._select_experts_with_group_limit(selection_scores)

        expert_weights = gate_scores.gather(1, selected_experts)
        if self.norm_topk_prob:
            expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-20)
        expert_weights = expert_weights * self.routed_scaling_factor
        return selected_experts, expert_weights

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = hidden_states.reshape(-1, self.config.hidden_size)
        # Megatron routes in float32 even when the rest of the model uses bf16/fp16.
        router_logits = F.linear(tokens.to(torch.float32), self.weight.to(torch.float32))
        selected_experts, expert_weights = self.route_tokens_to_experts(router_logits)
        return router_logits, expert_weights, selected_experts


# RMSNorm.
@use_kernel_forward_from_hub("RMSNorm")
class Apertus2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """Normalize each token by its root-mean-square over the final feature axis."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        float_states = hidden_states.to(torch.float32)
        mean_square = float_states.pow(2).mean(dim=-1, keepdim=True)
        normalized_states = float_states * torch.rsqrt(mean_square + self.variance_epsilon)
        return self.weight * normalized_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# Routed experts.
@use_experts_implementation
class Apertus2NaiveMoe(nn.Module):
    """Apply only the experts selected for each token, then sum their weighted outputs.

    The input is ``[tokens, expert_input_dim]``.  ``top_k_index`` and ``top_k_weights`` are both
    ``[tokens, top_k]``.  Parameters are stacked as ``[E, ...]`` for Transformers checkpoint
    conversion, while this readable fallback loops over only the experts that received tokens.
    Optimized kernels can replace this method through ``use_experts_implementation``.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.moe_latent_size if config.moe_latent_size else config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = resolve_activation(config.hidden_act)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        combined_output = torch.zeros_like(hidden_states)

        # [tokens, top_k] -> [E, top_k, tokens].  Each true entry tells us that a token chose an
        # expert in a particular route slot.  Routing decisions have no gradients.
        # ``num_experts`` is the LOCAL expert count under expert parallelism (the plan rewrites
        # the attribute after sharding), and index ``num_experts`` is the EP sentinel for
        # "routed to another rank's expert": encode it as one extra class, then drop it.
        with torch.no_grad():
            assignment_mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1)[..., : self.num_experts]
            assignment_mask = assignment_mask.permute(2, 1, 0)
            active_expert_indices = torch.greater(assignment_mask.sum(dim=(-1, -2)), 0).nonzero().flatten()

        for expert_index in active_expert_indices:
            route_slots, token_indices = torch.where(assignment_mask[expert_index])
            expert_inputs = hidden_states[token_indices]

            # One fused parameter holds both input projections: [token, 2 * intermediate].
            gate_features, up_features = F.linear(expert_inputs, self.gate_up_proj[expert_index]).chunk(2, dim=-1)
            expert_features = self.act_fn(gate_features) * up_features
            expert_output = F.linear(expert_features, self.down_proj[expert_index])

            # A token can visit several experts.  Weight each visit, then accumulate by token.
            route_weights = top_k_weights[token_indices, route_slots, None]
            weighted_output = expert_output * route_weights
            combined_output.index_add_(0, token_indices, weighted_output.to(combined_output.dtype))

        return combined_output


# Complete MoE block; selection lives on the router.
class Apertus2MoE(nn.Module):
    """Combine token-routed experts with one shared expert path.

    The order is deliberately explicit because LatentMoE has two parallel paths::

        full hidden [B, S, H] -> router -> top-k expert ids and weights
                              -> optional down projection -> routed experts
                              -> optional up projection ---------------------+
        full hidden [B, S, H] -> shared expert ------------------------------+ -> add

    The router and shared expert always receive the original full-width tensor.  Only routed
    experts use ``moe_latent_size``.  The returned tensor is always ``[B, S, H]``.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = Apertus2NaiveMoe(config)
        self.gate = Apertus2TopkRouter(config)
        self.shared_experts = Apertus2MLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )
        self.moe_latent_size = config.moe_latent_size
        if self.moe_latent_size:
            self.latent_down_proj = nn.Linear(config.hidden_size, self.moe_latent_size, bias=False)
            self.latent_up_proj = nn.Linear(self.moe_latent_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        full_hidden_states = hidden_states
        original_shape = hidden_states.shape

        # 1. Route with full-width features before any latent projection.  The gate selects
        #    as well as scores; its raw logits are not needed further here.
        _, expert_weights, selected_experts = self.gate(full_hidden_states)

        # 2. Run routed experts, optionally in latent space.
        if self.moe_latent_size:
            hidden_states = self.latent_down_proj(hidden_states)
        expert_inputs = hidden_states.reshape(-1, hidden_states.shape[-1])
        routed_output = self.experts(expert_inputs, selected_experts, expert_weights)
        routed_output = routed_output.view(*original_shape[:-1], -1)
        if self.moe_latent_size:
            routed_output = self.latent_up_proj(routed_output)

        # 3. The shared expert sees full-width input and is added after routed output returns to H.
        shared_output = self.shared_experts(full_hidden_states)
        return routed_output + shared_output


# Decoder layer.
class Apertus2DecoderLayer(GradientCheckpointingLayer):
    """One attention block followed by one dense or MoE feed-forward block.

    Every input, branch output, and result has shape ``[B, S, H]``.  The layer supports three
    residual rules, selected once by the config:

    - plain: ``x + residual_multiplier * branch(pre_norm(x))``;
    - sandwich: ``x + residual_multiplier * post_norm(branch(pre_norm(x)))``;
    - KEEL: ``post_norm(keel_alpha * x + branch(pre_norm(x)))``.

    KEEL layer zero intentionally omits the attention post-norm and uses carry scale 1, matching
    Megatron.  ``_merge_branch_with_residual`` is the single place implementing these formulas.
    """

    def __init__(self, config: Apertus2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.sandwich_norm = config.sandwich_norm
        self.keel = config.keel
        self.residual_multiplier = config.residual_multiplier

        self.self_attn = Apertus2Attention(config=config, layer_idx=layer_idx)

        if config.is_moe_layer(layer_idx):
            self.mlp = Apertus2MoE(config)
        else:
            self.mlp = Apertus2MLP(config)

        # Checkpoint naming differs from Llama/GLM: feedforward_layernorm is the MLP pre-norm;
        # post_attention_layernorm really is the attention post-norm.
        self.attention_layernorm = Apertus2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feedforward_layernorm = Apertus2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Absent modules stay absent rather than becoming Identity placeholders.  This keeps
        # state-dict keys an exact description of the architecture.
        if config.sandwich_norm:
            self.post_attention_layernorm = Apertus2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_feedforward_layernorm = Apertus2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        elif self.keel:
            self.keel_first_layer = layer_idx == 0
            keel_alpha = (
                config.keel_alpha if config.keel_alpha is not None else float(2 * config.num_hidden_layers)
            )
            self.keel_residual_scale = 1.0 if self.keel_first_layer else keel_alpha
            if not self.keel_first_layer:
                self.post_attention_layernorm = Apertus2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_feedforward_layernorm = Apertus2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _merge_branch_with_residual(
        self,
        residual_stream: torch.Tensor,
        branch_output: torch.Tensor,
        post_norm: nn.Module | None,
    ) -> torch.Tensor:
        """Apply the configured residual formula to one ``[B, S, H]`` branch output."""
        if self.keel:
            # KEEL normalizes after adding the branch.  ``post_norm is None`` is the intentional
            # first-layer attention special case.
            combined_states = self.keel_residual_scale * residual_stream + branch_output
            return combined_states if post_norm is None else post_norm(combined_states)

        # Sandwich norm is applied before residual_multiplier.  RMSNorm would cancel most of a
        # multiplier applied before it, so this order is part of the checkpoint's model math.
        if post_norm is not None:
            branch_output = post_norm(branch_output)
        return residual_stream + self.residual_multiplier * branch_output

    def _run_attention_block(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None,
        past_key_values: Cache | None,
        use_cache: bool | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """Read context from earlier tokens, then update the residual stream."""
        residual_stream = hidden_states
        normalized_states = self.attention_layernorm(hidden_states)
        attention_output, _ = self.self_attn(
            hidden_states=normalized_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        post_norm = getattr(self, "post_attention_layernorm", None)
        return self._merge_branch_with_residual(residual_stream, attention_output, post_norm)

    def _run_feed_forward_block(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Transform each token independently with a dense MLP or the MoE block."""
        residual_stream = hidden_states
        normalized_states = self.feedforward_layernorm(hidden_states)
        feed_forward_output = self.mlp(normalized_states)
        post_norm = getattr(self, "post_feedforward_layernorm", None)
        return self._merge_branch_with_residual(residual_stream, feed_forward_output, post_norm)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        hidden_states = self._run_attention_block(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        return self._run_feed_forward_block(hidden_states)


# Transformers integration and initialization
# Model integration hooks.
class Apertus2PreTrainedModel(PreTrainedModel):
    """Shared Transformers hooks for loading, attention backends, and weight initialization."""

    config: Apertus2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Apertus2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Apertus2DecoderLayer,
        "attentions": Apertus2Attention,
    }
    # Router offsets must survive low-precision checkpoint loading in float32.
    _keep_in_fp32_modules_strict = ["e_score_correction_bias", "qb_beta"]

    def _reject_tensor_parallel(self) -> None:
        """Fail loudly on TP ranks: tensor parallelism is intentionally unsupported for now.

        No ``base_model_tp_plan`` is published, and with an absent plan
        ``from_pretrained(tp_plan="auto")`` silently replicates the full model on every rank
        instead of sharding it.  Commit a465e6e holds a working replicated-KV TP
        implementation and tests if support is revived.  Expert parallelism is exempt: it
        runs on the same device mesh but shards via ``base_model_ep_plan``.
        """
        if self.tp_size is None or self.tp_size <= 1:
            return
        distributed_config = getattr(self.config, "distributed_config", None)
        if distributed_config is not None and distributed_config.enable_expert_parallel:
            return
        raise ValueError(
            f"Apertus2 supports only TP=1 for now (got tp_size={self.tp_size}); "
            "load the model without tp_plan/tp_size, or use expert parallelism via "
            "distributed_config=DistributedConfig(enable_expert_parallel=True)."
        )

    @torch.no_grad()
    def _init_weights(self, module):
        # Transformers initializes normal Linear/Embedding/RMSNorm modules.
        super()._init_weights(module)
        # The router and stacked expert parameters are raw nn.Parameters, so initialize them here.
        if isinstance(module, Apertus2TopkRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            init.zeros_(module.e_score_correction_bias)
            if getattr(module, "qb_beta", None) is not None:  # QB only (buffer absent otherwise)
                init.zeros_(module.qb_beta)
        elif isinstance(module, Apertus2NaiveMoe):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)


# Base decoder model.
class Apertus2Model(Apertus2PreTrainedModel):
    """Map token ids to contextual hidden states of shape ``[B, S, H]``."""

    def __init__(self, config: Apertus2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Apertus2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Apertus2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.has_sliding_layers = "sliding_attention" in config.layer_types
        self.gradient_checkpointing = False

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        self._reject_tensor_parallel()
        # Supplying both is ambiguous; supplying neither leaves the model without an input.
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # One mask per distinct layer type, because sliding layers admit fewer keys than full
        # ones.  `generate` with a compiled cache prepares this dictionary itself and passes it in
        # as attention_mask, so an already-built mapping is used as-is.
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        # [B, S, H].  The fixed Megatron embedding scale applies to ids and prebuilt embeddings.
        hidden_states = inputs_embeds * self.config.embedding_multiplier
        # RoPE tables depend only on positions, so all layers share this pair [B, S, rotary_dim].
        # NoPE layers ignore it; the tables are still built once because most layers use them.
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        # Every layer preserves [B, S, H].  The slice is retained for Transformers pipeline plans,
        # which may adjust the configured layer range.
        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[layer_idx]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


# Causal language-model head.
class Apertus2ForCausalLM(Apertus2PreTrainedModel, GenerationMixin):
    """Add an untied vocabulary projection and optional next-token loss to the decoder."""

    # Untied (--untie-embeddings-and-output-weights); lm_head has its own parameter.
    _tied_weights_keys = {}
    def __init__(self, config):
        super().__init__(config)
        self.model = Apertus2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()
        # PreTrainedModel.post_init() rebuilds composite plans from child modules, so class-level
        # entries would be discarded. Add top-level modules after that merge.
        self._pp_plan["lm_head"] = (["hidden_states"], ["logits"])

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        self._reject_tensor_parallel()
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # logits_to_keep=0 means all positions.  Generation usually requests only the last token,
        # avoiding a large [B, S, vocab] projection for positions it will not use.
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        selected_hidden_states = hidden_states[:, slice_indices, :]
        logits = self.lm_head(selected_hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# This import-time registration makes save_pretrained copy this implementation and lets
# AutoModelForCausalLM reload the directory with trust_remote_code=True.
Apertus2ForCausalLM.register_for_auto_class("AutoModelForCausalLM")
# The bare decoder is registered too, so a saved directory also serves consumers that want the
# BACKBONE rather than the LM head. Without this, AutoModel.from_pretrained on an exported dir
# raises "Unrecognized configuration class ... for this kind of AutoModel".
Apertus2Model.register_for_auto_class("AutoModel")


__all__ = ["Apertus2PreTrainedModel", "Apertus2Model", "Apertus2ForCausalLM"]
