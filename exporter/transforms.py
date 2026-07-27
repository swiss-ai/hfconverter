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
"""Small, pure tensor transforms used by the checkpoint exporter.

Megatron fuses some matrices for efficient training, while Hugging Face stores the logical
projections separately. These functions only rearrange or slice values; they never change dtype.
"""

import torch


def split_qkv(
    qkv: torch.Tensor, num_q_heads: int, num_kv_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a Megatron fused QKV weight or bias into Hugging Face q, k, and v tensors.

    Megatron groups rows by KV head. Each group is laid out as::

        [query head rows..., key head rows, value head rows]

    Hugging Face instead expects all query rows together, then all key rows, then all value rows.
    For a weight the input shape is ``[(Q + 2*K)*D, hidden]``; a bias omits ``hidden``.
    """
    if num_q_heads <= 0 or num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"split_qkv: num_q_heads={num_q_heads} must be a positive multiple of "
            f"num_kv_heads={num_kv_heads}"
        )
    if qkv.dim() not in (1, 2):
        raise ValueError(f"split_qkv: expected a 2D weight or 1D bias, got shape {tuple(qkv.shape)}")
    query_heads_per_kv_group = num_q_heads // num_kv_heads
    total_head_blocks = num_q_heads + 2 * num_kv_heads
    blocks_per_kv_group = query_heads_per_kv_group + 2
    expected_first_dimension = total_head_blocks * head_dim
    if qkv.shape[0] != expected_first_dimension:
        raise ValueError(
            f"split_qkv: dim 0 is {qkv.shape[0]}, expected (num_q_heads + 2*num_kv_heads) * "
            f"head_dim = ({num_q_heads} + 2*{num_kv_heads}) * {head_dim} = "
            f"{expected_first_dimension}"
        )

    remaining_shape = qkv.shape[1:]  # (hidden,) for a weight, empty for a bias
    head_blocks = qkv.reshape(total_head_blocks, head_dim, *remaining_shape)
    query_indices = torch.cat(
        [
            torch.arange(
                blocks_per_kv_group * group,
                blocks_per_kv_group * group + query_heads_per_kv_group,
            )
            for group in range(num_kv_heads)
        ]
    )
    key_indices = torch.arange(
        query_heads_per_kv_group, total_head_blocks, blocks_per_kv_group
    )
    value_indices = torch.arange(
        query_heads_per_kv_group + 1, total_head_blocks, blocks_per_kv_group
    )

    query = head_blocks[query_indices].reshape(num_q_heads * head_dim, *remaining_shape)
    key = head_blocks[key_indices].reshape(num_kv_heads * head_dim, *remaining_shape)
    value = head_blocks[value_indices].reshape(num_kv_heads * head_dim, *remaining_shape)

    return query, key, value


def split_gated_fc1(fc1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split Megatron's fused SwiGLU input projection into HF gate and up projections.

    Megatron stores ``[gate rows, up rows]`` along dimension 0. Both returned tensors are views and
    keep the input dtype; the writer later makes owned contiguous copies for safetensors.
    """
    if fc1.dim() < 1 or fc1.shape[0] % 2 != 0:
        raise ValueError(f"split_gated_fc1: dim 0 of shape {tuple(fc1.shape)} is not even")
    gate, up = torch.chunk(fc1, 2, dim=0)
    return gate, up
