"""Layout unit tests for ``exporter/transforms.py``.

INDEPENDENCE RULE: every expected layout here is built BY HAND from the fork spec —
per-KV-group [q ... q, k, v] blocks of head_dim rows each (Megatron-LM-MoE
attention.py:1408-1409 "q1 q2 k1 v1 | q3 q4 k2 v2 | ...") and [gate; up] row halves
(fused_bias_swiglu.py:15-18, mlp.py:432, gate stored first). megatron_mock's merge helpers
are deliberately NOT used: a shared wrong assumption there would cancel out here.

Covered: 2D weights + 1D bias qkv split; GQA / MQA-ish (kv=1) / kv==heads geometries;
gate/up split order; stacked-expert per-slice splitting; dtype preservation (bf16 in ->
bf16 out); shape-assert failures (indivisible head counts, wrong row totals, odd fc1 rows).
"""

import pytest

pytest.importorskip("megatron.core")

import torch  # noqa: E402

from exporter.transforms import split_gated_fc1, split_qkv, split_qkv_gate  # noqa: E402

HIDDEN = 32
HEAD_DIM = 8

# Distinct per-head constants, all < 256 so they are exact in bf16 too.
Q_FILL = 1  # q head h -> Q_FILL + h
K_FILL = 101  # kv group g -> K_FILL + g
V_FILL = 201  # kv group g -> V_FILL + g
G_FILL = 51  # gate head h -> G_FILL + h (attention_output_gate)


def build_fused_qkv_2d(num_q_heads, num_kv_heads, head_dim=HEAD_DIM, hidden=HIDDEN,
                       dtype=torch.float32):
    """Hand-built fork layout: per KV group, heads_per_group q blocks, then k, then v."""
    heads_per_group = num_q_heads // num_kv_heads
    blocks = []
    for group in range(num_kv_heads):
        for j in range(heads_per_group):
            q_head = group * heads_per_group + j
            blocks.append(torch.full((head_dim, hidden), float(Q_FILL + q_head), dtype=dtype))
        blocks.append(torch.full((head_dim, hidden), float(K_FILL + group), dtype=dtype))
        blocks.append(torch.full((head_dim, hidden), float(V_FILL + group), dtype=dtype))
    return torch.cat(blocks, dim=0)


def build_fused_qkv_1d(num_q_heads, num_kv_heads, head_dim=HEAD_DIM, dtype=torch.float32):
    heads_per_group = num_q_heads // num_kv_heads
    blocks = []
    for group in range(num_kv_heads):
        for j in range(heads_per_group):
            q_head = group * heads_per_group + j
            blocks.append(torch.full((head_dim,), float(Q_FILL + q_head), dtype=dtype))
        blocks.append(torch.full((head_dim,), float(K_FILL + group), dtype=dtype))
        blocks.append(torch.full((head_dim,), float(V_FILL + group), dtype=dtype))
    return torch.cat(blocks, dim=0)


def _assert_qkv_pieces(q, k, v, num_q_heads, num_kv_heads, head_dim, hidden=None):
    """q must be per-head-ordered [q0..qN], k/v per-group-ordered [k0..kG]/[v0..vG]."""
    lead_q = num_q_heads * head_dim
    lead_kv = num_kv_heads * head_dim
    if hidden is None:  # 1D bias
        assert q.shape == (lead_q,)
        assert k.shape == (lead_kv,)
        assert v.shape == (lead_kv,)
    else:
        assert q.shape == (lead_q, hidden)
        assert k.shape == (lead_kv, hidden)
        assert v.shape == (lead_kv, hidden)
    for head in range(num_q_heads):
        block = q[head * head_dim : (head + 1) * head_dim]
        assert torch.equal(block, torch.full_like(block, float(Q_FILL + head))), f"q head {head}"
    for group in range(num_kv_heads):
        k_block = k[group * head_dim : (group + 1) * head_dim]
        v_block = v[group * head_dim : (group + 1) * head_dim]
        assert torch.equal(k_block, torch.full_like(k_block, float(K_FILL + group))), (
            f"k group {group}"
        )
        assert torch.equal(v_block, torch.full_like(v_block, float(V_FILL + group))), (
            f"v group {group}"
        )


GEOMETRIES = [
    pytest.param(4, 2, id="gqa-4q-2kv"),
    pytest.param(4, 1, id="mqa-ish-4q-1kv"),
    pytest.param(4, 4, id="mha-4q-4kv"),
]


class TestSplitQkv:
    @pytest.mark.parametrize("num_q_heads,num_kv_heads", GEOMETRIES)
    def test_2d_weight_split(self, num_q_heads, num_kv_heads):
        fused = build_fused_qkv_2d(num_q_heads, num_kv_heads)
        q, k, v = split_qkv(fused, num_q_heads, num_kv_heads, HEAD_DIM)
        _assert_qkv_pieces(q, k, v, num_q_heads, num_kv_heads, HEAD_DIM, hidden=HIDDEN)

    @pytest.mark.parametrize("num_q_heads,num_kv_heads", GEOMETRIES)
    def test_1d_bias_split(self, num_q_heads, num_kv_heads):
        fused = build_fused_qkv_1d(num_q_heads, num_kv_heads)
        q, k, v = split_qkv(fused, num_q_heads, num_kv_heads, HEAD_DIM)
        _assert_qkv_pieces(q, k, v, num_q_heads, num_kv_heads, HEAD_DIM, hidden=None)

    def test_random_content_exact_row_gather(self):
        """Bitwise row-level check on random data: rows are gathered per the fork layout,
        never transformed."""
        torch.manual_seed(7)
        num_q_heads, num_kv_heads, head_dim = 4, 2, HEAD_DIM
        heads_per_group = num_q_heads // num_kv_heads
        fused = torch.randn((num_q_heads + 2 * num_kv_heads) * head_dim, HIDDEN)
        q, k, v = split_qkv(fused, num_q_heads, num_kv_heads, head_dim)
        group_rows = (heads_per_group + 2) * head_dim
        for group in range(num_kv_heads):
            base = group * group_rows
            expected_q = fused[base : base + heads_per_group * head_dim]
            expected_k = fused[base + heads_per_group * head_dim : base + (heads_per_group + 1) * head_dim]
            expected_v = fused[base + (heads_per_group + 1) * head_dim : base + group_rows]
            assert torch.equal(
                q[group * heads_per_group * head_dim : (group + 1) * heads_per_group * head_dim],
                expected_q,
            )
            assert torch.equal(k[group * head_dim : (group + 1) * head_dim], expected_k)
            assert torch.equal(v[group * head_dim : (group + 1) * head_dim], expected_v)

    def test_dtype_preserved_bf16(self):
        fused = build_fused_qkv_2d(4, 2, dtype=torch.bfloat16)
        q, k, v = split_qkv(fused, 4, 2, HEAD_DIM)
        assert q.dtype == k.dtype == v.dtype == torch.bfloat16
        _assert_qkv_pieces(q, k, v, 4, 2, HEAD_DIM, hidden=HIDDEN)

    def test_rejects_indivisible_head_counts(self):
        fused = build_fused_qkv_2d(4, 2)
        with pytest.raises((AssertionError, ValueError)):
            split_qkv(fused, 4, 3, HEAD_DIM)  # 4 q heads not divisible into 3 kv groups

    def test_rejects_wrong_row_total(self):
        fused = torch.randn(60, HIDDEN)  # (4 + 2*2) * 8 == 64 expected
        with pytest.raises((AssertionError, ValueError)):
            split_qkv(fused, 4, 2, HEAD_DIM)


def build_fused_qkv_gate(num_q_heads, num_kv_heads, head_dim=HEAD_DIM, hidden=HIDDEN,
                         dtype=torch.float32):
    """Hand-built gated fork layout (attention.py "(2 * np/ng + 2) * hn"): per KV group,
    heads_per_group q blocks, then heads_per_group gate blocks, then k, then v."""
    heads_per_group = num_q_heads // num_kv_heads
    blocks = []
    for group in range(num_kv_heads):
        for j in range(heads_per_group):
            q_head = group * heads_per_group + j
            blocks.append(torch.full((head_dim, hidden), float(Q_FILL + q_head), dtype=dtype))
        for j in range(heads_per_group):
            g_head = group * heads_per_group + j
            blocks.append(torch.full((head_dim, hidden), float(G_FILL + g_head), dtype=dtype))
        blocks.append(torch.full((head_dim, hidden), float(K_FILL + group), dtype=dtype))
        blocks.append(torch.full((head_dim, hidden), float(V_FILL + group), dtype=dtype))
    return torch.cat(blocks, dim=0)


class TestSplitQkvGate:
    @pytest.mark.parametrize("num_q_heads,num_kv_heads", GEOMETRIES)
    def test_2d_weight_split(self, num_q_heads, num_kv_heads):
        fused = build_fused_qkv_gate(num_q_heads, num_kv_heads)
        q, g, k, v = split_qkv_gate(fused, num_q_heads, num_kv_heads, HEAD_DIM)
        _assert_qkv_pieces(q, k, v, num_q_heads, num_kv_heads, HEAD_DIM, hidden=HIDDEN)
        # gate blocks come out in the same global head order as query blocks
        assert g.shape == (num_q_heads * HEAD_DIM, HIDDEN)
        for head in range(num_q_heads):
            block = g[head * HEAD_DIM : (head + 1) * HEAD_DIM]
            assert torch.equal(block, torch.full_like(block, float(G_FILL + head))), (
                f"gate head {head}"
            )

    def test_random_content_exact_row_gather(self):
        torch.manual_seed(13)
        num_q_heads, num_kv_heads, head_dim = 4, 2, HEAD_DIM
        heads_per_group = num_q_heads // num_kv_heads
        fused = torch.randn((2 * num_q_heads + 2 * num_kv_heads) * head_dim, HIDDEN)
        q, g, k, v = split_qkv_gate(fused, num_q_heads, num_kv_heads, head_dim)
        group_rows = (2 * heads_per_group + 2) * head_dim
        q_rows = heads_per_group * head_dim
        for group in range(num_kv_heads):
            base = group * group_rows
            assert torch.equal(q[group * q_rows : (group + 1) * q_rows], fused[base : base + q_rows])
            assert torch.equal(
                g[group * q_rows : (group + 1) * q_rows], fused[base + q_rows : base + 2 * q_rows]
            )
            assert torch.equal(
                k[group * head_dim : (group + 1) * head_dim],
                fused[base + 2 * q_rows : base + 2 * q_rows + head_dim],
            )
            assert torch.equal(
                v[group * head_dim : (group + 1) * head_dim],
                fused[base + 2 * q_rows + head_dim : base + group_rows],
            )

    def test_dtype_preserved_bf16(self):
        fused = build_fused_qkv_gate(4, 2, dtype=torch.bfloat16)
        q, g, k, v = split_qkv_gate(fused, 4, 2, HEAD_DIM)
        assert q.dtype == g.dtype == k.dtype == v.dtype == torch.bfloat16

    def test_rejects_ungated_row_total(self):
        """A fused weight WITHOUT the gate must not slip through the gated split."""
        fused = build_fused_qkv_2d(4, 2)  # (4 + 2*2) * 8 rows; gated expects (8 + 4) * 8
        with pytest.raises((AssertionError, ValueError)):
            split_qkv_gate(fused, 4, 2, HEAD_DIM)

    def test_rejects_indivisible_head_counts(self):
        fused = build_fused_qkv_gate(4, 2)
        with pytest.raises((AssertionError, ValueError)):
            split_qkv_gate(fused, 4, 3, HEAD_DIM)


class TestSplitGatedFc1:
    def test_gate_first_then_up(self):
        intermediate = 16
        gate = torch.full((intermediate, HIDDEN), 7.0)
        up = torch.full((intermediate, HIDDEN), 11.0)
        fused = torch.cat([gate, up], dim=0)
        out_gate, out_up = split_gated_fc1(fused)
        assert torch.equal(out_gate, gate)
        assert torch.equal(out_up, up)

    def test_random_content_exact_halves(self):
        torch.manual_seed(11)
        fused = torch.randn(2 * 64, HIDDEN)
        out_gate, out_up = split_gated_fc1(fused)
        assert torch.equal(out_gate, fused[:64])
        assert torch.equal(out_up, fused[64:])

    def test_per_expert_slice_of_stacked_tensor(self):
        """mapping consumes the stacked [E, 2F, in_e] tensor one expert-slice at a time."""
        num_experts, intermediate, in_dim = 8, 16, 24
        expert_slices = []
        for e in range(num_experts):
            gate = torch.full((intermediate, in_dim), float(10 * e + 7))
            up = torch.full((intermediate, in_dim), float(10 * e + 11))
            expert_slices.append(torch.cat([gate, up], dim=0))
        stacked = torch.stack(expert_slices, dim=0)
        for e in range(num_experts):
            out_gate, out_up = split_gated_fc1(stacked[e])
            assert torch.equal(out_gate, torch.full((intermediate, in_dim), float(10 * e + 7)))
            assert torch.equal(out_up, torch.full((intermediate, in_dim), float(10 * e + 11)))

    def test_dtype_preserved_bf16(self):
        fused = torch.cat(
            [torch.full((16, HIDDEN), 7.0), torch.full((16, HIDDEN), 11.0)], dim=0
        ).to(torch.bfloat16)
        out_gate, out_up = split_gated_fc1(fused)
        assert out_gate.dtype == out_up.dtype == torch.bfloat16
        assert torch.equal(out_gate, torch.full((16, HIDDEN), 7.0, dtype=torch.bfloat16))

    def test_rejects_odd_row_count(self):
        with pytest.raises((AssertionError, ValueError)):
            split_gated_fc1(torch.randn(33, HIDDEN))
