"""Shared fixtures and tiny geometry for the exported Hugging Face model tests."""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest
import torch

from configuration_apertus_moe import ApertusMoeConfig
from modeling_apertus_moe import ApertusMoeForCausalLM

# ---------------------------------------------------------------------------
# Tiny geometry shared by the model tests:
#   hidden_size=32, intermediate_size=64, num_hidden_layers=3, heads=4, kv=2,
#   head_dim=8, n_routed_experts=8, num_experts_per_tok=2,
#   moe_intermediate_size=16, moe_latent_size=24 (when on), vocab_size=128,
#   max_position_embeddings=64, first_k_dense_replace=1.
# Multipliers: model defaults for math tests; 1.0 only for the stock oracle.
# ---------------------------------------------------------------------------
TINY_VOCAB = 128
TINY_HIDDEN = 32
TINY_INTERMEDIATE = 64
TINY_LAYERS = 3
TINY_HEADS = 4
TINY_KV_HEADS = 2
TINY_HEAD_DIM = 8
TINY_N_EXPERTS = 8
TINY_TOPK = 2
TINY_MOE_INTERMEDIATE = 16
TINY_LATENT = 24
TINY_MAX_POS = 64
TINY_FIRST_K_DENSE = 1

# ApertusMoeConfig defaults used by the model tests.
EMBEDDING_MULTIPLIER = 27.712812921102035
RESIDUAL_MULTIPLIER = 0.22360679774997896
ROPE_THETA = 500000.0

TINY_KWARGS = dict(
    vocab_size=TINY_VOCAB,
    hidden_size=TINY_HIDDEN,
    intermediate_size=TINY_INTERMEDIATE,
    num_hidden_layers=TINY_LAYERS,
    num_attention_heads=TINY_HEADS,
    num_key_value_heads=TINY_KV_HEADS,
    head_dim=TINY_HEAD_DIM,
    max_position_embeddings=TINY_MAX_POS,
    rms_norm_eps=1e-5,
    hidden_act="silu",
    attention_bias=False,
    attention_dropout=0.0,
    tie_word_embeddings=False,
    rope_parameters={
        "rope_type": "default",
        "rope_theta": ROPE_THETA,
        "partial_rotary_factor": 1.0,
    },
    use_qk_norm=True,
    n_routed_experts=TINY_N_EXPERTS,
    num_experts_per_tok=TINY_TOPK,
    moe_intermediate_size=TINY_MOE_INTERMEDIATE,
    n_shared_experts=1,
    first_k_dense_replace=TINY_FIRST_K_DENSE,
    routed_scaling_factor=2.5,
    norm_topk_prob=True,
    n_group=1,
    topk_group=1,
    embedding_multiplier=EMBEDDING_MULTIPLIER,
    residual_multiplier=RESIDUAL_MULTIPLIER,
    initializer_range=0.02,
    use_cache=True,
)

# Full {sandwich_norm} x {moe_latent_size} matrix.
SANDWICH_LATENT_MATRIX = [
    pytest.param(False, None, id="plain"),
    pytest.param(False, TINY_LATENT, id="latent"),
    pytest.param(True, None, id="sandwich"),
    pytest.param(True, TINY_LATENT, id="sandwich-latent"),
]


@pytest.fixture(autouse=True)
def _seed_torch():
    """Deterministic weights/inputs for every test."""
    torch.manual_seed(0)
    yield


@pytest.fixture
def make_config():
    """Factory for tiny ApertusMoeConfig covering the 2x2 feature matrix."""

    def _make(sandwich_norm=False, moe_latent_size=None, **overrides):
        kwargs = dict(TINY_KWARGS)
        kwargs["sandwich_norm"] = sandwich_norm
        kwargs["moe_latent_size"] = moe_latent_size
        kwargs.update(overrides)
        return ApertusMoeConfig(**kwargs)

    return _make


@pytest.fixture
def make_model(make_config):
    """Factory for a tiny, seeded, eval-mode ApertusMoeForCausalLM."""

    def _make(sandwich_norm=False, moe_latent_size=None, seed=0, **overrides):
        torch.manual_seed(seed)
        config = make_config(sandwich_norm, moe_latent_size, **overrides)
        model = ApertusMoeForCausalLM(config)
        model.eval()
        return model

    return _make


@pytest.fixture
def input_ids():
    """Fixed token ids, [batch=2, seq=7]."""
    generator = torch.Generator().manual_seed(1234)
    return torch.randint(0, TINY_VOCAB, (2, 7), generator=generator)
