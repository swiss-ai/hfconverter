"""Acceptance tests for the custom Hugging Face Apertus MoE implementation."""

import glob
import json
import os

import pytest
import torch
from safetensors import safe_open

from transformers import AutoModelForCausalLM, Glm4MoeConfig, Glm4MoeForCausalLM
from transformers.conversion_mapping import get_checkpoint_conversion_mapping

from configuration_apertus_moe import ApertusMoeConfig
from modeling_apertus_moe import (
    ApertusMoeForCausalLM,
    ApertusMoeMLP,
    ApertusMoeMoE,
    ApertusMoeRMSNorm,
    resolve_activation,
)
from conftest import (
    EMBEDDING_MULTIPLIER,
    RESIDUAL_MULTIPLIER,
    ROPE_THETA,
    SANDWICH_LATENT_MATRIX,
    TINY_FIRST_K_DENSE,
    TINY_HEAD_DIM,
    TINY_HEADS,
    TINY_HIDDEN,
    TINY_INTERMEDIATE,
    TINY_KV_HEADS,
    TINY_LATENT,
    TINY_LAYERS,
    TINY_MAX_POS,
    TINY_MOE_INTERMEDIATE,
    TINY_N_EXPERTS,
    TINY_TOPK,
    TINY_VOCAB,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _as_tensor(out):
    """Decoder layers / attention may return a tensor or a tuple whose first
    element is the hidden states."""
    return out[0] if isinstance(out, tuple) else out


def _first_tensor(args, kwargs):
    for a in args:
        if isinstance(a, torch.Tensor):
            return a
    for a in kwargs.values():
        if isinstance(a, torch.Tensor):
            return a
    raise AssertionError("hooked module received no tensor input")


def _rope(model, x):
    """Position embeddings for a direct decoder-layer call."""
    position_ids = torch.arange(x.shape[1]).unsqueeze(0)
    return model.model.rotary_emb(x, position_ids)


def _handrolled_layer_forward(
    layer,
    x,
    position_embeddings,
    *,
    sandwich,
    alpha_attn,
    alpha_mlp,
    post_norm_first=True,
):
    """Reference forward math:

        h = pre_norm(x); h = sublayer(h)
        if sandwich: h = post_norm(h)   # norm FIRST
        h = h * residual_multiplier     # alpha AFTER the norm
        x = residual + h

    ``post_norm_first=False`` computes the WRONG order (multiplier before the
    post-norm) used to prove the correct order is load-bearing.
    """

    def _branch(h, post_norm, alpha):
        if sandwich and post_norm_first:
            h = post_norm(h)
            h = h * alpha
        elif sandwich:  # wrong order: alpha then norm
            h = h * alpha
            h = post_norm(h)
        else:
            h = h * alpha
        return h

    residual = x
    h = layer.attention_layernorm(x)
    h = _as_tensor(layer.self_attn(h, position_embeddings, None))
    h = _branch(h, getattr(layer, "post_attention_layernorm", None), alpha_attn)
    x = residual + h

    residual = x
    h = layer.feedforward_layernorm(x)
    h = _as_tensor(layer.mlp(h))
    h = _branch(h, getattr(layer, "post_feedforward_layernorm", None), alpha_mlp)
    return residual + h


def _call_layer(layer, x, position_embeddings):
    return _as_tensor(
        layer(x, attention_mask=None, position_embeddings=position_embeddings)
    )


def _load_saved_state_dict(directory):
    """All tensors from every *.safetensors shard save_pretrained wrote."""
    state_dict = {}
    files = sorted(glob.glob(os.path.join(str(directory), "*.safetensors")))
    assert files, f"no safetensors files written in {directory}"
    for path in files:
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict


def _expected_disk_keys(config):
    """The exact per-expert on-disk checkpoint layout."""
    keys = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for L in range(config.num_hidden_layers):
        p = f"model.layers.{L}."
        keys.add(p + "attention_layernorm.weight")
        keys.add(p + "feedforward_layernorm.weight")
        for proj in ("q", "k", "v", "o"):
            keys.add(p + f"self_attn.{proj}_proj.weight")
        keys.add(p + "self_attn.q_norm.weight")
        keys.add(p + "self_attn.k_norm.weight")
        if config.sandwich_norm:
            keys.add(p + "post_attention_layernorm.weight")
            keys.add(p + "post_feedforward_layernorm.weight")
        if config.is_moe_layer(L):
            keys.add(p + "mlp.gate.weight")
            keys.add(p + "mlp.gate.e_score_correction_bias")
            if config.use_quantile_balancing:
                keys.add(p + "mlp.gate.qb_beta")
            if config.moe_latent_size is not None:
                keys.add(p + "mlp.latent_down_proj.weight")
                keys.add(p + "mlp.latent_up_proj.weight")
            for E in range(config.n_routed_experts):
                for proj in ("gate", "up", "down"):
                    keys.add(p + f"mlp.experts.{E}.{proj}_proj.weight")
            for proj in ("gate", "up", "down"):
                keys.add(p + f"mlp.shared_experts.{proj}_proj.weight")
        else:
            for proj in ("gate", "up", "down"):
                keys.add(p + f"mlp.{proj}_proj.weight")
    return keys


def _moe_layers(model):
    cfg = model.config
    return [
        model.model.layers[L]
        for L in range(cfg.num_hidden_layers)
        if cfg.is_moe_layer(L)
    ]


# ---------------------------------------------------------------------------
# contract plumbing: config defaults, class flags, conversion registration
# ---------------------------------------------------------------------------


class TestContractPlumbing:
    def test_config_defaults_match_contract(self):
        cfg = ApertusMoeConfig()
        assert cfg.model_type == "apertus_moe"
        assert cfg.vocab_size == 200064  # exactly 1563*128, zero padding
        assert cfg.hidden_size == 768
        assert cfg.intermediate_size == 1920
        assert cfg.num_hidden_layers == 10
        assert cfg.num_attention_heads == 6
        assert cfg.num_key_value_heads == 3
        assert cfg.head_dim == 128
        assert cfg.max_position_embeddings == 8192
        assert cfg.rms_norm_eps == 1e-5
        assert cfg.hidden_act == "silu"
        assert cfg.attention_bias is False
        assert cfg.attention_dropout == 0.0
        assert cfg.use_qk_norm is True
        assert cfg.n_routed_experts == 128
        assert cfg.num_experts_per_tok == 4
        assert cfg.moe_intermediate_size == 448
        assert cfg.n_shared_experts == 1
        assert cfg.first_k_dense_replace == 1
        assert cfg.moe_layer_freq is None
        assert cfg.routed_scaling_factor == 2.5
        assert cfg.norm_topk_prob is True
        assert cfg.n_group == 1
        assert cfg.topk_group == 1
        assert cfg.sandwich_norm is False
        assert cfg.moe_latent_size is None
        assert cfg.embedding_multiplier == EMBEDDING_MULTIPLIER
        assert cfg.residual_multiplier == RESIDUAL_MULTIPLIER
        assert cfg.initializer_range == 0.02
        assert cfg.use_cache is True

    def test_default_rope_parameters_full_rotary(self):
        # THE classic silent-garbage trap: the Glm4Moe pattern defaults
        # partial_rotary_factor to 0.5 — the contract demands an EXPLICIT 1.0.
        cfg = ApertusMoeConfig()
        assert cfg.rope_parameters["rope_theta"] == ROPE_THETA
        assert cfg.rope_parameters.get("partial_rotary_factor", None) == 1.0

    def test_untied_by_default(self):
        assert ApertusMoeConfig().tie_word_embeddings is False

    def test_pretrained_model_flags(self):
        assert "e_score_correction_bias" in ApertusMoeForCausalLM._keep_in_fp32_modules_strict
        assert "ApertusMoeDecoderLayer" in ApertusMoeForCausalLM._no_split_modules
        assert not ApertusMoeForCausalLM._tied_weights_keys  # untied: EMPTY

    def test_rms_norm_eps_wired_from_config(self, make_model):
        """Every RMSNorm in the model (q/k norms, pre-norms, sandwich post-norms,
        final norm) must take its epsilon from config.rms_norm_eps. A NON-default
        eps makes any hardcoded 1e-5/1e-6 fallback fail this test directly,
        independent of the stock-Glm4Moe oracle."""
        eps = 3.5e-4  # deliberately not 1e-5 (config default) nor 1e-6 (RMSNorm default)
        model = make_model(True, TINY_LATENT, rms_norm_eps=eps)
        assert model.model.norm.variance_epsilon == eps
        for layer in model.model.layers:
            assert layer.self_attn.q_norm.variance_epsilon == eps
            assert layer.self_attn.k_norm.variance_epsilon == eps
            assert layer.attention_layernorm.variance_epsilon == eps
            assert layer.feedforward_layernorm.variance_epsilon == eps
            assert layer.post_attention_layernorm.variance_epsilon == eps
            assert layer.post_feedforward_layernorm.variance_epsilon == eps

    def test_conversion_mapping_registered_at_import_time(self):
        # Guide sec 9 item 8: register_checkpoint_conversion_mapping("apertus_moe", ...)
        # at module top level is load-bearing — without it per-expert disk keys
        # never reach the fused runtime params.
        mapping = get_checkpoint_conversion_mapping("apertus_moe")
        assert mapping, (
            "no checkpoint conversion mapping registered for model_type "
            "'apertus_moe' — modeling_apertus_moe.py must call "
            "register_checkpoint_conversion_mapping at import time"
        )


# ---------------------------------------------------------------------------
# tensor parallelism is intentionally unsupported (rejected loudly, not silently replicated)
# ---------------------------------------------------------------------------


class TestTensorParallelUnsupported:
    """No TP plan is published, and TP ranks are rejected at forward time.

    With an absent plan, ``tp_plan="auto"`` would silently replicate the full model on every
    rank instead of sharding it, so the model must fail loudly instead.
    """

    def test_no_tensor_parallel_plan_is_published(self, make_model):
        model = make_model()
        assert model.tp_plan == {}
        assert model.model.tp_plan == {}
        assert ApertusMoeConfig.base_model_tp_plan is None

    def test_causal_lm_forward_rejects_tp_ranks(self, make_model, input_ids):
        model = make_model()
        model._tp_size = 2
        with pytest.raises(ValueError, match=r"only TP=1"):
            model(input_ids)

    def test_base_model_forward_rejects_tp_ranks(self, make_model, input_ids):
        model = make_model()
        model.model._tp_size = 2
        with pytest.raises(ValueError, match=r"only TP=1"):
            model(input_ids)


# ---------------------------------------------------------------------------
# config ValueError guards (fail loudly instead of silently diverging)
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_explicit_partial_rotary_factor_kwarg_rejected(self):
        with pytest.raises(ValueError, match="partial_rotary_factor"):
            ApertusMoeConfig(partial_rotary_factor=0.5)

    def test_partial_rotary_factor_in_rope_dict_rejected(self):
        with pytest.raises(ValueError, match="partial_rotary_factor"):
            ApertusMoeConfig(
                rope_parameters={
                    "rope_type": "default",
                    "rope_theta": ROPE_THETA,
                    "partial_rotary_factor": 0.5,
                }
            )

    def test_explicit_partial_rotary_factor_one_accepted(self):
        cfg = ApertusMoeConfig(partial_rotary_factor=1.0)
        assert cfg.rope_parameters["partial_rotary_factor"] == 1.0
        cfg = ApertusMoeConfig(
            rope_parameters={
                "rope_type": "default",
                "rope_theta": ROPE_THETA,
                "partial_rotary_factor": 1.0,
            }
        )
        assert cfg.rope_parameters["partial_rotary_factor"] == 1.0

    def test_tie_word_embeddings_rejected(self):
        # _tied_weights_keys is EMPTY on the model, so tie_word_embeddings=True
        # would silently do nothing — the config must refuse it.
        with pytest.raises(ValueError, match="tie_word_embeddings"):
            ApertusMoeConfig(tie_word_embeddings=True)

    def test_topk1_with_norm_topk_prob_rejected(self):
        # The Megatron fork skips top-k renorm when topk==1; norm_topk_prob=True
        # here would renormalize the single score to 1.0 and silently diverge.
        with pytest.raises(ValueError, match="norm_topk_prob"):
            ApertusMoeConfig(num_experts_per_tok=1, norm_topk_prob=True)
        cfg = ApertusMoeConfig(num_experts_per_tok=1, norm_topk_prob=False)
        assert cfg.num_experts_per_tok == 1

    def test_legacy_dense_cutoff_is_validated(self):
        with pytest.raises(ValueError, match="first_k_dense_replace"):
            ApertusMoeConfig(num_hidden_layers=3, first_k_dense_replace=4)

    def test_moe_layer_freq_requires_one_binary_entry_per_layer(self):
        with pytest.raises(ValueError, match="moe_layer_freq"):
            ApertusMoeConfig(num_hidden_layers=3, moe_layer_freq=[0, 1])
        with pytest.raises(ValueError, match="moe_layer_freq"):
            ApertusMoeConfig(num_hidden_layers=3, moe_layer_freq=[0, 2, 1])

    def test_explicit_moe_schedule_is_authoritative_and_normalizes_legacy_cutoff(self):
        cfg = ApertusMoeConfig(
            num_hidden_layers=5,
            first_k_dense_replace=1,
            moe_layer_freq=[0, 0, 1, 0, 1],
        )
        assert cfg.moe_layer_freq == [0, 0, 1, 0, 1]
        assert cfg.first_k_dense_replace == 2
        assert [cfg.is_moe_layer(index) for index in range(5)] == [
            False,
            False,
            True,
            False,
            True,
        ]


# ---------------------------------------------------------------------------
# requirement 1: forward smoke, full 2x2 matrix
# ---------------------------------------------------------------------------


class TestForwardSmoke:
    @pytest.mark.parametrize("sandwich_norm,moe_latent_size", SANDWICH_LATENT_MATRIX)
    def test_forward_smoke(self, make_model, input_ids, sandwich_norm, moe_latent_size):
        model = make_model(sandwich_norm, moe_latent_size)
        with torch.no_grad():
            out = model(input_ids)
        assert out.logits.shape == (2, 7, TINY_VOCAB)
        assert torch.isfinite(out.logits).all()


# ---------------------------------------------------------------------------
# requirements 2 + 3: decoder-layer math vs hand-rolled contract reference
# ---------------------------------------------------------------------------


class TestLayerMath:
    @pytest.mark.parametrize("sandwich_norm,moe_latent_size", SANDWICH_LATENT_MATRIX)
    @pytest.mark.parametrize("layer_idx", [0, 1], ids=["dense-layer0", "moe-layer1"])
    def test_layer_matches_handrolled_reference(
        self, make_model, sandwich_norm, moe_latent_size, layer_idx
    ):
        model = make_model(sandwich_norm, moe_latent_size)
        layer = model.model.layers[layer_idx]
        if sandwich_norm:
            # non-trivial post-norm weights so norm/multiplier order matters maximally
            with torch.no_grad():
                layer.post_attention_layernorm.weight.copy_(
                    torch.randn(TINY_HIDDEN)
                )
                layer.post_feedforward_layernorm.weight.copy_(
                    torch.randn(TINY_HIDDEN)
                )
        x = torch.randn(2, 5, TINY_HIDDEN)
        pos_emb = _rope(model, x)
        with torch.no_grad():
            out = _call_layer(layer, x, pos_emb)
            ref = _handrolled_layer_forward(
                layer,
                x,
                pos_emb,
                sandwich=sandwich_norm,
                alpha_attn=model.config.residual_multiplier,
                alpha_mlp=model.config.residual_multiplier,
            )
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_sandwich_wrong_order_differs(self, make_model):
        """Requirement 2, kill-switch half: multiplier BEFORE the post-norm is a
        different function (RMSNorm is scale-invariant, so the wrong order
        silently swallows the alpha)."""
        model = make_model(True, None)
        layer = model.model.layers[1]
        with torch.no_grad():
            layer.post_attention_layernorm.weight.copy_(torch.randn(TINY_HIDDEN))
            layer.post_feedforward_layernorm.weight.copy_(torch.randn(TINY_HIDDEN))
        x = torch.randn(2, 5, TINY_HIDDEN)
        pos_emb = _rope(model, x)
        alpha = model.config.residual_multiplier
        with torch.no_grad():
            out = _call_layer(layer, x, pos_emb)
            right = _handrolled_layer_forward(
                layer, x, pos_emb, sandwich=True, alpha_attn=alpha, alpha_mlp=alpha
            )
            wrong = _handrolled_layer_forward(
                layer,
                x,
                pos_emb,
                sandwich=True,
                alpha_attn=alpha,
                alpha_mlp=alpha,
                post_norm_first=False,
            )
        torch.testing.assert_close(out, right, rtol=1e-5, atol=1e-5)
        assert (out - wrong).abs().max().item() > 1e-2, (
            "layer forward matches the WRONG order (residual_multiplier applied "
            "before the post-norm) — alpha must come AFTER the post-norm"
        )

    @pytest.mark.parametrize("missing_branch", ["attn", "mlp"])
    def test_residual_multiplier_missing_on_either_branch_differs(
        self, make_model, missing_branch
    ):
        """Requirement 3, kill-switch half: dropping the multiplier on either
        branch (sandwich off) must change the output."""
        model = make_model(False, None)
        layer = model.model.layers[1]
        x = torch.randn(2, 5, TINY_HIDDEN)
        pos_emb = _rope(model, x)
        alpha = model.config.residual_multiplier
        alpha_attn = 1.0 if missing_branch == "attn" else alpha
        alpha_mlp = 1.0 if missing_branch == "mlp" else alpha
        with torch.no_grad():
            out = _call_layer(layer, x, pos_emb)
            wrong = _handrolled_layer_forward(
                layer,
                x,
                pos_emb,
                sandwich=False,
                alpha_attn=alpha_attn,
                alpha_mlp=alpha_mlp,
            )
        assert (out - wrong).abs().max().item() > 1e-3, (
            f"output matches a reference with residual_multiplier omitted on the "
            f"{missing_branch} branch — alpha must be applied on BOTH branches"
        )


# ---------------------------------------------------------------------------
# requirements 4 + 5: conditional module existence
# ---------------------------------------------------------------------------

_POST_NORM_NAMES = ("post_attention_layernorm", "post_feedforward_layernorm")
_LATENT_NAMES = ("latent_down_proj", "latent_up_proj")


class TestModuleExistence:
    @pytest.mark.parametrize("moe_latent_size", [None, TINY_LATENT], ids=["latent-off", "latent-on"])
    def test_sandwich_off_no_post_norms(self, make_model, moe_latent_size):
        model = make_model(False, moe_latent_size)
        for layer in model.model.layers:
            for name in _POST_NORM_NAMES:
                assert getattr(layer, name, None) is None, (
                    f"sandwich_norm=False but layer has a '{name}' module "
                    "(identity placeholders are forbidden by the contract)"
                )
        module_names = [n for n, _ in model.named_modules()]
        assert not any(any(p in n for p in _POST_NORM_NAMES) for n in module_names)
        assert not any(any(p in k for p in _POST_NORM_NAMES) for k in model.state_dict())

    @pytest.mark.parametrize("moe_latent_size", [None, TINY_LATENT], ids=["latent-off", "latent-on"])
    def test_sandwich_on_post_norms_on_all_layers(self, make_model, moe_latent_size):
        model = make_model(True, moe_latent_size)
        state_dict = model.state_dict()
        # ALL layers INCLUDING the dense layer 0 carry both post-norms
        for L, layer in enumerate(model.model.layers):
            for name in _POST_NORM_NAMES:
                module = getattr(layer, name, None)
                assert isinstance(module, ApertusMoeRMSNorm), (
                    f"sandwich_norm=True but layer {L} lacks '{name}'"
                )
                key = f"model.layers.{L}.{name}.weight"
                assert key in state_dict
                assert state_dict[key].shape == (TINY_HIDDEN,)

    def test_layer_type_split_dense_vs_moe(self, make_model):
        model = make_model(False, None)
        assert isinstance(model.model.layers[0].mlp, ApertusMoeMLP)
        for L in range(TINY_FIRST_K_DENSE, TINY_LAYERS):
            assert isinstance(model.model.layers[L].mlp, ApertusMoeMoE)

    def test_interleaved_layer_schedule_builds_the_exact_module_types(self, make_model):
        pattern = [0, 1, 0]
        model = make_model(False, None, moe_layer_freq=pattern)
        assert model.config.moe_layer_freq == pattern
        assert [
            isinstance(layer.mlp, ApertusMoeMoE) for layer in model.model.layers
        ] == [False, True, False]

    @pytest.mark.parametrize("sandwich_norm", [False, True], ids=["plain", "sandwich"])
    def test_latent_off_no_latent_projs(self, make_model, sandwich_norm):
        model = make_model(sandwich_norm, None)
        assert not any("latent" in k for k in model.state_dict()), (
            "moe_latent_size=None but latent projection tensors exist"
        )
        for layer in _moe_layers(model):
            for name in _LATENT_NAMES:
                assert getattr(layer.mlp, name, None) is None
            # expert container dims are hidden-sized when latent is off
            assert layer.mlp.experts.gate_up_proj.shape == (
                TINY_N_EXPERTS, 2 * TINY_MOE_INTERMEDIATE, TINY_HIDDEN,
            )
            assert layer.mlp.experts.down_proj.shape == (
                TINY_N_EXPERTS, TINY_HIDDEN, TINY_MOE_INTERMEDIATE,
            )

    @pytest.mark.parametrize("sandwich_norm", [False, True], ids=["plain", "sandwich"])
    def test_latent_on_projs_and_expert_dims(self, make_model, sandwich_norm):
        model = make_model(sandwich_norm, TINY_LATENT)
        state_dict = model.state_dict()
        # dense layer 0 never carries latent projections
        layer0_keys = [k for k in state_dict if k.startswith("model.layers.0.")]
        assert not any("latent" in k for k in layer0_keys)
        for L in range(TINY_FIRST_K_DENSE, TINY_LAYERS):
            mlp = model.model.layers[L].mlp
            # Linear(hidden, latent, bias=False) -> weight [latent, hidden]
            assert mlp.latent_down_proj.weight.shape == (TINY_LATENT, TINY_HIDDEN)
            assert mlp.latent_down_proj.bias is None
            # Linear(latent, hidden, bias=False) -> weight [hidden, latent]
            assert mlp.latent_up_proj.weight.shape == (TINY_HIDDEN, TINY_LATENT)
            assert mlp.latent_up_proj.bias is None
            assert f"model.layers.{L}.mlp.latent_down_proj.weight" in state_dict
            assert f"model.layers.{L}.mlp.latent_up_proj.weight" in state_dict
            # expert container in/out dims are LATENT-sized
            assert mlp.experts.gate_up_proj.shape == (
                TINY_N_EXPERTS, 2 * TINY_MOE_INTERMEDIATE, TINY_LATENT,
            )
            assert mlp.experts.down_proj.shape == (
                TINY_N_EXPERTS, TINY_LATENT, TINY_MOE_INTERMEDIATE,
            )


# ---------------------------------------------------------------------------
# requirement 6: the LatentMoE ordering trio (guide sec 3.5)
# ---------------------------------------------------------------------------


class TestLatentMoeOrdering:
    def _moe(self, make_model):
        model = make_model(False, TINY_LATENT)
        return model, model.model.layers[1].mlp

    def test_router_input_is_full_hidden(self, make_model):
        """6a: the router consumes the FULL hidden dim, never the latent dim."""
        _, moe = self._moe(make_model)
        assert moe.gate.weight.shape == (TINY_N_EXPERTS, TINY_HIDDEN), (
            "gate.weight must stay [n_routed_experts, hidden_size] even with "
            "moe_latent_size set"
        )
        seen = []
        handle = moe.gate.register_forward_pre_hook(
            lambda mod, args, kwargs: seen.append(_first_tensor(args, kwargs).shape[-1]),
            with_kwargs=True,
        )
        x = torch.randn(2, 5, TINY_HIDDEN)
        with torch.no_grad():
            moe(x)
        handle.remove()
        assert seen and all(d == TINY_HIDDEN for d in seen), (
            f"router received last-dim {seen}, expected hidden_size={TINY_HIDDEN} "
            "(routing must happen BEFORE latent_down_proj)"
        )

    def test_shared_experts_input_is_full_hidden(self, make_model):
        """6b: the shared expert bypasses both latent projections."""
        _, moe = self._moe(make_model)
        assert moe.shared_experts.gate_proj.weight.shape == (
            TINY_MOE_INTERMEDIATE, TINY_HIDDEN,
        ), "shared expert must be hidden-dim even with moe_latent_size set"
        seen = []
        handle = moe.shared_experts.register_forward_pre_hook(
            lambda mod, args, kwargs: seen.append(_first_tensor(args, kwargs).shape[-1]),
            with_kwargs=True,
        )
        x = torch.randn(2, 5, TINY_HIDDEN)
        with torch.no_grad():
            moe(x)
        handle.remove()
        assert seen and all(d == TINY_HIDDEN for d in seen), (
            f"shared_experts received last-dim {seen}, expected {TINY_HIDDEN} "
            "(shared expert operates on the un-projected input)"
        )

    def test_zeroed_routed_path_output_equals_shared(self, make_model):
        """6c part 1: with the expert container zeroed (latent projections left
        random!), the MoE output must equal shared_experts(x) exactly. If the
        shared output were added BEFORE latent_up_proj, the (random) up-proj
        would distort it and this fails."""
        _, moe = self._moe(make_model)
        with torch.no_grad():
            moe.experts.gate_up_proj.zero_()
            moe.experts.down_proj.zero_()
        x = torch.randn(2, 5, TINY_HIDDEN)
        with torch.no_grad():
            out = _as_tensor(moe(x))
            expected = moe.shared_experts(x)
        assert out.shape == x.shape
        torch.testing.assert_close(out, expected, rtol=0.0, atol=0.0)

    def test_zeroed_shared_output_equals_up_projected_expert_path(self, make_model):
        """6c part 2: with shared_experts zeroed, the output must equal
        latent_up_proj(experts(latent_down_proj(x), ...)) — hooks on the expert
        container pin the projections to their contract positions."""
        _, moe = self._moe(make_model)
        with torch.no_grad():
            moe.shared_experts.gate_proj.weight.zero_()
            moe.shared_experts.up_proj.weight.zero_()
            moe.shared_experts.down_proj.weight.zero_()
        captured = {}
        pre = moe.experts.register_forward_pre_hook(
            lambda mod, args, kwargs: captured.__setitem__("in", _first_tensor(args, kwargs)),
            with_kwargs=True,
        )
        post = moe.experts.register_forward_hook(
            lambda mod, args, out: captured.__setitem__("out", _as_tensor(out))
        )
        x = torch.randn(2, 5, TINY_HIDDEN)
        with torch.no_grad():
            out = _as_tensor(moe(x))
        pre.remove()
        post.remove()

        # experts run in the LATENT dim, on latent_down_proj(x)
        assert captured["in"].shape[-1] == TINY_LATENT, (
            "expert container input is not latent-sized — latent_down_proj must "
            "be applied before the experts"
        )
        with torch.no_grad():
            torch.testing.assert_close(
                captured["in"].reshape(-1, TINY_LATENT),
                moe.latent_down_proj(x).reshape(-1, TINY_LATENT),
                rtol=1e-6,
                atol=1e-6,
            )
            # final output == latent_up_proj(expert output) (+ zeroed shared)
            torch.testing.assert_close(
                out.reshape(-1, TINY_HIDDEN),
                moe.latent_up_proj(captured["out"]).reshape(-1, TINY_HIDDEN),
                rtol=1e-6,
                atol=1e-6,
            )


# ---------------------------------------------------------------------------
# Router bias semantics: the correction
# bias steers top-k SELECTION only; gate values are BIAS-FREE sigmoid scores,
# renormalized FIRST and scaled by routed_scaling_factor AFTER.
# ---------------------------------------------------------------------------


class TestRouterCorrectionBias:
    def _moe_and_logits(self, make_model, n_tokens=6):
        model = make_model(False, None)
        moe = model.model.layers[1].mlp
        generator = torch.Generator().manual_seed(7)
        x = torch.randn(n_tokens, TINY_HIDDEN, generator=generator)
        with torch.no_grad():
            logits, _, _ = moe.gate(x)  # fp32 pre-sigmoid router logits
        return moe, logits

    def _bias_free_reference(self, moe, logits, idx):
        """The contract gate values: gather BIAS-FREE sigmoid scores at the
        selected indices, renorm by their sum FIRST, then * routed_scaling_factor.
        (Scaling BEFORE the renorm would cancel the factor; skipping the renorm
        changes the values — both are pinned here.)"""
        scores = logits.sigmoid()
        ref = scores.gather(1, idx)
        ref = ref / (ref.sum(dim=-1, keepdim=True) + 1e-20)
        return ref * moe.gate.routed_scaling_factor

    def test_gate_values_ignore_correction_bias(self, make_model):
        moe, logits = self._moe_and_logits(make_model)
        generator = torch.Generator().manual_seed(11)
        with torch.no_grad():
            # NONZERO bias: gathering from the biased scores would visibly differ
            moe.gate.e_score_correction_bias.copy_(
                torch.randn(TINY_N_EXPERTS, generator=generator).abs() + 0.5
            )
            idx, weights = moe.gate.route_tokens_to_experts(logits)
            ref = self._bias_free_reference(moe, logits, idx)
        assert idx.shape == weights.shape == (logits.shape[0], TINY_TOPK)
        torch.testing.assert_close(weights, ref, rtol=1e-6, atol=1e-6)

    def test_correction_bias_changes_topk_selection(self, make_model):
        moe, logits = self._moe_and_logits(make_model)
        with torch.no_grad():
            moe.gate.e_score_correction_bias.zero_()
            idx0, _ = moe.gate.route_tokens_to_experts(logits)
            # the expert picked by the FEWEST tokens under zero bias (6 tokens x
            # top-2 = 12 slots over 8 experts -> some expert has count <= 1)
            counts = torch.zeros(TINY_N_EXPERTS)
            counts.scatter_add_(0, idx0.reshape(-1), torch.ones(idx0.numel()))
            j = int(counts.argmin())
            assert counts[j] < logits.shape[0], "least-picked expert already picked by every token"
            forced = torch.zeros(TINY_N_EXPERTS)
            forced[j] = 1e4
            moe.gate.e_score_correction_bias.copy_(forced)
            idx1, weights1 = moe.gate.route_tokens_to_experts(logits)
            ref1 = self._bias_free_reference(moe, logits, idx1)
        # the bias DOES steer selection: expert j is now in every token's top-k...
        assert (idx1 == j).any(dim=-1).all(), "large correction bias failed to force expert selection"
        assert not torch.equal(torch.sort(idx1, dim=-1)[0], torch.sort(idx0, dim=-1)[0])
        # ...while the gate values stay bias-free renormed-then-scaled sigmoid scores
        torch.testing.assert_close(weights1, ref1, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# group-limited routing: groups are ranked by the sum of their
# top-(topk // group_topk) scores, NOT a hardcoded top-2 (fork
# group_limited_topk, moe_utils.py:632-650). Dormant at the shipped
# n_group=1/topk_group=1 — one group, the mask is all-ones — so no other test in
# this suite ever executes the branch. The oracle below is transcribed from the
# fork and does not import modeling_apertus_moe.
# ---------------------------------------------------------------------------


def _fork_group_limited_topk(scores, topk, num_groups, group_topk):
    """Verbatim transcription of the fork's group_limited_topk
    (megatron/core/transformer/moe/moe_utils.py:632-650)."""
    num_tokens, num_experts = scores.shape
    group_scores = (
        scores.view(num_tokens, num_groups, -1).topk(topk // group_topk, dim=-1)[0].sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=group_topk, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_groups, num_experts // num_groups)
        .reshape(num_tokens, -1)
    )
    masked_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
    return torch.topk(masked_scores, k=topk, dim=-1)


def _fork_route(logits, expert_bias, topk, num_groups, group_topk, scaling_factor):
    """The fork's sigmoid + expert_bias + group-limited path, in order
    (moe_utils.py:823-840, topk_routing_with_score_function): bias steers
    SELECTION only, gate values gather bias-free scores, renorm only when
    topk > 1, scaling_factor LAST."""
    scores = torch.sigmoid(logits.float()).type_as(logits)
    _, top_indices = _fork_group_limited_topk(scores + expert_bias, topk, num_groups, group_topk)
    gathered = torch.gather(scores, dim=1, index=top_indices).type_as(logits)
    probs = gathered / (gathered.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else gathered
    return probs * scaling_factor, top_indices


def _dense(idx, weights, n_experts):
    """[tokens, topk] -> [tokens, n_experts], so a comparison is independent of
    top-k index ORDER (the fork's final topk sorts; route_tokens_to_experts
    passes sorted=False)."""
    out = torch.zeros(idx.shape[0], n_experts, dtype=weights.dtype)
    out.scatter_(1, idx, weights)
    return out


class TestGroupLimitedRouting:
    def test_group_ranking_uses_topk_over_group_topk_not_hardcoded_two(self, make_model):
        """Scores chosen so the two rankings DISAGREE: group 0 wins on its top-2 sum
        (1.8 > 1.6) but loses on its top-3 sum (1.9 < 2.4). With topk=3/group_topk=1
        the fork ranks by the top-3 sum and selects group 1; a hardcoded top-2 ranking
        selects group 0. Stating the scores and deriving the logits (rather than the
        reverse) is what makes the arithmetic above checkable by eye."""
        model = make_model(False, None, n_group=2, topk_group=1, num_experts_per_tok=3)
        moe = model.model.layers[1].mlp
        target_scores = torch.tensor([[0.9, 0.9, 0.1, 0.1, 0.8, 0.8, 0.8, 0.05]])
        logits = torch.logit(target_scores)
        with torch.no_grad():
            idx, _ = moe.gate.route_tokens_to_experts(logits)
        assert set(idx.reshape(-1).tolist()) == {4, 5, 6}, (
            "expected the top-(topk//group_topk) ranking to select group 1 (experts 4-7); "
            f"got {sorted(idx.reshape(-1).tolist())} — a top-2 group ranking would pick group 0"
        )

    @pytest.mark.parametrize(
        "n_group, topk_group, top_k",
        [
            pytest.param(2, 1, 3, id="g2-tg1-k3"),  # topk//group_topk = 3
            pytest.param(2, 2, 2, id="g2-tg2-k2"),  # = 1
            pytest.param(4, 2, 4, id="g4-tg2-k4"),  # = 2, the case a hardcoded 2 gets right
            pytest.param(2, 1, 4, id="g2-tg1-k4"),  # = 4, the whole group
        ],
    )
    def test_matches_fork_group_limited_reference(self, make_model, n_group, topk_group, top_k):
        model = make_model(
            False, None, n_group=n_group, topk_group=topk_group, num_experts_per_tok=top_k
        )
        moe = model.model.layers[1].mlp
        generator = torch.Generator().manual_seed(23)
        # 256 tokens, not a handful: whether a top-2 group ranking and the fork's
        # top-(topk//group_topk) ranking DIVERGE is a property of the draw, and at 6
        # tokens they routinely coincide — this test then passes against the very bug
        # test_group_ranking_uses_topk_over_group_topk_not_hardcoded_two exists to
        # catch. Enough tokens make divergence near-certain rather than lucky.
        logits = torch.randn(256, TINY_N_EXPERTS, generator=generator)
        with torch.no_grad():
            # NONZERO bias: it must steer selection without reaching the gate values
            bias = torch.randn(TINY_N_EXPERTS, generator=generator).abs() + 0.5
            moe.gate.e_score_correction_bias.copy_(bias)
            idx, weights = moe.gate.route_tokens_to_experts(logits)
            ref_weights, ref_idx = _fork_route(
                logits, bias, top_k, n_group, topk_group, moe.gate.routed_scaling_factor
            )
        assert idx.shape == weights.shape == (logits.shape[0], top_k)
        torch.testing.assert_close(
            _dense(idx, weights, TINY_N_EXPERTS),
            _dense(ref_idx, ref_weights, TINY_N_EXPERTS),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_selection_confined_to_the_chosen_groups(self, make_model):
        """topk_group=1 over 4 groups of 2 experts: the mask leaves exactly one group
        alive, so every token's top-2 must be that group's contiguous expert pair."""
        model = make_model(False, None, n_group=4, topk_group=1, num_experts_per_tok=2)
        moe = model.model.layers[1].mlp
        generator = torch.Generator().manual_seed(5)
        logits = torch.randn(6, TINY_N_EXPERTS, generator=generator)
        with torch.no_grad():
            idx, _ = moe.gate.route_tokens_to_experts(logits)
        for row in idx.tolist():
            groups = {expert // 2 for expert in row}
            assert len(groups) == 1, (
                f"topk_group=1 must confine selection to ONE group; row {sorted(row)} spans {sorted(groups)}"
            )


# ---------------------------------------------------------------------------
# Quantile-balancing router:
# selection = topk(RAW pre-sigmoid logits MINUS qb_beta) — the two traps vs the
# GLM correction-bias pattern are SUBTRACT (not add) and RAW logits (not
# post-sigmoid scores). Gate values stay bias-free renormed-then-scaled sigmoid
# scores; e_score_correction_bias is ignored for selection when QB is on.
# ---------------------------------------------------------------------------


class TestQuantileBalancing:
    def _sorted(self, idx):
        """Order-insensitive top-k index comparison (topk uses sorted=False)."""
        return torch.sort(idx, dim=-1)[0]

    def _qb_moe_and_logits(self, make_model, n_tokens=6, sandwich=False, latent=None):
        model = make_model(sandwich, latent, use_quantile_balancing=True)
        moe = model.model.layers[1].mlp
        generator = torch.Generator().manual_seed(7)
        x = torch.randn(n_tokens, TINY_HIDDEN, generator=generator)
        with torch.no_grad():
            logits, _, _ = moe.gate(x)  # fp32 pre-sigmoid router logits
        return moe, logits

    def _bias_free_reference(self, moe, logits, idx):
        """Contract gate values (identical to the non-QB path): gather BIAS-FREE
        sigmoid scores at the selected indices, renorm FIRST, scale AFTER —
        qb_beta must never touch these."""
        scores = logits.sigmoid()
        ref = scores.gather(1, idx)
        ref = ref / (ref.sum(dim=-1, keepdim=True) + 1e-20)
        return ref * moe.gate.routed_scaling_factor

    # -- 1. buffer existence ------------------------------------------------

    def test_flag_off_no_qb_beta_anywhere(self, make_model):
        model = make_model(False, None)
        assert not any("qb_beta" in k for k in model.state_dict()), (
            "use_quantile_balancing=False but qb_beta tensors exist "
            "(the buffer must be absent, not a zero placeholder)"
        )
        assert not any("qb_beta" in n for n, _ in model.named_buffers())
        for layer in _moe_layers(model):
            assert getattr(layer.mlp.gate, "qb_beta", None) is None

    def test_flag_on_qb_beta_on_moe_gates_only(self, make_model):
        model = make_model(False, None, use_quantile_balancing=True)
        state_dict = model.state_dict()
        buffer_names = dict(model.named_buffers())
        # dense layer 0 has no router, hence no qb_beta keys
        assert not any(
            "qb_beta" in k for k in state_dict if k.startswith("model.layers.0.")
        )
        for L in range(TINY_FIRST_K_DENSE, TINY_LAYERS):
            gate = model.model.layers[L].mlp.gate
            beta = getattr(gate, "qb_beta", None)
            assert beta is not None, f"layer {L} gate lacks qb_beta with QB on"
            assert beta.shape == (TINY_N_EXPERTS,)
            assert beta.dtype == torch.float32
            assert (beta == 0).all(), "qb_beta must be zeros-init"
            key = f"model.layers.{L}.mlp.gate.qb_beta"
            # in state_dict == persistent (non-persistent buffers are excluded)
            assert key in state_dict, f"{key} missing — qb_beta must be persistent"
            assert key in buffer_names, f"{key} is not a registered buffer"
        # qb_beta appears nowhere except MoE-layer gates
        expected = {
            f"model.layers.{L}.mlp.gate.qb_beta"
            for L in range(TINY_FIRST_K_DENSE, TINY_LAYERS)
        }
        assert {k for k in state_dict if "qb_beta" in k} == expected

    # -- 2. selection math: topk(RAW logits - qb_beta), and it moves top-k ---

    def test_selection_is_topk_of_raw_logits_minus_qb_beta(self, make_model):
        moe, logits = self._qb_moe_and_logits(make_model)
        with torch.no_grad():
            plain_idx = torch.topk(logits, k=TINY_TOPK, dim=-1).indices
            # Deterministic flip: a huge positive beta on every token's plain
            # top-1 expert pushes that expert out of the shifted top-k for ALL
            # tokens (8 experts, <=6 penalized, penalty ~1e4 >> logit range).
            top1 = logits.argmax(dim=-1)
            beta = torch.zeros(TINY_N_EXPERTS)
            beta[top1] = 1e4
            moe.gate.qb_beta.copy_(beta)
            idx, _ = moe.gate.route_tokens_to_experts(logits)
            ref_idx = torch.topk(logits - beta, k=TINY_TOPK, dim=-1).indices
        assert torch.equal(self._sorted(idx), self._sorted(ref_idx)), (
            "QB selection != topk(raw_logits - qb_beta) — the shift must be a "
            "SUBTRACTION applied to the RAW pre-sigmoid logits"
        )
        # and the crafted beta really flipped the selection for every token:
        # the formerly-top-1 expert is gone from each row
        assert not (idx == top1.unsqueeze(-1)).any(), (
            "a +1e4 qb_beta on the top-1 expert failed to evict it — qb_beta "
            "is not steering selection (sign flip? applied to sigmoid scores?)"
        )
        assert not torch.equal(self._sorted(idx), self._sorted(plain_idx))

    def test_selection_shift_applies_pre_sigmoid(self, make_model):
        # Saturation case: distinguishes topk(logits - beta) from the plausible-but-wrong
        # topk(sigmoid(logits) - beta). With logits A=10, B=5, C=4 (rest -10) and beta A=2:
        #   raw shifted:     A=8    > B=5     > C=4     -> top-2 = {A, B}
        #   sigmoid shifted: A=-1.0 < C=0.982 < B=0.993 -> top-2 = {B, C}
        # A huge eviction beta (test above) cannot tell these apart; this one can.
        moe, _ = self._qb_moe_and_logits(make_model)
        assert TINY_TOPK == 2 and TINY_N_EXPERTS >= 4
        n_tokens = 4
        logits = torch.full((n_tokens, TINY_N_EXPERTS), -10.0)
        beta = torch.zeros(TINY_N_EXPERTS)
        # vary the A/B/C expert slots per token so a lucky fixed permutation can't pass
        for t in range(n_tokens):
            a, b, c = t % TINY_N_EXPERTS, (t + 3) % TINY_N_EXPERTS, (t + 5) % TINY_N_EXPERTS
            logits[t, a], logits[t, b], logits[t, c] = 10.0, 5.0, 4.0
        for t in range(n_tokens):
            beta_t = torch.zeros(TINY_N_EXPERTS)
            beta_t[t % TINY_N_EXPERTS] = 2.0
            with torch.no_grad():
                moe.gate.qb_beta.copy_(beta_t)
                idx, _ = moe.gate.route_tokens_to_experts(logits[t : t + 1])
                ref_idx = torch.topk(logits[t : t + 1] - beta_t, k=TINY_TOPK, dim=-1).indices
            assert torch.equal(self._sorted(idx), self._sorted(ref_idx)), (
                "QB selection does not match topk(raw_logits - qb_beta) in the sigmoid-"
                "saturated regime — the shift is being applied post-sigmoid"
            )

    # -- 3. gate values are bias-free (qb_beta never touches them) ----------

    def test_gate_values_ignore_qb_beta(self, make_model):
        moe, logits = self._qb_moe_and_logits(make_model)
        generator = torch.Generator().manual_seed(11)
        with torch.no_grad():
            # NONZERO O(1) beta: shifts selection AND would visibly distort the
            # weights if it leaked into the gathered scores
            moe.gate.qb_beta.copy_(torch.randn(TINY_N_EXPERTS, generator=generator) * 2.0)
            idx, weights = moe.gate.route_tokens_to_experts(logits)
            ref = self._bias_free_reference(moe, logits, idx)
        assert idx.shape == weights.shape == (logits.shape[0], TINY_TOPK)
        torch.testing.assert_close(weights, ref, rtol=1e-6, atol=1e-6)

    # -- 4. e_score_correction_bias is ignored for selection under QB -------

    def test_correction_bias_ignored_when_qb_on(self, make_model):
        """With QB on and qb_beta=0, selection must be plain topk(logits) even
        under a huge e_score_correction_bias. The contrast (QB off: the same
        forced bias DOES change selection) is already pinned by
        TestRouterCorrectionBias::test_correction_bias_changes_topk_selection."""
        moe, logits = self._qb_moe_and_logits(make_model)
        with torch.no_grad():
            moe.gate.qb_beta.zero_()
            plain_idx = torch.topk(logits, k=TINY_TOPK, dim=-1).indices
            # same forced-bias construction as the QB-off contrast test
            counts = torch.zeros(TINY_N_EXPERTS)
            counts.scatter_add_(0, plain_idx.reshape(-1), torch.ones(plain_idx.numel()))
            j = int(counts.argmin())
            forced = torch.zeros(TINY_N_EXPERTS)
            forced[j] = 1e4
            moe.gate.e_score_correction_bias.copy_(forced)
            idx, _ = moe.gate.route_tokens_to_experts(logits)
        assert torch.equal(self._sorted(idx), self._sorted(plain_idx)), (
            "with QB on, a large e_score_correction_bias changed the selection "
            "— the expert bias must not participate in QB routing"
        )
        assert not (idx == j).any(dim=-1).all(), (
            "the forced-bias expert entered every token's top-k — the "
            "correction bias leaked into the QB selection path"
        )

    # -- 5. fp32 invariant ---------------------------------------------------

    def test_qb_beta_stays_fp32_under_bf16(self, make_model, tmp_path):
        model = make_model(True, TINY_LATENT, use_quantile_balancing=True)
        model.save_pretrained(str(tmp_path))
        reloaded = ApertusMoeForCausalLM.from_pretrained(
            str(tmp_path), dtype=torch.bfloat16
        )
        for layer in _moe_layers(reloaded):
            assert layer.mlp.gate.qb_beta.dtype == torch.float32
            assert layer.mlp.gate.e_score_correction_bias.dtype == torch.float32
            assert layer.mlp.gate.weight.dtype == torch.bfloat16
        assert "qb_beta" in ApertusMoeForCausalLM._keep_in_fp32_modules_strict

    # -- 6. save/load round trip ----------------------------------------------

    def test_save_load_round_trip_with_qb(self, make_model, input_ids, tmp_path):
        model = make_model(False, None, use_quantile_balancing=True)
        generator = torch.Generator().manual_seed(23)
        with torch.no_grad():
            # NONZERO values so equality below is meaningful (not zeros == zeros)
            for layer in _moe_layers(model):
                layer.mlp.gate.qb_beta.copy_(
                    torch.randn(TINY_N_EXPERTS, generator=generator)
                )
        model.save_pretrained(str(tmp_path))
        disk = _load_saved_state_dict(tmp_path)
        assert set(disk) == _expected_disk_keys(model.config)
        for L in range(TINY_FIRST_K_DENSE, TINY_LAYERS):
            key = f"model.layers.{L}.mlp.gate.qb_beta"
            assert disk[key].dtype == torch.float32
            torch.testing.assert_close(
                disk[key], model.model.layers[L].mlp.gate.qb_beta, rtol=0.0, atol=0.0
            )
        reloaded, info = ApertusMoeForCausalLM.from_pretrained(
            str(tmp_path), dtype=torch.float32, output_loading_info=True
        )
        reloaded.eval()
        assert not info["missing_keys"], info["missing_keys"]
        assert not info["unexpected_keys"], info["unexpected_keys"]
        assert not info["mismatched_keys"], info["mismatched_keys"]
        assert not info["error_msgs"], info["error_msgs"]
        original = model.state_dict()
        restored = reloaded.state_dict()
        assert set(original) == set(restored)
        for key in original:
            torch.testing.assert_close(
                restored[key], original[key], rtol=0.0, atol=0.0,
                msg=lambda m, key=key: f"{key} changed across the round trip: {m}",
            )
        with torch.no_grad():
            a = model(input_ids).logits
            b = reloaded(input_ids).logits
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)

    # -- 7. forward + generate smoke (sandwich + latent + QB compose) --------

    def test_forward_and_generate_smoke_sandwich_latent_qb(self, make_model, input_ids):
        model = make_model(True, TINY_LATENT, use_quantile_balancing=True)
        with torch.no_grad():
            out = model(input_ids)
            assert out.logits.shape == (2, 7, TINY_VOCAB)
            assert torch.isfinite(out.logits).all()
            gen = model.generate(
                input_ids,
                max_new_tokens=8,
                min_new_tokens=8,
                do_sample=False,
                use_cache=True,
                pad_token_id=0,
            )
        assert gen.shape == (2, input_ids.shape[1] + 8)
        assert torch.equal(gen[:, : input_ids.shape[1]], input_ids)

    # -- 8. config validation --------------------------------------------------

    def test_qb_with_group_limited_routing_rejected(self):
        with pytest.raises(ValueError, match="quantile"):
            ApertusMoeConfig(use_quantile_balancing=True, n_group=2)
        with pytest.raises(ValueError, match="quantile"):
            ApertusMoeConfig(use_quantile_balancing=True, n_group=2, topk_group=2)
        cfg = ApertusMoeConfig(use_quantile_balancing=True)  # n_group=topk_group=1: fine
        assert cfg.use_quantile_balancing is True
        assert ApertusMoeConfig().use_quantile_balancing is False


# ---------------------------------------------------------------------------
# requirement 7: embedding multiplier
# ---------------------------------------------------------------------------


class TestEmbeddingMultiplier:
    def test_hidden_after_embedding_is_scaled(self, make_model, input_ids):
        model = make_model(False, None)
        captured = []
        handle = model.model.layers[0].register_forward_pre_hook(
            lambda mod, args, kwargs: captured.append(_first_tensor(args, kwargs)),
            with_kwargs=True,
        )
        with torch.no_grad():
            model(input_ids)
        handle.remove()
        got = captured[0]
        with torch.no_grad():
            raw = model.model.embed_tokens(input_ids)
        torch.testing.assert_close(
            got, raw * model.config.embedding_multiplier, rtol=1e-6, atol=1e-6
        )
        # and the multiplier is actually load-bearing (not silently 1.0)
        assert not torch.allclose(got, raw, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# requirement 9 (guide sec 9 item 1): qk-norm placement equivalence
# ---------------------------------------------------------------------------


class TestQkNormPlacement:
    """RMSNorm over head_dim commutes with the [B,S,H,d] <-> [B,H,S,d]
    transpose. Glm4Moe norms before the transpose, Megatron after its own
    reshape — both are the same math. This is the one-time proof plus a
    regression guard so nobody "fixes" the placement."""

    @staticmethod
    def _rms_norm_ref(x, weight, eps=1e-5):
        input_dtype = x.dtype
        x32 = x.to(torch.float32)
        variance = x32.pow(2).mean(-1, keepdim=True)
        return weight * (x32 * torch.rsqrt(variance + eps)).to(input_dtype)

    def test_transpose_equivalence_pure_torch(self):
        x = torch.randn(2, 5, TINY_HEADS, TINY_HEAD_DIM)
        weight = torch.randn(TINY_HEAD_DIM)
        norm_then_transpose = self._rms_norm_ref(x, weight).transpose(1, 2)
        transpose_then_norm = self._rms_norm_ref(x.transpose(1, 2), weight)
        torch.testing.assert_close(
            norm_then_transpose, transpose_then_norm, rtol=0.0, atol=0.0
        )

    def test_transpose_equivalence_model_qk_norm(self, make_model):
        model = make_model(False, None)
        attn = model.model.layers[0].self_attn
        for norm in (attn.q_norm, attn.k_norm):
            # the norm is over head_dim, shared across heads
            assert tuple(norm.weight.shape) == (TINY_HEAD_DIM,)
        q_norm = attn.q_norm
        with torch.no_grad():
            q_norm.weight.copy_(torch.randn(TINY_HEAD_DIM))
        x = torch.randn(2, 5, TINY_HEADS, TINY_HEAD_DIM)
        with torch.no_grad():
            a = q_norm(x).transpose(1, 2)
            b = q_norm(x.transpose(1, 2))
        torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# requirements 8 + 10 (guide sec 9 items 6-7): save/load through the
# WeightConverter, per-expert disk keys, fp32 invariant
# ---------------------------------------------------------------------------


class TestSaveLoadRoundTrip:
    @pytest.mark.parametrize("sandwich_norm,moe_latent_size", SANDWICH_LATENT_MATRIX)
    def test_on_disk_keys_are_per_expert_and_contract_exact(
        self, make_model, tmp_path, sandwich_norm, moe_latent_size
    ):
        model = make_model(sandwich_norm, moe_latent_size)
        model.save_pretrained(str(tmp_path))
        disk = _load_saved_state_dict(tmp_path)

        assert set(disk) == _expected_disk_keys(model.config), (
            "on-disk state dict does not match the contract layout; "
            f"missing={sorted(_expected_disk_keys(model.config) - set(disk))[:10]} "
            f"extra={sorted(set(disk) - _expected_disk_keys(model.config))[:10]}"
        )
        # fused runtime keys must NOT leak to disk
        assert not any(k.endswith("mlp.experts.gate_up_proj") for k in disk)
        assert not any(k.endswith("mlp.experts.down_proj") for k in disk)

        in_dim = moe_latent_size if moe_latent_size is not None else TINY_HIDDEN
        e0 = "model.layers.1.mlp.experts.0."
        assert disk[e0 + "gate_proj.weight"].shape == (TINY_MOE_INTERMEDIATE, in_dim)
        assert disk[e0 + "up_proj.weight"].shape == (TINY_MOE_INTERMEDIATE, in_dim)
        assert disk[e0 + "down_proj.weight"].shape == (in_dim, TINY_MOE_INTERMEDIATE)
        assert disk["model.layers.1.mlp.gate.e_score_correction_bias"].dtype == torch.float32

        # save must be the exact inverse of the load-time fusion: expert i's
        # gate/up rows come from the fused container (gate first — the
        # qwen2_moe converter concatenates [gate, up] on dim 1)
        fused = model.model.layers[1].mlp.experts
        torch.testing.assert_close(
            disk[e0 + "gate_proj.weight"],
            fused.gate_up_proj[0, :TINY_MOE_INTERMEDIATE, :],
            rtol=0.0, atol=0.0,
        )
        torch.testing.assert_close(
            disk[e0 + "up_proj.weight"],
            fused.gate_up_proj[0, TINY_MOE_INTERMEDIATE:, :],
            rtol=0.0, atol=0.0,
        )
        torch.testing.assert_close(
            disk[e0 + "down_proj.weight"], fused.down_proj[0], rtol=0.0, atol=0.0
        )

    @pytest.mark.parametrize("sandwich_norm,moe_latent_size", SANDWICH_LATENT_MATRIX)
    def test_round_trip_repopulates_fused_runtime_params(
        self, make_model, input_ids, tmp_path, sandwich_norm, moe_latent_size
    ):
        model = make_model(sandwich_norm, moe_latent_size)
        model.save_pretrained(str(tmp_path))
        reloaded, info = ApertusMoeForCausalLM.from_pretrained(
            str(tmp_path), dtype=torch.float32, output_loading_info=True
        )
        reloaded.eval()
        assert not info["missing_keys"], info["missing_keys"]
        assert not info["unexpected_keys"], info["unexpected_keys"]
        assert not info["mismatched_keys"], info["mismatched_keys"]
        assert not info["error_msgs"], info["error_msgs"]

        original = model.state_dict()
        restored = reloaded.state_dict()
        assert set(original) == set(restored)
        for key in original:
            torch.testing.assert_close(
                restored[key], original[key], rtol=0.0, atol=0.0,
                msg=lambda m, key=key: f"{key} changed across the round trip: {m}",
            )
        with torch.no_grad():
            a = model(input_ids).logits
            b = reloaded(input_ids).logits
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)

    def test_save_pretrained_is_self_describing_for_auto_classes(
        self, make_model, tmp_path
    ):
        """register_for_auto_class at import time makes save_pretrained write
        auto_map into config.json AND copy the two module files next to the
        weights, so the saved dir loads via the Auto classes with
        trust_remote_code in a fresh process — no repo sys.path needed."""
        model = make_model(False, None)
        model.save_pretrained(str(tmp_path))

        with open(os.path.join(str(tmp_path), "config.json")) as f:
            saved_config = json.load(f)
        assert saved_config.get("auto_map") == {
            "AutoConfig": "configuration_apertus_moe.ApertusMoeConfig",
            # Two keys, not three: save_pretrained on a ForCausalLM emits only the auto class
            # THAT model registered. ApertusMoeModel registers AutoModel separately, which shows
            # up when saving a bare backbone -- and the EXPORTER writes all three explicitly
            # (exporter/writer.py), which is what a shipped directory carries.
            "AutoModelForCausalLM": "modeling_apertus_moe.ApertusMoeForCausalLM",
        }
        assert os.path.isfile(os.path.join(str(tmp_path), "configuration_apertus_moe.py"))
        assert os.path.isfile(os.path.join(str(tmp_path), "modeling_apertus_moe.py"))

        # and the dir actually loads through the Auto classes (dynamic module
        # path, NOT the repo import already in sys.path), with zero key issues
        reloaded, info = AutoModelForCausalLM.from_pretrained(
            str(tmp_path),
            trust_remote_code=True,
            dtype=torch.float32,
            output_loading_info=True,
        )
        assert type(reloaded).__name__ == "ApertusMoeForCausalLM"
        assert type(reloaded.config).__name__ == "ApertusMoeConfig"
        assert not info["missing_keys"], info["missing_keys"]
        assert not info["unexpected_keys"], info["unexpected_keys"]
        assert not info["mismatched_keys"], info["mismatched_keys"]
        assert not info["error_msgs"], info["error_msgs"]

    def test_e_score_correction_bias_stays_fp32_under_bf16(
        self, make_model, tmp_path
    ):
        """Requirement 8: _keep_in_fp32_modules_strict must protect the router
        bias — a bf16 cast silently shifts expert selection near top-k ties.

        NOTE: on transformers 5.8.1 this test passes even WITHOUT the
        _keep_in_fp32_modules_strict flag (the loader keeps fp32 buffers fp32
        under dtype=bf16 on its own); the literal flag assertion in
        TestContractPlumbing::test_pretrained_model_flags is the real contract
        pin. This test still guards the observable dtype behavior end to end."""
        model = make_model(True, TINY_LATENT)
        model.save_pretrained(str(tmp_path))
        reloaded = ApertusMoeForCausalLM.from_pretrained(
            str(tmp_path), dtype=torch.bfloat16
        )
        for layer in _moe_layers(reloaded):
            assert layer.mlp.gate.e_score_correction_bias.dtype == torch.float32
            assert layer.mlp.gate.weight.dtype == torch.bfloat16
        assert reloaded.model.embed_tokens.weight.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# requirement 11: untied embeddings
# ---------------------------------------------------------------------------


class TestUntiedEmbeddings:
    def test_untied_lm_head(self, make_model):
        model = make_model(False, None)
        embed = model.model.embed_tokens.weight
        lm_head = model.lm_head.weight
        assert embed.data_ptr() != lm_head.data_ptr(), "lm_head is tied to embed_tokens"
        before = lm_head.detach().clone()
        with torch.no_grad():
            embed.add_(1.0)
        torch.testing.assert_close(model.lm_head.weight, before, rtol=0.0, atol=0.0)
        state_dict = model.state_dict()
        assert "lm_head.weight" in state_dict
        assert "model.embed_tokens.weight" in state_dict


# ---------------------------------------------------------------------------
# requirement 12: stock-Glm4Moe equivalence oracle
# ---------------------------------------------------------------------------

# Explicit ApertusMoe -> Glm4Moe state-dict rename map (D3 naming decision):
# our pre-norms follow dense Apertus; GLM calls the attention pre-norm
# "input_layernorm" and — collision trap — the FFN PRE-norm
# "post_attention_layernorm". Everything else matches key-for-key.
APERTUS_TO_GLM4MOE_RENAMES = [
    (".attention_layernorm.", ".input_layernorm."),
    (".feedforward_layernorm.", ".post_attention_layernorm."),
]


class TestStockGlm4MoeOracle:
    def test_flagoff_model_matches_stock_glm4_moe(self, make_model):
        """With sandwich off, latent off and both multipliers 1.0, ApertusMoe
        must be exactly stock Glm4Moe — a free correctness proof for every
        non-novel module (attention, qk-norm, router, experts, shared expert,
        dense layer 0, rope)."""
        apertus = make_model(
            False, None, embedding_multiplier=1.0, residual_multiplier=1.0
        )
        glm_config = Glm4MoeConfig(
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
            n_routed_experts=TINY_N_EXPERTS,
            num_experts_per_tok=TINY_TOPK,
            moe_intermediate_size=TINY_MOE_INTERMEDIATE,
            n_shared_experts=1,
            first_k_dense_replace=TINY_FIRST_K_DENSE,
            routed_scaling_factor=2.5,
            norm_topk_prob=True,
            n_group=1,
            topk_group=1,
            use_qk_norm=True,
            tie_word_embeddings=False,  # Glm4Moe ties by default: force untied
            rope_parameters={
                "rope_type": "default",
                "rope_theta": ROPE_THETA,
                # Glm4Moe's pattern default is 0.5 — must be explicit 1.0
                "partial_rotary_factor": 1.0,
            },
        )
        glm = Glm4MoeForCausalLM(glm_config)
        glm.eval()
        # Glm4Moe must actually be untied for the lm_head transplant to hold
        assert (
            glm.lm_head.weight.data_ptr()
            != glm.model.embed_tokens.weight.data_ptr()
        )
        # same attention kernel on both sides
        apertus.config._attn_implementation = "eager"
        glm.config._attn_implementation = "eager"

        apertus_sd = apertus.state_dict()
        renamed = {}
        for key, value in apertus_sd.items():
            new_key = key
            for src, dst in APERTUS_TO_GLM4MOE_RENAMES:
                new_key = new_key.replace(src, dst)
            renamed[new_key] = value
        assert len(renamed) == len(apertus_sd), "rename map collided keys"

        # strict load IS the key-bijection assert between the two models
        glm.load_state_dict(renamed, strict=True)

        generator = torch.Generator().manual_seed(99)
        ids = torch.randint(0, TINY_VOCAB, (2, 9), generator=generator)
        with torch.no_grad():
            apertus_logits = apertus(ids).logits.float()
            glm_logits = glm(ids).logits.float()
        torch.testing.assert_close(apertus_logits, glm_logits, rtol=1e-5, atol=1e-5)
        assert torch.equal(
            apertus_logits.argmax(-1), glm_logits.argmax(-1)
        )


# ---------------------------------------------------------------------------
# requirement 13: generate smoke
# ---------------------------------------------------------------------------


class TestGenerate:
    @pytest.mark.parametrize("sandwich_norm,moe_latent_size", SANDWICH_LATENT_MATRIX)
    def test_generate_smoke(self, make_model, input_ids, sandwich_norm, moe_latent_size):
        model = make_model(sandwich_norm, moe_latent_size)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=8,
                min_new_tokens=8,  # random weights may emit eos; force 8 tokens
                do_sample=False,
                use_cache=True,
                pad_token_id=0,
            )
        assert out.shape == (2, input_ids.shape[1] + 8)
        assert torch.equal(out[:, : input_ids.shape[1]], input_ids)
        assert ((out >= 0) & (out < TINY_VOCAB)).all()


# ---------------------------------------------------------------------------
# KEEL: highway Post-LN residual mode (Megatron fork `--keel`)
#
# Contract (as implemented in ApertusMoeDecoderLayer.__init__ / .forward):
#   keel_first_layer = (layer_idx == 0)
#   keel_alpha_val   = config.keel_alpha if not None else float(2 * num_hidden_layers)
#   keel_residual_scale = 1.0 if first layer else keel_alpha_val
#   attention:  h = post_norm( scale*residual + attn(pre_norm(x)) )
#               but layer 0 has NO attention post-norm and scale 1.0 -> plain pre-norm
#   mlp:        h = post_norm( scale*residual + mlp(pre_norm(x)) )   on ALL layers
#   The post-norm is applied AFTER the residual add (Post-LN), unlike sandwich
#   which norms the sublayer output BEFORE the add.
# ---------------------------------------------------------------------------


def _keel_model(make_model, moe_latent_size=None, keel_alpha=None, seed=0, **overrides):
    """Build a tiny KEEL model. sandwich must be OFF and residual_multiplier
    must be 1.0 (else the config raises)."""
    kwargs = dict(keel=True, residual_multiplier=1.0)
    if keel_alpha is not None:
        kwargs["keel_alpha"] = keel_alpha
    kwargs.update(overrides)
    return make_model(False, moe_latent_size, seed=seed, **kwargs)


def _randomize_keel_post_norms(model):
    """Give the KEEL post-norms non-trivial (non-ones) weights so norm/scale
    ordering and the Post-LN placement matter maximally."""
    with torch.no_grad():
        for layer in model.model.layers:
            pa = getattr(layer, "post_attention_layernorm", None)
            if pa is not None:
                pa.weight.copy_(torch.randn(TINY_HIDDEN))
            layer.post_feedforward_layernorm.weight.copy_(torch.randn(TINY_HIDDEN))


def _handrolled_keel_layer_forward(
    layer, x, position_embeddings, *, alpha, first_layer, post_norm_before_add=False
):
    """KEEL Post-LN reference:

        residual = x; h = pre_norm(x); h = sublayer(h)
        h = post_norm( scale*residual + h )     # norm AFTER the add (Post-LN)

    where ``scale`` is 1.0 on the first layer else ``alpha``. The first layer's
    ATTENTION branch has no post-norm at all (plain pre-norm block); every MLP
    branch (incl. layer 0) carries a post-norm.

    ``post_norm_before_add=True`` computes the WRONG (sandwich-style) order —
    norm the sublayer output first, then add the scaled residual — used to prove
    the Post-LN ordering is load-bearing.
    """
    scale = 1.0 if first_layer else alpha

    # attention
    residual = x
    h = layer.attention_layernorm(x)
    h = _as_tensor(layer.self_attn(h, position_embeddings, None))
    if first_layer:
        # plain pre-norm block: scale 1.0, no attention post-norm
        h = scale * residual + h
    else:
        post_attn = layer.post_attention_layernorm
        if post_norm_before_add:
            h = scale * residual + post_attn(h)
        else:
            h = post_attn(scale * residual + h)
    x = h

    # mlp
    residual = x
    h = layer.feedforward_layernorm(x)
    h = _as_tensor(layer.mlp(h))
    post_ff = layer.post_feedforward_layernorm
    if post_norm_before_add:
        h = scale * residual + post_ff(h)
    else:
        h = post_ff(scale * residual + h)
    return h


class TestKeel:
    # -- 1. config guards ---------------------------------------------------

    def test_keel_with_sandwich_norm_rejected(self, make_config):
        with pytest.raises(ValueError, match="sandwich_norm"):
            make_config(sandwich_norm=True, keel=True, residual_multiplier=1.0)

    def test_keel_with_residual_multiplier_rejected(self, make_config):
        with pytest.raises(ValueError, match="residual_multiplier"):
            make_config(keel=True, residual_multiplier=2.0)
        # the contract default residual_multiplier (!= 1.0) is likewise refused
        with pytest.raises(ValueError, match="residual_multiplier"):
            make_config(keel=True)

    def test_keel_alpha_without_keel_rejected(self, make_config):
        with pytest.raises(ValueError, match="keel_alpha"):
            make_config(keel_alpha=3.0)  # keel defaults to False

    def test_valid_keel_config_constructs(self, make_config):
        cfg = make_config(keel=True, residual_multiplier=1.0)
        assert cfg.keel is True
        assert cfg.sandwich_norm is False
        assert cfg.residual_multiplier == 1.0
        assert cfg.keel_alpha is None  # unset -> resolves to 2*num_hidden_layers
        cfg2 = make_config(keel=True, residual_multiplier=1.0, keel_alpha=5.0)
        assert cfg2.keel_alpha == 5.0

    def test_keel_off_by_default(self):
        cfg = ApertusMoeConfig()
        assert cfg.keel is False
        assert cfg.keel_alpha is None

    # -- 2. conditional module existence ------------------------------------

    @pytest.mark.parametrize(
        "moe_latent_size", [None, TINY_LATENT], ids=["latent-off", "latent-on"]
    )
    def test_keel_post_norm_module_existence(self, make_model, moe_latent_size):
        model = _keel_model(make_model, moe_latent_size)
        state_dict = model.state_dict()
        for L, layer in enumerate(model.model.layers):
            # post_feedforward_layernorm exists on ALL layers (incl. layer 0)
            ff = getattr(layer, "post_feedforward_layernorm", None)
            assert isinstance(ff, ApertusMoeRMSNorm), f"layer {L} lacks post_feedforward_layernorm"
            ff_key = f"model.layers.{L}.post_feedforward_layernorm.weight"
            assert ff_key in state_dict
            assert state_dict[ff_key].shape == (TINY_HIDDEN,)

            # post_attention_layernorm exists on L>=1 only; ABSENT on layer 0
            pa = getattr(layer, "post_attention_layernorm", None)
            pa_key = f"model.layers.{L}.post_attention_layernorm.weight"
            if L == 0:
                assert pa is None, "layer 0 must NOT carry an attention post-norm under KEEL"
                assert pa_key not in state_dict
            else:
                assert isinstance(pa, ApertusMoeRMSNorm), f"layer {L} lacks post_attention_layernorm"
                assert pa_key in state_dict
                assert state_dict[pa_key].shape == (TINY_HIDDEN,)

        # module-tree cross-check: exactly one missing post-attention norm (layer 0)
        module_names = {n for n, _ in model.named_modules()}
        assert "model.layers.0.post_feedforward_layernorm" in module_names
        assert "model.layers.0.post_attention_layernorm" not in module_names
        for L in range(1, TINY_LAYERS):
            assert f"model.layers.{L}.post_attention_layernorm" in module_names
            assert f"model.layers.{L}.post_feedforward_layernorm" in module_names

        # the first-layer flag and carry scale wiring
        assert model.model.layers[0].keel_first_layer is True
        assert model.model.layers[0].keel_residual_scale == 1.0
        for L in range(1, TINY_LAYERS):
            assert model.model.layers[L].keel_first_layer is False
            assert model.model.layers[L].keel_residual_scale == float(2 * TINY_LAYERS)

    # -- 3. forward math vs a hand-rolled KEEL reference --------------------

    @pytest.mark.parametrize("keel_alpha", [None, 3.0], ids=["alpha-default", "alpha-3"])
    @pytest.mark.parametrize(
        "layer_idx", [0, 1, 2], ids=["first-layer0", "moe-layer1", "moe-layer2"]
    )
    @pytest.mark.parametrize(
        "moe_latent_size", [None, TINY_LATENT], ids=["latent-off", "latent-on"]
    )
    def test_layer_matches_handrolled_keel_reference(
        self, make_model, keel_alpha, layer_idx, moe_latent_size
    ):
        model = _keel_model(make_model, moe_latent_size, keel_alpha=keel_alpha)
        _randomize_keel_post_norms(model)
        layer = model.model.layers[layer_idx]
        # resolved carry scale: default keel_alpha is 2*num_hidden_layers
        alpha = keel_alpha if keel_alpha is not None else float(2 * TINY_LAYERS)
        x = torch.randn(2, 5, TINY_HIDDEN)
        pos_emb = _rope(model, x)
        with torch.no_grad():
            out = _call_layer(layer, x, pos_emb)
            ref = _handrolled_keel_layer_forward(
                layer, x, pos_emb, alpha=alpha, first_layer=(layer_idx == 0)
            )
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_full_model_matches_handrolled_keel_reference(self, make_model, input_ids):
        """End-to-end: stacking the per-layer KEEL reference through the whole
        model (embedding multiplier + final norm + lm_head) reproduces the
        logits, pinning the first-layer special case AND the regular layers at
        once."""
        model = _keel_model(make_model, None)
        _randomize_keel_post_norms(model)
        alpha = float(2 * TINY_LAYERS)
        with torch.no_grad():
            h = model.model.embed_tokens(input_ids) * model.config.embedding_multiplier
            pos_emb = _rope(model, h)
            for L, layer in enumerate(model.model.layers):
                h = _handrolled_keel_layer_forward(
                    layer, h, pos_emb, alpha=alpha, first_layer=(L == 0)
                )
            h = model.model.norm(h)
            ref_logits = model.lm_head(h)
            out_logits = model(input_ids).logits
        torch.testing.assert_close(out_logits, ref_logits, rtol=1e-4, atol=1e-4)

    # -- 4. keel_alpha default == 2*num_hidden_layers -----------------------

    def test_keel_alpha_default_equals_two_num_layers(self, make_model, input_ids):
        """An unset keel_alpha must behave exactly as keel_alpha=2*num_hidden_layers."""
        default_model = _keel_model(make_model, None, keel_alpha=None, seed=0)
        explicit_model = _keel_model(
            make_model, None, keel_alpha=float(2 * TINY_LAYERS), seed=0
        )
        # same seed -> identical weights; the only question is the carry scale
        for L in range(TINY_LAYERS):
            assert (
                default_model.model.layers[L].keel_residual_scale
                == explicit_model.model.layers[L].keel_residual_scale
            )
        with torch.no_grad():
            a = default_model(input_ids).logits
            b = explicit_model(input_ids).logits
        torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)

        # ...and it is genuinely load-bearing: a DIFFERENT alpha changes outputs
        other_model = _keel_model(make_model, None, keel_alpha=1.0, seed=0)
        with torch.no_grad():
            c = other_model(input_ids).logits
        assert (a - c).abs().max().item() > 1e-3, (
            "keel_alpha did not affect the output — the carry scale is not applied"
        )

    # -- 5. Post-LN vs sandwich ordering (norm AFTER the add) ---------------

    def test_keel_norm_after_add_not_before(self, make_model):
        """KEEL normalizes the SUM (Post-LN); norming the sublayer output BEFORE
        the add (sandwich order) is a different function. Uses a non-first layer
        so both branches carry a post-norm."""
        model = _keel_model(make_model, None)
        _randomize_keel_post_norms(model)
        layer = model.model.layers[1]
        alpha = float(2 * TINY_LAYERS)
        x = torch.randn(2, 5, TINY_HIDDEN)
        pos_emb = _rope(model, x)
        with torch.no_grad():
            out = _call_layer(layer, x, pos_emb)
            right = _handrolled_keel_layer_forward(
                layer, x, pos_emb, alpha=alpha, first_layer=False
            )
            wrong = _handrolled_keel_layer_forward(
                layer, x, pos_emb, alpha=alpha, first_layer=False,
                post_norm_before_add=True,
            )
        torch.testing.assert_close(out, right, rtol=1e-5, atol=1e-5)
        assert (out - wrong).abs().max().item() > 1e-2, (
            "KEEL layer matches the WRONG (norm-before-add) order — the post-norm "
            "must be applied AFTER the residual add (Post-LN, not sandwich)"
        )

    # -- 6. smoke + save/load round trip (KEEL composes with LatentMoE) ------

    @pytest.mark.parametrize(
        "moe_latent_size", [None, TINY_LATENT], ids=["latent-off", "latent-on"]
    )
    def test_keel_forward_and_generate_smoke(self, make_model, input_ids, moe_latent_size):
        model = _keel_model(make_model, moe_latent_size)
        with torch.no_grad():
            out = model(input_ids)
            assert out.logits.shape == (2, 7, TINY_VOCAB)
            assert torch.isfinite(out.logits).all()
            gen = model.generate(
                input_ids,
                max_new_tokens=8,
                min_new_tokens=8,
                do_sample=False,
                use_cache=True,
                pad_token_id=0,
            )
        assert gen.shape == (2, input_ids.shape[1] + 8)
        assert torch.equal(gen[:, : input_ids.shape[1]], input_ids)

    def test_keel_save_load_round_trip_with_latent(self, make_model, input_ids, tmp_path):
        """Bitwise state-dict + logits equality across save/load, with KEEL and
        moe_latent_size both on — proves KEEL composes with LatentMoE and that
        the KEEL post-norm tensors persist and reload cleanly."""
        model = _keel_model(make_model, TINY_LATENT)
        _randomize_keel_post_norms(model)  # non-ones so equality below is meaningful

        # on-disk keys carry post_feedforward on all layers, post_attention on L>=1 only
        model.save_pretrained(str(tmp_path))
        disk = _load_saved_state_dict(tmp_path)
        for L in range(TINY_LAYERS):
            assert f"model.layers.{L}.post_feedforward_layernorm.weight" in disk
        assert "model.layers.0.post_attention_layernorm.weight" not in disk
        for L in range(1, TINY_LAYERS):
            assert f"model.layers.{L}.post_attention_layernorm.weight" in disk

        reloaded, info = ApertusMoeForCausalLM.from_pretrained(
            str(tmp_path), dtype=torch.float32, output_loading_info=True
        )
        reloaded.eval()
        assert not info["missing_keys"], info["missing_keys"]
        assert not info["unexpected_keys"], info["unexpected_keys"]
        assert not info["mismatched_keys"], info["mismatched_keys"]
        assert not info["error_msgs"], info["error_msgs"]
        assert reloaded.config.keel is True
        assert reloaded.config.moe_latent_size == TINY_LATENT

        original = model.state_dict()
        restored = reloaded.state_dict()
        assert set(original) == set(restored)
        for key in original:
            torch.testing.assert_close(
                restored[key], original[key], rtol=0.0, atol=0.0,
                msg=lambda m, key=key: f"{key} changed across the round trip: {m}",
            )
        with torch.no_grad():
            a = model(input_ids).logits
            b = reloaded(input_ids).logits
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# sssglu activation (chonk-SWA gap 1)
# ---------------------------------------------------------------------------


class TestSSSGLUActivation:
    """The gate differs from SwiGLU's; everything around it, including tensor layout, does not."""

    def test_matches_the_fork_formula(self):
        # megatron/core/activations.py sssglu_act: softsign(x - 1) + 0.5.
        x = torch.linspace(-40.0, 40.0, 4001, dtype=torch.float64)
        expected = (x - 1) / (1 + (x - 1).abs()) + 0.5
        torch.testing.assert_close(resolve_activation("sssglu")(x), expected, rtol=0.0, atol=0.0)

    def test_range_is_not_zero_to_one(self):
        # softsign spans (-1, 1), so the shifted gate spans (-0.5, 1.5). A sanity assertion
        # written against a mistaken (0, 1) range would reject correct values.
        values = resolve_activation("sssglu")(torch.linspace(-1e4, 1e4, 20001, dtype=torch.float64))
        assert values.min() > -0.5 and values.max() < 1.5
        assert values.min() < 0.0 and values.max() > 1.0
        assert resolve_activation("sssglu")(torch.ones(1, dtype=torch.float64)).item() == 0.5

    def test_activation_is_applied_to_the_gate_branch_only(self, make_config):
        # gate and up have identical shapes, so swapping them is shape-invariant and would
        # survive every other check in the suite. Distinguish the two branches by value.
        config = make_config(hidden_act="sssglu")
        torch.manual_seed(0)
        mlp = ApertusMoeMLP(config)
        activation = resolve_activation("sssglu")
        x = torch.randn(4, config.hidden_size)
        with torch.no_grad():
            gate = mlp.gate_proj(x)
            up = mlp.up_proj(x)
            expected = mlp.down_proj(activation(gate) * up)
            swapped = mlp.down_proj(activation(up) * gate)
            torch.testing.assert_close(mlp(x), expected, rtol=0.0, atol=0.0)
        assert not torch.allclose(expected, swapped, rtol=1e-3, atol=1e-3), (
            "fixture is too symmetric to detect a gate/up swap"
        )

    def test_activation_resolves_to_an_instance_without_touching_act2fn(self):
        from transformers.activations import ACT2FN

        assert "sssglu" not in ACT2FN, (
            "a trust_remote_code artifact must not insert into the library's global ACT2FN"
        )
        activation = resolve_activation("sssglu")
        assert isinstance(activation, torch.nn.Module)
        # ACT2FN is a ClassInstantier: indexing builds an instance while .get() returns the
        # class. The fallback must not hand back an uninstantiated class.
        assert isinstance(resolve_activation("silu"), torch.nn.Module)

    def test_model_forward_uses_sssglu_everywhere(self, make_model, input_ids):
        model = make_model(hidden_act="sssglu")
        dense = model.model.layers[0].mlp
        moe = model.model.layers[TINY_FIRST_K_DENSE].mlp
        for owner in (dense, moe.shared_experts, moe.experts):
            assert type(owner.act_fn).__name__ == "ApertusMoeSSSGLU"
        with torch.no_grad():
            assert torch.isfinite(model(input_ids).logits).all()


# ---------------------------------------------------------------------------
# sliding-window attention (chonk-SWA gap 2)
# ---------------------------------------------------------------------------


def _fork_sliding_window_mask(sq: int, window_size: tuple[int, int]) -> torch.Tensor:
    """megatron/core/transformer/utils.py get_sliding_window_causal_mask, device dropped.

    Returns True where a key is MASKED OUT, which is the fork's convention.
    """
    m = torch.ones(sq, sq, dtype=torch.bool)
    mu = torch.triu(m, diagonal=-window_size[0])
    ml = torch.tril(mu, diagonal=window_size[1])
    return ~ml


class TestSlidingWindowAttention:
    """Megatron's window is inclusive at both ends; HF's counts admitted keys."""

    @staticmethod
    def _build_mask(config, seq_len):
        from transformers.masking_utils import create_sliding_window_causal_mask

        config._attn_implementation = "eager"
        return create_sliding_window_causal_mask(
            config=config,
            inputs_embeds=torch.zeros(1, seq_len, config.hidden_size),
            attention_mask=None,
            past_key_values=None,
            position_ids=torch.arange(seq_len).unsqueeze(0),
        )

    @pytest.mark.parametrize("left_context", [0, 1, 4, 15])
    def test_window_matches_the_fork_bitwise_after_adding_one(self, make_config, left_context):
        seq_len = 24
        config = make_config(
            sliding_window=left_context + 1,
            layer_types=["sliding_attention"] * TINY_LAYERS,
        )
        admitted = self._build_mask(config, seq_len)[0, 0] == 0
        fork_admitted = ~_fork_sliding_window_mask(seq_len, (left_context, 0))
        assert torch.equal(admitted, fork_admitted)

    def test_off_by_one_is_detected(self, make_config):
        # The whole point of the +1: writing the Megatron number verbatim silently drops one key.
        seq_len = 24
        config = make_config(
            sliding_window=8, layer_types=["sliding_attention"] * TINY_LAYERS
        )
        admitted = self._build_mask(config, seq_len)[0, 0] == 0
        assert not torch.equal(admitted, ~_fork_sliding_window_mask(seq_len, (8, 0)))
        assert torch.equal(admitted, ~_fork_sliding_window_mask(seq_len, (7, 0)))
        # Stated as a count, which is what "sliding_window" actually means in Transformers.
        assert admitted[-1].sum().item() == 8

    def test_per_layer_wiring(self, make_model):
        layer_types = ["sliding_attention", "full_attention", "sliding_attention"]
        model = make_model(sliding_window=5, layer_types=layer_types)
        attentions = [layer.self_attn for layer in model.model.layers]
        assert [a.layer_type for a in attentions] == layer_types
        # A full-attention layer must carry None, not the window: the value is forwarded to the
        # attention backend, where flash-attention reads it and nothing else encodes the window.
        assert [a.sliding_window for a in attentions] == [5, None, 5]
        assert model.model.has_sliding_layers is True

    def test_no_sliding_layers_builds_no_sliding_mask(self, make_model):
        model = make_model()
        assert model.model.has_sliding_layers is False
        assert all(layer.self_attn.sliding_window is None for layer in model.model.layers)

    def test_sliding_layer_ignores_tokens_outside_the_window(self, make_config):
        # One layer, so nothing propagates across positions: the last position must be blind to
        # a token further back than the window.
        window = 3
        config = make_config(
            num_hidden_layers=1, layer_types=["sliding_attention"], sliding_window=window,
            first_k_dense_replace=1,
        )
        torch.manual_seed(0)
        model = ApertusMoeForCausalLM(config).eval()
        ids = torch.randint(0, TINY_VOCAB, (1, 8))
        far = ids.clone()
        far[0, 0] = (far[0, 0] + 1) % TINY_VOCAB          # outside the last query's window
        near = ids.clone()
        near[0, 6] = (near[0, 6] + 1) % TINY_VOCAB        # inside it
        with torch.no_grad():
            base = model(input_ids=ids).logits[0, -1]
            changed_far = model(input_ids=far).logits[0, -1]
            changed_near = model(input_ids=near).logits[0, -1]
        torch.testing.assert_close(base, changed_far, rtol=1e-6, atol=1e-6)
        assert not torch.allclose(base, changed_near, rtol=1e-4, atol=1e-4)

    def test_cache_uses_sliding_layers_only_where_declared(self, make_config):
        from transformers.cache_utils import DynamicCache

        config = make_config(
            sliding_window=5,
            layer_types=["sliding_attention", "full_attention", "sliding_attention"],
        )
        cache = DynamicCache(config=config)
        is_sliding = [getattr(layer, "sliding_window", None) is not None for layer in cache.layers]
        assert is_sliding == [True, False, True], (
            "Transformers builds the per-layer cache class from config.layer_types; a full cache "
            "on a sliding layer would keep keys the mask then hides, wasting memory and diverging "
            "from Megatron at long context"
        )

    def test_window_reaches_the_attention_backend(self, make_model, input_ids, monkeypatch):
        # flash-attention receives the window through this kwarg and through nothing else, so a
        # mask-only implementation would silently run full attention on every sliding layer.
        import modeling_apertus_moe as modeling

        seen = []
        original = modeling.eager_attention_forward

        def recording(module, *args, **kwargs):
            seen.append((module.layer_idx, kwargs.get("sliding_window", "ABSENT")))
            return original(module, *args, **kwargs)

        monkeypatch.setattr(modeling, "eager_attention_forward", recording)
        model = make_model(
            sliding_window=5,
            layer_types=["sliding_attention", "full_attention", "sliding_attention"],
        )
        model.config._attn_implementation = "eager"
        with torch.no_grad():
            model(input_ids=input_ids)
        assert seen == [(0, 5), (1, None), (2, 5)]


# ---------------------------------------------------------------------------
# per-layer NoPE (chonk-SWA gap 3)
# ---------------------------------------------------------------------------


class TestNoRopeLayers:
    """HF's 1 means "rotate"; Megatron's no_rope_freq 1 means "skip". Polarity is everything."""

    def test_rotation_is_skipped_on_exactly_the_declared_layers(
        self, make_model, input_ids, monkeypatch
    ):
        # A logits comparison alone cannot localise an inverted list, so record the call sites.
        import modeling_apertus_moe as modeling

        model = make_model(no_rope_layers=[1, 0, 1])
        model.config._attn_implementation = "eager"

        # Layers run in order, so the call order identifies the layer that rotated.
        rotated: list[int] = []
        original = modeling.apply_rotary_pos_emb
        attention_order = {id(layer.self_attn): i for i, layer in enumerate(model.model.layers)}
        calls = {"n": 0}

        def recording(q, k, cos, sin, *args, **kwargs):
            rotated.append(calls["n"])
            calls["n"] += 1
            return original(q, k, cos, sin, *args, **kwargs)

        monkeypatch.setattr(modeling, "apply_rotary_pos_emb", recording)
        with torch.no_grad():
            model(input_ids=input_ids)

        assert len(rotated) == 2, (
            f"expected RoPE on 2 of {TINY_LAYERS} layers (no_rope_layers=[1, 0, 1]), "
            f"got {len(rotated)} calls"
        )
        assert len(attention_order) == TINY_LAYERS
        assert [layer.self_attn.use_rope for layer in model.model.layers] == [True, False, True]

    @staticmethod
    def _logits_under_two_position_spacings(config):
        """Run one layer over the same tokens at two different position SPACINGS.

        A uniform shift is not a discriminator: RoPE is relative, so shifting every position by
        the same amount leaves a rotating layer's output unchanged too. Stretching the spacing
        changes the relative offsets, which only a rotating layer can see.
        """
        torch.manual_seed(0)
        model = ApertusMoeForCausalLM(config).eval()
        ids = torch.randint(0, TINY_VOCAB, (1, 6))
        with torch.no_grad():
            close = model(input_ids=ids, position_ids=torch.arange(6).unsqueeze(0)).logits
            spread = model(input_ids=ids, position_ids=(torch.arange(6) * 3).unsqueeze(0)).logits
        return close, spread

    def test_nope_layer_is_blind_to_position(self, make_config):
        close, spread = self._logits_under_two_position_spacings(
            make_config(num_hidden_layers=1, no_rope_layers=[0], first_k_dense_replace=1)
        )
        torch.testing.assert_close(close, spread, rtol=1e-6, atol=1e-6)

    def test_rope_layer_is_not_blind_to_position(self, make_config):
        # The control: without it, a model that silently skipped RoPE everywhere would pass the
        # test above for the wrong reason.
        close, spread = self._logits_under_two_position_spacings(
            make_config(num_hidden_layers=1, no_rope_layers=[1], first_k_dense_replace=1)
        )
        assert not torch.allclose(close, spread, rtol=1e-4, atol=1e-4)

    def test_default_rotates_every_layer(self, make_model):
        model = make_model()
        assert model.config.no_rope_layers == [1] * TINY_LAYERS
        assert all(layer.self_attn.use_rope for layer in model.model.layers)

    def test_schedules_survive_a_save_load_round_trip(self, make_model, tmp_path):
        model = make_model(
            sliding_window=5,
            layer_types=["sliding_attention", "full_attention", "sliding_attention"],
            no_rope_layers=[1, 0, 1],
            hidden_act="sssglu",
        )
        model.save_pretrained(str(tmp_path))
        reloaded = ApertusMoeForCausalLM.from_pretrained(str(tmp_path), dtype=torch.float32)
        assert reloaded.config.sliding_window == 5
        assert reloaded.config.layer_types == model.config.layer_types
        assert reloaded.config.no_rope_layers == [1, 0, 1]
        assert reloaded.config.hidden_act == "sssglu"
        assert [a.self_attn.use_rope for a in reloaded.model.layers] == [True, False, True]
