"""Namespace-in / kwargs-out tests for ``exporter/config_from_args.py``.

The good namespace comes from megatron_mock.make_args_namespace (fork spellings), then each
test perturbs exactly one attribute. Canonical value checks go through an actually
constructed Apertus2Config, so they hold whether the kwargs spell a field explicitly or
lean on a config default, and whether rope is passed as rope_theta or rope_parameters.

Covered: geometry derivations; multipliers on/off with the exact tiny-
geometry floats sqrt(32) and 1/sqrt(6); kv_channels None fallback; moe_ffn fallback;
n_shared derivation; explicit/interleaved moe_layer_freq and homogeneous-int rejection; QB
detection from list AND scalar
string; expert_bias_present passthrough; every hard assert firing with a ValueError that
names the offending arg (one bad arg at a time on an otherwise-good namespace).
"""

import math

import pytest

pytest.importorskip("megatron.core")

import exporter.config_from_args as config_from_args_module  # noqa: E402

import megatron_mock  # noqa: E402
from configuration_apertus2 import Apertus2Config  # noqa: E402

# ---------------------------------------------------------------------------
# entry-point resolution + result normalization (function name is not pinned by the
# contract; resolve defensively and fail with an actionable message if nothing matches)
# ---------------------------------------------------------------------------

_ENTRY_CANDIDATES = (
    "config_from_args",
    "config_kwargs_from_args",
    "derive_config_kwargs",
    "derive_config",
    "args_to_config_kwargs",
    "from_args",
    "derive",
    "convert",
)


def _entry_point():
    for name in _ENTRY_CANDIDATES:
        candidate = getattr(config_from_args_module, name, None)
        if callable(candidate) and not isinstance(candidate, type):
            return candidate
    module_functions = [
        name
        for name, value in vars(config_from_args_module).items()
        if callable(value)
        and not isinstance(value, type)
        and not name.startswith("_")
        and getattr(value, "__module__", None) == config_from_args_module.__name__
    ]
    if len(module_functions) == 1:
        return getattr(config_from_args_module, module_functions[0])
    raise AssertionError(
        "cannot locate the derivation entry point in exporter.config_from_args; "
        f"tried {_ENTRY_CANDIDATES}, module defines {module_functions}"
    )


def derive(args):
    """-> (kwargs: dict, expert_bias_present: bool | None, raw_result)."""
    result = _entry_point()(args)
    if isinstance(result, dict):
        return result, None, result
    if isinstance(result, tuple):
        kwargs = next((item for item in result if isinstance(item, dict)), None)
        assert kwargs is not None, f"no kwargs dict in result tuple: {result!r}"
        expert_bias = next((item for item in result if isinstance(item, bool)), None)
        return kwargs, expert_bias, result
    for attr in ("kwargs", "config_kwargs"):
        if hasattr(result, attr):
            return (
                getattr(result, attr),
                getattr(result, "expert_bias_present", None),
                result,
            )
    raise AssertionError(f"unrecognized config_from_args result shape: {type(result)}")


def derive_config(args):
    kwargs, _, _ = derive(args)
    return Apertus2Config(**kwargs)


def good_args(sandwich=False, latent=None, qb=False, expert_bias=True, **overrides):
    config = megatron_mock.tiny_export_config(sandwich, latent, qb)
    return megatron_mock.make_args_namespace(
        config, expert_bias_present=expert_bias, **overrides
    )


# ---------------------------------------------------------------------------
# derivations
# ---------------------------------------------------------------------------


class TestDerivations:
    def test_tiny_geometry_round_trips(self):
        config = derive_config(good_args())
        assert config.vocab_size == 128
        assert config.hidden_size == 32
        assert config.num_hidden_layers == 3
        assert config.num_attention_heads == 4
        assert config.num_key_value_heads == 2
        assert config.head_dim == 8
        assert config.intermediate_size == 64
        assert config.moe_intermediate_size == 16
        assert config.n_routed_experts == megatron_mock.TINY_N_EXPERTS
        assert config.num_experts_per_tok == 2
        assert config.n_shared_experts == 1
        assert config.first_k_dense_replace == 1
        assert config.moe_layer_freq == [0, 1, 1]
        assert config.routed_scaling_factor == 2.5
        assert config.rms_norm_eps == 1e-5
        assert config.max_position_embeddings == 64
        assert config.attention_dropout == 0.0
        assert config.initializer_range == 0.02
        assert config.use_qk_norm is True
        assert config.sandwich_norm is False
        assert config.moe_latent_size is None
        assert config.use_quantile_balancing is False
        assert config.tie_word_embeddings is False
        assert config.norm_topk_prob is True
        assert config.n_group == 1 and config.topk_group == 1
        # rope: theta 500000.0, FULL rotary (the partial_rotary_factor trap)
        assert float(config.rope_parameters["rope_theta"]) == 500000.0
        assert float(config.rope_parameters.get("partial_rotary_factor", 1.0)) == 1.0

    def test_flag_combos_land_in_config(self):
        config = derive_config(good_args(sandwich=True, latent=24, qb=True))
        assert config.sandwich_norm is True
        assert config.moe_latent_size == 24
        assert config.use_quantile_balancing is True

    def test_topk_one_disables_probability_renormalization(self):
        config = derive_config(good_args(moe_router_topk=1))
        assert config.num_experts_per_tok == 1
        assert config.norm_topk_prob is False

    @pytest.mark.parametrize(
        "n_group,topk_group",
        [
            pytest.param(2, 1, id="two-groups-select-one"),
            pytest.param(4, 2, id="four-groups-select-two"),
        ],
    )
    def test_group_limited_routing_lands_in_config(self, n_group, topk_group):
        config = derive_config(
            good_args(
                moe_router_num_groups=n_group,
                moe_router_group_topk=topk_group,
            )
        )
        assert config.n_group == n_group
        assert config.topk_group == topk_group

    def test_group_count_without_group_topk_is_an_inactive_megatron_setting(self):
        config = derive_config(
            good_args(moe_router_num_groups=4, moe_router_group_topk=None)
        )
        assert config.n_group == 1
        assert config.topk_group == 1

    @pytest.mark.parametrize(
        "overrides,expected_message",
        [
            pytest.param(
                {"moe_router_num_groups": None, "moe_router_group_topk": 1},
                "moe_router_num_groups",
                id="missing-group-count",
            ),
            pytest.param(
                {"moe_router_num_groups": 5, "moe_router_group_topk": 1},
                "num_experts is divisible",
                id="experts-not-divisible-by-groups",
            ),
            pytest.param(
                {"moe_router_num_groups": 2, "moe_router_group_topk": 3},
                "moe_router_group_topk <= args.moe_router_num_groups",
                id="too-many-selected-groups",
            ),
            pytest.param(
                {
                    "moe_router_num_groups": 6,
                    "moe_router_group_topk": 2,
                    "moe_router_topk": 1,
                },
                "moe_router_group_topk <= args.moe_router_topk",
                id="fewer-experts-than-selected-groups",
            ),
            pytest.param(
                {
                    "moe_router_num_groups": 6,
                    "moe_router_group_topk": 1,
                    "moe_router_topk": 3,
                },
                "fits inside the selected expert groups",
                id="selected-groups-have-too-few-experts",
            ),
        ],
    )
    def test_invalid_group_limited_geometry_is_rejected(self, overrides, expected_message):
        with pytest.raises(ValueError, match=expected_message):
            derive_config(good_args(**overrides))

    def test_quantile_balancing_and_group_limited_routing_are_rejected(self):
        with pytest.raises(ValueError, match="quantile balancing"):
            derive_config(
                good_args(
                    qb=True,
                    moe_router_num_groups=2,
                    moe_router_group_topk=1,
                )
            )

    def test_quantile_balancing_rejects_even_an_inactive_group_count(self):
        with pytest.raises(ValueError, match="quantile balancing"):
            derive_config(
                good_args(
                    qb=True,
                    moe_router_num_groups=2,
                    moe_router_group_topk=None,
                )
            )

    def test_multipliers_on_exact_floats(self):
        config = derive_config(good_args())
        assert config.embedding_multiplier == math.sqrt(32)
        assert config.residual_multiplier == 1.0 / math.sqrt(6)

    def test_multipliers_off_are_exactly_one(self):
        args = good_args(
            scale_embeddings_by_sqrt_hidden=False, residual_output_scaling=False
        )
        config = derive_config(args)
        assert config.embedding_multiplier == 1.0
        assert config.residual_multiplier == 1.0

    def test_kv_channels_none_falls_back_to_hidden_over_heads(self):
        config = derive_config(good_args(kv_channels=None))
        assert config.head_dim == 32 // 4

    def test_group_query_attention_off_means_kv_equals_heads(self):
        config = derive_config(good_args(group_query_attention=False))
        assert config.num_key_value_heads == 4

    def test_moe_ffn_none_falls_back_to_ffn_hidden_size(self):
        args = good_args(
            moe_ffn_hidden_size=None,
            moe_shared_expert_intermediate_size=64,  # keep n_shared == 1 after fallback
        )
        config = derive_config(args)
        assert config.moe_intermediate_size == 64
        assert config.n_shared_experts == 1

    def test_n_shared_derivation(self):
        config = derive_config(good_args(moe_shared_expert_intermediate_size=32))
        assert config.n_shared_experts == 2

    def test_routed_scaling_factor_none_defaults_to_one(self):
        config = derive_config(good_args(moe_router_topk_scaling_factor=None))
        assert config.routed_scaling_factor == 1.0

    @pytest.mark.parametrize(
        "freq,expected_pattern,expected_first_k_dense",
        [
            pytest.param([0, 1, 1], [0, 1, 1], 1, id="one-dense"),
            pytest.param([0, 0, 1], [0, 0, 1], 2, id="two-dense"),
            pytest.param([1, 1, 1], [1, 1, 1], 0, id="all-moe-list"),
            pytest.param([0, 1, 0], [0, 1, 0], 1, id="interleaved-list"),
            pytest.param(2, [1, 0, 1], 0, id="integer-every-two"),
        ],
    )
    def test_moe_layer_freq_patterns(self, freq, expected_pattern, expected_first_k_dense):
        config = derive_config(good_args(moe_layer_freq=freq))
        assert config.moe_layer_freq == expected_pattern
        assert config.first_k_dense_replace == expected_first_k_dense

    def test_int_moe_layer_freq_is_rejected_because_it_changes_the_key_namespace(self):
        # An int freq makes the fork write HOMOGENEOUS keys (layer index folded into a tensor
        # axis), which the per-layer mapping table cannot read. Pinned against the real fork in
        # test_fork_real_keys.py::TestHomogeneousKeys.
        with pytest.raises(ValueError, match="moe_layer_freq"):
            derive(good_args(moe_layer_freq=1))

    @pytest.mark.parametrize(
        "balancing,expected_qb",
        [
            pytest.param(["seq_aux_loss", "quantile_balancing"], True, id="list-with-qb"),
            pytest.param(["seq_aux_loss"], False, id="list-without-qb"),
            pytest.param("quantile_balancing", True, id="scalar-qb"),
            pytest.param("aux_loss", False, id="scalar-aux"),
            pytest.param(["aux_loss", "seq_aux_loss"], False, id="list-two-non-qb"),
        ],
    )
    def test_quantile_balancing_detection(self, balancing, expected_qb):
        config = derive_config(good_args(moe_router_load_balancing_type=balancing))
        assert config.use_quantile_balancing is expected_qb

    @pytest.mark.parametrize("enabled", [True, False])
    def test_expert_bias_present_returned_alongside(self, enabled):
        _, expert_bias_present, raw = derive(good_args(expert_bias=enabled))
        assert expert_bias_present is not None, (
            "expert_bias_present (= args.moe_router_enable_expert_bias) must "
            f"be returned alongside the kwargs; could not locate a bool in {raw!r}"
        )
        assert expert_bias_present is enabled

    @pytest.mark.parametrize("enabled", [True, False])
    def test_attention_output_gate_is_derived(self, enabled):
        config = derive_config(good_args(attention_output_gate=enabled))
        assert config.attention_output_gate is enabled

    def test_attention_output_gate_absent_from_args_defaults_off(self):
        args = good_args()
        delattr(args, "attention_output_gate")
        config = derive_config(args)
        assert config.attention_output_gate is False

    def test_mtp_num_layers_zero_is_accepted(self):
        derive_config(good_args(mtp_num_layers=0))

    def test_softmax_type_none_is_accepted(self):
        derive_config(good_args(softmax_type=None))


# ---------------------------------------------------------------------------
# hard asserts: each fires a ValueError naming the offending arg (one at a time)
# ---------------------------------------------------------------------------

HARD_ASSERT_CASES = [
    ("normalization", "LayerNorm", "normalization"),
    ("swiglu", False, "swiglu"),
    ("glu_linear_offset", 1.0, "glu_linear_offset"),
    ("activation_func_clamp_value", 6.0, "activation_func_clamp_value"),
    ("add_bias_linear", True, "add_bias_linear"),
    ("add_qkv_bias", True, "add_qkv_bias"),
    ("position_embedding_type", "learned_absolute", "position_embedding_type"),
    ("rotary_percent", 0.5, "rotary_percent"),
    ("rotary_interleaved", True, "rotary_interleaved"),
    ("multi_latent_attention", True, "multi_latent_attention"),
    ("mtp_num_layers", 2, "mtp_num_layers"),
    ("pnglu", True, "pnglu"),
    ("use_mup", True, "use_mup"),
    ("moe_router_score_function", "softmax", "moe_router_score_function"),
    ("moe_router_topk_limited_devices", 1, "moe_router_topk_limited_devices"),
    ("moe_shared_expert_gate", True, "moe_shared_expert_gate"),
    ("softmax_scale", 0.125, "softmax_scale"),
    ("apply_query_key_layer_scaling", True, "apply_query_key_layer_scaling"),
    # Sliding-window attention IS supported; a right-hand context is not, because a causal
    # decoder cannot look ahead and HF's sliding_window has nowhere to put it.
    ("window_size", (128, 128), "window_size"),
    ("softmax_type", "off_by_one", "softmax_type"),
    ("num_experts", None, "num_experts"),
    ("moe_apply_probs_on_input", True, "moe_apply_probs_on_input"),
    ("moe_input_jitter_eps", 0.1, "moe_input_jitter_eps"),
    ("moe_router_force_load_balancing", True, "moe_router_force_load_balancing"),
    ("moe_router_force_biased", 0.5, "moe_router_force_biased"),
    ("moe_expert_capacity_factor", 1.0, "moe_expert_capacity_factor"),
    ("fp32_residual_connection", True, "fp32_residual_connection"),
    ("untie_embeddings_and_output_weights", False, "untie_embeddings_and_output_weights"),
    # silent-semantics flags: they change model math WITHOUT changing any tensor or key, so the
    # bijection cannot see them and only these asserts stand between us and a wrong export
    # mcore's DEFAULT (None) routes in the activation dtype (bf16); the HF router is fp32, so the
    # exported model would pick different experts near ties. No tensor/key change => bijection-blind.
    ("moe_router_dtype", None, "moe_router_dtype"),
    ("moe_router_dtype", "fp64", "moe_router_dtype"),
    ("layernorm_zero_centered_gamma", True, "layernorm_zero_centered_gamma"),  # gain = 1 + weight
    ("qk_l2_norm", True, "qk_l2_norm"),  # parameter-free L2Norm overrides qk_layernorm
    ("rotary_seq_len_interpolation_factor", 4.0, "rotary_seq_len_interpolation_factor"),
    # Per-layer NoPE IS supported; a schedule that does not describe every layer is not.
    ("no_rope_freq", [1, 0], "no_rope_freq"),  # length != num_layers
    ("no_rope_freq", 2, "no_rope_freq"),  # int form must divide num_layers (3 here)
    # sssglu IS supported, but only one activation flag may be set at a time.
    ("sssglu", True, "swiglu"),
    ("apply_residual_connection_post_layernorm", True, "apply_residual_connection_post_layernorm"),
    ("transformer_impl", "local", "transformer_impl"),  # different on-disk norm key namespace
    # derivation-level asserts
    ("moe_router_topk", 0, "moe_router_topk"),
    ("moe_layer_freq", [1, 2, 1], "moe_layer_freq"),
    ("moe_layer_freq", [1, 0], "moe_layer_freq"),
    ("moe_layer_freq", 1, "moe_layer_freq"),  # int -> homogeneous keys, unreadable
    ("moe_shared_expert_intermediate_size", 24, "moe_shared_expert_intermediate_size"),
    ("moe_shared_expert_intermediate_size", 0, "moe_shared_expert_intermediate_size"),
]


class TestHardAsserts:
    @pytest.mark.parametrize(
        "attribute,bad_value,expected_in_message",
        [pytest.param(*case, id=f"{case[0]}={case[1]!r}") for case in HARD_ASSERT_CASES],
    )
    def test_assert_fires_naming_the_arg(self, attribute, bad_value, expected_in_message):
        args = good_args(**{attribute: bad_value})
        with pytest.raises(ValueError, match=expected_in_message):
            derive(args)


class TestUnsupportedRoutingModes:
    @pytest.mark.parametrize(
        "balancing",
        [
            pytest.param("sinkhorn", id="scalar"),
            pytest.param(["seq_aux_loss", "sinkhorn"], id="composed"),
        ],
    )
    def test_sinkhorn_is_rejected(self, balancing):
        with pytest.raises(ValueError, match="sinkhorn"):
            derive(good_args(moe_router_load_balancing_type=balancing))


class TestUnsupportedResidualModes:
    def test_keel_is_rejected(self):
        with pytest.raises(ValueError, match="args.keel is falsy"):
            derive(good_args(keel=True))

    def test_keel_alpha_is_rejected(self):
        with pytest.raises(ValueError, match="args.keel_alpha is None"):
            derive(good_args(keel_alpha=3.0))


class TestRopeScaling:
    """RULED 2026-07-24: active rope scaling is TRANSLATED, not rejected.

    Megatron's ``_apply_scaling`` was adapted from HF's ``_compute_llama3_parameters`` and passes
    only ``factor``; the other three llama3 parameters are hardcoded defaults in the fork. So an
    active scaling maps onto ``rope_type='llama3'`` exactly, and the chonk-SWA checkpoint (factor
    2.5) needs it. An INERT scaling still has to stay ``'default'``, or every previously exported
    config.json would churn.

    (Until 2026-07-24 active scaling was fatal; the earlier ruling only ever said the *inert*
    combinations must be accepted, which is still asserted below.)
    """

    def test_active_rope_scaling_becomes_llama3(self):
        kwargs, _, _ = derive(good_args(use_rope_scaling=True, rope_scaling_factor=2.5))
        rope = kwargs["rope_parameters"]
        assert rope["rope_type"] == "llama3"
        assert rope["factor"] == 2.5
        # The fork hardcodes these three; they are NOT read from args. In particular
        # original_max_position_embeddings must not become max_position_embeddings, which HF
        # would otherwise silently default it to, moving every frequency band.
        assert rope["low_freq_factor"] == 1.0
        assert rope["high_freq_factor"] == 4.0
        assert rope["original_max_position_embeddings"] == 8192

    def test_llama3_parameters_reach_the_config_and_validate(self):
        config = derive_config(good_args(use_rope_scaling=True, rope_scaling_factor=2.5))
        assert config.rope_parameters["rope_type"] == "llama3"
        assert config.rope_parameters["partial_rotary_factor"] == 1.0
        # Nothing in Transformers 5.8.1 validates rope_parameters on construction, so the
        # exporter's dictionary is checked explicitly here: a missing key would otherwise only
        # surface as a KeyError deep inside _compute_llama3_parameters at load time.
        config.validate_rope()

    def test_exported_llama3_frequencies_match_the_fork(self):
        torch = pytest.importorskip("torch")
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        config = derive_config(good_args(use_rope_scaling=True, rope_scaling_factor=2.5))
        hf_inv_freq, _ = ROPE_INIT_FUNCTIONS["llama3"](config, device="cpu")

        # megatron/core/models/common/embeddings/rotary_pos_embedding.py _apply_scaling,
        # transcribed verbatim with the fork's default low/high/original arguments.
        head_dim, base, factor = config.head_dim, config.rope_parameters["rope_theta"], 2.5
        freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        low_freq_factor, high_freq_factor, old_context_len = 1, 4, 8192
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        wavelen = 2 * math.pi / freqs
        fork = torch.where(wavelen > low_freq_wavelen, freqs / factor, freqs)
        smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed = (1 - smooth) * fork / factor + smooth * fork
        is_medium = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        fork = torch.where(is_medium, smoothed, fork)

        assert torch.equal(hf_inv_freq, fork)

    @pytest.mark.parametrize(
        "overrides",
        [
            pytest.param({"use_rope_scaling": False, "rope_scaling_factor": 8.0},
                         id="inert-factor-while-flag-off"),
            pytest.param({"use_rope_scaling": True, "rope_scaling_factor": 1.0},
                         id="identity-factor-while-flag-on"),
            pytest.param({"use_rope_scaling": False, "rope_scaling_factor": 1.0}, id="both-off"),
        ],
    )
    def test_inert_rope_scaling_args_are_accepted(self, overrides):
        # These are the configurations real fork checkpoints actually carry; rejecting them would
        # make the exporter refuse valid work.
        kwargs, _, _ = derive(good_args(**overrides))
        assert kwargs["rope_parameters"]["rope_theta"] > 0
        # Inert scaling must leave the dictionary alone, or every earlier export churns.
        assert kwargs["rope_parameters"]["rope_type"] == "default"
        assert "factor" not in kwargs["rope_parameters"]


class TestActivationDerivation:
    """The exporter must WRITE hidden_act; the config default would silently be silu."""

    @pytest.mark.parametrize(
        "flags,expected",
        [
            pytest.param({"swiglu": True, "sssglu": False}, "silu", id="swiglu"),
            pytest.param({"swiglu": False, "sssglu": True}, "sssglu", id="sssglu"),
        ],
    )
    def test_glu_flag_maps_to_hidden_act(self, flags, expected):
        kwargs, _, _ = derive(good_args(**flags))
        assert kwargs["hidden_act"] == expected, (
            "hidden_act must be derived from the fork's activation flag; falling through to the "
            "Apertus2Config default would export sssglu weights as a silu model, and nothing "
            "downstream — shapes, keys, counts — would notice"
        )
        assert derive_config(good_args(**flags)).hidden_act == expected

    @pytest.mark.parametrize("flag", ["ssglu", "rlglu", "lglu", "situ", "reglu", "quick_geglu"])
    def test_other_gated_activations_are_named_and_rejected(self, flag):
        # These are structurally identical GLUs with a different gate, so shapes and keys look
        # perfectly normal; only this check stands between them and a wrong export.
        with pytest.raises(ValueError, match=flag):
            derive(good_args(swiglu=False, **{flag: True}))


class TestSlidingWindowDerivation:
    """Megatron window_size is inclusive at both ends; HF sliding_window counts admitted keys."""

    def test_no_window_means_full_attention_everywhere(self):
        kwargs, _, _ = derive(good_args())
        assert kwargs["sliding_window"] is None
        assert kwargs["layer_types"] == ["full_attention"] * 3

    def test_window_size_gains_one_key(self):
        kwargs, _, _ = derive(good_args(window_size=(1024, 0)))
        assert kwargs["sliding_window"] == 1025, (
            "Megatron's (1024, 0) admits the query plus 1024 earlier tokens = 1025 keys; "
            "writing 1024 would silently drop one key per query"
        )

    def test_absent_skip_freq_slides_every_layer(self):
        # is_layer_window_attention returns True when window_attn_skip_freq is None: a window
        # without a schedule applies to the whole model, NOT to no layers.
        kwargs, _, _ = derive(good_args(window_size=(8, 0), window_attn_skip_freq=None))
        assert kwargs["layer_types"] == ["sliding_attention"] * 3

    def test_list_skip_freq_is_one_indexed_by_position(self):
        # 1 means "this layer slides" (arguments.py: "1 indicates SWA and 0 indicates full").
        kwargs, _, _ = derive(good_args(window_size=(8, 0), window_attn_skip_freq=[1, 1, 0]))
        assert kwargs["layer_types"] == ["sliding_attention", "sliding_attention", "full_attention"]

    def test_int_skip_freq_makes_every_nth_layer_full(self):
        # The fork slides where layer_number % freq != 0, with layer_number 1-indexed.
        kwargs, _, _ = derive(good_args(window_size=(8, 0), window_attn_skip_freq=3))
        assert kwargs["layer_types"] == ["sliding_attention", "sliding_attention", "full_attention"]

    def test_chonk_schedule_lands_on_the_expected_layers(self):
        kwargs, _, _ = derive(
            good_args(num_layers=42, moe_layer_freq=[0] * 3 + [1] * 39,
                      window_size=(1024, 0), window_attn_skip_freq=[1, 1, 1, 1, 1, 0] * 7)
        )
        full = [i for i, kind in enumerate(kwargs["layer_types"]) if kind == "full_attention"]
        assert full == [5, 11, 17, 23, 29, 35, 41]

    def test_skip_freq_must_describe_every_layer(self):
        with pytest.raises(ValueError, match="window_attn_skip_freq"):
            derive(good_args(window_size=(8, 0), window_attn_skip_freq=[1, 0]))

    def test_skip_freq_is_ignored_without_a_window(self):
        # is_layer_window_attention short-circuits on a falsy window_size, so a stale schedule
        # left in the args must not conjure sliding layers.
        kwargs, _, _ = derive(good_args(window_size=None, window_attn_skip_freq=[1, 0, 1]))
        assert kwargs["layer_types"] == ["full_attention"] * 3
        assert kwargs["sliding_window"] is None


class TestNoRopeDerivation:
    """Megatron marks the layers that SKIP RoPE; HF marks the layers that KEEP it."""

    def test_absent_no_rope_freq_rotates_every_layer(self):
        kwargs, _, _ = derive(good_args())
        assert kwargs["no_rope_layers"] == [1, 1, 1]

    def test_polarity_is_inverted(self):
        kwargs, _, _ = derive(good_args(no_rope_freq=[0, 0, 1]))
        assert kwargs["no_rope_layers"] == [1, 1, 0], (
            "Megatron's 1 means SKIP rope, HF's 1 means USE rope; an un-inverted list loads "
            "cleanly and generates fluent text while every layer rotates the wrong way"
        )

    def test_int_form_marks_every_nth_layer(self):
        # TransformerConfig expands N to ([0]*(N-1) + [1]) repeated: NoPE on every Nth layer.
        kwargs, _, _ = derive(good_args(no_rope_freq=3))
        assert kwargs["no_rope_layers"] == [1, 1, 0]

    def test_chonk_schedule_lands_on_the_expected_layers(self):
        kwargs, _, _ = derive(
            good_args(num_layers=42, moe_layer_freq=[0] * 3 + [1] * 39,
                      no_rope_freq=[0, 0, 0, 0, 0, 1] * 7)
        )
        nope = [i for i, rope in enumerate(kwargs["no_rope_layers"]) if not rope]
        assert nope == [5, 11, 17, 23, 29, 35, 41]

    def test_window_and_nope_schedules_stay_independent(self):
        # The two lists coincide in the chonk checkpoint, and upstream HF configs derive one from
        # the other. Keep them separate: the fork allows them to differ.
        kwargs, _, _ = derive(
            good_args(window_size=(8, 0), window_attn_skip_freq=[1, 0, 1], no_rope_freq=[0, 0, 1])
        )
        assert kwargs["layer_types"] == ["sliding_attention", "full_attention", "sliding_attention"]
        assert kwargs["no_rope_layers"] == [1, 1, 0]


class TestOffloadedExpertStorage:
    """Offloading is a training memory optimization; it only selects a mapping-table variant."""

    def test_offloading_is_accepted_and_reported(self):
        _, _, result = derive(good_args(moe_use_offloading_experts=True))
        assert result.offloaded_experts is True

    def test_default_is_the_grouped_layout(self):
        _, _, result = derive(good_args())
        assert result.offloaded_experts is False

    def test_offloading_never_reaches_the_hf_config(self):
        kwargs, _, _ = derive(good_args(moe_use_offloading_experts=True))
        assert not any("offload" in key for key in kwargs), (
            "expert offloading describes how training stored the weights, not what the exported "
            f"model is: {sorted(kwargs)}"
        )

    def test_inplace_fp8_without_extra_storage_is_fatal(self):
        # Without extra storage the fused master is overwritten with packed fp8 bytes, so the
        # bf16 weights the exporter expects are simply not there. The fork asserts the same.
        with pytest.raises(ValueError, match="fp8"):
            derive(good_args(moe_use_offloading_experts=True, moe_use_inplace_fp8_param=True,
                             moe_use_extra_fp8_param_storage=False))

    def test_inplace_fp8_with_extra_storage_is_accepted(self):
        _, _, result = derive(good_args(moe_use_offloading_experts=True,
                                        moe_use_inplace_fp8_param=True,
                                        moe_use_extra_fp8_param_storage=True))
        assert result.offloaded_experts is True
