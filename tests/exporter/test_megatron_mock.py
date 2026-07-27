"""Self-checks for the independent synthetic Megatron checkpoint generator."""

import glob
import os

import pytest

pytest.importorskip("megatron.core")

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

import megatron_mock  # noqa: E402
from megatron_mock import (  # noqa: E402
    ROUNDTRIP_COMBOS,
    TINY_HEAD_DIM,
    TINY_HEADS,
    TINY_HIDDEN,
    TINY_INTERMEDIATE,
    TINY_KV_HEADS,
    TINY_LATENT,
    TINY_LAYERS,
    TINY_MOE_INTERMEDIATE,
    TINY_N_EXPERTS,
    TINY_VOCAB,
)

COMBO_PARAMS = [
    pytest.param(sandwich, latent, qb, expert_bias, keel, id=combo_id)
    for combo_id, sandwich, latent, qb, expert_bias, keel in ROUNDTRIP_COMBOS
]


# ---------------------------------------------------------------------------
# Expected fork key set, generated independently of the mock's key construction.
# (kept independent of megatron_mock's own key construction)
# ---------------------------------------------------------------------------


def expected_fork_keys(
    num_layers=TINY_LAYERS,
    first_k_dense=1,
    sandwich=False,
    latent=False,
    qb=False,
    expert_bias=True,
    keel=False,
):
    keys = {
        "embedding.word_embeddings.weight",
        "output_layer.weight",
        "decoder.final_layernorm.weight",
    }
    for L in range(num_layers):
        p = f"decoder.layers.{L}."
        keys |= {
            p + "self_attention.linear_qkv.layer_norm_weight",
            p + "self_attention.linear_qkv.weight",
            p + "self_attention.q_layernorm.weight",
            p + "self_attention.k_layernorm.weight",
            p + "self_attention.linear_proj.weight",
        }
        if sandwich:
            keys |= {p + "post_self_attn_layernorm.weight", p + "post_mlp_layernorm.weight"}
        if keel:
            # post_self_attn only for L>=1 (IdentityOp on layer 0); post_mlp on all layers
            if L > 0:
                keys.add(p + "post_self_attn_layernorm.weight")
            keys.add(p + "post_mlp_layernorm.weight")
        if L < first_k_dense:
            keys |= {
                p + "mlp.linear_fc1.layer_norm_weight",
                p + "mlp.linear_fc1.weight",
                p + "mlp.linear_fc2.weight",
            }
        else:
            keys |= {
                p + "pre_mlp_layernorm.weight",
                p + "mlp.router.weight",
                p + "mlp.experts.experts.linear_fc1.weight",
                p + "mlp.experts.experts.linear_fc2.weight",
                p + "mlp.shared_experts.linear_fc1.weight",
                p + "mlp.shared_experts.linear_fc2.weight",
            }
            if expert_bias:
                keys.add(p + "mlp.router.expert_bias")
            if qb:
                keys.add(p + "mlp.router.qb_beta")
            if latent:
                keys |= {p + "mlp.fc1_latent_proj.weight", p + "mlp.fc2_latent_proj.weight"}
    return keys


def _mock_pair(sandwich, latent, qb, expert_bias, keel=False, seed=0):
    model = megatron_mock.build_tiny_model(
        sandwich, latent, qb, seed=seed, zero_expert_bias=not expert_bias, keel=keel
    )
    tensors = megatron_mock.to_megatron_tensors(
        model, model.config, expert_bias_present=expert_bias
    )
    return model, tensors


# ---------------------------------------------------------------------------
# key set / shapes / dtypes
# ---------------------------------------------------------------------------


class TestKeySet:
    @pytest.mark.parametrize("sandwich,latent,qb,expert_bias,keel", COMBO_PARAMS)
    def test_key_set_matches_contract(self, sandwich, latent, qb, expert_bias, keel):
        _, tensors = _mock_pair(sandwich, latent, qb, expert_bias, keel=keel)
        expected = expected_fork_keys(
            sandwich=sandwich, latent=latent is not None, qb=qb, expert_bias=expert_bias, keel=keel
        )
        assert set(tensors) == expected, (
            f"missing={sorted(expected - set(tensors))} "
            f"extra={sorted(set(tensors) - expected)}"
        )

    def test_shapes_and_dtypes_latent_combo(self):
        _, tensors = _mock_pair(True, TINY_LATENT, True, True)
        H, D, F, E = TINY_HIDDEN, TINY_HEAD_DIM, TINY_MOE_INTERMEDIATE, TINY_N_EXPERTS
        qkv_rows = (TINY_HEADS + 2 * TINY_KV_HEADS) * D  # 64
        assert tensors["embedding.word_embeddings.weight"].shape == (TINY_VOCAB, H)
        assert tensors["output_layer.weight"].shape == (TINY_VOCAB, H)
        assert tensors["decoder.layers.0.self_attention.linear_qkv.weight"].shape == (qkv_rows, H)
        assert tensors["decoder.layers.0.self_attention.linear_proj.weight"].shape == (
            H,
            TINY_HEADS * D,
        )
        assert tensors["decoder.layers.0.self_attention.q_layernorm.weight"].shape == (D,)
        assert tensors["decoder.layers.0.mlp.linear_fc1.weight"].shape == (
            2 * TINY_INTERMEDIATE,
            H,
        )
        assert tensors["decoder.layers.0.mlp.linear_fc2.weight"].shape == (H, TINY_INTERMEDIATE)
        # MoE layer with latent: in_e == Lat
        assert tensors["decoder.layers.1.mlp.experts.experts.linear_fc1.weight"].shape == (
            E,
            2 * F,
            TINY_LATENT,
        )
        assert tensors["decoder.layers.1.mlp.experts.experts.linear_fc2.weight"].shape == (
            E,
            TINY_LATENT,
            F,
        )
        assert tensors["decoder.layers.1.mlp.fc1_latent_proj.weight"].shape == (TINY_LATENT, H)
        assert tensors["decoder.layers.1.mlp.fc2_latent_proj.weight"].shape == (H, TINY_LATENT)
        assert tensors["decoder.layers.1.mlp.router.weight"].shape == (E, H)
        assert tensors["decoder.layers.1.mlp.shared_experts.linear_fc1.weight"].shape == (
            2 * F,
            H,
        )
        assert tensors["decoder.layers.1.mlp.shared_experts.linear_fc2.weight"].shape == (H, F)
        # fp32 buffers stay fp32 and non-trivial (randomized by the model factory)
        for L in (1, 2):
            bias = tensors[f"decoder.layers.{L}.mlp.router.expert_bias"]
            beta = tensors[f"decoder.layers.{L}.mlp.router.qb_beta"]
            assert bias.dtype == torch.float32 and bias.shape == (E,)
            assert beta.dtype == torch.float32 and beta.shape == (E,)
            assert bias.abs().sum() > 0, "factory must randomize expert_bias"
            assert beta.abs().sum() > 0, "factory must randomize qb_beta"

    def test_experts_in_dim_is_hidden_without_latent(self):
        _, tensors = _mock_pair(False, None, False, True)
        assert tensors["decoder.layers.1.mlp.experts.experts.linear_fc1.weight"].shape == (
            TINY_N_EXPERTS,
            2 * TINY_MOE_INTERMEDIATE,
            TINY_HIDDEN,
        )
        assert tensors["decoder.layers.1.mlp.experts.experts.linear_fc2.weight"].shape == (
            TINY_N_EXPERTS,
            TINY_HIDDEN,
            TINY_MOE_INTERMEDIATE,
        )


# ---------------------------------------------------------------------------
# merge-layout hand checks
# ---------------------------------------------------------------------------


class TestMergeLayout:
    def test_qkv_merge_per_group_row_blocks(self):
        """Fused row order for heads=4, kv=2, D=8 must be q0 q1 k0 v0 | q2 q3 k1 v1."""
        model, tensors = _mock_pair(False, None, False, True)
        state = model.state_dict()
        D = TINY_HEAD_DIM
        for L in range(TINY_LAYERS):
            fused = tensors[f"decoder.layers.{L}.self_attention.linear_qkv.weight"]
            q = state[f"model.layers.{L}.self_attn.q_proj.weight"]
            k = state[f"model.layers.{L}.self_attn.k_proj.weight"]
            v = state[f"model.layers.{L}.self_attn.v_proj.weight"]
            assert torch.equal(fused[0 * D : 2 * D], q[0 * D : 2 * D])  # q0 q1
            assert torch.equal(fused[2 * D : 3 * D], k[0 * D : 1 * D])  # k0
            assert torch.equal(fused[3 * D : 4 * D], v[0 * D : 1 * D])  # v0
            assert torch.equal(fused[4 * D : 6 * D], q[2 * D : 4 * D])  # q2 q3
            assert torch.equal(fused[6 * D : 7 * D], k[1 * D : 2 * D])  # k1
            assert torch.equal(fused[7 * D : 8 * D], v[1 * D : 2 * D])  # v1

    def test_gate_up_and_expert_stack_match_save_pretrained_disk_layout(self, tmp_path):
        """Anchor the mock to the REAL on-disk per-expert layout: megatron fc1[e] must equal
        cat([gate_proj.{e}, up_proj.{e}], dim=0) of the tensors save_pretrained writes, and
        fc2[e] must equal down_proj.{e} — for dense mlp and shared expert likewise."""
        model, tensors = _mock_pair(False, TINY_LATENT, False, True)
        model.save_pretrained(str(tmp_path))
        disk = {}
        for path in sorted(glob.glob(os.path.join(str(tmp_path), "*.safetensors"))):
            with safe_open(path, framework="pt") as f:
                for key in f.keys():
                    disk[key] = f.get_tensor(key)

        # dense layer 0
        assert torch.equal(
            tensors["decoder.layers.0.mlp.linear_fc1.weight"],
            torch.cat(
                [
                    disk["model.layers.0.mlp.gate_proj.weight"],
                    disk["model.layers.0.mlp.up_proj.weight"],
                ],
                dim=0,
            ),
        )
        assert torch.equal(
            tensors["decoder.layers.0.mlp.linear_fc2.weight"],
            disk["model.layers.0.mlp.down_proj.weight"],
        )
        # routed experts, stacked on axis 0
        for L in (1, 2):
            fc1 = tensors[f"decoder.layers.{L}.mlp.experts.experts.linear_fc1.weight"]
            fc2 = tensors[f"decoder.layers.{L}.mlp.experts.experts.linear_fc2.weight"]
            for e in range(TINY_N_EXPERTS):
                prefix = f"model.layers.{L}.mlp.experts.{e}."
                assert torch.equal(
                    fc1[e],
                    torch.cat(
                        [disk[prefix + "gate_proj.weight"], disk[prefix + "up_proj.weight"]],
                        dim=0,
                    ),
                ), f"layer {L} expert {e} fc1"
                assert torch.equal(fc2[e], disk[prefix + "down_proj.weight"]), (
                    f"layer {L} expert {e} fc2"
                )
            # shared expert
            shared = f"model.layers.{L}.mlp.shared_experts."
            assert torch.equal(
                tensors[f"decoder.layers.{L}.mlp.shared_experts.linear_fc1.weight"],
                torch.cat(
                    [disk[shared + "gate_proj.weight"], disk[shared + "up_proj.weight"]], dim=0
                ),
            )
            assert torch.equal(
                tensors[f"decoder.layers.{L}.mlp.shared_experts.linear_fc2.weight"],
                disk[shared + "down_proj.weight"],
            )

    def test_copies_are_decoupled_from_the_model(self):
        model, tensors = _mock_pair(False, None, False, True)
        with torch.no_grad():
            model.model.embed_tokens.weight.add_(1.0)
        assert not torch.equal(
            tensors["embedding.word_embeddings.weight"], model.model.embed_tokens.weight
        )


# ---------------------------------------------------------------------------
# Args namespace consumed by config derivation.
# ---------------------------------------------------------------------------

# Every attribute config derivation reads, using fork spellings.
REQUIRED_ARGS_ATTRS = [
    "padded_vocab_size",
    "hidden_size",
    "num_layers",
    "num_attention_heads",
    "group_query_attention",
    "num_query_groups",
    "kv_channels",
    "ffn_hidden_size",
    "max_position_embeddings",
    "seq_length",
    "num_experts",
    "moe_router_topk",
    "moe_ffn_hidden_size",
    "moe_shared_expert_intermediate_size",
    "moe_layer_freq",
    "moe_latent_size",
    "moe_router_load_balancing_type",
    "moe_router_topk_scaling_factor",
    "moe_router_score_function",
    "moe_router_enable_expert_bias",
    "moe_router_num_groups",
    "moe_router_group_topk",
    "moe_router_topk_limited_devices",
    "moe_shared_expert_gate",
    "moe_use_offloading_experts",
    "normalization",
    "layernorm_epsilon",
    "swiglu",
    "qk_layernorm",
    "sandwich_norm",
    "add_bias_linear",
    "add_qkv_bias",
    "position_embedding_type",
    "rotary_base",
    "rotary_percent",
    "rotary_interleaved",
    "use_rope_scaling",
    "rope_scaling_factor",
    "scale_embeddings_by_sqrt_hidden",
    "residual_output_scaling",
    "multi_latent_attention",
    "mtp_num_layers",
    "keel",
    "pnglu",
    "use_mup",
    "attention_output_gate",
    "softmax_type",
    "untie_embeddings_and_output_weights",
    "attention_dropout",
    "init_method_std",
    "bf16",
]


class TestArgsNamespace:
    def test_every_contract_attr_present(self):
        config = megatron_mock.tiny_export_config(True, TINY_LATENT, True)
        args = megatron_mock.make_args_namespace(config, expert_bias_present=True)
        missing = [a for a in REQUIRED_ARGS_ATTRS if not hasattr(args, a)]
        assert not missing, f"make_args_namespace missing contract attrs: {missing}"

    def test_fork_spellings_and_values(self):
        config = megatron_mock.tiny_export_config(False, None, False)
        args = megatron_mock.make_args_namespace(config, expert_bias_present=True)
        assert args.padded_vocab_size == TINY_VOCAB
        assert args.hidden_size == TINY_HIDDEN
        assert args.num_layers == TINY_LAYERS
        assert args.num_attention_heads == TINY_HEADS
        assert args.group_query_attention is True
        assert args.num_query_groups == TINY_KV_HEADS
        assert args.kv_channels == TINY_HEAD_DIM
        assert args.ffn_hidden_size == TINY_INTERMEDIATE
        assert args.num_experts == TINY_N_EXPERTS  # fork name, not num_moe_experts
        assert args.moe_router_topk == 2
        assert args.moe_ffn_hidden_size == TINY_MOE_INTERMEDIATE
        assert args.moe_shared_expert_intermediate_size == TINY_MOE_INTERMEDIATE  # n_shared 1
        assert args.moe_layer_freq == [0, 1, 1]  # [0]*k + [1]*(n-k), k == first_k_dense == 1
        assert args.moe_latent_size is None
        assert args.moe_router_load_balancing_type == "aux_loss"  # scalar string when no QB
        assert args.moe_router_topk_scaling_factor == 2.5
        assert args.layernorm_epsilon == 1e-5
        assert isinstance(args.rotary_base, int) and args.rotary_base == 500000
        assert args.rotary_percent == 1.0
        assert args.untie_embeddings_and_output_weights is True
        assert args.swiglu is True
        assert args.add_bias_linear is False and args.add_qkv_bias is False
        assert args.qk_layernorm is True
        assert args.normalization == "RMSNorm"
        assert args.position_embedding_type == "rope"
        assert args.moe_router_score_function == "sigmoid"
        assert args.softmax_type == "vanilla"
        assert args.mtp_num_layers is None
        assert args.moe_router_enable_expert_bias is True
        assert args.moe_router_num_groups is None
        assert args.moe_router_group_topk is None
        assert args.moe_router_topk_limited_devices is None
        # tiny config carries the derivable multipliers -> both flags on
        assert args.scale_embeddings_by_sqrt_hidden is True
        assert args.residual_output_scaling is True
        assert args.attention_dropout == 0.0
        assert args.init_method_std == 0.02
        assert args.max_position_embeddings == 64 and args.seq_length == 64
        assert args.sandwich_norm is False

    def test_qb_flag_switches_balancing_to_list(self):
        config = megatron_mock.tiny_export_config(False, None, True)
        args = megatron_mock.make_args_namespace(config, expert_bias_present=True)
        assert args.moe_router_load_balancing_type == ["seq_aux_loss", "quantile_balancing"]

    def test_expert_bias_and_overrides(self):
        config = megatron_mock.tiny_export_config(False, None, False)
        args = megatron_mock.make_args_namespace(
            config, expert_bias_present=False, normalization="LayerNorm", bf16=True
        )
        assert args.moe_router_enable_expert_bias is False
        assert args.normalization == "LayerNorm"  # overrides win
        assert args.bf16 is True

    def test_inexpressible_multiplier_is_rejected(self):
        # The production defaults do not match this tiny geometry.
        config = megatron_mock.tiny_export_config(
            False, None, False, embedding_multiplier=27.712812921102035
        )
        with pytest.raises(AssertionError, match="embedding_multiplier"):
            megatron_mock.make_args_namespace(config, expert_bias_present=True)


# ---------------------------------------------------------------------------
# dist_checkpointing save/load round trip (probe recipe)
# ---------------------------------------------------------------------------


class TestSyntheticCheckpoint:
    def test_save_load_round_trip_bitwise(self, dist_env, tmp_path):
        import megatron.core.dist_checkpointing as dist_checkpointing

        model, tensors = _mock_pair(True, TINY_LATENT, True, True)
        args = megatron_mock.make_args_namespace(model.config, expert_bias_present=True)
        ckpt_dir = tmp_path / "ckpt"
        megatron_mock.save_synthetic_checkpoint(tensors, args, ckpt_dir, iteration=100)

        assert dist_checkpointing.check_is_distributed_checkpoint(str(ckpt_dir))
        common = dist_checkpointing.load_common_state_dict(str(ckpt_dir))
        assert vars(common["args"]) == vars(args)
        assert common["iteration"] == 100
        assert common["checkpoint_version"] == 3.0

        metadata = dist_checkpointing.load_tensors_metadata(str(ckpt_dir))
        assert set(metadata) == set(tensors)

        loaded = megatron_mock.load_plain_tensor_dict(ckpt_dir)
        assert set(loaded) == set(tensors)
        for key, reference in tensors.items():
            assert loaded[key].dtype == reference.dtype, key
            assert torch.equal(loaded[key], reference), key

    def test_extra_tensors_are_injected(self, dist_env, tmp_path):
        _, tensors = _mock_pair(False, None, False, True)
        args = megatron_mock.make_args_namespace(
            megatron_mock.tiny_export_config(False, None, False), expert_bias_present=True
        )
        extras = {
            "optimizer.state.exp_avg.decoder.layers.1.mlp.router.weight": torch.randn(8, 32),
            "decoder.layers.1.mlp.mystery_gadget.weight": torch.randn(4, 4),
        }
        ckpt_dir = tmp_path / "ckpt"
        megatron_mock.save_synthetic_checkpoint(tensors, args, ckpt_dir, extra_tensors=extras)
        loaded = megatron_mock.load_plain_tensor_dict(ckpt_dir)
        assert set(loaded) == set(tensors) | set(extras)
        for key, reference in extras.items():
            assert torch.equal(loaded[key], reference), key

    def test_expert_bias_absent_combo_omits_key_and_keeps_zeros(self):
        model, tensors = _mock_pair(False, None, False, expert_bias=False)
        assert not any(key.endswith("router.expert_bias") for key in tensors)
        assert not any(key.endswith("router.qb_beta") for key in tensors)
        # the comparison target for the exporter's zero-synthesis path must BE zeros
        for layer in model.model.layers[1:]:
            assert torch.equal(
                layer.mlp.gate.e_score_correction_bias,
                torch.zeros_like(layer.mlp.gate.e_score_correction_bias),
            )
