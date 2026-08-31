"""KDA layers map by pure renames; softmax layers keep their split transforms.

The KDA key layout is pinned independently in megatron_mock.kda_attention_key_triples (the
fork pre-splits the fused in_proj/conv1d at save time, so every KDA tensor is a COPY row).
These tests require the production plan to consume and produce exactly those names, push
tensors through convert() demanding bit-identity, and run one full in-process export whose
config.json must carry the KDA fields. Model-level round trips (from_pretrained) need
flash-linear-attention: modeling_apertus2's KDA module runs on its kernels and refuses to
construct without the package, which the last test pins for both environments.
"""

import json

import pytest

pytest.importorskip("megatron.core")

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

import megatron_mock  # noqa: E402
from exporter import config_from_args, mapping  # noqa: E402


def _plan_and_fixture(expert_bias=True, seed=0):
    tensors, args, expected_hf, donor = megatron_mock.build_tiny_kda_checkpoint(
        seed=seed, expert_bias_present=expert_bias
    )
    derived = config_from_args.derive_config(args)
    plan = mapping.build_plan(
        derived.kwargs, derived.expert_bias_present, derived.offloaded_experts
    )
    return tensors, expected_hf, donor, plan


class TestKdaPlan:
    def test_plan_consumes_exactly_the_checkpoint_keys(self):
        tensors, _, _, plan = _plan_and_fixture()
        mapping.check_consumed(set(tensors), plan)  # checks both directions

    def test_kda_and_softmax_layers_never_mix_key_namespaces(self):
        _, _, _, plan = _plan_and_fixture()
        consumed = mapping.expected_consumed(plan)
        for layer, is_kda in enumerate(megatron_mock.TINY_KDA_PATTERN):
            attention_keys = {
                key for key in consumed
                if key.startswith(f"decoder.layers.{layer}.self_attention.")
            }
            softmax_markers = ("linear_qkv", "q_layernorm", "k_layernorm", "linear_proj")
            kda_markers = (".A_log", ".dt_bias", "conv1d", "in_proj", "out_norm")
            if is_kda:
                assert not any(m in key for key in attention_keys for m in softmax_markers), (
                    f"softmax keys claimed on KDA layer {layer}: {sorted(attention_keys)}"
                )
                assert f"decoder.layers.{layer}.self_attention.A_log" in attention_keys
            else:
                assert any("linear_qkv.weight" in key for key in attention_keys)
                assert not any(m in key for key in attention_keys for m in kda_markers), (
                    f"KDA keys claimed on softmax layer {layer}: {sorted(attention_keys)}"
                )

    def test_kda_rows_are_all_unpinned_pure_copies(self):
        tensors, expected_hf, _, plan = _plan_and_fixture()
        kda_source_keys = {
            triple[0]
            for layer, is_kda in enumerate(megatron_mock.TINY_KDA_PATTERN) if is_kda
            for triple in megatron_mock.kda_attention_key_triples(layer)
        }
        kda_rows = [row for row in plan.rows if row.megatron_key in kda_source_keys]
        assert len(kda_rows) == len(kda_source_keys) == 17 * sum(
            megatron_mock.TINY_KDA_PATTERN
        )
        assert all(row.transform == mapping.COPY for row in kda_rows)
        # Uniform bf16 checkpoints: nothing on a KDA layer is dtype-pinned (qb_beta/expert_bias
        # remain the only fp32-pinned rows in the table).
        assert all(row.dtype is None for row in kda_rows)
        assert {key for row in kda_rows for key in row.hf_keys} == set(expected_hf)

    def test_output_gate_bias_row_follows_the_config_field(self):
        # derive_config pins linear_attn_output_gate_bias=True for this fork (gate_out_proj
        # always trains a bias); an explicit False must drop exactly the bias row on every
        # KDA layer so a Kimi-Linear-style checkpoint without the key still bijects.
        _, args, _, _ = megatron_mock.build_tiny_kda_checkpoint()
        derived = config_from_args.derive_config(args)
        assert derived.kwargs["linear_attn_output_gate_bias"] is True

        plan = mapping.build_plan(derived.kwargs, derived.expert_bias_present)
        no_bias_kwargs = {**derived.kwargs, "linear_attn_output_gate_bias": False}
        no_bias_plan = mapping.build_plan(no_bias_kwargs, derived.expert_bias_present)

        dropped = mapping.expected_consumed(plan) - mapping.expected_consumed(no_bias_plan)
        assert dropped == {
            f"decoder.layers.{layer}.self_attention.gate_out_proj.bias"
            for layer, is_kda in enumerate(megatron_mock.TINY_KDA_PATTERN) if is_kda
        }
        assert not any(
            key.endswith("g_b_proj.bias")
            for row in no_bias_plan.rows for key in row.hf_keys
        )

    def test_incomplete_kda_geometry_is_rejected(self):
        _, args, _, _ = megatron_mock.build_tiny_kda_checkpoint()
        derived = config_from_args.derive_config(args)
        broken = {**derived.kwargs, "linear_num_key_heads": None}
        with pytest.raises(ValueError, match="linear_num_key_heads"):
            mapping.build_plan(broken, derived.expert_bias_present)

    def test_layer_types_must_describe_every_layer(self):
        _, args, _, _ = megatron_mock.build_tiny_kda_checkpoint()
        derived = config_from_args.derive_config(args)
        broken = {**derived.kwargs, "layer_types": ["linear_attention", "full_attention"]}
        with pytest.raises(ValueError, match="layer_types"):
            mapping.build_plan(broken, derived.expert_bias_present)


class TestKdaConvert:
    def test_kda_tensors_convert_bit_identically(self):
        tensors, expected_hf, _, plan = _plan_and_fixture()
        produced = mapping.convert(plan, tensors)
        mapping.check_produced(set(produced), plan)
        for hf_key, reference in expected_hf.items():
            assert torch.equal(produced[hf_key], reference), hf_key
            assert produced[hf_key].dtype == reference.dtype, hf_key

    def test_softmax_layer_attention_still_round_trips(self):
        # Layer 2 stays a gated softmax layer; its exported attention tensors must equal the
        # donor model's own, proving SPLIT_QKV_GATE is untouched by the KDA branch.
        tensors, _, donor, plan = _plan_and_fixture()
        produced = mapping.convert(plan, tensors)
        state = donor.state_dict()
        for name in ("q_proj", "g_proj", "k_proj", "v_proj", "o_proj"):
            key = f"model.layers.2.self_attn.{name}.weight"
            assert torch.equal(produced[key], state[key]), key


class TestKdaExportEndToEnd:
    def test_export_writes_kda_weights_and_config(self, dist_env, export_api, tmp_path):
        tensors, args, expected_hf, _ = megatron_mock.build_tiny_kda_checkpoint()
        checkpoint_dir = tmp_path / "ckpt"
        megatron_mock.save_synthetic_checkpoint(tensors, args, checkpoint_dir)
        output_dir = tmp_path / "out"
        export_api(checkpoint_dir, output_dir)

        with open(output_dir / "config.json") as handle:
            config = json.load(handle)
        assert config["layer_types"] == [
            "linear_attention", "linear_attention", "full_attention"
        ]
        assert config["linear_num_key_heads"] == megatron_mock.TINY_KDA_HEADS
        assert config["linear_num_value_heads"] == megatron_mock.TINY_KDA_HEADS
        assert config["linear_key_head_dim"] == megatron_mock.TINY_KDA_HEAD_DIM
        assert config["linear_value_head_dim"] == megatron_mock.TINY_KDA_HEAD_DIM
        assert config["linear_conv_kernel_dim"] == megatron_mock.TINY_KDA_CONV_KERNEL
        assert config["gate_lower_bound"] == -5.0

        written = load_file(str(output_dir / "model.safetensors"))
        for hf_key, reference in expected_hf.items():
            assert hf_key in written, hf_key
            assert torch.equal(written[hf_key], reference), hf_key

    def test_verify_load_requires_flash_linear_attention(
        self, dist_env, export_api, export_expect_failure, tmp_path
    ):
        # --verify-load builds the HF model, whose KDA layers run on flash-linear-attention:
        # with the package installed the exported dir must load back cleanly, and without it
        # the load must fail loudly instead of silently building softmax attention around KDA
        # weights. The branch condition is the environment itself, so neither environment
        # skips.
        from modeling_apertus2 import chunk_kda

        tensors, args, _, _ = megatron_mock.build_tiny_kda_checkpoint()
        checkpoint_dir = tmp_path / "ckpt"
        megatron_mock.save_synthetic_checkpoint(tensors, args, checkpoint_dir)
        if chunk_kda is None:
            blob = export_expect_failure(
                checkpoint_dir, tmp_path / "out", "--verify-load"
            )
            assert "flash-linear-attention" in blob
        else:
            export_api(checkpoint_dir, tmp_path / "out", "--verify-load")
