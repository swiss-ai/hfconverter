"""Bitwise tiny-model round trip through a synthetic Megatron checkpoint and the exporter.

Per flag combo (megatron_mock.ROUNDTRIP_COMBOS): (a) the exported dir loads with zero
missing/unexpected/mismatched keys, (b) EVERY runtime state-dict tensor is bitwise equal
(torch.equal + dtype) to the source model's, (c) config.json carries the combo's flags, the
exact tiny-geometry multipliers and geometry, (d) exporting into a non-empty output dir
fails. One combo additionally runs through the real CLI (`python -m exporter.export`) in a
subprocess, asserting exit 0 and the expected output file set.
"""

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("megatron.core")

import torch  # noqa: E402

import exporter.export  # noqa: E402,F401  (module-level on purpose: collection surfaces the dependency)
import megatron_mock  # noqa: E402
from modeling_apertus2 import Apertus2ForCausalLM  # noqa: E402

COMBO_PARAMS = [
    pytest.param(sandwich, latent, qb, expert_bias, id=combo_id)
    for combo_id, sandwich, latent, qb, expert_bias in megatron_mock.ROUNDTRIP_COMBOS
]


def _build_and_save(tmp_path, sandwich, latent, qb, expert_bias, seed=0, **model_overrides):
    model = megatron_mock.build_tiny_model(
        sandwich,
        latent,
        qb,
        seed=seed,
        zero_expert_bias=not expert_bias,
        **model_overrides,
    )
    tensors = megatron_mock.to_megatron_tensors(
        model, model.config, expert_bias_present=expert_bias
    )
    args = megatron_mock.make_args_namespace(model.config, expert_bias_present=expert_bias)
    checkpoint_dir = tmp_path / "iter_0000100"
    megatron_mock.save_synthetic_checkpoint(tensors, args, checkpoint_dir, iteration=100)
    return model, checkpoint_dir


def _assert_weight_files(output_dir):
    single = os.path.join(str(output_dir), "model.safetensors")
    index = os.path.join(str(output_dir), "model.safetensors.index.json")
    shards = [
        name
        for name in os.listdir(str(output_dir))
        if name.startswith("model-") and name.endswith(".safetensors")
    ]
    assert os.path.isfile(single) or (shards and os.path.isfile(index)), (
        f"no model.safetensors and no sharded set + index in {sorted(os.listdir(str(output_dir)))}"
    )


class TestRoundtrip:
    @pytest.mark.parametrize("sandwich,latent,qb,expert_bias", COMBO_PARAMS)
    def test_bitwise_roundtrip(
        self, dist_env, export_api, tmp_path, sandwich, latent, qb, expert_bias
    ):
        model, checkpoint_dir = _build_and_save(tmp_path, sandwich, latent, qb, expert_bias)
        output_dir = tmp_path / "hf_export"

        export_api(checkpoint_dir, output_dir)

        # (a) loads with zero key issues
        reloaded, info = Apertus2ForCausalLM.from_pretrained(
            str(output_dir), dtype=torch.float32, output_loading_info=True
        )
        reloaded.eval()
        assert not info["missing_keys"], info["missing_keys"]
        assert not info["unexpected_keys"], info["unexpected_keys"]
        assert not info["mismatched_keys"], info["mismatched_keys"]
        assert not info["error_msgs"], info["error_msgs"]

        # (b) bitwise equality of every runtime tensor, dtype included
        source_state = model.state_dict()
        restored_state = reloaded.state_dict()
        assert set(source_state) == set(restored_state), (
            sorted(set(source_state) ^ set(restored_state))
        )
        for key, source in source_state.items():
            restored = restored_state[key]
            assert restored.dtype == source.dtype, f"{key}: {restored.dtype} != {source.dtype}"
            assert torch.equal(restored, source), f"{key}: values differ across the roundtrip"

        # (c) config.json: flags, multipliers, geometry
        with open(output_dir / "config.json") as f:
            saved = json.load(f)
        assert saved["model_type"] == "apertus2"
        assert saved["architectures"] == ["Apertus2ForCausalLM"]
        assert saved.get("auto_map") == {
            "AutoConfig": "configuration_apertus2.Apertus2Config",
            # AutoModel too: a consumer that wants the BACKBONE rather than the LM head must be
            # able to call AutoModel.from_pretrained on the exported dir.
            "AutoModel": "modeling_apertus2.Apertus2Model",
            "AutoModelForCausalLM": "modeling_apertus2.Apertus2ForCausalLM",
        }
        assert saved["sandwich_norm"] is sandwich
        assert "keel" not in saved
        assert "keel_alpha" not in saved
        assert saved["moe_latent_size"] == latent
        assert saved["use_quantile_balancing"] is qb
        assert saved["embedding_multiplier"] == megatron_mock.TINY_EMBEDDING_MULTIPLIER
        assert saved["residual_multiplier"] == megatron_mock.TINY_RESIDUAL_MULTIPLIER
        assert saved["vocab_size"] == 128
        assert saved["hidden_size"] == 32
        assert saved["num_hidden_layers"] == 3
        assert saved["num_attention_heads"] == 4
        assert saved["num_key_value_heads"] == 2
        assert saved["head_dim"] == 8
        assert saved["intermediate_size"] == 64
        assert saved["moe_intermediate_size"] == 16
        assert saved["n_routed_experts"] == megatron_mock.TINY_N_EXPERTS
        assert saved["num_experts_per_tok"] == 2
        assert saved["n_shared_experts"] == 1
        assert saved["first_k_dense_replace"] == 1
        assert saved["routed_scaling_factor"] == 2.5
        assert saved["rms_norm_eps"] == 1e-5
        assert saved["max_position_embeddings"] == 64
        assert saved["tie_word_embeddings"] is False
        rope = saved.get("rope_parameters") or {}
        theta = rope.get("rope_theta", saved.get("rope_theta"))
        assert float(theta) == 500000.0
        assert (saved.get("dtype") or saved.get("torch_dtype")) == "float32"

        # The exported directory is self-contained.
        assert os.path.isfile(output_dir / "configuration_apertus2.py")
        assert os.path.isfile(output_dir / "modeling_apertus2.py")
        assert os.path.isfile(output_dir / "conversion_info.json")
        _assert_weight_files(output_dir)

        if not expert_bias:
            # zero-synthesis path: buffers exist, are fp32 zeros, and the synthesis is
            # The choice is recorded in conversion_info.json.
            for layer in reloaded.model.layers[1:]:
                bias = layer.mlp.gate.e_score_correction_bias
                assert bias.dtype == torch.float32
                assert torch.equal(bias, torch.zeros_like(bias))
            with open(output_dir / "conversion_info.json") as f:
                assert "expert_bias" in f.read(), (
                    "synthesized expert_bias keys should be recorded in conversion_info.json"
                )

    def test_interleaved_dense_moe_schedule_round_trips_bitwise(
        self, dist_env, export_api, tmp_path
    ):
        pattern = [0, 1, 0]
        model, checkpoint_dir = _build_and_save(
            tmp_path,
            sandwich=True,
            latent=24,
            qb=True,
            expert_bias=True,
            moe_layer_freq=pattern,
        )
        output_dir = tmp_path / "hf_interleaved"
        export_api(checkpoint_dir, output_dir)

        reloaded, info = Apertus2ForCausalLM.from_pretrained(
            str(output_dir), dtype=torch.float32, output_loading_info=True
        )
        assert not any(info.values()), info
        assert reloaded.config.moe_layer_freq == pattern
        assert reloaded.config.first_k_dense_replace == 1
        assert [
            reloaded.config.is_moe_layer(index) for index in range(len(pattern))
        ] == [False, True, False]

        source_state = model.state_dict()
        restored_state = reloaded.state_dict()
        assert set(source_state) == set(restored_state)
        for key, source in source_state.items():
            restored = restored_state[key]
            assert restored.dtype == source.dtype, key
            assert torch.equal(restored, source), key

        with open(output_dir / "config.json") as handle:
            saved = json.load(handle)
        assert saved["moe_layer_freq"] == pattern

    def test_attention_output_gate_round_trips_bitwise(self, dist_env, export_api, tmp_path):
        """Gated attention: the fused QKV carries an extra per-query-head gate slice that must
        come back bitwise as self_attn.g_proj on every layer."""
        model, checkpoint_dir = _build_and_save(
            tmp_path,
            sandwich=True,
            latent=24,
            qb=True,
            expert_bias=True,
            attention_output_gate=True,
        )
        output_dir = tmp_path / "hf_gated"
        export_api(checkpoint_dir, output_dir)

        reloaded, info = Apertus2ForCausalLM.from_pretrained(
            str(output_dir), dtype=torch.float32, output_loading_info=True
        )
        assert not any(info.values()), info
        assert reloaded.config.attention_output_gate is True

        source_state = model.state_dict()
        restored_state = reloaded.state_dict()
        assert set(source_state) == set(restored_state), (
            sorted(set(source_state) ^ set(restored_state))
        )
        gate_keys = [key for key in restored_state if key.endswith("self_attn.g_proj.weight")]
        assert len(gate_keys) == model.config.num_hidden_layers, gate_keys
        for key, source in source_state.items():
            restored = restored_state[key]
            assert restored.dtype == source.dtype, key
            assert torch.equal(restored, source), key

        with open(output_dir / "config.json") as handle:
            saved = json.load(handle)
        assert saved["attention_output_gate"] is True

    def test_output_dir_must_not_exist_or_be_empty(self, dist_env, export_api, tmp_path):
        _, checkpoint_dir = _build_and_save(tmp_path, False, None, False, True)
        output_dir = tmp_path / "occupied"
        output_dir.mkdir()
        sentinel = output_dir / "precious.txt"
        sentinel.write_text("do not clobber")
        with pytest.raises((SystemExit, ValueError, FileExistsError, OSError)):
            export_api(checkpoint_dir, output_dir)
        assert sentinel.read_text() == "do not clobber", "existing files must not be clobbered"

    def test_cli_subprocess_full_run(self, dist_env, tmp_path):
        """One combo (sandwich+latent+QB, the final-model shape) through the real CLI."""
        model, checkpoint_dir = _build_and_save(tmp_path, True, 24, True, True)
        output_dir = tmp_path / "hf_cli"
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "exporter.export",
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--output-dir",
                str(output_dir),
            ],
            cwd=megatron_mock.REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=570,
        )
        assert process.returncode == 0, (
            f"CLI failed rc={process.returncode}\nstdout:\n{process.stdout}\n"
            f"stderr:\n{process.stderr}"
        )
        assert os.path.isfile(output_dir / "config.json")
        assert os.path.isfile(output_dir / "conversion_info.json")
        assert os.path.isfile(output_dir / "configuration_apertus2.py")
        assert os.path.isfile(output_dir / "modeling_apertus2.py")
        _assert_weight_files(output_dir)
        # and the CLI output actually loads back bitwise
        reloaded = Apertus2ForCausalLM.from_pretrained(str(output_dir), dtype=torch.float32)
        source_state = model.state_dict()
        restored_state = reloaded.state_dict()
        assert set(source_state) == set(restored_state)
        for key, source in source_state.items():
            assert torch.equal(restored_state[key], source), key
