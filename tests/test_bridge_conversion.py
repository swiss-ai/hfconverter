"""CPU-only checks for exporter inputs and reference modes."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import convert_checkpoint as exporter


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "iter_0001"
        self.reference = self.root / "reference"
        self.output = self.root / "output"
        self.checkpoint.mkdir()
        self.reference.mkdir()
        (self.checkpoint / "common.pt").touch()
        (self.reference / "config.json").write_text("{}", encoding="utf-8")

    def test_default_reference_mode_is_config_only(self) -> None:
        args = exporter._parser().parse_args(
            ["export", "checkpoint", "reference", "output"]
        )
        self.assertEqual(args.reference_mode, "config-only")
        self.assertEqual(args.max_shard_size_gib, 4)

    def test_streaming_shard_size(self) -> None:
        argv = ["export", "checkpoint", "reference", "output", "--max-shard-size-gib"]
        self.assertEqual(exporter._parser().parse_args([*argv, "16"]).max_shard_size_gib, 16)
        for value in ("0", "-1", "nan", "1.5"):
            with self.subTest(value=value), mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                exporter._parser().parse_args([*argv, value])

    def test_directional_cli_arguments(self) -> None:
        imported = exporter._parser().parse_args(
            [
                "import",
                "owner/model",
                "output",
                "--hf-revision",
                "pinned-revision",
                "--pipeline-layout",
                "Et*16|t*15|t*15|t*15",
                "--tp",
                "2",
                "--pp",
                "3",
                "--ep",
                "4",
                "--etp",
                "2",
                "--dtype",
                "bf16",
                "--trust-remote-code",
                "--bridge-root",
                "/bridge",
                "--megatron-root",
                "/megatron",
            ]
        )
        self.assertEqual(imported.direction, "import")
        self.assertEqual(imported.reference, "owner/model")
        self.assertEqual(imported.hf_revision, "pinned-revision")
        self.assertEqual(imported.pipeline_layout, "Et*16|t*15|t*15|t*15")
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "24"}):
            self.assertEqual(exporter._validate_topology(imported), 24)
        self.assertEqual(
            (imported.tp, imported.pp, imported.ep, imported.etp), (2, 3, 4, 2)
        )
        self.assertTrue(imported.trust_remote_code)
        exported = exporter._parser().parse_args(
            [
                "export",
                "checkpoint",
                "reference",
                "output",
                "--pipeline-layout",
                "Et*2|t*2",
            ]
        )
        self.assertEqual(exported.pipeline_layout, "Et*2|t*2")
        self.assertEqual(exported.model_override, [])
        imported_override = exporter._parser().parse_args(
            ["import", "owner/model", "output", "--model-override", "foo=10000"]
        )
        self.assertEqual(imported_override.model_override, [("foo", 10000)])

    def test_model_overrides_reject_unknown_and_topology_fields(self) -> None:
        provider = mock.Mock(spec_set=["moe_router_quantile_balancing_method"])
        with self.assertRaisesRegex(ValueError, "not supported by this provider"):
            exporter._apply_model_overrides(provider, {"not_a_provider_field": True})
        with self.assertRaisesRegex(ValueError, "topology/dtype"):
            exporter._apply_model_overrides(provider, {"pipeline_model_parallel_size": 4})

    def test_import_applies_explicit_layout_before_provider_finalize(self) -> None:
        calls: list[str] = []
        provider = mock.Mock()
        provider.finalize.side_effect = lambda: calls.append(
            (provider.pipeline_model_parallel_layout, provider.moe_router_quantile_balancing_method)
        )
        bridge = mock.Mock()
        bridge.to_megatron_provider.return_value = provider
        bridge._model_bridge = mock.Mock()
        bridge._model_bridge.USE_MODEL_CONFIG_FOR_CONVERSION = False

        backend = mock.Mock()
        backend.import_checkpoint = mock.Mock()
        backend._ensure_distributed_initialized = mock.Mock()
        backend._prepare_distributed_output = mock.Mock()
        backend._configure_model_provider = mock.Mock()
        backend._uses_model_builder = mock.Mock(return_value=False)
        backend._hf_tokenizer_kwargs = mock.Mock(return_value={})
        bridge_module = mock.Mock()
        bridge_module.AutoBridge.from_hf_pretrained.return_value = bridge
        utils_module = mock.Mock()
        utils_module.is_safe_repo.return_value = False
        torch_module = mock.Mock()
        torch_module.bfloat16 = object()
        modules = {
            "gpu_backend": backend,
            "megatron": mock.Mock(),
            "megatron.bridge": bridge_module,
            "megatron.bridge.models": mock.Mock(),
            "megatron.bridge.models.hf_pretrained": mock.Mock(),
            "megatron.bridge.models.hf_pretrained.utils": utils_module,
            "torch": torch_module,
        }
        args = exporter._parser().parse_args(
            [
                "import",
                str(self.reference),
                str(self.output),
                "--pipeline-layout",
                "Et*16|t*15|t*15|t*15",
                "--model-override",
                'moe_router_quantile_balancing_method="marin_histogram"',
            ]
        )
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(exporter, "_check_runtime_roots", return_value=(Path("/bridge"), Path("/megatron"))),
        ):
            exporter._run_import(args)

        self.assertEqual(
            calls,
            [("Et*16|t*15|t*15|t*15", "marin_histogram")],
        )
        backend.import_checkpoint.assert_not_called()
        bridge.save_megatron_model.assert_called_once()

    def test_config_only_reference_rejects_weights_and_symlinks(self) -> None:
        exporter._validate_reference(self.reference, "config-only")
        weight = self.reference / "weights.safetensors"
        weight.touch()
        with self.assertRaisesRegex(ValueError, "weight or symlink"):
            exporter._validate_reference(self.reference, "config-only")
        weight.unlink()
        (self.reference / "config-link.json").symlink_to(self.reference / "config.json")
        with self.assertRaisesRegex(ValueError, "weight or symlink"):
            exporter._validate_reference(self.reference, "config-only")

    def test_weighted_reference_requires_safe_existing_shards(self) -> None:
        index = self.reference / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"w": "model.safetensors"}}))
        with self.assertRaisesRegex(ValueError, "incomplete or unsafe indexed shards"):
            exporter._validate_reference(self.reference, "weighted")
        (self.reference / "model.safetensors").touch()
        exporter._validate_reference(self.reference, "weighted")

        (self.root / "outside.safetensors").touch()
        index.write_text(json.dumps({"weight_map": {"w": "../outside.safetensors"}}))
        with self.assertRaisesRegex(ValueError, "incomplete or unsafe indexed shards"):
            exporter._validate_reference(self.reference, "weighted")

    def test_checkpoint_and_output_paths(self) -> None:
        self.assertEqual(
            exporter._validate_paths(self.checkpoint, self.reference, self.output),
            "native",
        )
        (self.checkpoint / "run_config.yaml").touch()
        self.assertEqual(
            exporter._validate_paths(self.checkpoint, self.reference, self.output),
            "bridge",
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            exporter._validate_paths(
                self.checkpoint, self.reference, self.reference / "output"
            )

    def test_nonempty_output_is_not_overwritten(self) -> None:
        self.output.mkdir()
        (self.output / "existing").touch()
        with self.assertRaises(FileExistsError):
            exporter._validate_paths(self.checkpoint, self.reference, self.output)

    def test_launch_world_size_matches_topology(self) -> None:
        args = exporter._parser().parse_args(
            [
                "export",
                "checkpoint",
                "reference",
                "output",
                "--tp",
                "4",
                "--pp",
                "2",
                "--ep",
                "4",
            ]
        )
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "8"}):
            self.assertEqual(exporter._validate_topology(args), 8)
        with (
            mock.patch.dict(os.environ, {"WORLD_SIZE": "4"}),
            self.assertRaisesRegex(ValueError, "does not match"),
        ):
            exporter._validate_topology(args)


if __name__ == "__main__":
    unittest.main()
