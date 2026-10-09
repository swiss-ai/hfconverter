"""CPU-only checks for exporter inputs and reference modes."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import convert_megatron_to_hf as exporter


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
        args = exporter._parser().parse_args(["checkpoint", "reference", "output"])
        self.assertEqual(args.reference_mode, "config-only")

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
            ["checkpoint", "reference", "output", "--tp", "4", "--pp", "2", "--ep", "4"]
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
