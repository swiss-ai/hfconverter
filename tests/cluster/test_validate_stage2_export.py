"""Completion-certificate checks for the Stage 2 conversion wrapper."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from cluster.validate_stage2_export import validate


REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE2 = REPO_ROOT / "cluster" / "stage2_export.sbatch"


def completed_export(tmp_path: Path, *, verify_load: bool = True) -> tuple[Path, Path]:
    source = tmp_path / "torch_dist" / "iter_0000012"
    source.mkdir(parents=True)
    (source / "common.pt").touch()
    output = tmp_path / "hf"
    output.mkdir()
    info = {
        "source_checkpoint": str(source),
        "iteration": 12,
        "verified": True,
        "settings": {"verify": True, "verify_load": verify_load},
    }
    (output / "conversion_info.json").write_text(json.dumps(info))
    return source, output


def test_completed_verified_export_is_accepted(tmp_path: Path):
    source, output = completed_export(tmp_path)
    validate(output, source)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("source_checkpoint", "source_checkpoint"),
        ("iteration", "iteration"),
        ("verified", "verified"),
        ("verify", "settings.verify"),
        ("verify_load", "settings.verify_load"),
    ],
)
def test_invalid_certificate_field_is_rejected(tmp_path: Path, field: str, message: str):
    source, output = completed_export(tmp_path)
    info_path = output / "conversion_info.json"
    info = json.loads(info_path.read_text())
    if field == "source_checkpoint":
        info[field] = str(tmp_path / "wrong" / source.name)
    elif field == "iteration":
        info[field] = 13
    elif field == "verified":
        info[field] = False
    else:
        info["settings"][field] = False
    info_path.write_text(json.dumps(info))

    with pytest.raises(ValueError, match=message):
        validate(output, source)


def test_incomplete_marker_is_rejected(tmp_path: Path):
    source, output = completed_export(tmp_path)
    (output / ".export_incomplete").touch()

    with pytest.raises(ValueError, match="export_incomplete"):
        validate(output, source)


def test_generic_export_can_explicitly_allow_skipped_full_load(tmp_path: Path):
    source, output = completed_export(tmp_path, verify_load=False)
    validate(output, source, require_verify_load=False)


def test_stage2_wrapper_has_valid_syntax_and_runs_the_validator():
    result = subprocess.run(
        ["bash", "-n", str(STAGE2)], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    text = STAGE2.read_text()
    assert '[ ! -e "$HF_OUT_DIR/.export_incomplete" ]' in text
    assert "validate_stage2_export.py" in text
