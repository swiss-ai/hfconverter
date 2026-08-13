"""Checks for the generic Stage 2 submitter."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMITTER = REPO_ROOT / "cluster" / "convert.sh"


def run_submitter(
    tmp_path: Path, args: list[str], env_extra: dict[str, str] | None = None
) -> tuple[subprocess.CompletedProcess[str], Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    calls = tmp_path / "sbatch.calls"
    sbatch = fake_bin / "sbatch"
    sbatch.write_text(
        "#!/bin/bash\n"
        f"printf '%s\\n' \"$*\" >> {calls}\n"
        "printf 'Submitted batch job 123\\n'\n"
    )
    sbatch.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env.update(env_extra or {})
    result = subprocess.run(
        [str(SUBMITTER), *args], env=env, check=False, capture_output=True, text=True
    )
    return result, calls


def test_script_has_valid_bash_syntax_and_help():
    syntax = subprocess.run(
        ["bash", "-n", str(SUBMITTER)], check=False, capture_output=True, text=True
    )
    assert syntax.returncode == 0, syntax.stderr
    help_result = subprocess.run(
        [str(SUBMITTER), "--help"], check=False, capture_output=True, text=True
    )
    assert help_result.returncode == 0
    assert "convert.sh" in help_result.stdout


def test_missing_arguments_do_not_submit(tmp_path: Path):
    result, calls = run_submitter(tmp_path, ["/only/one/path"])
    assert result.returncode != 0
    assert not calls.exists()


def test_submits_stage2_with_paths_and_extra_sbatch_args(tmp_path: Path):
    result, calls = run_submitter(
        tmp_path,
        ["/ckpt/iter_0000001", "/out/hf", "--reservation=res1", "--partition=normal"],
        env_extra={"VERIFY_LOAD": "1"},
    )

    assert result.returncode == 0, result.stderr
    recorded = calls.read_text()
    assert "--reservation=res1 --partition=normal" in recorded
    assert "TD_ITER_DIR=/ckpt/iter_0000001" in recorded
    assert "HF_OUT_DIR=/out/hf" in recorded
    assert "VERIFY_LOAD=1" in recorded
    assert recorded.strip().endswith("cluster/stage2_export.sbatch")


def test_verify_load_defaults_off(tmp_path: Path):
    result, calls = run_submitter(tmp_path, ["/ckpt/iter_0000001", "/out/hf"])
    assert result.returncode == 0, result.stderr
    assert "VERIFY_LOAD=0" in calls.read_text()
