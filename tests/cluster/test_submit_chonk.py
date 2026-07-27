"""Checks for the pinned Chonk 120B Stage 2 submitter."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMITTER = REPO_ROOT / "cluster" / "submit_chonk_120b_s1_iter1000.sh"


def make_environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    repo = tmp_path / "repo"
    (repo / "cluster").mkdir(parents=True)
    (repo / "cluster" / "stage2_export.sbatch").touch()

    checkpoint = tmp_path / "iter_0001000"
    checkpoint.mkdir()
    (checkpoint / "common.pt").touch()

    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")

    image = tmp_path / "image.sqsh"
    image.touch()
    hf_env = tmp_path / "hf.toml"
    hf_env.write_text(f'image = "{image}"\n')

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "sbatch.calls"
    sbatch = fake_bin / "sbatch"
    sbatch.write_text(
        "#!/bin/bash\n"
        f"printf '%s | VERIFY_LOAD=%s\\n' \"$*\" \"${{VERIFY_LOAD-}}\" >> {calls}\n"
        "printf 'Submitted batch job 123\\n'\n"
    )
    sbatch.chmod(0o755)

    output = tmp_path / "hf"
    env = os.environ.copy()
    env.update(
        PATH=f"{fake_bin}:{env['PATH']}",
        REPO=str(repo),
        TD_ITER_DIR=str(checkpoint),
        TOKENIZER_DIR=str(tokenizer),
        HF_OUT_DIR=str(output),
        HF_ENV=str(hf_env),
        LOG_DIR=str(tmp_path / "logs"),
    )
    return env, output, calls


def test_script_has_valid_bash_syntax_and_help():
    syntax = subprocess.run(
        ["bash", "-n", str(SUBMITTER)], check=False, capture_output=True, text=True
    )
    assert syntax.returncode == 0, syntax.stderr
    help_result = subprocess.run(
        [str(SUBMITTER), "--help"], check=False, capture_output=True, text=True
    )
    assert help_result.returncode == 0
    assert "submit_chonk_120b_s1_iter1000.sh" in help_result.stdout


def test_test_only_uses_overrides_and_does_not_submit(tmp_path: Path):
    env, output, calls = make_environment(tmp_path)
    result = subprocess.run(
        [str(SUBMITTER), "--test-only"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--test-only" in calls.read_text()
    assert "VERIFY_LOAD=0" in calls.read_text()
    assert str(output) in result.stdout


def test_nonempty_output_is_rejected_before_sbatch(tmp_path: Path):
    env, output, calls = make_environment(tmp_path)
    output.mkdir()
    (output / "existing").touch()

    result = subprocess.run(
        [str(SUBMITTER), "--test-only"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "non-empty" in result.stderr
    assert not calls.exists()
