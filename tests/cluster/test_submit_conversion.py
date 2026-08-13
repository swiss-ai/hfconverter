"""Login-node tests for the generic two-job conversion submitter.

No SLURM job or model is run here.  ``sbatch`` is replaced by a small recorder.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from gate_helpers import write_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMITTER = REPO_ROOT / "cluster" / "submit_conversion.sh"


def make_layout(tmp_path: Path) -> tuple[dict[str, str], Path, Path, Path]:
    source = tmp_path / "source-checkpoint"
    source.mkdir()
    write_checkpoint(
        source,
        {
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "expert_model_parallel_size": 4,
            "expert_tensor_parallel_size": 1,
            "context_parallel_size": 1,
            "bf16": True,
            "fp16": False,
            "moe_router_load_balancing_type": "seq_aux_loss",
            "moe_aux_loss_coeff": 1e-3,
            "init_method_std": 0.0360844,
            "layernorm_epsilon": 1e-5,
            "sandwich_norm": True,
            "moe_router_dtype": "fp32",
            "transformer_impl": "transformer_engine",
            "window_size": (1024, 0),
            "window_attn_skip_freq": [1, 1, 1, 0],
            "no_rope_freq": [0, 0, 0, 1],
        },
        iteration=730,
    )

    fork = tmp_path / "Megatron-LM-MoE"
    tokenizer = fork / "_research" / "data" / "tokenizer"
    tokenizer.mkdir(parents=True)
    (fork / "pretrain_gpt.py").touch()
    (tokenizer / "tokenizer.json").write_text("{}")

    stage1_image = tmp_path / "stage1.sqsh"
    stage1_image.touch()
    stage1_env = tmp_path / "stage1.toml"
    stage1_env.write_text(f'image = "{stage1_image}"\n')
    image = tmp_path / "hf.sqsh"
    image.touch()
    hf_env = tmp_path / "hf.toml"
    hf_env.write_text(f'image = "{image}"\n')

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "sbatch.calls"
    count = tmp_path / "sbatch.count"
    sbatch = fake_bin / "sbatch"
    sbatch.write_text(
        "#!/bin/bash\n"
        f"printf '%s | SRC_EP=%s SANDWICH_NORM=%s MOE_AUX_LOSS_COEFF=%s "
        f"STAGE1_ENV=%s TRUST_LEGACY_CHECKPOINT=%s HF_ENV=%s TD_ITER_DIR=%s HF_OUT_DIR=%s VERIFY_LOAD=%s "
        f"SKIP_INSPECT=<%s> FORCE_NO=<%s> FORCE_YES=<%s> "
        f"WINDOW_SIZE=<%s> WINDOW_ATTN_SKIP_FREQ=<%s> NO_ROPE_FREQ=<%s>\\n' "
        f"\"$*\" \"${{SRC_EP-}}\" \"${{SANDWICH_NORM-}}\" "
        f"\"${{MOE_AUX_LOSS_COEFF-}}\" \"${{STAGE1_ENV-}}\" \"${{TRUST_LEGACY_CHECKPOINT-}}\" \"${{HF_ENV-}}\" \"${{TD_ITER_DIR-}}\" "
        f"\"${{HF_OUT_DIR-}}\" \"${{VERIFY_LOAD-}}\" \"${{SKIP_INSPECT-}}\" "
        f"\"${{TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD-__UNSET__}}\" \"${{TORCH_FORCE_WEIGHTS_ONLY_LOAD-__UNSET__}}\" "
        f"\"${{WINDOW_SIZE-__UNSET__}}\" \"${{WINDOW_ATTN_SKIP_FREQ-__UNSET__}}\" "
        f"\"${{NO_ROPE_FREQ-__UNSET__}}\" >> {calls}\n"
        f"n=0; [ ! -f {count} ] || n=$(<{count}); n=$((n + 1)); "
        f"printf '%s' \"$n\" > {count}\n"
        "printf '4100%s\\n' \"$n\"\n"
    )
    sbatch.chmod(0o755)

    env = os.environ.copy()
    env.update(
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        REPO=str(REPO_ROOT),
        MEGATRON_PATH=str(fork),
        TOKENIZER_DIR=str(tokenizer),
        STAGE1_ENV=str(stage1_env),
        TRUST_LEGACY_CHECKPOINT="1",
        HF_ENV=str(hf_env),
        LOG_DIR=str(tmp_path / "logs"),
        SRC_TP="1",
        SRC_PP="1",
        SRC_EP="4",
        SRC_ETP="1",
        SRC_CP="1",
        PRECISION="bf16",
        ROUTING_TYPE="seq_aux_loss",
        MOE_AUX_LOSS_COEFF="1e-3",
        INIT_METHOD_STD="0.0360844",
        NORM_EPSILON="1e-5",
        SANDWICH_NORM="1",
        WINDOW_SIZE="(1024,0)",
        WINDOW_ATTN_SKIP_FREQ="([1,1,1,0])",
        NO_ROPE_FREQ="([0,0,0,1])",
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
        TORCH_FORCE_WEIGHTS_ONLY_LOAD="1",
    )
    return env, source, tmp_path / "converted", calls


def run_submitter(
    env: dict[str, str], source: Path, output: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SUBMITTER), str(source), str(output)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_script_has_valid_bash_syntax():
    result = subprocess.run(
        ["bash", "-n", str(SUBMITTER)], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_submits_only_stage1_then_dependent_stage2(tmp_path: Path):
    env, source, output, calls = make_layout(tmp_path)

    result = run_submitter(env, source, output)

    assert result.returncode == 0, result.stderr
    submitted = calls.read_text().splitlines()
    assert len(submitted) == 2
    assert "stage1_torchdist.sbatch" in submitted[0]
    assert f" {source} {output}" in submitted[0]
    assert "SRC_EP=4" in submitted[0]
    assert "MOE_AUX_LOSS_COEFF=1e-3" in submitted[0]
    assert "SANDWICH_NORM=1" in submitted[0]
    assert f"STAGE1_ENV={env['STAGE1_ENV']}" in submitted[0]
    assert "TRUST_LEGACY_CHECKPOINT=1" in submitted[0]
    assert f"HF_ENV={env['HF_ENV']}" in submitted[1]
    assert env["STAGE1_ENV"] != env["HF_ENV"]

    assert "stage2_export.sbatch" in submitted[1]
    assert "--dependency=afterok:41001" in submitted[1]
    assert f"TD_ITER_DIR={output}/torch_dist/iter_0000730" in submitted[1]
    assert f"HF_OUT_DIR={output}/hf" in submitted[1]
    assert "VERIFY_LOAD=1" in submitted[1]
    assert "SKIP_INSPECT=<>" in submitted[1]
    assert "FORCE_NO=<__UNSET__>" in submitted[1]
    assert "FORCE_YES=<__UNSET__>" in submitted[1]
    assert "Stage 1 job: 41001" in result.stdout
    assert "Stage 2 job: 41002" in result.stdout


def test_missing_profile_value_fails_before_any_submission(tmp_path: Path):
    env, source, output, calls = make_layout(tmp_path)
    env.pop("SANDWICH_NORM")

    result = run_submitter(env, source, output)

    assert result.returncode != 0
    assert "SANDWICH_NORM is required" in result.stderr
    assert not calls.exists()


def test_sliding_window_and_nope_patterns_reach_stage1(tmp_path: Path):
    env, source, output, calls = make_layout(tmp_path)

    result = run_submitter(env, source, output)

    assert result.returncode == 0, result.stderr
    stage1 = calls.read_text().splitlines()[0]
    assert "WINDOW_SIZE=<(1024,0)>" in stage1
    assert "WINDOW_ATTN_SKIP_FREQ=<([1,1,1,0])>" in stage1
    assert "NO_ROPE_FREQ=<([0,0,0,1])>" in stage1


@pytest.mark.parametrize(
    "name", ["WINDOW_SIZE", "WINDOW_ATTN_SKIP_FREQ", "NO_ROPE_FREQ"]
)
def test_undeclared_attention_pattern_fails_before_any_submission(
    tmp_path: Path, name: str
):
    # Unset must not be readable as "the source has none": Megatron restores none of these and
    # they leave no tensor evidence, so the wrong answer exports silently.
    env, source, output, calls = make_layout(tmp_path)
    env.pop(name)

    result = run_submitter(env, source, output)

    assert result.returncode != 0
    assert f"{name} must be set" in result.stderr
    assert not calls.exists()


def test_empty_attention_patterns_are_accepted_for_a_source_without_them(tmp_path: Path):
    env, source, output, calls = make_layout(tmp_path)
    write_checkpoint(
        source,
        {
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "expert_model_parallel_size": 4,
            "expert_tensor_parallel_size": 1,
            "context_parallel_size": 1,
            "bf16": True,
            "fp16": False,
            "moe_router_load_balancing_type": "seq_aux_loss",
            "moe_aux_loss_coeff": 1e-3,
            "init_method_std": 0.0360844,
            "layernorm_epsilon": 1e-5,
            "sandwich_norm": True,
            "moe_router_dtype": "fp32",
            "transformer_impl": "transformer_engine",
        },
        iteration=731,
    )
    (source / "latest_checkpointed_iteration.txt").write_text("731\n")
    env.update(WINDOW_SIZE="", WINDOW_ATTN_SKIP_FREQ="", NO_ROPE_FREQ="")

    result = run_submitter(env, source, output)

    assert result.returncode == 0, result.stderr
    stage1 = calls.read_text().splitlines()[0]
    assert "WINDOW_SIZE=<>" in stage1
    assert "NO_ROPE_FREQ=<>" in stage1


def test_source_context_parallelism_above_one_is_converted(tmp_path: Path):
    # p2 of the 9.3b ctxext row trains at CP=4.  CP shards the sequence, not the parameters,
    # so the iteration directory holds the same mp_rank_<tp>_<ep> shards as a CP=1 run.
    env, source, output, calls = make_layout(tmp_path)
    write_checkpoint(
        source,
        {
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "expert_model_parallel_size": 4,
            "expert_tensor_parallel_size": 1,
            "context_parallel_size": 4,
            "bf16": True,
            "fp16": False,
            "moe_router_load_balancing_type": "seq_aux_loss",
            "moe_aux_loss_coeff": 1e-3,
            "init_method_std": 0.0360844,
            "layernorm_epsilon": 1e-5,
            "sandwich_norm": True,
            "moe_router_dtype": "fp32",
            "transformer_impl": "transformer_engine",
            "window_size": (1024, 0),
            "window_attn_skip_freq": [1, 1, 1, 0],
            "no_rope_freq": [0, 0, 0, 1],
        },
        iteration=732,
    )
    (source / "latest_checkpointed_iteration.txt").write_text("732\n")
    env["SRC_CP"] = "4"

    result = run_submitter(env, source, output)

    assert result.returncode == 0, result.stderr
    assert len(calls.read_text().splitlines()) == 2


def test_missing_trusted_checkpoint_assertion_fails_before_submission(tmp_path: Path):
    env, source, output, calls = make_layout(tmp_path)
    env.pop("TRUST_LEGACY_CHECKPOINT")

    result = run_submitter(env, source, output)

    assert result.returncode != 0
    assert "set TRUST_LEGACY_CHECKPOINT=1" in result.stderr
    assert not calls.exists()
    assert not output.exists()


def test_missing_stage1_environment_fails_before_any_submission(tmp_path: Path):
    env, source, output, calls = make_layout(tmp_path)
    missing_environment = tmp_path / "missing-stage1.toml"
    env["STAGE1_ENV"] = str(missing_environment)

    result = run_submitter(env, source, output)

    assert result.returncode != 0
    assert f"Stage 1 environment file is not readable: {missing_environment}" in result.stderr
    assert not calls.exists()
    assert not output.exists()


def test_relative_stage1_environment_fails_before_any_submission(tmp_path: Path):
    env, source, output, calls = make_layout(tmp_path)
    env["STAGE1_ENV"] = "relative-stage1.toml"

    result = run_submitter(env, source, output)

    assert result.returncode != 0
    assert "STAGE1_ENV must be an absolute EDF path" in result.stderr
    assert not calls.exists()
    assert not output.exists()


def test_stage1_environment_is_canonicalized_before_submission(tmp_path: Path):
    env, source, output, calls = make_layout(tmp_path)
    real_environment = Path(env["STAGE1_ENV"])
    environment_link = tmp_path / "stage1-link.toml"
    environment_link.symlink_to(real_environment)
    env["STAGE1_ENV"] = str(environment_link)

    result = run_submitter(env, source, output)

    assert result.returncode == 0, result.stderr
    stage1 = calls.read_text().splitlines()[0]
    assert f"STAGE1_ENV={real_environment.resolve()}" in stage1
    assert str(environment_link) not in stage1


def test_nonempty_output_fails_before_any_submission(tmp_path: Path):
    env, source, output, calls = make_layout(tmp_path)
    output.mkdir()
    (output / "old-result").touch()

    result = run_submitter(env, source, output)

    assert result.returncode != 0
    assert "output root is not empty" in result.stderr
    assert not calls.exists()


def test_output_inside_source_is_rejected(tmp_path: Path):
    env, source, _, calls = make_layout(tmp_path)

    result = run_submitter(env, source, source / "converted")

    assert result.returncode != 0
    assert "must not be inside the source checkpoint" in result.stderr
    assert not calls.exists()


def test_dotdot_cannot_bypass_source_containment(tmp_path: Path):
    env, source, _, calls = make_layout(tmp_path)
    disguised = source / ".." / source.name / "converted"

    result = run_submitter(env, source, disguised)

    assert result.returncode != 0
    assert "must not be inside the source checkpoint" in result.stderr
    assert not calls.exists()


def test_symlink_cannot_bypass_source_containment(tmp_path: Path):
    env, source, _, calls = make_layout(tmp_path)
    link = tmp_path / "checkpoint-link"
    link.symlink_to(source, target_is_directory=True)

    result = run_submitter(env, source, link / "converted")

    assert result.returncode != 0
    assert "must not be inside the source checkpoint" in result.stderr
    assert not calls.exists()


def test_profile_mismatch_fails_before_any_submission(tmp_path: Path):
    env, source, output, calls = make_layout(tmp_path)
    env["MOE_AUX_LOSS_COEFF"] = "1e-4"

    result = run_submitter(env, source, output)

    assert result.returncode != 0
    assert "moe_aux_loss_coeff" in result.stderr
    assert not calls.exists()


def test_unsupported_source_topology_fails_before_any_submission(tmp_path: Path):
    env, source, output, calls = make_layout(tmp_path)
    env["SRC_TP"] = "2"

    result = run_submitter(env, source, output)

    assert result.returncode != 0
    assert "needs 8 ranks" in result.stderr
    assert not calls.exists()


def test_no_reservation_is_requested_by_default(tmp_path: Path):
    env, source, output, calls = make_layout(tmp_path)
    for stale in ("RESERVATION", "STAGE2_PARTITION"):
        env.pop(stale, None)

    result = run_submitter(env, source, output)

    assert result.returncode == 0, result.stderr
    assert "--reservation" not in calls.read_text()
    assert "--partition" not in calls.read_text()


def test_reservation_reaches_both_jobs(tmp_path: Path):
    env, source, output, calls = make_layout(tmp_path)
    env.update(RESERVATION="SD-69241-apertus-1-5-0", STAGE2_PARTITION="normal")

    result = run_submitter(env, source, output)

    assert result.returncode == 0, result.stderr
    stage1, stage2 = calls.read_text().splitlines()
    assert "--reservation=SD-69241-apertus-1-5-0" in stage1
    assert "--reservation=SD-69241-apertus-1-5-0" in stage2
    # Only Stage 2 needs the partition override; Stage 1's #SBATCH header is already `normal`.
    assert "--partition=normal" in stage2


def test_reservation_without_a_partition_is_refused(tmp_path: Path):
    # A reservation is scoped to one partition, and Stage 2 defaults to `debug`.  Accepting the
    # reservation alone would queue a job the reservation cannot admit.
    env, source, output, calls = make_layout(tmp_path)
    env["RESERVATION"] = "SD-69241-apertus-1-5-0"
    env.pop("STAGE2_PARTITION", None)

    result = run_submitter(env, source, output)

    assert result.returncode != 0
    assert "also needs STAGE2_PARTITION" in result.stderr
    assert not calls.exists()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RESERVATION", "res --uid=0"),
        ("STAGE2_PARTITION", "normal;whoami"),
    ],
)
def test_reservation_names_are_restricted(tmp_path: Path, name: str, value: str):
    env, source, output, calls = make_layout(tmp_path)
    env.update(RESERVATION="SD-69241-apertus-1-5-0", STAGE2_PARTITION="normal")
    env[name] = value

    result = run_submitter(env, source, output)

    assert result.returncode != 0
    assert name in result.stderr
    assert not calls.exists()


def test_stale_ambient_stage2_values_are_overridden(tmp_path: Path):
    env, source, output, calls = make_layout(tmp_path)
    env.update(
        TD_ITER_DIR="/stale/torch-dist",
        HF_OUT_DIR="/stale/hf",
        VERIFY_LOAD="0",
        SKIP_INSPECT="1",
    )

    result = run_submitter(env, source, output)

    assert result.returncode == 0, result.stderr
    stage2 = calls.read_text().splitlines()[1]
    assert f"TD_ITER_DIR={output}/torch_dist/iter_0000730" in stage2
    assert f"HF_OUT_DIR={output}/hf" in stage2
    assert "VERIFY_LOAD=1" in stage2
    assert "SKIP_INSPECT=<>" in stage2
    assert "/stale/" not in stage2
