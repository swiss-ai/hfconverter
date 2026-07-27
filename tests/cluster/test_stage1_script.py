"""Static and early-exit checks for the small Stage 1 SLURM wrapper.

These tests do not launch SLURM or load a checkpoint. They protect the interface that can be
checked on a login node; model-equivalence still requires a real run with the selected
Megatron checkout.
"""

from pathlib import Path
import os
import subprocess
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "cluster" / "stage1_torchdist.sbatch"


def run_script(*args: str, **env_overrides: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(env_overrides)
    if "TRUST_LEGACY_CHECKPOINT" not in env_overrides:
        env["TRUST_LEGACY_CHECKPOINT"] = "1"
    # Fake-srun tests still exercise the EDF preflight. Build a temporary valid
    # local EDF unless the test deliberately supplied another one.
    with tempfile.TemporaryDirectory() as environment_dir:
        if "STAGE1_ENV" not in env_overrides:
            image = Path(environment_dir) / "stage1.sqsh"
            image.touch()
            environment = Path(environment_dir) / "stage1.toml"
            environment.write_text(f'image = "{image}"\n')
            env["STAGE1_ENV"] = str(environment)
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )


def make_source(tmp_path: Path, iteration: int = 7) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    # Real Megatron trackers do not necessarily end with a newline.
    (source / "latest_checkpointed_iteration.txt").write_text(str(iteration))
    return source


def test_script_has_valid_bash_syntax():
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_script_uses_the_small_native_megatron_path():
    text = SCRIPT.read_text()

    assert "SRC_CKPT=${1:" in text
    assert "DST_ROOT=${2:" in text
    assert '"${MEGATRON_PATH}/pretrain_gpt.py"' in text
    assert "--ckpt-convert-format torch_dist" in text
    assert "--use-checkpoint-args" in text
    assert "--use-mp-args-from-checkpoint-args" in text
    assert '--init-method-std "${INIT_METHOD_STD}"' in text
    assert '--norm-epsilon "${NORM_EPSILON}"' in text
    assert '"${sandwich_args[@]}"' in text
    assert "--moe-router-dtype fp32" in text
    assert '--moe-router-load-balancing-type "${routing_type_args[@]}"' in text
    assert '--moe-aux-loss-coeff "${aux_loss_coeff_args[@]}"' in text
    assert "convert_entry.py" not in text
    assert "ckpt_argcopy" not in text
    assert "/iopsstor/scratch/cscs/${USER}/Megatron-LM-MoE" in text
    assert "/users/anowak" not in text
    assert "STAGE1_ENV=${STAGE1_ENV:-" in text
    assert '--environment="${STAGE1_ENV}"' in text
    assert "--environment=apertus2-alps4-temp" not in text
    assert "export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1" in text
    assert "unset TORCH_FORCE_WEIGHTS_ONLY_LOAD" in text


def test_script_fills_native_router_restore_gaps_with_readable_defaults():
    text = SCRIPT.read_text()

    assert "ROUTING_TYPE=${ROUTING_TYPE:-seq_aux_loss}" in text
    assert "MOE_AUX_LOSS_COEFF=${MOE_AUX_LOSS_COEFF:-1e-4}" in text
    assert "INIT_METHOD_STD=${INIT_METHOD_STD:-0.0360844}" in text
    assert "NORM_EPSILON=${NORM_EPSILON:-1e-5}" in text
    assert "SANDWICH_NORM=${SANDWICH_NORM:-0}" in text
    assert 'read -r -a routing_type_args <<< "${ROUTING_TYPE}"' in text
    assert 'read -r -a aux_loss_coeff_args <<< "${MOE_AUX_LOSS_COEFF}"' in text


def test_missing_positional_paths_prints_usage():
    result = run_script()

    assert result.returncode != 0
    assert "usage: sbatch cluster/stage1_torchdist.sbatch SRC_CKPT DST_ROOT" in result.stderr


def test_more_than_four_source_ranks_is_rejected(tmp_path):
    result = run_script(
        str(tmp_path / "source"),
        str(tmp_path / "destination"),
        SRC_TP="1",
        SRC_PP="1",
        SRC_EP="5",
    )

    assert result.returncode != 0
    assert "requires 5 ranks, but this job has 4 GPUs" in result.stderr


def test_topology_values_must_be_positive_integers(tmp_path):
    result = run_script(
        str(tmp_path / "source"),
        str(tmp_path / "destination"),
        SRC_TP="0",
    )

    assert result.returncode != 0
    assert "SRC_TP must be a positive integer" in result.stderr


def test_expert_tensor_parallel_size_must_divide_tensor_parallel_size(tmp_path):
    result = run_script(
        str(tmp_path / "source"),
        str(tmp_path / "destination"),
        SRC_TP="1",
        SRC_ETP="2",
    )

    assert result.returncode != 0
    assert "SRC_ETP must divide SRC_TP" in result.stderr


def test_unknown_precision_is_rejected_before_launch(tmp_path):
    source = tmp_path / "source"
    source.mkdir()

    result = run_script(
        str(source),
        str(tmp_path / "destination"),
        PRECISION="fp32",
    )

    assert result.returncode != 0
    assert "PRECISION must be bf16 or fp16" in result.stderr


def test_legacy_pickle_requires_an_explicit_trust_assertion(tmp_path):
    source = make_source(tmp_path)
    destination = tmp_path / "destination"

    result = run_script(
        str(source),
        str(destination),
        TRUST_LEGACY_CHECKPOINT="0",
    )

    assert result.returncode != 0
    assert "Set TRUST_LEGACY_CHECKPOINT=1" in result.stderr
    assert not destination.exists()


def test_missing_stage1_environment_is_rejected_before_launch(tmp_path):
    source = make_source(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    srun_was_called = tmp_path / "srun-was-called"
    fake_srun = fake_bin / "srun"
    fake_srun.write_text(f"#!/bin/bash\ntouch {srun_was_called}\n")
    fake_srun.chmod(0o755)
    missing_environment = tmp_path / "missing-stage1.toml"
    destination = tmp_path / "destination"

    result = run_script(
        str(source),
        str(destination),
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        STAGE1_ENV=str(missing_environment),
    )

    assert result.returncode != 0
    assert f"Stage 1 environment file is not readable: {missing_environment}" in result.stderr
    assert not srun_was_called.exists()
    assert not destination.exists()


def test_relative_stage1_environment_is_rejected_before_launch(tmp_path):
    source = make_source(tmp_path)
    destination = tmp_path / "destination"

    result = run_script(
        str(source),
        str(destination),
        STAGE1_ENV="relative-stage1.toml",
    )

    assert result.returncode != 0
    assert "STAGE1_ENV must be an absolute EDF path" in result.stderr
    assert not destination.exists()


def test_missing_stage1_image_is_rejected_before_launch(tmp_path):
    source = make_source(tmp_path)
    missing_image = tmp_path / "missing-stage1.sqsh"
    environment = tmp_path / "stage1.toml"
    environment.write_text(f'image = "{missing_image}"\n')
    destination = tmp_path / "destination"

    result = run_script(
        str(source),
        str(destination),
        STAGE1_ENV=str(environment),
    )

    assert result.returncode != 0
    assert f"Stage 1 image is not readable: {missing_image}" in result.stderr
    assert not destination.exists()


def test_zero_srun_without_an_output_is_rejected(tmp_path):
    source = make_source(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_srun = fake_bin / "srun"
    fake_srun.write_text("#!/bin/bash\nexit 0\n")
    fake_srun.chmod(0o755)

    result = run_script(
        str(source),
        str(tmp_path / "destination"),
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        SCRATCH=str(tmp_path),
    )

    assert result.returncode != 0
    assert "without a complete distributed checkpoint" in result.stderr
    assert "iter_0000007" in result.stderr


def test_existing_torch_dist_destination_is_rejected(tmp_path):
    source = make_source(tmp_path)
    destination = tmp_path / "destination"
    (destination / "torch_dist").mkdir(parents=True)

    result = run_script(str(source), str(destination))

    assert result.returncode != 0
    assert "preserve it for diagnosis and use a fresh root" in result.stderr


def test_log_only_destination_is_preserved_instead_of_overwritten(tmp_path):
    source = make_source(tmp_path)
    destination = tmp_path / "destination"
    destination.mkdir()
    old_log = destination / "stage1-megatron.log"
    old_log.write_text("failure evidence\n")

    result = run_script(str(source), str(destination))

    assert result.returncode != 0
    assert "preserve it for diagnosis" in result.stderr
    assert old_log.read_text() == "failure evidence\n"


def test_common_pt_alone_is_not_a_completion_marker(tmp_path):
    source = make_source(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_srun = fake_bin / "srun"
    fake_srun.write_text(
        "#!/bin/bash\n"
        'mkdir -p "${DST_ROOT}/torch_dist/${SOURCE_ITER_DIR}"\n'
        'touch "${DST_ROOT}/torch_dist/${SOURCE_ITER_DIR}/common.pt"\n'
    )
    fake_srun.chmod(0o755)
    destination = tmp_path / "destination"

    result = run_script(
        str(source),
        str(destination),
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        SCRATCH=str(tmp_path),
    )

    assert result.returncode != 0
    assert "without a complete distributed checkpoint" in result.stderr
    assert not (destination / "stage1-complete").exists()


def test_non_strict_megatron_fallback_is_rejected(tmp_path):
    source = make_source(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_srun = fake_bin / "srun"
    fake_srun.write_text(
        "#!/bin/bash\n"
        'mkdir -p "${DST_ROOT}/torch_dist/${SOURCE_ITER_DIR}"\n'
        'touch "${DST_ROOT}/torch_dist/${SOURCE_ITER_DIR}/common.pt"\n'
        'echo "load_return: missing checkpoint tensors"\n'
    )
    fake_srun.chmod(0o755)

    result = run_script(
        str(source),
        str(tmp_path / "destination"),
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        SCRATCH=str(tmp_path),
    )

    assert result.returncode != 0
    assert "retried a strict model load non-strictly" in result.stderr
    assert not (tmp_path / "destination" / "stage1-complete").exists()


def test_final_metadata_and_tracker_certify_wrapper_success(tmp_path):
    source = make_source(tmp_path, iteration=19)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_srun = fake_bin / "srun"
    srun_args = tmp_path / "srun.args"
    fake_srun.write_text(
        "#!/bin/bash\n"
        f"printf '%s\\n' \"$*\" > {srun_args}\n"
        'mkdir -p "${DST_ROOT}/torch_dist/${SOURCE_ITER_DIR}"\n'
        'touch "${DST_ROOT}/torch_dist/${SOURCE_ITER_DIR}/common.pt"\n'
        'touch "${DST_ROOT}/torch_dist/${SOURCE_ITER_DIR}/.metadata"\n'
        'touch "${DST_ROOT}/torch_dist/${SOURCE_ITER_DIR}/metadata.json"\n'
        'printf "%s" "19" > "${DST_ROOT}/torch_dist/latest_checkpointed_iteration.txt"\n'
    )
    fake_srun.chmod(0o755)
    destination = tmp_path / "destination"
    image = tmp_path / "stage1.sqsh"
    image.touch()
    environment = tmp_path / "stage1.toml"
    environment.write_text(f'image = "{image}"\n')

    result = run_script(
        str(source),
        str(destination),
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        SCRATCH=str(tmp_path),
        STAGE1_ENV=str(environment),
    )

    assert result.returncode == 0, result.stderr
    assert f"Converted checkpoint: {destination}/torch_dist/iter_0000019" in result.stdout
    assert (destination / "stage1-complete").read_text() == "iter_0000019\n"
    assert f"--environment={environment.resolve()}" in srun_args.read_text()
    assert "--container-mounts" not in srun_args.read_text()
