"""Convert fixed, read-only Capstor fixtures into scratch and check parity."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CAPSTOR = Path("/capstor/store/cscs/swissai/infra01/apertus_checkpoints/v1p5")
NATIVE = CAPSTOR / "megatron/8B/posttraining/longctx_iter_0004200"
HF = CAPSTOR / "hf/8B/posttraining/longctx_iter_0004200"
APERTUS2 = CAPSTOR.parent / "v2"
NATIVE_APERTUS2 = APERTUS2 / "megatron/chonk/pretraining/long_context/iter_0003576"
HF_APERTUS2 = APERTUS2 / "hf/chonk/pretraining/long_context/long-context-iter-0003576"
PROMPT = "The capital of Switzerland is"


def _fixture_tree_state(root):
    # Lustre statvfs can report filesystem flags rather than bind-mount flags.
    mounts = (
        line.split() for line in Path("/proc/self/mountinfo").read_text().splitlines()
    )
    readonly_bind = any(
        fields[4] == str(root) and "ro" in fields[5].split(",") for fields in mounts
    )
    if not (os.statvfs(root).f_flag & os.ST_RDONLY or readonly_bind):
        raise ValueError(f"Checkpoint fixture must be mounted read-only: {root}")
    state = {}
    for path in (root, *root.rglob("*")):
        if path.is_symlink():
            raise ValueError(
                f"Fixture symlink could escape the read-only mount: {path}"
            )
        info = path.stat()
        state[str(path.relative_to(root))] = (
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
    return state


def _run(script, arguments, directory, label, environment, *, processes=1):
    command = [
        "uv",
        "run",
        "--no-project",
        "--python",
        sys.executable,
        "python",
    ]
    if processes > 1:
        command += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={processes}",
        ]
    command += [str(script), *arguments]
    with (directory / f"{label}.log").open("w") as log:
        subprocess.run(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=3600,
        )


def _new_run(prefix):
    base = Path(
        os.environ.get("HFCONVERTER_TEST_TMP")
        or str(Path(os.environ["SCRATCH"]) / "tmp")
    ).resolve()
    assert "tmp" in base.parts and any(
        base.is_relative_to(path)
        for path in (Path("/iopsstor/scratch"), Path("/ritom/scratch"))
    )
    base.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=prefix, dir=base))
    print(f"Parity artifacts: {directory}", flush=True)
    environment = dict(
        os.environ,
        RANK="0",
        LOCAL_RANK="0",
        WORLD_SIZE="1",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="26499",
        NVIDIA_TF32_OVERRIDE="0",
        PYTHONDONTWRITEBYTECODE="1",
    )
    bridge = Path(environment["BRIDGE_ROOT"])
    megatron = Path(environment["MEGATRON_ROOT"])
    environment["PYTHONPATH"] = (
        f"{megatron}:{bridge / 'src'}:{ROOT}:{environment.get('PYTHONPATH', '')}"
    )
    return directory, environment


def _copy_reference(hf, directory):
    reference = directory / "reference"
    reference.mkdir()
    for source in hf.iterdir():
        if (
            source.is_file()
            and source.suffix not in (".safetensors", ".bin")
            and not source.name.endswith(".index.json")
        ):
            shutil.copy2(source, reference / source.name)
    return reference


pytestmark = pytest.mark.skipif(
    os.environ.get("HFCONVERTER_RUN_PARITY") != "1", reason="opt-in Capstor/GPU test"
)


def test_apertus15_conversion_and_logit_parity():
    import torch
    from transformers import AutoConfig

    for path in (
        NATIVE / "common.pt",
        HF / "config.json",
        HF / "model.safetensors.index.json",
    ):
        assert path.is_file(), f"Required fixed fixture is missing: {path}"
    tracked = [
        NATIVE / "common.pt",
        HF / "config.json",
        HF / "tokenizer_config.json",
        HF / "model.safetensors.index.json",
    ]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked}
    fixture_state = {str(path): _fixture_tree_state(path) for path in (NATIVE, HF)}
    directory, environment = _new_run("apertus15-parity-")
    (directory / "fixture-trees-before.json").write_text(
        json.dumps(fixture_state, indent=2)
    )
    bridge = Path(environment["BRIDGE_ROOT"])
    megatron = Path(environment["MEGATRON_ROOT"])
    reference = _copy_reference(HF, directory)
    output = directory / "converted-hf"
    _run(
        ROOT / "convert_checkpoint.py",
        [
            "export",
            str(NATIVE),
            str(reference),
            str(output),
            "--expected-hf-index",
            str(HF / "model.safetensors.index.json"),
            "--bridge-root",
            str(bridge),
            "--megatron-root",
            str(megatron),
        ],
        directory,
        "conversion",
        environment,
    )
    assert (output / "CONFIG_ONLY_EXPORT_COMPLETE").is_file()
    _run(
        ROOT / "tests/compare_hf_weights.py",
        [str(HF), str(output), "--report-dir", str(directory / "weights")],
        directory,
        "weights",
        environment,
    )
    expected_config = AutoConfig.from_pretrained(HF)
    actual_config = AutoConfig.from_pretrained(output)
    for field in (
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "max_position_embeddings",
        "rms_norm_eps",
        "qk_norm",
        "rope_parameters",
        "tie_word_embeddings",
    ):
        assert getattr(actual_config, field) == getattr(expected_config, field), field
    assert (
        getattr(actual_config, "head_dim", None)
        or actual_config.hidden_size // actual_config.num_attention_heads
    ) == (
        getattr(expected_config, "head_dim", None)
        or expected_config.hidden_size // expected_config.num_attention_heads
    )
    for label, checkpoint in (("hf-capstor", HF), ("hf-converted", output)):
        _run(
            ROOT / "logits_tools/get_hf_logits.py",
            [
                "--hf-dir",
                str(checkpoint),
                "--dtype",
                "fp32",
                "--prompt",
                PROMPT,
                "--out-pt",
                str(directory / f"{label}.pt"),
                "--out-report",
                str(directory / f"{label}.json"),
            ],
            directory,
            label,
            environment,
        )
    view = directory / "native-view"
    view.mkdir()
    (view / "iter_0004200").symlink_to(NATIVE, target_is_directory=True)
    (view / "latest_checkpointed_iteration.txt").write_text("4200\n")
    _run(
        ROOT / "logits_tools/get_native_dist_logits.py",
        [
            "--load",
            str(view),
            "--ckpt-step",
            "4200",
            "--ckpt-format",
            "torch_dist",
            "--auto-detect-ckpt-format",
            "--use-checkpoint-args",
            "--tensor-model-parallel-size",
            "1",
            "--pipeline-model-parallel-size",
            "1",
            "--tokenizer-type",
            "HuggingFaceTokenizer",
            "--tokenizer-model",
            str(HF),
            "--no-use-tokenizer-model-from-checkpoint-args",
            "--micro-batch-size",
            "1",
            "--no-load-optim",
            "--no-load-rng",
            "--exit-on-missing-checkpoint",
            "--no-initialization",
            "--inference-dtype",
            "fp32",
            "--prompt",
            PROMPT,
            "--out-pt",
            str(directory / "native-capstor.pt"),
            "--out-report",
            str(directory / "native-capstor.json"),
        ],
        directory,
        "native-capstor",
        environment,
    )
    converted = torch.load(
        directory / "hf-converted.pt", map_location="cpu", weights_only=True
    )
    checks = []
    for label in ("hf-capstor", "native-capstor"):
        baseline = torch.load(
            directory / f"{label}.pt", map_location="cpu", weights_only=True
        )
        assert baseline["token_ids"] == converted["token_ids"], (
            f"{label}: token IDs differ"
        )
        a, b = baseline["last_token_logits"], converted["last_token_logits"]
        assert a.shape == b.shape
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        checks.append(
            {
                "baseline": label,
                "max_abs": (a - b).abs().max().item(),
                "mean_abs": (a - b).abs().mean().item(),
                "atol": 1e-4,
                "rtol": 1e-4,
                "passed": torch.allclose(a, b, atol=1e-4, rtol=1e-4),
            }
        )
    (directory / "parity.json").write_text(json.dumps(checks, indent=2))
    assert all(check["passed"] for check in checks), checks
    assert before == {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked
    }
    after = {str(path): _fixture_tree_state(path) for path in (NATIVE, HF)}
    (directory / "fixture-trees-after.json").write_text(json.dumps(after, indent=2))
    assert fixture_state == after, "Canonical fixture tree changed during the test"


def test_apertus2_conversion_weight_parity():
    native, hf = NATIVE_APERTUS2, HF_APERTUS2
    for path in (
        native / "common.pt",
        hf / "config.json",
        hf / "model.safetensors.index.json",
    ):
        assert path.is_file(), f"Required fixed fixture is missing: {path}"
    before = {str(path): _fixture_tree_state(path) for path in (native, hf)}
    directory, environment = _new_run("apertus2-parity-")
    (directory / "fixture-trees-before.json").write_text(json.dumps(before, indent=2))
    reference = _copy_reference(hf, directory)
    output = directory / "converted-hf"
    _run(
        ROOT / "convert_checkpoint.py",
        [
            "export",
            str(native),
            str(reference),
            str(output),
            "--bridge-root",
            environment["BRIDGE_ROOT"],
            "--megatron-root",
            environment["MEGATRON_ROOT"],
            "--ep",
            "4",
            "--trust-remote-code",
            "--model-override",
            "kda_legacy_gate_out_proj_bias=true",
            "--model-override",
            "linear_attn_a_log_per_channel=false",
            "--expected-hf-index",
            str(hf / "model.safetensors.index.json"),
        ],
        directory,
        "conversion",
        environment,
        processes=4,
    )
    assert (output / "CONFIG_ONLY_EXPORT_COMPLETE").is_file()
    _run(
        ROOT / "tests/compare_hf_weights.py",
        [str(hf), str(output), "--report-dir", str(directory / "weights")],
        directory,
        "weights",
        environment,
        processes=4,
    )
    after = {str(path): _fixture_tree_state(path) for path in (native, hf)}
    (directory / "fixture-trees-after.json").write_text(json.dumps(after, indent=2))
    assert before == after, "Canonical fixture tree changed during the test"
