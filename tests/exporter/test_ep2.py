"""EP2 sharded-save test: the stacked expert tensors are
written by TWO torchrun ranks as expert-axis halves (ShardedTensor.from_rank_offsets with an
axis-0 offset — probe_ep2.py recipe, worker: _ep2_worker.py); the merged torch_dist
checkpoint must single-process-export bitwise-identically to a rank-0-only reference export.
This exercises the distributed-checkpoint merge path.

Kept under ~60s: the payload is built once in-process and shared with both the torchrun
worker and the reference save via a torch.save pickle, so the two checkpoints are
bit-identical by construction and the workers import only torch + megatron-core.
"""

import glob
import importlib.util
import os
import subprocess
import sys

import pytest

pytest.importorskip("megatron.core")

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

import exporter.export  # noqa: E402,F401  (module-level on purpose: collection surfaces the dependency)
import megatron_mock  # noqa: E402

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ep2_worker.py")

torchrun_available = (
    importlib.util.find_spec("torch.distributed.run") is not None and os.path.isfile(WORKER)
)


def _read_all_safetensors(directory):
    state_dict = {}
    files = sorted(glob.glob(os.path.join(str(directory), "*.safetensors")))
    assert files, f"no safetensors written in {directory}"
    for path in files:
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict


@pytest.mark.skipif(not torchrun_available, reason="torch.distributed.run unavailable")
def test_ep2_expert_axis_sharded_save_exports_bitwise(dist_env, export_api, tmp_path):
    # one payload, shared bit-identically by the 2-rank save and the reference save
    model = megatron_mock.build_tiny_model(False, None, False, seed=0)
    tensors = megatron_mock.to_megatron_tensors(model, model.config, expert_bias_present=True)
    args = megatron_mock.make_args_namespace(model.config, expert_bias_present=True)
    payload_path = tmp_path / "payload.pt"
    torch.save({"tensors": tensors, "args": args, "iteration": 100}, payload_path)

    # 2-rank EP-style save via torchrun
    ckpt_ep2 = tmp_path / "ckpt_ep2"
    env = os.environ.copy()
    for stale in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(stale, None)
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--nnodes=1",
            "--master_addr=127.0.0.1",
            f"--master_port={megatron_mock.find_free_port()}",
            WORKER,
            str(payload_path),
            str(ckpt_ep2),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(tmp_path),
    )
    assert process.returncode == 0, (
        f"torchrun worker failed rc={process.returncode}\nstdout:\n{process.stdout}\n"
        f"stderr:\n{process.stderr}"
    )

    # sanity: the DCP merge itself is bitwise (isolates worker bugs from exporter bugs)
    dist_env()  # the torchrun subprocess never touches our in-process pg; re-ensure anyway
    merged = megatron_mock.load_plain_tensor_dict(ckpt_ep2)
    assert set(merged) == set(tensors)
    for key, reference in tensors.items():
        assert merged[key].dtype == reference.dtype, key
        assert torch.equal(merged[key], reference), f"{key} corrupted by the sharded save"

    # reference: identical tensors saved rank-0-only, single process
    ckpt_ref = tmp_path / "ckpt_ref"
    megatron_mock.save_synthetic_checkpoint(tensors, args, ckpt_ref, iteration=100)

    # export both, compare every on-disk tensor bitwise
    out_ep2 = tmp_path / "hf_ep2"
    out_ref = tmp_path / "hf_ref"
    export_api(ckpt_ep2, out_ep2)
    export_api(ckpt_ref, out_ref)

    disk_ep2 = _read_all_safetensors(out_ep2)
    disk_ref = _read_all_safetensors(out_ref)
    assert set(disk_ep2) == set(disk_ref), sorted(set(disk_ep2) ^ set(disk_ref))
    for key, reference in disk_ref.items():
        assert disk_ep2[key].dtype == reference.dtype, key
        assert torch.equal(disk_ep2[key], reference), (
            f"{key}: EP2-sharded export differs from rank-0-only export"
        )
