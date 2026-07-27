#!/usr/bin/env python
"""2-rank torchrun worker for test_ep2.py: saves the tiny mock checkpoint with the STACKED
expert tensors contributed as two expert-axis halves (EP2-style), everything else from
rank 0 only.

Usage (launched by the test):
  python -m torch.distributed.run --nproc_per_node=2 --master_addr=127.0.0.1 \
      --master_port=<free> _ep2_worker.py <payload.pt> <ckpt_dir>

payload.pt (torch.save, pickled): {"tensors": {key: Tensor}, "args": argparse.Namespace,
"iteration": int} — produced single-process by the test so both ranks (and the reference
checkpoint) share bit-identical data. This worker deliberately imports NOTHING from the
repo: torch + megatron-core only.

Recipe provenance: scratchpad dcprobe/probe_ep2.py. Everything lives under __main__ —
mcore's save spawns a multiprocessing Manager ('spawn' context) whose child re-imports
__main__; an unguarded module-level save recurses (probe_ep2.py:23-26).
"""

import os
import sys

import torch
import torch.distributed as dist

# CPU shim: mcore's save path calls torch.cuda.synchronize()/current_device() unconditionally
if not torch.cuda.is_available():
    torch.cuda.synchronize = lambda *a, **kw: None
    torch.cuda.current_device = lambda: torch.device("cpu")

EXPERT_KEY_MARKER = ".mlp.experts.experts.linear_fc"


def main():
    payload_path, ckpt_dir = sys.argv[1], sys.argv[2]

    import megatron.core.dist_checkpointing as dist_checkpointing
    from megatron.core.dist_checkpointing import ShardedTensor

    dist.init_process_group(backend="gloo", init_method="env://")  # torchrun env vars
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, world_size

    payload = torch.load(payload_path, weights_only=False)  # Namespace needs full pickle
    tensors = payload["tensors"]

    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
    dist.barrier()

    sharded_state_dict = {}
    expert_keys = sorted(key for key in tensors if EXPERT_KEY_MARKER in key)
    assert expert_keys, "payload has no stacked expert tensors"

    # expert-stacked tensors: rank r owns experts [r*E/2 : (r+1)*E/2] along axis 0
    for key in expert_keys:
        full = tensors[key]
        num_experts = full.shape[0]
        assert num_experts % world_size == 0, (key, full.shape)
        chunk = num_experts // world_size
        local = full[rank * chunk : (rank + 1) * chunk].contiguous()
        sharded_state_dict[key] = ShardedTensor.from_rank_offsets(
            key, local, (0, rank, world_size), replica_id=0
        )

    # everything else: fully owned by rank 0 (unique-keys-per-rank pattern, probe_ep2.py)
    if rank == 0:
        for key, tensor in tensors.items():
            if key in set(expert_keys):
                continue
            sharded_state_dict[key] = ShardedTensor.from_rank_offsets(
                key, tensor, replica_id=0
            )

    # common part must be identical on all ranks (consistency-checked); rank 0 writes it
    sharded_state_dict["args"] = payload["args"]
    sharded_state_dict["iteration"] = payload["iteration"]
    sharded_state_dict["checkpoint_version"] = 3.0

    dist_checkpointing.save(sharded_state_dict, ckpt_dir)
    print(f"[rank {rank}] ep2 save done", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
