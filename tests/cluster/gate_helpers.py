"""Synthetic-checkpoint builders for the arg-gate tests.

A MODULE, not conftest.py: `tests/conftest.py` and `tests/cluster/conftest.py` are both
importable as `conftest`, and pytest puts both directories on sys.path -- `from conftest
import ...` would resolve to whichever landed first. This name is unambiguous.

Everything here is stdlib pickle+zipfile. That is the point: the gate must be testable with
no torch, no megatron, no GPU and no real checkpoint, so the suite runs in any container
(cluster/tests.sbatch counts a skip as a failure).
"""

import argparse
import io
import os
import pickle
import zipfile


class ForeignClass:
    """Stands in for a class the reader cannot import (a megatron enum, a torch dtype).

    Its pickled module is this test module, which is not on the reader's safe-module list,
    so it takes exactly the stub path a real megatron enum takes.
    """

    def __init__(self, value):
        self.value = value

    def __reduce__(self):
        return (ForeignClass, (self.value,))


class _StorageSentinel:
    """Stands in for a torch tensor storage: reached via persistent_id, never inlined."""


class _TorchLikePickler(pickle.Pickler):
    """Writes storages as persistent ids, the way torch.save does.

    Without this the synthetic checkpoint would contain no persistent ids and the tests
    would never exercise persistent_load -- the mechanism that keeps the read O(args)
    rather than O(checkpoint).
    """

    def persistent_id(self, obj):
        if isinstance(obj, _StorageSentinel):
            return ("storage", "FloatStorage", "0", "cpu", 1024)
        return None


def write_checkpoint(root, args_dict, iteration=1192, tracker=True, with_storage=True):
    """Build a legacy-`torch`-shaped checkpoint dir. -> the root path (str)."""
    iter_dir = os.path.join(str(root), "iter_{:07d}".format(iteration))
    rank_dir = os.path.join(iter_dir, "mp_rank_00_000")
    os.makedirs(rank_dir)
    if tracker:
        with open(os.path.join(str(root), "latest_checkpointed_iteration.txt"), "w") as handle:
            handle.write("{}\n".format(iteration))

    payload = {"args": argparse.Namespace(**args_dict), "checkpoint_version": 3.0}
    if with_storage:
        # A real model_optim_rng.pt is mostly this: hundreds of MB of storages.
        payload["model"] = {"embedding.weight": _StorageSentinel(), "layer.0.w": _StorageSentinel()}

    buf = io.BytesIO()
    _TorchLikePickler(buf, protocol=2).dump(payload)
    with zipfile.ZipFile(os.path.join(rank_dir, "model_optim_rng.pt"), "w") as zf:
        zf.writestr("archive/data.pkl", buf.getvalue())
    return str(root)


