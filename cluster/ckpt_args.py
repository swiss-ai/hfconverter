# Copyright 2026 the Swiss AI Initiative. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Read a legacy Megatron ``torch`` checkpoint's stored args without loading tensors.

WHY NOT ``torch.load``: ``model_optim_rng.pt`` holds the ENTIRE model shard, not just the
args. ``torch.load`` would materialize hundreds of MB of tensors (and needs ``megatron``
importable to unpickle the fork's own classes) purely to read one Namespace. The unpickler
here returns ``None`` from ``persistent_load``, which is where every tensor storage lives,
so the tensor data is never read off disk at all. Stdlib only, no torch, no megatron: it
runs unchanged in any container.

WHAT DOES NOT SURVIVE: values whose class lives in torch/megatron cannot be reconstructed.
They come back as ``_Stub`` instances (``AttnBackend(5)``) or, for module-level globals
pickled by reference, as the stub CLASS itself (``torch.float32``). ``is_opaque()`` detects
both. None of them is architecture-critical -- they are kernel/dtype selection, derived
from --bf16 -- and consumers must handle them by NAME (via ``is_opaque()``), never by value.

SECURITY: skipping tensor storages saves memory and I/O; it does not make Python pickle a
security boundary. A pickle reducer can still execute while metadata is decoded. Run this
helper only on a checkpoint whose files you trust.

    python ckpt_args.py <ckpt-root-or-iter-dir> [--grep PATTERN]
"""

import argparse
import fnmatch
import os
import pickle
import re
import sys
import zipfile


class _Stub(object):
    """Stand-in for a class this process cannot import (torch.*, megatron.*).

    ``__setstate__`` is load-bearing: pickle hands some objects a non-dict state, and the
    default ``object.__setstate__`` rejects it with "state is not a dictionary". Swallowing
    it here is what lets an arbitrary foreign object decode into an inert placeholder
    instead of aborting the whole read.
    """

    __stub_module__ = "?"
    __stub_name__ = "?"

    def __init__(self, *args, **kwargs):
        self.__stub_args__ = args
        self.__stub_kwargs__ = kwargs

    def __setstate__(self, state):
        pass

    def __repr__(self):
        inner = ", ".join(repr(a) for a in self.__stub_args__)
        return "{}({})".format(self.__stub_name__, inner)


def _make_stub(module, name):
    return type(
        str(name),
        (_Stub,),
        {"__stub_module__": module, "__stub_name__": name, "__module__": module},
    )


def is_opaque(value):
    """True if `value` is a placeholder for a class we could not import.

    Two shapes, both real: an INSTANCE (an enum the pickle constructed, e.g.
    ``AttnBackend(5)``) and the CLASS ITSELF (a module-level global pickled by reference,
    e.g. ``torch.float32``).
    """
    if isinstance(value, _Stub):
        return True
    return isinstance(value, type) and issubclass(value, _Stub)


# Classes from these modules are imported for real; everything else is stubbed. argparse is
# the one that matters -- the stored args object IS an argparse.Namespace, and decoding it
# for real is what makes vars() below work.
_SAFE_MODULES = frozenset(["argparse", "builtins", "__builtin__", "collections", "copyreg", "copy_reg"])


class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in _SAFE_MODULES:
            return super(_StubUnpickler, self).find_class(module, name)
        return _make_stub(module, name)

    def persistent_load(self, pid):
        # Every tensor storage in a torch save arrives here as a persistent id. Returning
        # None is what keeps this read O(args) instead of O(checkpoint).
        return None


def _find_data_pkl(zf):
    """torch writes `<archive-name>/data.pkl`; the archive name is not fixed."""
    matches = [n for n in zf.namelist() if n.endswith("/data.pkl") or n == "data.pkl"]
    if not matches:
        raise ValueError("no data.pkl in the archive -- not a torch-saved file")
    return sorted(matches, key=len)[0]


def resolve_model_optim_rng(path):
    """Accept a checkpoint ROOT (with latest_checkpointed_iteration.txt) or an iter dir.

    Returns the path of the rank-0 `model_optim_rng.pt`. Args are replicated across ranks,
    so rank 0 is authoritative for everything this module reads.
    """
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise ValueError("{} is not a directory".format(path))

    iter_dir = path
    tracker = os.path.join(path, "latest_checkpointed_iteration.txt")
    if os.path.isfile(tracker):
        with open(tracker) as handle:
            tag = handle.read().strip()
        iter_dir = os.path.join(path, tag if tag == "release" else "iter_{:07d}".format(int(tag)))
        if not os.path.isdir(iter_dir):
            raise ValueError("tracker names {} but {} does not exist".format(tag, iter_dir))

    ranks = sorted(d for d in os.listdir(iter_dir) if d.startswith("mp_rank_"))
    if not ranks:
        raise ValueError(
            "{} has no mp_rank_* subdir -- not a legacy `torch` checkpoint "
            "(a torch_dist checkpoint has metadata.json + *.distcp instead)".format(iter_dir)
        )
    pt = os.path.join(iter_dir, ranks[0], "model_optim_rng.pt")
    if not os.path.isfile(pt):
        raise ValueError("no model_optim_rng.pt in {}".format(os.path.join(iter_dir, ranks[0])))
    return pt


def read_checkpoint_args(path):
    """-> dict of the checkpoint's stored args. `path` is a ckpt root or an iter dir."""
    pt = resolve_model_optim_rng(path)
    with zipfile.ZipFile(pt) as zf:
        with zf.open(_find_data_pkl(zf)) as handle:
            payload = _StubUnpickler(handle).load()

    if not isinstance(payload, dict) or "args" not in payload:
        raise ValueError(
            "{} has no 'args' key (top-level keys: {}) -- not a Megatron "
            "model_optim_rng.pt".format(pt, sorted(payload.keys())[:8] if isinstance(payload, dict) else type(payload))
        )
    args = payload["args"]
    if is_opaque(args) or not hasattr(args, "__dict__"):
        raise ValueError("the stored 'args' did not decode to a Namespace (got {!r})".format(args))
    return vars(args)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python cluster/ckpt_args.py",
        description="Print a legacy Megatron torch checkpoint's stored args (no torch, no tensors read).",
    )
    parser.add_argument("checkpoint", help="checkpoint root (with latest_checkpointed_iteration.txt) or an iter_XXXXXXX dir")
    parser.add_argument("--grep", default=None, help="only print keys matching this glob (e.g. 'moe_*')")
    ns = parser.parse_args(argv)

    try:
        args = read_checkpoint_args(ns.checkpoint)
    except ValueError as exc:
        sys.stderr.write("FATAL: {}\n".format(exc))
        return 1

    keys = sorted(args)
    if ns.grep:
        keys = [k for k in keys if fnmatch.fnmatch(k, ns.grep)]
    for key in keys:
        value = args[key]
        mark = "  # opaque (class not importable here)" if is_opaque(value) else ""
        sys.stdout.write("{} = {!r}{}\n".format(key, value, mark))
    sys.stderr.write("\n{} of {} keys shown\n".format(len(keys), len(args)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
