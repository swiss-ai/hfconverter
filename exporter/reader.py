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
"""Read Megatron ``torch_dist`` checkpoints through Megatron Core's DCP API.

A distributed checkpoint stores pieces produced by tensor, pipeline, and expert parallel ranks.
Megatron Core reconstructs those pieces into each tensor's global shape; the exporter must not try
to concatenate rank files itself. This module keeps that framework-specific setup out of the
mapping and writing code.
"""

import logging
import pickle
import socket
import warnings
from argparse import Namespace
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

with warnings.catch_warnings():
    # pip mcore without TransformerEngine/Apex warns at import; harmless on this CPU-only host.
    warnings.simplefilter("ignore")
    from megatron.core import dist_checkpointing

logger = logging.getLogger(__name__)


def install_cpu_shim() -> None:
    """Make Megatron Core's synchronous DCP load usable with a CPU-only PyTorch build.

    Megatron Core 0.18 routes synchronous loads through code that unconditionally calls two CUDA
    helpers. Replacing only those helpers is a compatibility boundary, not a model transformation;
    tensor loading and DCP's global-shard merge still run through Megatron Core. The operation is
    idempotent and does nothing when CUDA is available.
    """
    if not torch.cuda.is_available():
        torch.cuda.synchronize = lambda *a, **kw: None
        torch.cuda.current_device = lambda: torch.device("cpu")


def find_free_port() -> int:
    """Ask the operating system for a free localhost port for the one-rank process group."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def init_single_rank_pg() -> bool:
    """Initialize the one-rank Gloo group required by Megatron Core's DCP loader.

    Returns ``True`` only when this call created the group, which transfers cleanup ownership to
    the caller. An existing process group is reused and never destroyed here.
    """
    if dist.is_available() and dist.is_initialized():
        return False
    dist.init_process_group(
        backend="gloo",
        world_size=1,
        rank=0,
        init_method=f"tcp://127.0.0.1:{find_free_port()}",
    )
    return True


def destroy_pg() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


@contextmanager
def single_rank_pg() -> Iterator[None]:
    """Provide Megatron Core with a process group and clean up only groups created here."""
    created = init_single_rank_pg()
    try:
        yield
    finally:
        if created:
            destroy_pg()


def check_checkpoint(ckpt_dir: str | Path) -> None:
    """Require a Megatron distributed checkpoint rather than a legacy checkpoint directory."""
    if not dist_checkpointing.check_is_distributed_checkpoint(str(ckpt_dir)):
        raise ValueError(f"not a Megatron torch_dist distributed checkpoint: {ckpt_dir}")


class _ForkObjectStub:
    """Placeholder for fork-only classes pickled into ``common.pt``.

    Newer fork training runs pickle fork-defined objects into the common state dict (first seen:
    ``megatron.core.tokenizers.utils.tokenizer_extra_metadata.TokenizerExtraMetadata`` in the KDA
    smoke run). HF-side runs pin stock Megatron Core, which lacks fork-added modules, so those
    globals cannot resolve — but :func:`load_args` only needs ``common['args']``, a stdlib
    ``Namespace``. Auxiliary fork objects just have to unpickle into *something*.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        self._pickled_state = state


class _TolerantUnpickler(pickle.Unpickler):
    """Resolve unimportable pickle globals to :class:`_ForkObjectStub` instead of failing."""

    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            logger.warning(
                "common.pt references %s.%s, which is unavailable outside the fork; "
                "loading it as a stub", module, name,
            )
            return _ForkObjectStub


# torch.load wants a module-shaped object; only .Unpickler and .__name__ are touched for
# zip-format checkpoints.
_TOLERANT_PICKLE = SimpleNamespace(Unpickler=_TolerantUnpickler, __name__="tolerant_pickle")


def load_args(ckpt_dir: str | Path) -> tuple[Namespace, dict[str, Any]]:
    """Read training arguments from ``common.pt`` without loading tensor data."""
    try:
        common = dist_checkpointing.load_common_state_dict(str(ckpt_dir))
    except pickle.UnpicklingError:
        # Megatron Core 0.18's load_common defers to torch.load's weights_only=True default,
        # which refuses any pickled non-tensor class. The checkpoint comes from the user's own
        # training run, so load permissively, stubbing fork-only globals (see _ForkObjectStub).
        common = torch.load(
            Path(ckpt_dir) / "common.pt",
            map_location="cpu",
            weights_only=False,
            pickle_module=_TOLERANT_PICKLE,
        )
    if "args" not in common:
        raise ValueError(f"checkpoint has no 'args' entry in its common state dict: {ckpt_dir}")
    return common["args"], common


def load_metadata(ckpt_dir: str | Path) -> dict[str, Any]:
    """Read tensor names, global shapes, and dtypes without materializing values.

    The returned Megatron ``ShardedTensor`` descriptors have ``data=None``. Their ``global_shape``
    already describes the tensor after tensor/expert/pipeline-parallel pieces are merged.
    """
    return dist_checkpointing.load_tensors_metadata(str(ckpt_dir))


def load_tensors(
    ckpt_dir: str | Path, keys: Iterable[str] | None = None
) -> dict[str, torch.Tensor]:
    """Load selected global tensors, merging their distributed checkpoint pieces.

    A process group must already exist; :func:`single_rank_pg` handles that for normal callers.
    ``keys`` is a memory limit, not merely a convenience filter. Without it Megatron Core loads
    every tensor, including large fp32 optimizer state. Filtering the metadata before ``load`` means
    those bytes are never materialized.

    Megatron Core 0.18 also merges non-tensor ``common.pt`` entries into the result, so the final
    comprehension intentionally retains tensors only.
    """
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError(
            "load_tensors requires an initialized process group; wrap the call in "
            "exporter.reader.single_rank_pg()"
        )
    install_cpu_shim()
    metadata = load_metadata(ckpt_dir)
    if keys is not None:
        wanted = set(keys)
        missing = wanted - set(metadata)
        if missing:
            raise ValueError(
                f"checkpoint is missing {len(missing)} expected tensor(s): {sorted(missing)[:8]}"
            )
        metadata = {k: v for k, v in metadata.items() if k in wanted}
    plain = dist_checkpointing.load(
        metadata, str(ckpt_dir), validate_access_integrity=False
    )
    return {k: v for k, v in plain.items() if isinstance(v, torch.Tensor)}

