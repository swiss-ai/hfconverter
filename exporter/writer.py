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
"""Write and verify the Hugging Face output directory.

This module owns file-format details: safetensors shards and index, ``config.json``, tokenizer and
remote-code files, and the final conversion provenance record. Name and tensor transformations live
in :mod:`exporter.mapping`, keeping file I/O separate from model geometry.
"""

import importlib
import json
import logging
import math
import shutil
import subprocess
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

# The generic shard packer accepts size/id callbacks. This lets us make Hugging Face's normal shard
# decision from tensor metadata, before loading the model. The import path differs across versions.
try:
    from huggingface_hub import split_state_dict_into_shards_factory
except ImportError:  # pragma: no cover - depends on installed huggingface_hub layout
    from huggingface_hub.serialization import split_state_dict_into_shards_factory

from . import __version__
from .mapping import HFTensorSpec, SynthesizedTensor

# export must work when the repo root is not the cwd: resolve it from __file__ and put it on
# sys.path before importing the configuration/modeling modules (modeling falls back to a
# top-level `from configuration_apertus_moe import ...`).
REPO_ROOT = Path(__file__).resolve().parent.parent

SAFETENSORS_INDEX_NAME = "model.safetensors.index.json"
SAFETENSORS_SINGLE_NAME = "model.safetensors"
# huggingface_hub's SAFETENSORS_WEIGHTS_FILE_PATTERN: single shard -> "model.safetensors"; sharded
# -> "model-00001-of-00005.safetensors". Passed to the packer so filenames match the torch path.
SHARD_FILENAME_PATTERN = "model{suffix}.safetensors"
_MODULE_FILES = ("configuration_apertus_moe.py", "modeling_apertus_moe.py")
_TOKENIZER_GLOBS = (
    "tokenizer*",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab*.json",
    "merges.txt",
    "*.tiktoken",
    "chat_template*",
)

logger = logging.getLogger(__name__)


def import_repo_module(name: str):
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module(name)


def _owned_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    """Return a safetensors-ready tensor that owns exactly its visible storage.

    Split outputs are often views into a much larger fused Megatron tensor. Safetensors requires
    contiguous data, and shard sizing must count the visible output rather than the source storage.
    Cloning a remaining view is bit-preserving and releases that source after the shard is written.
    """
    tensor = tensor.contiguous()
    visible_bytes = tensor.numel() * tensor.element_size()
    if tensor.storage_offset() != 0 or tensor.untyped_storage().nbytes() != visible_bytes:
        tensor = tensor.clone()
    return tensor


@dataclass(frozen=True)
class _ShardStub:
    """Stand-in fed to the packer in place of a real tensor: it reports only size and id."""

    key: str
    nbytes: int


def _spec_nbytes(spec: HFTensorSpec) -> int:
    """Calculate an output tensor's owned storage size without allocating the tensor."""
    return math.prod(spec.shape) * torch.empty((), dtype=spec.dtype).element_size()


def plan_shards(specs: list[HFTensorSpec], max_shard_size: int | str):
    """Ask Hugging Face's shard packer for a layout using only planned shapes and dtypes.

    Each stand-in reports the same owned byte size the eventual tensor will have. Its unique key is
    also its storage identity because :func:`_owned_contiguous` gives every written tensor separate
    storage. The result therefore matches packing real materialized tensors without the memory cost.
    """
    stubs = {spec.hf_key: _ShardStub(spec.hf_key, _spec_nbytes(spec)) for spec in specs}
    return split_state_dict_into_shards_factory(
        stubs,
        get_storage_size=lambda stub: stub.nbytes,
        get_storage_id=lambda stub: stub.key,
        filename_pattern=SHARD_FILENAME_PATTERN,
        max_shard_size=max_shard_size,
    )


def write_one_shard(
    out_dir: Path, filename: str, tensors: dict[str, torch.Tensor]
) -> tuple[int, dict[str, torch.Tensor]]:
    """Make owned tensor copies and write one safetensors shard.

    The returned dictionary is the exact in-memory reference used for bitwise verification. The
    caller discards it immediately afterward, bounding extra memory to one shard.
    """
    owned = {key: _owned_contiguous(tensor) for key, tensor in tensors.items()}
    save_file(owned, str(out_dir / filename), metadata={"format": "pt"})
    nbytes = sum(t.numel() * t.element_size() for t in owned.values())
    logger.info("write: %s (%d tensors, %.2f GiB)", filename, len(owned), nbytes / 2**30)
    return nbytes, owned


def write_index(out_dir: Path, split, total_bytes: int) -> None:
    """Write model.safetensors.index.json for a sharded export (skip for a single shard)."""
    metadata = dict(split.metadata)
    metadata.setdefault("total_size", total_bytes)
    index = {"metadata": metadata, "weight_map": split.tensor_to_filename}
    (out_dir / SAFETENSORS_INDEX_NAME).write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )


def write_config(
    out_dir: Path,
    cfg_kwargs: dict[str, Any],
    params_dtype: torch.dtype,
    token_ids: dict[str, int] | None = None,
) -> None:
    """Build ``ApertusMoeConfig`` from checkpoint-derived values and save ``config.json``.

    ``token_ids`` (from the tokenizer, when one was supplied) is folded in HERE, not only into
    generation_config.json: otherwise the two files in the same exported directory disagree and
    config.json silently ships the ApertusMoeConfig class defaults.
    """
    configuration = import_repo_module("configuration_apertus_moe")
    if token_ids:
        cfg_kwargs = {**cfg_kwargs, **token_ids}
    else:
        logger.warning(
            "no --tokenizer-dir given: config.json keeps the ApertusMoeConfig DEFAULT token ids "
            "(bos/eos/pad). Verify them against the training tokenizer before using this model."
        )
    config = configuration.ApertusMoeConfig(**cfg_kwargs)
    config.architectures = ["ApertusMoeForCausalLM"]
    # AutoModel is listed as well as AutoModelForCausalLM: without it,
    # AutoModel.from_pretrained(dir, trust_remote_code=True) raises "Unrecognized configuration
    # class ... for this kind of AutoModel" and any recipient whose tooling loads the BACKBONE
    # (embeddings, feature extraction, a custom head) is stuck, even though the class exists and
    # is exported. Found on the chonk-SWA export, which shipped without it.
    config.auto_map = {
        "AutoConfig": "configuration_apertus_moe.ApertusMoeConfig",
        "AutoModel": "modeling_apertus_moe.ApertusMoeModel",
        "AutoModelForCausalLM": "modeling_apertus_moe.ApertusMoeForCausalLM",
    }
    # The embedding tensor determines the main parameter dtype; fp32 router buffers are exceptions.
    config.dtype = str(params_dtype).removeprefix("torch.")
    config.save_pretrained(str(out_dir))


def copy_module_files(out_dir: Path) -> None:
    for name in _MODULE_FILES:
        shutil.copyfile(REPO_ROOT / name, out_dir / name)


def _git(*argv: str) -> str | None:
    # Provenance is best-effort by construction: it runs AFTER the weights are on disk, so no
    # failure here (missing git, stalled git -> TimeoutExpired, which is a SubprocessError and
    # NOT an OSError) may be allowed to fail the export.
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *argv], capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _exporter_git_hash() -> str:
    """HEAD sha, suffixed '-dirty' when the exporter/model sources differ from that commit.

    Without the dirty check the record can name a commit whose tree does not contain the code
    that produced the weights.
    """
    head = _git("rev-parse", "HEAD")
    if head is None:
        return "unknown"
    status = _git("status", "--porcelain", "--", "exporter", *_MODULE_FILES)
    return f"{head}-dirty" if status else head


def write_conversion_info(
    out_dir: Path,
    *,
    source_checkpoint: str,
    iteration: Any,
    checks: list[str],
    dropped_key_count: int,
    synthesized: list[SynthesizedTensor],
    settings: dict[str, Any] | None = None,
    verified: bool = True,
) -> None:
    """Write the final provenance record after every requested operation succeeds.

    File presence means the conversion finished. The separate ``verified`` field records whether
    the optional safetensors round-trip check ran.
    """
    import megatron.core
    import transformers

    info = {
        "source_checkpoint": source_checkpoint,
        "iteration": iteration,
        "verified": verified,
        "exporter_version": __version__,
        "exporter_git_hash": _exporter_git_hash(),
        "megatron_core_version": megatron.core.__version__,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "performed_checks": checks,
        "dropped_optimizer_key_count": dropped_key_count,
        "synthesized_zero_keys": [{"key": s.hf_key, "reason": s.reason} for s in synthesized],
        "settings": settings or {},
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "conversion_info.json").write_text(json.dumps(info, indent=2) + "\n")


def _token_ids_from_tokenizer_files(tokenizer_dir: Path) -> dict[str, int]:
    """Best-effort {bos,eos,pad}_token_id extraction from tokenizer_config/tokenizer.json."""
    ids: dict[str, int] = {}
    cfg_path = tokenizer_dir / "tokenizer_config.json"
    if not cfg_path.is_file():
        return ids
    cfg = json.loads(cfg_path.read_text())
    added: dict[str, int] = {}
    tok_json = tokenizer_dir / "tokenizer.json"
    if tok_json.is_file():
        for entry in json.loads(tok_json.read_text()).get("added_tokens", []):
            if isinstance(entry, dict) and "content" in entry and "id" in entry:
                added[entry["content"]] = entry["id"]
    for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = cfg.get(field)
        if isinstance(value, int):
            ids[field] = value
            continue
        token = cfg.get(field.removesuffix("_id"))
        if isinstance(token, dict):
            token = token.get("content")
        if isinstance(token, str) and token in added:
            ids[field] = added[token]
    return ids


def _highest_token_id(tokenizer_dir: Path) -> int | None:
    """Largest id the tokenizer can emit, or ``None`` when it cannot be determined."""
    tok_json = tokenizer_dir / "tokenizer.json"
    if not tok_json.is_file():
        return None
    tokenizer = json.loads(tok_json.read_text())
    ids = [entry["id"] for entry in tokenizer.get("added_tokens", [])
           if isinstance(entry, dict) and isinstance(entry.get("id"), int)]
    vocab = tokenizer.get("model", {}).get("vocab")
    if isinstance(vocab, dict):
        ids.extend(value for value in vocab.values() if isinstance(value, int))
    elif isinstance(vocab, list):  # some models store vocab as a list of (token, score) pairs
        ids.append(len(vocab) - 1)
    return max(ids) if ids else None


def check_tokenizer_fits_vocab(tokenizer_dir: str | Path, vocab_size: int) -> None:
    """Refuse a tokenizer that can emit ids the embedding matrix does not have.

    Nothing else in the pipeline compares the two, and the mismatch is silent: the export
    finishes, ``config.json`` carries the checkpoint's ``vocab_size``, and the model only fails
    later at the embedding lookup on ordinary text. Both cluster entry points default
    ``TOKENIZER_DIR`` to a 1.5b-era 200064-entry tokenizer, so forgetting the knob on a
    131072-vocab checkpoint is the easy mistake this exists to catch.
    """
    highest = _highest_token_id(Path(tokenizer_dir))
    if highest is None:
        logger.warning(
            "cannot read a vocabulary from %s; tokenizer/vocab_size agreement is unverified",
            tokenizer_dir,
        )
        return
    if highest >= vocab_size:
        raise ValueError(
            f"tokenizer in {tokenizer_dir} can emit token id {highest}, but the checkpoint's "
            f"vocab_size is {vocab_size} (valid ids are 0..{vocab_size - 1}). Pass the tokenizer "
            "this checkpoint was trained with (args.tokenizer_model names it); the shipped "
            "default belongs to the 1.5b models."
        )
    logger.info("tokenizer: highest token id %d < vocab_size %d", highest, vocab_size)


def copy_tokenizer(out_dir: Path, tokenizer_dir: str | Path) -> dict[str, int]:
    """Copy tokenizer files, write generation config, and return special-token ids.

    Returning the ids lets :func:`write_config` place identical values in ``config.json``.
    """
    tokenizer_dir = Path(tokenizer_dir)
    if not tokenizer_dir.is_dir():
        raise ValueError(f"--tokenizer-dir is not a directory: {tokenizer_dir}")
    copied: list[str] = []
    for pattern in _TOKENIZER_GLOBS:
        for path in sorted(tokenizer_dir.glob(pattern)):
            if path.is_file() and path.name not in copied:
                shutil.copyfile(path, out_dir / path.name)
                copied.append(path.name)
    if not copied:
        logger.warning("no tokenizer files matched in %s; nothing copied", tokenizer_dir)
    ids = _token_ids_from_tokenizer_files(tokenizer_dir)
    if ids:
        (out_dir / "generation_config.json").write_text(json.dumps(ids, indent=2) + "\n")
        logger.info("tokenizer: copied %d file(s); token ids %s", len(copied), ids)
    else:
        logger.warning(
            "no token ids found in %s/tokenizer_config.json; skipping generation_config.json",
            tokenizer_dir,
        )
    return ids


def verify_shard(out_dir: Path, filename: str, written: dict[str, torch.Tensor]) -> None:
    """Reload one shard and compare dtype and values against exactly what was written."""
    reloaded = load_file(str(out_dir / filename))
    if reloaded.keys() != written.keys():
        raise ValueError(
            f"verify: shard {filename} reloaded key set differs from what was written; "
            f"missing={sorted(written.keys() - reloaded.keys())} "
            f"extra={sorted(reloaded.keys() - written.keys())}"
        )
    for key, ref in written.items():
        got = reloaded[key]
        if got.dtype != ref.dtype:
            raise ValueError(f"verify: {key} dtype changed on disk: {got.dtype} != {ref.dtype}")
        if not torch.equal(got, ref):
            raise ValueError(f"verify: {key} is not bit-identical after safetensors round trip")


class DiskBackedTensors(Mapping[str, torch.Tensor]):
    """Read-only tensor mapping that loads one safetensors value at a time.

    Full ``from_pretrained`` verification already holds the HF model in memory. Keeping references
    disk-backed avoids holding a second full copy; only the tensor currently being compared is read.
    """

    def __init__(self, out_dir: Path, weight_map: dict[str, str]) -> None:
        self._out_dir = out_dir
        self._weight_map = weight_map

    def __contains__(self, key: str) -> bool:
        return key in self._weight_map

    def __iter__(self) -> Iterator[str]:
        return iter(self._weight_map)

    def __len__(self) -> int:
        return len(self._weight_map)

    def __getitem__(self, key: str) -> torch.Tensor:
        with safe_open(str(self._out_dir / self._weight_map[key]), framework="pt") as handle:
            return handle.get_tensor(key)


def verify_from_pretrained(
    out_dir: Path, refs: Mapping[str, torch.Tensor], geometry: dict[str, Any]
) -> None:
    """Load the HF model and compare its runtime state with the exported on-disk tensors.

    The HF runtime fuses expert matrices into 3D parameters, while its on-disk format uses one key
    per expert. The two expert branches below bridge that intentional representation difference.
    """
    modeling = import_repo_module("modeling_apertus_moe")
    model, loading_info = modeling.ApertusMoeForCausalLM.from_pretrained(
        str(out_dir), dtype="auto", output_loading_info=True
    )
    problems = {k: v for k, v in loading_info.items() if v}
    if problems:
        raise ValueError(f"verify-load: from_pretrained reported {problems}")

    expert_ffn_size = geometry["moe_intermediate_size"]

    def compare(hf_key: str, got: torch.Tensor) -> None:
        reference = refs[hf_key]
        if got.dtype != reference.dtype:
            raise ValueError(
                f"verify-load: {hf_key} dtype {got.dtype} != exported {reference.dtype}"
            )
        if not torch.equal(got, reference):
            raise ValueError(f"verify-load: {hf_key} differs bitwise from the exported tensor")

    matched: set[str] = set()
    for key, tensor in model.state_dict().items():
        if key in refs:
            compare(key, tensor)
            matched.add(key)
        elif key.endswith(".mlp.experts.gate_up_proj"):
            # Runtime [E,2*F,input] <-> on-disk expert gate/up matrices [F,input].
            prefix = key.removesuffix("gate_up_proj")
            for expert_index in range(tensor.shape[0]):
                pieces = (
                    ("gate", tensor[expert_index, :expert_ffn_size]),
                    ("up", tensor[expert_index, expert_ffn_size:]),
                )
                for projection_name, piece in pieces:
                    hf_key = f"{prefix}{expert_index}.{projection_name}_proj.weight"
                    compare(hf_key, piece)
                    matched.add(hf_key)
        elif key.endswith(".mlp.experts.down_proj"):
            # Runtime [E,input,F] <-> one on-disk [input,F] matrix per expert.
            prefix = key.removesuffix("down_proj")
            for expert_index in range(tensor.shape[0]):
                hf_key = f"{prefix}{expert_index}.down_proj.weight"
                compare(hf_key, tensor[expert_index])
                matched.add(hf_key)
        else:
            raise ValueError(f"verify-load: unexpected runtime state dict key {key}")
    unmatched = sorted(refs.keys() - matched)
    if unmatched:
        raise ValueError(f"verify-load: exported keys not found in the loaded model: {unmatched}")


# Compatibility alias for the in-progress streaming implementation that originally introduced the
# private name. New code should use the ordinary Mapping-style public name above.
_DiskShardTensors = DiskBackedTensors
