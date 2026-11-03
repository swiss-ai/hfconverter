"""Stream and compare two Hugging Face safetensors checkpoints.

Each process scans shard headers, then compares only the keys assigned to its
rank. Tensor payloads are loaded one at a time on CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any

_INDEX_NAME = "model.safetensors.index.json"
_SHARD_NAME = re.compile(r"^model-\d+-of-\d+\.safetensors$")


def validate_index(
    index: Mapping[str, Any], available: Mapping[str, str]
) -> dict[str, str]:
    """Validate an HF weight-map index against discovered headers and return it."""
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or any(
        not isinstance(key, str) or not isinstance(filename, str)
        for key, filename in weight_map.items()
    ):
        raise ValueError("index must contain a string-to-string 'weight_map'")

    missing = sorted(set(weight_map) - set(available))
    unexpected = sorted(set(available) - set(weight_map))
    wrong_shard = sorted(
        key
        for key, filename in weight_map.items()
        if key in available and available[key] != filename
    )
    if missing or unexpected or wrong_shard:
        details = []
        if missing:
            details.append(f"indexed keys absent from shard headers: {missing}")
        if unexpected:
            details.append(f"header keys absent from index: {unexpected}")
        if wrong_shard:
            details.append(f"index points to wrong shard for keys: {wrong_shard}")
        raise ValueError("invalid safetensors index: " + "; ".join(details))
    return dict(weight_map)


def _safetensors_module() -> Any:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is required to compare checkpoints") from exc
    return safe_open


def _scan_checkpoint(
    directory: Path, safe_open: Any
) -> dict[str, tuple[Path, tuple[int, ...], str]]:
    if not directory.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {directory}")

    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise ValueError(f"no .safetensors files found in {directory}")
    discovered: dict[str, tuple[Path, tuple[int, ...], str]] = {}
    filenames: dict[str, str] = {}
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118 - safe_open is not a mapping
                if key in discovered:
                    raise ValueError(f"duplicate tensor key {key!r} in {directory}")
                tensor_slice = handle.get_slice(key)
                discovered[key] = (
                    path,
                    tuple(tensor_slice.get_shape()),
                    str(tensor_slice.get_dtype()),
                )
                filenames[key] = path.name

    index_path = directory / _INDEX_NAME
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read index {index_path}: {exc}") from exc
        if not isinstance(index, dict):
            raise ValueError(f"index must be a JSON object: {index_path}")
        validate_index(index, filenames)
    elif len(files) > 1:
        if not all(_SHARD_NAME.match(path.name) for path in files):
            raise ValueError(
                f"multiple safetensors files require {_INDEX_NAME} in {directory}"
            )

    return discovered


def _metadata(info: tuple[Path, tuple[int, ...], str]) -> dict[str, Any]:
    return {"shape": list(info[1]), "dtype": info[2]}


def compare_checkpoints(
    source: str | Path,
    target: str | Path,
    *,
    report_dir: str | Path,
    rank: int | None = None,
    world_size: int | None = None,
    allow_fp32_to_bf16: bool = False,
) -> dict[str, Any]:
    """Compare checkpoint tensors assigned to this rank and write its JSON report."""
    rank = int(os.environ.get("RANK", "0")) if rank is None else rank
    world_size = (
        int(os.environ.get("WORLD_SIZE", "1")) if world_size is None else world_size
    )
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError(
            f"invalid rank/world_size: rank={rank}, world_size={world_size}"
        )

    safe_open = _safetensors_module()
    source_map = _scan_checkpoint(Path(source), safe_open)
    target_map = _scan_checkpoint(Path(target), safe_open)
    source_keys, target_keys = set(source_map), set(target_map)
    shared_keys = sorted(source_keys & target_keys)
    assigned = [
        key for index, key in enumerate(shared_keys) if index % world_size == rank
    ]

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to compare checkpoint tensors") from exc
    torch.set_num_threads(min(torch.get_num_threads(), 8))

    report: dict[str, Any] = {
        "source": str(source),
        "target": str(target),
        "rank": rank,
        "world_size": world_size,
        "allow_fp32_to_bf16": allow_fp32_to_bf16,
        "global_key_counts": {"source": len(source_keys), "target": len(target_keys)},
        "missing_keys": sorted(source_keys - target_keys),
        "unexpected_keys": sorted(target_keys - source_keys),
        "rank_key_count": len(assigned),
        "tensor_counts": {"matched": 0, "cast_matches": 0, "mismatches": 0},
        "matched": [],
        "cast_matches": [],
        "mismatches": [],
    }

    # Reuse one file handle while consecutive sorted keys belong to the same shard.
    handles: dict[Path, Any] = {}
    handle_stack = ExitStack()
    for key in assigned:
        src_info, dst_info = source_map[key], target_map[key]
        src_meta, dst_meta = _metadata(src_info), _metadata(dst_info)
        cast_match = (
            allow_fp32_to_bf16
            and src_info[2] == "F32"
            and dst_info[2] == "BF16"
            and src_info[1] == dst_info[1]
        )
        if src_info[1] != dst_info[1] or (
            src_info[2] != dst_info[2] and not cast_match
        ):
            report["mismatches"].append(
                {
                    "key": key,
                    "reason": "metadata",
                    "source": src_meta,
                    "target": dst_meta,
                }
            )
            continue

        for path in (src_info[0], dst_info[0]):
            if path not in handles:
                handles[path] = handle_stack.enter_context(
                    safe_open(str(path), framework="pt", device="cpu")
                )
        src_tensor = handles[src_info[0]].get_tensor(key)
        dst_tensor = handles[dst_info[0]].get_tensor(key)
        if cast_match:
            equal = torch.equal(src_tensor.to(dtype=torch.bfloat16), dst_tensor)
            count_name = "cast_matches"
            record_name = "cast_matches"
        else:
            equal = torch.equal(src_tensor, dst_tensor)
            count_name = "matched"
            record_name = "matched"
        if equal:
            report["tensor_counts"][count_name] += 1
            report[record_name].append(
                {
                    "key": key,
                    "shape": src_meta["shape"],
                    "source_dtype": src_meta["dtype"],
                    "target_dtype": dst_meta["dtype"],
                }
            )
        else:
            report["mismatches"].append(
                {"key": key, "reason": "values", "source": src_meta, "target": dst_meta}
            )
        del src_tensor, dst_tensor

    handle_stack.close()
    report["tensor_counts"]["mismatches"] = len(report["mismatches"])
    report_path = Path(report_dir) / f"rank{rank:04d}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="source HF checkpoint directory")
    parser.add_argument("target", help="target HF checkpoint directory")
    parser.add_argument(
        "--report-dir", required=True, help="directory for rankNNNN.json"
    )
    parser.add_argument(
        "--allow-fp32-to-bf16",
        action="store_true",
        help="allow exact source-FP32 cast-to-BF16 equality (reported separately)",
    )
    args = parser.parse_args(argv)
    try:
        report = compare_checkpoints(
            args.source,
            args.target,
            report_dir=args.report_dir,
            allow_fp32_to_bf16=args.allow_fp32_to_bf16,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"comparison failed: {exc}", file=sys.stderr)
        return 2
    counts = report["tensor_counts"]
    print(
        f"rank {report['rank']}/{report['world_size']}: "
        f"matched={counts['matched']} cast_matches={counts['cast_matches']} "
        f"mismatches={counts['mismatches']} missing={len(report['missing_keys'])} "
        f"unexpected={len(report['unexpected_keys'])}"
    )
    return int(
        bool(
            report["missing_keys"] or report["unexpected_keys"] or counts["mismatches"]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
