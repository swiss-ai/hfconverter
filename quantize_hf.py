"""Safe local-checkpoint wrapper around llm-compressor's model-free FP8 PTQ."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

LOG = logging.getLogger("checkpoint_conversion")
EXPERT_TARGET = (
    r"re:^model\.layers\.\d+\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"
)


def _read_json(path: Path) -> dict[str, Any]:
    def unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {path}: {key}")
            result[key] = value

        return result

    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_fields)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object: {path}")

    return data


def _source_shards(source: Path) -> dict[str, set[str]]:
    from safetensors import safe_open

    index = source / "model.safetensors.index.json"
    if not index.exists():
        weights = source / "model.safetensors"
        if not weights.is_file() or list(source.glob("*.safetensors")) != [weights]:
            raise ValueError(
                "Source needs an HF safetensors index or model.safetensors"
            )
        with safe_open(weights, framework="pt", device="cpu") as reader:
            return {weights.name: set(reader.keys())}

    weight_map = _read_json(index).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("HF index needs a non-empty weight_map")

    shards: dict[str, set[str]] = {}
    for key, filename in weight_map.items():
        if not isinstance(key, str) or not key or not isinstance(filename, str):
            raise ValueError("Invalid key or shard filename in HF index")
        if Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise ValueError(f"Unsafe or unsupported shard filename: {filename}")
        shards.setdefault(filename, set()).add(key)

    if {p.name for p in source.glob("*.safetensors")} != set(shards):
        raise ValueError("HF index and source safetensors files do not match")

    return shards


def _validate_selection(
    tensors: dict[str, Any], selected: list[tuple[str, str]]
) -> None:
    from compressed_tensors.utils import match_quantizable_tensors
    from llmcompressor.entrypoints.model_free.microscale import get_fused_names

    selected_keys = {key for _, key in selected}
    groups = [set(group.values()) for group in get_fused_names(tensors)]
    routed_layers: dict[str, set[str]] = {}

    for module, key in match_quantizable_tensors(tensors, [], [EXPERT_TARGET]):
        layer = module.partition(".experts.")[0]
        routed_layers.setdefault(layer, set()).add(key)

    groups.extend(routed_layers.values())

    for group in groups:
        if selected_keys.intersection(group) and not group.issubset(selected_keys):
            raise ValueError(
                "Select or ignore entire fused projection groups and routed-expert "
                f"layers for vLLM: {sorted(group)}"
            )


def quantize_checkpoint(
    source: Path,
    output: Path,
    *,
    include: list[str] | None = None,
    ignore: list[str] | None = None,
    device: list[str] | None = None,
    max_workers: int = 1,
) -> None:
    """Use FP8_BLOCK on routed experts plus explicitly included module targets."""
    source = source.expanduser()
    output = output.expanduser()
    source_path = source.resolve()
    output_path = output.resolve()

    if source.is_symlink() or not source.is_dir():
        raise ValueError("Source must be a local HF directory, not a symlink")
    if output_path.is_relative_to(source_path) or source_path.is_relative_to(
        output_path
    ):
        raise ValueError("Source and output directories must not overlap")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output must be a new directory: {output}")
    if any(p.is_symlink() for p in source.rglob("*")):
        raise ValueError("Source checkpoint must not contain symlinks")

    config = _read_json(source / "config.json")
    if config.get("quantization_config") is not None:
        raise ValueError("Source already has a quantization_config")
    if (source / "QUANTIZATION_COMPLETE").exists():
        raise ValueError("Source already has a quantization completion marker")

    targets = [EXPERT_TARGET, *(include or [])]
    ignore = ignore or []

    for pattern in [*targets, *ignore]:
        if not pattern:
            raise ValueError("Quantization targets and ignores must not be empty")
        if pattern.startswith("re:"):
            re.compile(pattern[3:])

    if max_workers < 1:
        raise ValueError("max_workers must be positive")

    try:
        import torch
        from compressed_tensors.quantization import preset_name_to_scheme
        from compressed_tensors.utils import match_quantizable_tensors
        from llmcompressor import model_free_ptq
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError(
            "Quantization requires llmcompressor==0.15.0 and its dependencies; "
            "install requirements-quantization.txt in the selected runtime"
        ) from error

    shards = _source_shards(source)

    # Meta tensors let the library match targets without loading weights.
    tensors = {}
    dtypes = {}
    for filename, expected in sorted(shards.items()):
        with safe_open(source / filename, framework="pt", device="cpu") as reader:
            if set(reader.keys()) != expected:
                raise ValueError(f"Index/header key mismatch in {filename}")
            for key in expected:
                view = reader.get_slice(key)
                tensors[key] = torch.empty(view.get_shape(), device="meta")
                dtypes[key] = view.get_dtype()

    selected = list(match_quantizable_tensors(tensors, ignore, targets))
    if not selected:
        raise ValueError("No weights match the quantization targets after ignores")
    if any(dtypes[key] not in {"BF16", "F16", "F32"} for _, key in selected):
        raise ValueError("Selected weights must be BF16, FP16 or FP32")

    _validate_selection(tensors, selected)
    del tensors

    LOG.info(
        "FP8_BLOCK: %d matrices; source=%s output=%s targets=%s ignore=%s workers=%d",
        len(selected),
        source,
        output,
        targets,
        ignore,
        max_workers,
    )
    scheme = preset_name_to_scheme("FP8_BLOCK", targets=targets)

    output.mkdir(parents=True)

    try:
        model_free_ptq(
            model_stub=str(source),
            save_directory=str(output),
            scheme=scheme,
            ignore=ignore,
            device=device,
            max_workers=max_workers,
        )
    finally:
        # Copied export markers must not certify this quantization run.
        for marker in output.glob("*_EXPORT_COMPLETE"):
            marker.unlink()
        (output / "QUANTIZATION_COMPLETE").unlink(missing_ok=True)

    (output / "QUANTIZATION_COMPLETE").write_text(
        f"llm-compressor FP8_BLOCK: {len(selected)} matrices; "
        f"targets={targets!r}; ignore={ignore!r}\n",
        encoding="utf-8",
    )

    LOG.info("FP8 quantization complete: %s", output)
