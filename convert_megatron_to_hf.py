"""Export Megatron checkpoints with Megatron-Bridge's strict HF conversion APIs.

Megatron Bridge conversion idea: Yixuan Xu (@Alvorecer721).
Export flow adapted from Andres Nowak / Swiss AI's Apertus2 Chonk experiments:
https://github.com/swiss-ai/apertus2_experiments/tree/eval-kda-120b-chonk/conversions/chonk.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import inspect
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

LOG = logging.getLogger("megatron_to_hf")
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Megatron iteration directory",
    )
    parser.add_argument(
        "reference",
        type=Path,
        help="Local HF reference directory",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="New/empty HF output directory",
    )

    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallel size (default: 1)",
    )
    parser.add_argument(
        "--pp",
        type=int,
        default=1,
        help="Pipeline parallel size (default: 1)",
    )
    parser.add_argument(
        "--ep",
        type=int,
        default=1,
        help="Expert parallel size (default: 1)",
    )
    parser.add_argument(
        "--etp",
        type=int,
        default=1,
        help="Expert tensor parallel size (default: 1)",
    )
    parser.add_argument(
        "--vpp",
        type=int,
        help="Virtual pipeline parallel size",
    )

    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
    )
    parser.add_argument(
        "--reference-mode",
        choices=("config-only", "weighted"),
        default="config-only",
    )
    parser.add_argument(
        "--model-override",
        action="append",
        type=_parse_model_override,
        default=[],
        metavar="NAME=JSON",
        help="Override a Megatron config field; repeat for multiple fields",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow trusted reference code execution",
    )
    parser.add_argument(
        "--megatron-root",
        type=Path,
        help="Expected Megatron-LM checkout (or MEGATRON_ROOT)",
    )
    parser.add_argument(
        "--bridge-root",
        type=Path,
        help="Expected Megatron-Bridge checkout (or BRIDGE_ROOT)",
    )

    return parser


def _resolved_root(value: Path | None, variable: str) -> Path:
    candidate = value or os.environ.get(variable)
    if not candidate:
        raise ValueError(
            f"Supply --{variable.lower().replace('_', '-')} or set {variable}"
        )

    root = Path(candidate).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"{variable} is not a directory: {root}")

    return root


def _within(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def _check_runtime_roots(args: argparse.Namespace) -> tuple[Path, Path]:
    """Reject ambiguous imports that could silently use another Bridge checkout."""
    bridge_root = _resolved_root(args.bridge_root, "BRIDGE_ROOT")
    megatron_root = _resolved_root(args.megatron_root, "MEGATRON_ROOT")
    conversion_path = bridge_root / "scripts" / "conversion"
    if not (conversion_path / "gpu_backend.py").is_file():
        raise ValueError(f"Bridge conversion helpers not found under {bridge_root}")
    for source_path in (conversion_path, bridge_root / "src", megatron_root):
        if str(source_path) not in sys.path:
            sys.path.insert(0, str(source_path))
    expected_modules = {
        "megatron.core": megatron_root / "megatron" / "core",
        "megatron.bridge": bridge_root / "src" / "megatron" / "bridge",
        "gpu_backend": conversion_path,
    }
    for name, expected_root in expected_modules.items():
        spec = importlib.util.find_spec(name)
        if (
            spec is None
            or not spec.origin
            or not _within(Path(spec.origin), expected_root)
        ):
            origin = spec.origin if spec else None
            raise RuntimeError(
                f"{name} resolved outside expected root {expected_root}: {origin}"
            )
        LOG.info("Runtime source %s=%s", name, spec.origin)
    return bridge_root, megatron_root


def _validate_reference(reference: Path, mode: str) -> None:
    if (
        reference.is_symlink()
        or not reference.is_dir()
        or not (reference / "config.json").is_file()
    ):
        raise ValueError(
            f"Reference must be a local HF directory with config.json: {reference}"
        )
    if mode == "config-only":
        for entry in reference.rglob("*"):
            if entry.is_symlink() or entry.suffix.lower() in WEIGHT_SUFFIXES:
                raise ValueError(
                    f"Config-only reference contains a weight or symlink: {entry}"
                )
        if (reference / "model.safetensors.index.json").exists() or (
            reference / "pytorch_model.bin.index.json"
        ).exists():
            raise ValueError(
                f"Config-only reference contains a weights index: {reference}"
            )
        return

    index_path = reference / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ValueError(
            f"Weighted reference needs model.safetensors.index.json: {index_path}"
        )
    try:
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        shards = set(weight_map.values())
    except (OSError, json.JSONDecodeError, KeyError, AttributeError, TypeError) as exc:
        raise ValueError(f"Invalid safetensors index: {index_path}") from exc
    if not shards:
        raise ValueError(
            f"Weighted reference has incomplete indexed shards: {reference}"
        )
    for shard in shards:
        if not isinstance(shard, str):
            raise TypeError(f"Weighted reference has an invalid shard path: {shard!r}")
        shard_path = (reference / shard).resolve()
        if not _within(shard_path, reference) or not shard_path.is_file():
            raise ValueError(
                f"Weighted reference has incomplete or unsafe indexed shards: {shard}"
            )


def _validate_paths(checkpoint: Path, reference: Path, output: Path) -> str:
    checkpoint = checkpoint.resolve()
    reference = reference.resolve()
    output = output.resolve()
    native = (checkpoint / "common.pt").is_file()
    bridge = (checkpoint / "run_config.yaml").is_file()
    if not native and not bridge:
        raise ValueError(
            "Checkpoint iteration must contain common.pt (legacy) or run_config.yaml (Bridge)"
        )
    for source in (checkpoint, reference):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError(
                f"Output and source paths must not overlap: {output} and {source}"
            )
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    return "bridge" if bridge else "native"


def _validate_topology(args: argparse.Namespace) -> int:
    sizes = (args.tp, args.pp, args.ep, args.etp)
    if any(size < 1 for size in sizes) or (args.vpp is not None and args.vpp < 1):
        raise ValueError("TP, PP, EP, ETP, and optional VPP must be positive")
    # Dense and expert parallelism share ranks.
    world_size = args.pp * math.lcm(args.tp, args.etp * args.ep)
    launched = os.environ.get("WORLD_SIZE") or os.environ.get("SLURM_NTASKS")
    if launched is not None and int(launched) != world_size:
        raise ValueError(
            f"Launch world size {launched} does not match PP*lcm(TP, ETP*EP)={world_size}"
        )
    return world_size


def _parse_model_override(value: str) -> tuple[str, Any]:
    name, separator, json_value = value.partition("=")

    if not separator or not name.isidentifier():
        raise argparse.ArgumentTypeError("Use NAME=JSON for a model override")

    try:
        parsed_value = json.loads(json_value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(
            f"Model override {name} must have a valid JSON value"
        ) from error

    return name, parsed_value


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown (source checkout)"


def _checkpoint_bridge(
    auto_bridge: Any,
    args: argparse.Namespace,
    overrides: dict[str, Any],
) -> Any:
    parameters = inspect.signature(auto_bridge.from_auto_config).parameters
    supports_overrides = "mp_overrides" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if not supports_overrides and not (args.checkpoint / "run_config.yaml").is_file():
        raise ValueError(
            "This Bridge version requires run_config.yaml; native common.pt needs the Swiss AI fork"
        )
    options: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if supports_overrides:
        options["mp_overrides"] = overrides

    return auto_bridge.from_auto_config(
        str(args.checkpoint),
        str(args.reference),
        **options,
    )


def _distributed_barrier(torch: Any) -> None:
    if torch.distributed.get_backend() == "gloo":
        torch.distributed.barrier()
    else:
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])


def run(args: argparse.Namespace) -> None:
    source_kind = _validate_paths(
        args.checkpoint,
        args.reference,
        args.output,
    )
    _validate_reference(args.reference, args.reference_mode)
    world_size = _validate_topology(args)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    bridge_root, megatron_root = _check_runtime_roots(args)

    import torch
    from gpu_backend import (
        _configure_model_provider,
        _ensure_distributed_initialized,
        _maybe_restore_pipeline_layout,
    )
    from megatron.bridge import AutoBridge
    from transformers import PretrainedConfig

    dtypes = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtypes[args.dtype]

    LOG.info(
        "Runtime python=%s bridge=%s bridge_version=%s megatron=%s torch=%s transformers=%s",
        sys.executable,
        bridge_root,
        _package_version("megatron-bridge"),
        megatron_root,
        torch.__version__,
        _package_version("transformers"),
    )
    LOG.info(
        "Source=%s kind=%s reference=%s mode=%s output=%s world_size=%d",
        args.checkpoint.resolve(),
        source_kind,
        args.reference.resolve(),
        args.reference_mode,
        args.output.resolve(),
        world_size,
    )

    _ensure_distributed_initialized(55)
    rank = torch.distributed.get_rank()
    if rank == 0:
        try:
            args.output.mkdir(parents=True, exist_ok=False)
            status: list[Any] = [None]
        except FileExistsError:
            if args.output.is_dir() and not any(args.output.iterdir()):
                status = [None]
            else:
                status = [f"Refusing to overwrite non-empty output: {args.output}"]
        except OSError as exc:
            status = [str(exc)]
    else:
        status = [None]
    torch.distributed.broadcast_object_list(status, src=0)
    if status[0]:
        raise FileExistsError(status[0])
    _distributed_barrier(torch)

    mp_overrides: dict[str, Any] = {
        "tensor_model_parallel_size": args.tp,
        "pipeline_model_parallel_size": args.pp,
        "pipeline_model_parallel_layout": None,
        "virtual_pipeline_model_parallel_size": args.vpp,
        "context_parallel_size": 1,
        "expert_model_parallel_size": args.ep,
        "expert_tensor_parallel_size": args.etp,
        "pipeline_dtype": dtype,
        "params_dtype": dtype,
    }
    model_overrides = dict(args.model_override)
    conflicting_fields = model_overrides.keys() & mp_overrides.keys()

    if conflicting_fields:
        raise ValueError(
            "Use the topology/dtype options instead of overriding: "
            + ", ".join(sorted(conflicting_fields))
        )

    mp_overrides.update(model_overrides)

    checkpoint_bridge = _checkpoint_bridge(AutoBridge, args, mp_overrides)

    if args.reference_mode == "config-only":
        bridge = checkpoint_bridge
        if not isinstance(bridge.hf_pretrained, PretrainedConfig):
            raise TypeError("Expected config-only HF reference")
    else:
        bridge = AutoBridge.from_hf_pretrained(
            str(args.reference),
            trust_remote_code=args.trust_remote_code,
            torch_dtype=dtype,
        )
        if isinstance(bridge.hf_pretrained, PretrainedConfig):
            raise TypeError(
                "Expected weighted HF reference with a streaming state source"
            )
        bridge.hf_pretrained.config = checkpoint_bridge.hf_pretrained

    provider = bridge.to_megatron_provider(load_weights=False)
    _configure_model_provider(
        provider,
        tp=args.tp,
        pp=args.pp,
        ep=args.ep,
        etp=args.etp,
        dtype=dtype,
    )
    provider.context_parallel_size = 1
    _maybe_restore_pipeline_layout(
        bridge,
        provider,
        str(args.checkpoint),
        args.pp,
    )
    if args.vpp is not None:
        provider.virtual_pipeline_model_parallel_size = args.vpp
    provider.finalize()
    mp_overrides["pipeline_model_parallel_layout"] = (
        provider.pipeline_model_parallel_layout
    )
    mp_overrides["virtual_pipeline_model_parallel_size"] = (
        provider.virtual_pipeline_model_parallel_size
    )
    provider.initialize_model_parallel(seed=0)

    model = bridge.load_megatron_model(
        str(args.checkpoint),
        mp_overrides=mp_overrides,
        wrap_with_ddp=False,
    )

    save_args: dict[str, bool | Path] = {
        "distributed_save": args.reference_mode == "weighted"
    }
    if args.reference_mode == "config-only":
        save_args["source_path"] = args.reference

    bridge.save_hf_pretrained(
        model,
        args.output,
        show_progress=rank == 0,
        strict=True,
        weight_dtype=dtype,
        **save_args,
    )

    _distributed_barrier(torch)

    if rank == 0:
        marker = f"{args.reference_mode.upper().replace('-', '_')}_EXPORT_COMPLETE"
        (args.output / marker).write_text(
            f"{args.reference_mode} export completed\n", encoding="utf-8"
        )
        LOG.info("Export complete: %s (%s)", args.output, marker)

    _distributed_barrier(torch)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args = _parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
