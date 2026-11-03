"""Convert checkpoints between Megatron and Hugging Face with Megatron Bridge.

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
import re
import sys
from pathlib import Path
from typing import Any

LOG = logging.getLogger("checkpoint_conversion")
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}


def _positive_int(value: str) -> int:
    size = int(value)
    if size <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return size


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="direction", required=True)
    export_parser = commands.add_parser("export", help="Export Megatron weights to HF")
    export_parser.add_argument(
        "checkpoint", type=Path, help="Megatron iteration directory"
    )
    export_parser.add_argument(
        "reference", type=Path, help="Local HF reference directory"
    )
    export_parser.add_argument(
        "output", type=Path, help="New/empty HF output directory"
    )
    export_parser.add_argument(
        "--pipeline-layout", help="Explicit Megatron pipeline layout"
    )
    export_parser.add_argument(
        "--expected-hf-index", type=Path, help="HF index JSON for strict key validation"
    )
    export_parser.add_argument(
        "--expected-hf-keys", type=Path, help="JSON list of expected HF keys"
    )

    import_parser = commands.add_parser("import", help="Import HF weights to Megatron")
    import_parser.add_argument(
        "reference", help="Local HF model directory or Hugging Face model ID"
    )
    import_parser.add_argument(
        "output", type=Path, help="New/empty Megatron checkpoint directory"
    )

    import_parser.add_argument("--hf-revision", help="Hugging Face model revision")
    import_parser.add_argument("--pipeline-layout", help="Explicit Megatron pipeline layout")

    for subparser in (export_parser, import_parser):
        subparser.add_argument("--tp", type=int, default=1)
        subparser.add_argument("--pp", type=int, default=1)
        subparser.add_argument("--ep", type=int, default=1)
        subparser.add_argument("--etp", type=int, default=1)
        subparser.add_argument(
            "--dtype", choices=("bf16", "fp16", "fp32"), default="bf16"
        )
        subparser.add_argument("--trust-remote-code", action="store_true")
        subparser.add_argument("--megatron-root", type=Path)
        subparser.add_argument("--bridge-root", type=Path)
    export_parser.add_argument("--vpp", type=int)
    export_parser.add_argument(
        "--max-shard-size-gib",
        type=_positive_int,
        default=4,
        help="Shard target in GiB for config-only streaming export (default: 4). "
        "Not a total-memory limit; a single tensor may exceed it.",
    )
    export_parser.add_argument(
        "--reference-mode", choices=("config-only", "weighted"), default="config-only"
    )
    for subparser in (export_parser, import_parser):
        subparser.add_argument(
            "--model-override",
            action="append",
            type=_parse_model_override,
            default=[],
            metavar="NAME=JSON",
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
    if any(size < 1 for size in sizes) or (
        getattr(args, "vpp", None) is not None and args.vpp < 1
    ):
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


def _apply_model_overrides(provider: Any, overrides: dict[str, Any]) -> None:
    topology_fields = {
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "pipeline_model_parallel_layout",
        "virtual_pipeline_model_parallel_size",
        "context_parallel_size",
        "sequence_parallel",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
        "pipeline_dtype",
        "params_dtype",
    }
    conflicts = overrides.keys() & topology_fields
    if conflicts:
        raise ValueError(
            "Use the topology/dtype options instead of overriding: "
            + ", ".join(sorted(conflicts))
        )
    unknown = sorted(name for name in overrides if not hasattr(provider, name))
    if unknown:
        raise ValueError(
            "Model override fields are not supported by this provider: "
            + ", ".join(unknown)
        )
    for name, value in overrides.items():
        setattr(provider, name, value)


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


def _run_import(args: argparse.Namespace) -> None:
    _validate_topology(args)
    reference = Path(args.reference).expanduser()
    if reference.is_dir():
        if not (reference / "config.json").is_file():
            raise ValueError(f"HF input has no config.json: {reference}")
        source, output = reference.resolve(), args.output.resolve()
        if source.is_relative_to(output) or output.is_relative_to(source):
            raise ValueError(f"Input and output must not overlap: {source}, {output}")
        args.reference = str(reference)
    elif not re.fullmatch(r"[\w.-]+/[\w.-]+", args.reference):
        raise ValueError(
            f"HF input must be a local directory or owner/model ID: {args.reference}"
        )
    if args.output.exists() and (
        not args.output.is_dir() or any(args.output.iterdir())
    ):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
    bridge_root, megatron_root = _check_runtime_roots(args)
    from gpu_backend import import_checkpoint

    torch_dtype = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[args.dtype]
    LOG.info(
        "Import reference=%s output=%s dtype=%s bridge=%s megatron=%s",
        args.reference,
        args.output,
        torch_dtype,
        bridge_root,
        megatron_root,
    )
    import_options = {
        "hf_model": str(args.reference),
        "hf_revision": args.hf_revision,
        "megatron_path": str(args.output),
        "tp": args.tp,
        "pp": args.pp,
        "ep": args.ep,
        "etp": args.etp,
        "torch_dtype": torch_dtype,
        "trust_remote_code": args.trust_remote_code,
        "low_memory_save": True,
        "distributed_timeout_minutes": 55,
        "overwrite": False,
    }
    if args.pipeline_layout is None and not args.model_override:
        import_checkpoint(**import_options)
        return

    # gpu_backend.import_checkpoint does not expose a layout parameter. For an
    # explicit layout, use its generic Bridge/provider APIs so the layout is
    # applied before finalize() performs the layer-to-stage validation.
    import torch
    from gpu_backend import (
        _configure_model_config,
        _configure_model_provider,
        _ensure_distributed_initialized,
        _hf_tokenizer_kwargs,
        _prepare_distributed_output,
        _uses_model_builder,
    )
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.hf_pretrained.utils import is_safe_repo

    _ensure_distributed_initialized(55)
    _prepare_distributed_output(str(args.output), overwrite=False, source_paths=[str(args.reference)])
    dtype = getattr(torch, torch_dtype)
    revision_kwargs = {"revision": args.hf_revision} if args.hf_revision is not None else {}
    bridge = AutoBridge.from_hf_pretrained(
        str(args.reference),
        trust_remote_code=is_safe_repo(
            trust_remote_code=args.trust_remote_code, hf_path=str(args.reference)
        ),
        torch_dtype=dtype,
        **revision_kwargs,
    )
    if _uses_model_builder(bridge):
        model_config = bridge.get_model_config()
        _configure_model_config(
            model_config, tp=args.tp, pp=args.pp, ep=args.ep, etp=args.etp, dtype=dtype
        )
        if args.pipeline_layout is not None:
            model_config.transformer.pipeline_model_parallel_layout = args.pipeline_layout
        _apply_model_overrides(model_config.transformer, dict(args.model_override))
        model = bridge.get_model(
            model_config, wrap_with_ddp=False, mixed_precision_wrapper=None
        )
    else:
        provider = bridge.to_megatron_provider(load_weights=True)
        _configure_model_provider(
            provider, tp=args.tp, pp=args.pp, ep=args.ep, etp=args.etp, dtype=dtype
        )
        if args.pipeline_layout is not None:
            provider.pipeline_model_parallel_layout = args.pipeline_layout
        _apply_model_overrides(provider, dict(args.model_override))
        provider.finalize()
        provider.initialize_model_parallel(seed=0, create_gloo_process_groups=False)
        model = provider.provide_distributed_model(wrap_with_ddp=False)
    bridge.save_megatron_model(
        model,
        str(args.output),
        hf_tokenizer_path=str(args.reference),
        hf_tokenizer_kwargs=_hf_tokenizer_kwargs(
            bridge, trust_remote_code=args.trust_remote_code
        ),
        low_memory_save=True,
    )


def _expected_keys(
    path: Path | None, expected_keys_fn: Any = None, bridge: Any = None
) -> set[str] | None:
    if path is not None:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "weight_map" in data:
            keys = set(data["weight_map"])
        elif isinstance(data, list):
            keys = set(data)
        else:
            raise ValueError(
                "Expected-key JSON must be an HF index or a JSON string list"
            )
        if not keys or not all(isinstance(key, str) for key in keys):
            raise ValueError("Expected-key manifest must contain non-empty string keys")
        return keys
    if expected_keys_fn is not None:
        return set(expected_keys_fn(bridge))
    return None


# TODO (@andresnowak): Will try to make a pr to megatron bridge upstream.
def _save_bounded_stream(
    generator: Any,
    output: Path,
    expected: set[str] | None,
    rank: int,
    max_shard_bytes: int,
) -> None:
    from safetensors.torch import save_file
    shard: dict[str, Any] = {}
    shard_bytes = 0
    total = 0
    shard_id = 0
    seen: set[str] = set()
    weight_map: dict[str, str] = {}

    def flush() -> None:
        nonlocal shard_bytes, total, shard_id
        if not shard:
            return
        filename = f"model-{shard_id + 1:05d}.safetensors"
        save_file(shard, output / filename)
        weight_map.update({key: filename for key in shard})
        total += shard_bytes
        shard.clear()
        shard_bytes = 0
        shard_id += 1

    for key, tensor in generator:
        if key in seen:
            raise ValueError(f"Duplicate HF tensor from Bridge: {key}")
        seen.add(key)
        if expected is not None and key not in expected:
            raise KeyError(f"Unexpected HF tensor from Bridge: {key}")
        if rank == 0:
            item = tensor.detach().to(device="cpu").contiguous().clone()
            size = item.numel() * item.element_size()
            if shard and shard_bytes + size > max_shard_bytes:
                flush()
            shard[key] = item
            shard_bytes += size
    if expected is not None and seen != expected:
        missing = sorted(expected - seen)[:20]
        unexpected = sorted(seen - expected)[:20]
        raise RuntimeError(
            f"HF keys mismatch: missing={missing}, unexpected={unexpected}"
        )
    if rank == 0:
        flush()
        if not weight_map:
            raise RuntimeError("Bridge yielded no HF weights")
        (output / "model.safetensors.index.json").write_text(
            json.dumps(
                {"metadata": {"total_size": total}, "weight_map": weight_map},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def run(args: argparse.Namespace, *, expected_keys_fn: Any = None) -> None:
    if args.direction == "import":
        _run_import(args)
        return
    source_kind = _validate_paths(
        args.checkpoint,
        args.reference,
        args.output,
    )
    _validate_reference(args.reference, args.reference_mode)
    world_size = _validate_topology(args)
    if args.expected_hf_index and args.expected_hf_keys:
        raise ValueError("Choose only one of --expected-hf-index or --expected-hf-keys")

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
        "Source=%s kind=%s reference=%s mode=%s output=%s dtype=%s world_size=%d",
        args.checkpoint.resolve(),
        source_kind,
        args.reference.resolve(),
        args.reference_mode,
        args.output.resolve(),
        args.dtype,
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
        "pipeline_model_parallel_layout": args.pipeline_layout,
        "virtual_pipeline_model_parallel_size": args.vpp,
        "context_parallel_size": 1,
        "sequence_parallel": args.tp > 1,
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
    _apply_model_overrides(provider, model_overrides)
    if args.pipeline_layout is None:
        _maybe_restore_pipeline_layout(bridge, provider, str(args.checkpoint), args.pp)
    else:
        provider.pipeline_model_parallel_layout = args.pipeline_layout
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

    if args.reference_mode == "weighted":
        bridge.save_hf_pretrained(
            model,
            args.output,
            show_progress=rank == 0,
            strict=True,
            weight_dtype=None,
            distributed_save=True,
        )
    else:
        model_bridge = bridge._model_bridge
        tasks = model_bridge.build_conversion_tasks(
            bridge.hf_pretrained, model, weight_dtype=None
        )
        expected_path = args.expected_hf_index or args.expected_hf_keys
        expected = _expected_keys(expected_path, expected_keys_fn, bridge)
        if rank == 0:
            from megatron.bridge.models.hf_pretrained.causal_lm import (
                PreTrainedCausalLM,
            )

            artifact_source = PreTrainedCausalLM.from_pretrained(
                str(args.reference), trust_remote_code=args.trust_remote_code
            )
            artifact_source.config = bridge.hf_pretrained
            additional_files = (
                getattr(model_bridge, "ADDITIONAL_FILE_PATTERNS", None) or None
            )
            artifact_source.save_artifacts(
                args.output,
                original_source_path=args.reference,
                additional_files=additional_files,
            )
        _distributed_barrier(torch)
        generator = model_bridge.stream_weights_megatron_to_hf(
            model,
            bridge.hf_pretrained,
            cpu=True,
            show_progress=rank == 0,
            conversion_tasks=tasks,
            merge_adapter_weights=True,
        )
        _save_bounded_stream(
            generator, args.output, expected, rank, args.max_shard_size_gib * 1024**3
        )

    _distributed_barrier(torch)

    if rank == 0:
        marker = f"{args.reference_mode.upper().replace('-', '_')}_EXPORT_COMPLETE"
        (args.output / marker).write_text(
            f"{args.reference_mode} export completed\n", encoding="utf-8"
        )
        LOG.info("Export complete: %s (%s)", args.output, marker)

    _distributed_barrier(torch)


def main(argv: list[str] | None = None, *, expected_keys_fn: Any = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args = _parser().parse_args(argv)
    run(args, expected_keys_fn=expected_keys_fn)


if __name__ == "__main__":
    main()
