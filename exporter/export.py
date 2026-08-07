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
"""Command line and Python entry points for checkpoint conversion.

The exporter follows four visible stages:

1. Read Megatron's checkpoint arguments and tensor metadata (no tensor data yet).
2. Map each Megatron source name to the Hugging Face names and shapes it will create.
3. Load, transform, and write one Hugging Face shard at a time.
4. Write the Hugging Face config/module files and verify the finished model.

Stage 3 deliberately streams shards. A full Apertus checkpoint and its optimizer state are too
large to materialize together, even though loading everything would make the code shorter.
"""

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

# export must work when the repo root is not the cwd (see writer.REPO_ROOT for the module
# imports themselves; this makes `exporter` submodule imports and by-path runs consistent).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from exporter import __version__, config_from_args, mapping, output_claim, reader, writer

logger = logging.getLogger("exporter")


@dataclass
class ExportSummary:
    output_dir: str
    num_tensors: int
    total_bytes: int
    shard_files: list[str] = field(default_factory=list)
    dropped_optimizer_key_count: int = 0
    synthesized_keys: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _SourceResidency:
    """When each Megatron tensor must enter and leave memory.

    One fused Megatron tensor can create several Hugging Face tensors. Those outputs can cross a
    shard boundary, so the source must remain loaded until its final output has been written.
    ``load_at_shard[i]`` and ``release_after_shard[i]`` make that lifetime explicit.
    """

    expected_shape: dict[str, tuple[int, ...]]
    load_at_shard: tuple[tuple[str, ...], ...]
    release_after_shard: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class _PreparedExport:
    """Everything that can be decided from checkpoint metadata before loading tensor bytes."""

    common_state: dict[str, Any]
    derived_config: config_from_args.DerivedConfig
    checks: list[str]
    mapping_plan: mapping.Plan
    tensor_specs: list[mapping.HFTensorSpec]
    spec_by_hf_key: dict[str, mapping.HFTensorSpec]
    shard_layout: Any
    shard_filenames: list[str]
    source_residency: _SourceResidency
    model_keys: set[str]
    dropped_keys: set[str]
    parameter_dtype: torch.dtype
    checkpoint_tensor_bytes: int


@dataclass(frozen=True)
class _WrittenWeights:
    produced_keys: set[str]
    total_bytes: int


def _validate_paths(
    checkpoint_dir: Path, output_dir: Path, tokenizer_dir: str | Path | None
) -> None:
    """Fail before expensive work or output creation when an input path is unusable."""
    if not checkpoint_dir.is_dir():
        raise ValueError(f"--checkpoint-dir is not a directory: {checkpoint_dir}")
    reader.check_checkpoint(checkpoint_dir)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"--output-dir exists and is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise ValueError(f"--output-dir must not exist or must be empty: {output_dir}")
    if tokenizer_dir is not None and not Path(tokenizer_dir).is_dir():
        raise ValueError(f"--tokenizer-dir is not a directory: {tokenizer_dir}")


def _plan_source_residency(
    tensor_specs: list[mapping.HFTensorSpec], mapping_plan: mapping.Plan, shard_layout: Any
) -> _SourceResidency:
    """Plan source lifetimes from the already-decided Hugging Face shard layout."""
    shard_filenames = list(shard_layout.filename_to_tensors)
    shard_index_for_hf_key = {
        hf_key: shard_index
        for shard_index, filename in enumerate(shard_filenames)
        for hf_key in shard_layout.filename_to_tensors[filename]
    }

    # source -> [first shard, last shard]. Dict insertion order follows the mapping table, which
    # keeps checkpoint reads deterministic and easy to compare with the generated plan.
    lifetime: dict[str, list[int]] = {}
    for spec in tensor_specs:
        if spec.megatron_key is None:  # synthesized zeros have no checkpoint source
            continue
        shard_index = shard_index_for_hf_key[spec.hf_key]
        if spec.megatron_key not in lifetime:
            lifetime[spec.megatron_key] = [shard_index, shard_index]
        else:
            lifetime[spec.megatron_key][1] = shard_index

    load_at_shard: list[list[str]] = [[] for _ in shard_filenames]
    release_after_shard: list[list[str]] = [[] for _ in shard_filenames]
    for megatron_key, (first_shard, last_shard) in lifetime.items():
        load_at_shard[first_shard].append(megatron_key)
        release_after_shard[last_shard].append(megatron_key)

    return _SourceResidency(
        expected_shape={row.megatron_key: row.shape for row in mapping_plan.rows},
        load_at_shard=tuple(tuple(keys) for keys in load_at_shard),
        release_after_shard=tuple(tuple(keys) for keys in release_after_shard),
    )


def _prepare_export(
    checkpoint_dir: Path, max_shard_size: int | str, strict_optimizer: bool
) -> _PreparedExport:
    """Stages 1-2: inspect Megatron metadata and plan every Hugging Face output tensor."""
    checkpoint_args, common_state = reader.load_args(checkpoint_dir)
    derived = config_from_args.derive_config(checkpoint_args)
    checks = list(derived.checks)
    logger.info(
        "1/4 read config: %d checks; layers=%d hidden=%d experts=%d",
        len(checks),
        derived.kwargs["num_hidden_layers"],
        derived.kwargs["hidden_size"],
        derived.kwargs["n_routed_experts"],
    )

    metadata = reader.load_metadata(checkpoint_dir)
    model_keys, dropped_keys = mapping.partition_universe(set(metadata), strict_optimizer)
    mapping.assert_no_unsupported(model_keys, derived.offloaded_experts)

    mapping_plan = mapping.build_plan(
        derived.kwargs, derived.expert_bias_present, derived.offloaded_experts
    )
    mapping.check_consumed(model_keys, mapping_plan)
    checks.extend(mapping.validate_metadata(mapping_plan, metadata))

    parameter_dtype = mapping.params_dtype(mapping_plan, lambda key: metadata[key].dtype)
    tensor_specs = mapping.plan_hf_tensors(mapping_plan, parameter_dtype)
    spec_by_hf_key = {spec.hf_key: spec for spec in tensor_specs}
    shard_layout = writer.plan_shards(tensor_specs, max_shard_size)
    shard_filenames = list(shard_layout.filename_to_tensors)
    source_residency = _plan_source_residency(tensor_specs, mapping_plan, shard_layout)

    checkpoint_tensor_bytes = sum(
        math.prod(tuple(metadata[key].global_shape))
        * torch.empty((), dtype=metadata[key].dtype).element_size()
        for key in model_keys
    )
    logger.info(
        "2/4 map names: %d Megatron tensor(s), %.2f GiB -> %d Hugging Face tensor(s), "
        "%d shard(s); ignored %d training-state tensor(s)",
        len(model_keys),
        checkpoint_tensor_bytes / 2**30,
        len(tensor_specs),
        len(shard_filenames),
        len(dropped_keys),
    )
    return _PreparedExport(
        common_state=common_state,
        derived_config=derived,
        checks=checks,
        mapping_plan=mapping_plan,
        tensor_specs=tensor_specs,
        spec_by_hf_key=spec_by_hf_key,
        shard_layout=shard_layout,
        shard_filenames=shard_filenames,
        source_residency=source_residency,
        model_keys=model_keys,
        dropped_keys=dropped_keys,
        parameter_dtype=parameter_dtype,
        checkpoint_tensor_bytes=checkpoint_tensor_bytes,
    )


def _load_sources_for_shard(
    checkpoint_dir: Path,
    source_keys: tuple[str, ...],
    expected_shapes: dict[str, tuple[int, ...]],
) -> dict[str, torch.Tensor]:
    """Load exactly the Megatron tensors first needed by one output shard."""
    if not source_keys:
        return {}
    loaded = reader.load_tensors(checkpoint_dir, keys=source_keys)
    for megatron_key in source_keys:
        tensor = loaded.get(megatron_key)
        if tensor is None:
            raise ValueError(f"checkpoint loader did not return requested tensor {megatron_key}")
        expected = expected_shapes[megatron_key]
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"loaded tensor {megatron_key} has shape {tuple(tensor.shape)}, expected {expected}"
            )
    return loaded


def _make_hf_shard(
    hf_keys: list[str],
    prepared: _PreparedExport,
    resident_sources: dict[str, torch.Tensor],
    produced_keys: set[str],
) -> dict[str, torch.Tensor]:
    """Transform the resident Megatron sources into one shard of Hugging Face tensors."""
    config = prepared.derived_config.kwargs
    shard: dict[str, torch.Tensor] = {}
    for hf_key in hf_keys:
        spec = prepared.spec_by_hf_key[hf_key]
        if hf_key in produced_keys:
            raise ValueError(f"HF key produced twice across shards: {hf_key}")
        source = (
            resident_sources.get(spec.megatron_key) if spec.megatron_key is not None else None
        )
        shard[hf_key] = mapping.produce_one(
            spec,
            source,
            config["num_attention_heads"],
            config["num_key_value_heads"],
            config["head_dim"],
        )
        produced_keys.add(hf_key)
    return shard


def _write_weight_shards(
    checkpoint_dir: Path, output_dir: Path, prepared: _PreparedExport, verify: bool
) -> _WrittenWeights:
    """Stage 3: stream Megatron sources through transforms into Hugging Face shards."""
    produced_keys: set[str] = set()
    resident_sources: dict[str, torch.Tensor] = {}
    total_bytes = 0

    with reader.single_rank_pg():
        for shard_index, filename in enumerate(prepared.shard_filenames):
            newly_loaded = _load_sources_for_shard(
                checkpoint_dir,
                prepared.source_residency.load_at_shard[shard_index],
                prepared.source_residency.expected_shape,
            )
            resident_sources.update(newly_loaded)

            hf_keys = prepared.shard_layout.filename_to_tensors[filename]
            shard = _make_hf_shard(hf_keys, prepared, resident_sources, produced_keys)
            shard_bytes, written_tensors = writer.write_one_shard(output_dir, filename, shard)
            total_bytes += shard_bytes
            if verify:
                writer.verify_shard(output_dir, filename, written_tensors)

            # ``shard`` can contain views of fused source storage; discard it before releasing
            # sources whose final output was in this shard.
            del shard, written_tensors
            for megatron_key in prepared.source_residency.release_after_shard[shard_index]:
                del resident_sources[megatron_key]

    if resident_sources:
        raise RuntimeError(
            "internal exporter error: sources remained resident after the final shard: "
            f"{sorted(resident_sources)}"
        )
    mapping.check_produced(produced_keys, prepared.mapping_plan)
    if prepared.shard_layout.is_sharded:
        writer.write_index(output_dir, prepared.shard_layout, total_bytes)
    logger.info(
        "3/4 transform + write: %d Hugging Face tensor(s), %d synthesized, %d shard(s)",
        len(produced_keys),
        len(prepared.mapping_plan.synthesized),
        len(prepared.shard_filenames),
    )
    return _WrittenWeights(produced_keys=produced_keys, total_bytes=total_bytes)


def _verify_loaded_model(
    output_dir: Path, prepared: _PreparedExport, produced_keys: set[str]
) -> None:
    """Load the exported HF model and compare it against disk-backed reference tensors."""
    weight_map = (
        prepared.shard_layout.tensor_to_filename
        if prepared.shard_layout.is_sharded
        else {key: writer.SAFETENSORS_SINGLE_NAME for key in produced_keys}
    )
    references = writer.DiskBackedTensors(output_dir, weight_map)
    writer.verify_from_pretrained(output_dir, references, prepared.mapping_plan.geometry)


def _finish_hf_directory(
    output_dir: Path,
    tokenizer_dir: str | Path | None,
    prepared: _PreparedExport,
    written: _WrittenWeights,
    *,
    verify: bool,
    verify_load: bool,
    checkpoint_dir: Path,
    max_shard_size: int | str,
    strict_optimizer: bool,
) -> None:
    """Stage 4: write HF support files, perform final verification, and certify completion."""
    token_ids = {}
    if tokenizer_dir is not None:
        # Before copying: a tokenizer from a different model exits 0 and only fails much later,
        # at the embedding lookup, on ordinary text.
        writer.check_tokenizer_fits_vocab(tokenizer_dir, prepared.derived_config.kwargs["vocab_size"])
        token_ids = writer.copy_tokenizer(output_dir, tokenizer_dir)
    writer.write_config(
        output_dir, prepared.derived_config.kwargs, prepared.parameter_dtype, token_ids
    )
    writer.copy_module_files(output_dir)

    if verify:
        logger.info(
            "4/4 verify: safetensors round trip was bit-identical for %d tensor(s)",
            len(written.produced_keys),
        )
    else:
        logger.warning(
            "--no-verify: the safetensors round trip was NOT checked. conversion_info.json will "
            "record verified=false; do not treat this export as validated."
        )
    if verify_load:
        _verify_loaded_model(output_dir, prepared, written.produced_keys)
        logger.info("4/4 verify: from_pretrained loaded and matched every exported tensor")

    # This certificate is intentionally last. Its presence means all requested work completed;
    # ``verified`` separately records whether the bitwise safetensors check was requested.
    writer.write_conversion_info(
        output_dir,
        source_checkpoint=str(checkpoint_dir),
        iteration=prepared.common_state.get("iteration"),
        checks=prepared.checks,
        dropped_key_count=len(prepared.dropped_keys),
        synthesized=prepared.mapping_plan.synthesized,
        verified=verify,
        settings={
            "max_shard_size": max_shard_size,
            "verify": verify,
            "verify_load": verify_load,
            "strict_optimizer": strict_optimizer,
            "tokenizer_dir": str(tokenizer_dir) if tokenizer_dir is not None else None,
        },
    )


def export_checkpoint(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    *,
    tokenizer_dir: str | Path | None = None,
    max_shard_size: int | str = "5GB",
    verify: bool = True,
    verify_load: bool = False,
    strict_optimizer: bool = False,
) -> ExportSummary:
    """Convert one Megatron ``torch_dist`` checkpoint into a Hugging Face directory.

    Raises ``ValueError`` when the checkpoint cannot be represented exactly. The returned summary
    describes only a completed export; interrupted runs retain ``.export_incomplete``.
    """
    checkpoint_path = Path(checkpoint_dir)
    output_path = Path(output_dir)
    _validate_paths(checkpoint_path, output_path, tokenizer_dir)
    prepared = _prepare_export(checkpoint_path, max_shard_size, strict_optimizer)

    claim = output_claim.claim_output_dir(output_path)
    # A failure after claiming deliberately leaves the marker and any partial files for diagnosis.
    written = _write_weight_shards(checkpoint_path, output_path, prepared, verify)
    _finish_hf_directory(
        output_path,
        tokenizer_dir,
        prepared,
        written,
        verify=verify,
        verify_load=verify_load,
        checkpoint_dir=checkpoint_path,
        max_shard_size=max_shard_size,
        strict_optimizer=strict_optimizer,
    )
    output_claim.release_output_claim(claim)

    logger.info(
        "complete: %d tensors, %d bytes, %d shard(s) -> %s",
        len(written.produced_keys),
        written.total_bytes,
        len(prepared.shard_filenames),
        output_path,
    )
    return ExportSummary(
        output_dir=str(output_path),
        num_tensors=len(written.produced_keys),
        total_bytes=written.total_bytes,
        shard_files=prepared.shard_filenames,
        dropped_optimizer_key_count=len(prepared.dropped_keys),
        synthesized_keys=[tensor.hf_key for tensor in prepared.mapping_plan.synthesized],
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m exporter.export",
        description="Apertus 2 MoE Megatron torch_dist checkpoint -> "
                    "Hugging Face safetensors directory (trust_remote_code).",
    )
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Megatron torch_dist checkpoint directory (the iter_XXXXXXX dir)")
    parser.add_argument("--output-dir", required=True,
                        help="HF output directory; must not exist or must be empty")
    parser.add_argument("--tokenizer-dir", default=None,
                        help="optional: copy tokenizer files verbatim from this directory")
    parser.add_argument("--max-shard-size", default="5GB",
                        help="max safetensors shard size (e.g. 5GB); default %(default)s")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the bitwise safetensors reload verification (default: on)")
    parser.add_argument("--verify-load", action="store_true",
                        help="additionally from_pretrained() the export and compare bitwise (heavy)")
    parser.add_argument("--strict-optimizer", action="store_true",
                        help="fail instead of dropping optimizer.*/opt_param_scheduler keys")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="stderr logging level; DEBUG adds full tracebacks on failure")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    ns = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        stream=sys.stderr,
        level=getattr(logging, ns.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        export_checkpoint(
            ns.checkpoint_dir,
            ns.output_dir,
            tokenizer_dir=ns.tokenizer_dir,
            max_shard_size=ns.max_shard_size,
            verify=not ns.no_verify,
            verify_load=ns.verify_load,
            strict_optimizer=ns.strict_optimizer,
        )
    except Exception as exc:  # clean error, no traceback spam (traceback at DEBUG)
        logger.debug("export failed with traceback:", exc_info=True)
        # The class name matters: str(MemoryError()) is the EMPTY STRING, and OOM is the most
        # likely failure on a real multi-GB checkpoint.
        logger.error(
            "export failed: %s: %s (re-run with --log-level DEBUG for the traceback)",
            type(exc).__name__, exc,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
