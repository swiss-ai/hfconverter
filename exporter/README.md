# Exporter

The exporter converts one Megatron Core `torch_dist` iteration directory into
a self-contained Hugging Face directory. It derives the architecture from the
checkpoint, maps every model tensor explicitly, streams safetensors shards, and
records the verification result.

## Input and output

`--checkpoint-dir` must point to the actual `iter_XXXXXXX` directory, not its
parent. A valid source contains `common.pt` and the distributed-checkpoint
metadata. Optimizer and scheduler entries may be present; they are counted and
excluded from the model export.

`--output-dir` must be absent or empty. A completed directory contains:

- `config.json`, `generation_config.json`, and the custom configuration/model
  Python files;
- one or more `model*.safetensors` files and an index for a sharded model;
- tokenizer files when `--tokenizer-dir` was supplied;
- `conversion_info.json`, written only after all requested checks pass.

An interrupted or failed run leaves `.export_incomplete` and any completed
shards in place for diagnosis. The directory is not resumable; use a new empty
output directory for the next attempt.

## Conversion flow

The conversion has four stages:

1. Read the saved Megatron arguments and tensor metadata without loading all
   tensor data.
2. Derive the Hugging Face config, reject unsupported semantics, and construct
   a complete source-to-output tensor plan.
3. Load only the tensors needed for the current output shard, transform them,
   write safetensors, and verify the shard.
4. Write model/config/tokenizer files, optionally reload the full Hugging Face
   model, and create the conversion certificate.

The mapping covers embeddings, the output head, final normalization, QKV
splitting, Q/K normalization, dense fused gate/up projections, routed experts,
shared experts, expert latent projections, router weights and buffers, and
post-branch norms. Offloaded expert checkpoints are converted from their fused,
transposed `weight1`/`weight2` representation into the stacked Hugging Face
expert tensors. With `attention_output_gate` the fused QKV weight also carries
one gate block per query head (per KV group `[q..., gate..., k, v]`); the gate
blocks become `self_attn.g_proj.weight`, ordered exactly like the query rows.

Splits and transposes change tensor layout but not values. Every source model
key must be consumed exactly once and every expected Hugging Face key must be
produced exactly once.

## Dtypes and losslessness

The exporter does not cast model parameters. It requires one consistent source
parameter dtype and preserves the FP32 router buffers separately. A checkpoint
using in-place FP8 experts is accepted only when it also persisted the
additional master-parameter storage; packed FP8-only expert storage is
rejected.

For the Chonk 120B iter-1000 checkpoint:

- the source model has 696 mapped tensors: 618 BF16 tensors and 78 FP32 router
  buffers;
- the Hugging Face model has 30,696 tensors after experts and fused projections
  are split: 30,618 BF16 tensors and the same 78 FP32 buffers;
- both sides contain 228,784,602,496 tensor bytes;
- all 46 safetensors shards passed bitwise reload verification, followed by a
  successful full `from_pretrained()` reload.

The larger Hugging Face tensor count is expected. It is caused by expanding
stacked or fused Megatron tensors into the per-module names used by
Transformers, not by synthesizing or changing weights.

## Command line

```bash
python -m exporter.export \
  --checkpoint-dir /path/to/torch_dist/iter_XXXXXXX \
  --output-dir /path/to/fresh-hf-output \
  --tokenizer-dir /path/to/tokenizer \
  --max-shard-size 5GB \
  --verify-load
```

Options that affect verification:

- shard-level bitwise verification is on by default;
- `--verify-load` additionally loads the complete model and compares its state
  against the exported tensors;
- `--no-verify` disables the shard check and produces
  `"verified": false`; this is not a validated export;
- `--strict-optimizer` rejects optimizer/scheduler entries instead of dropping
  them.

`--max-shard-size` is also a memory control. Smaller shards reduce the data
resident during streaming. `--verify-load` is different: it requires enough
CPU memory for the entire completed model plus overhead.

For quantile-balanced checkpoints, the exporter persists the selection score
space as `moe_router_quantile_balancing_method`: `sigmoid` (the default)
selects experts from `sigmoid(logits) - qb_beta`, `legacy` from raw
`logits - qb_beta`. Megatron's estimator names are accepted and normalized
(`average`/`histogram` -> `sigmoid`, `legacy_average` -> `legacy`). Current
Megatron does not save the method in the checkpoint arguments; the exporter
then defaults to `sigmoid`, which matches every current estimator. Early
raw-logit runs must be exported with
`--moe-router-quantile-balancing-method=legacy` — the exporter cannot detect
them from metadata alone.

The cluster wrappers forward exporter-only options through `EXTRA_EXPORT_ARGS`:

```bash
EXTRA_EXPORT_ARGS='--moe-router-quantile-balancing-method legacy' \
  cluster/convert.sh /path/to/torch_dist/iter_XXXXXXX /path/to/hf-output
```

The same entry point is available from Python:

```python
from exporter.export import export_checkpoint

summary = export_checkpoint(
    checkpoint_dir="/path/to/torch_dist/iter_XXXXXXX",
    output_dir="/path/to/fresh-hf-output",
    tokenizer_dir="/path/to/tokenizer",
    max_shard_size="5GB",
    verify=True,
    verify_load=True,
)
```

For an early raw-logit QB checkpoint, add
`moe_router_quantile_balancing_method="legacy"` to the Python call.

## Supported checkpoint features

The current config and mapping support the Apertus 2 variants exercised by
the test suite and production export, including:

- SwiGLU and SSSGLU;
- explicit per-layer dense/MoE schedules;
- standard and latent routed experts, a shared expert, expert bias, and
  method-aware quantile-balancing state;
- regular and offloaded expert checkpoint layouts;
- plain and sandwich-norm residual layouts;
- full/sliding attention schedules and per-layer RoPE/NoPE schedules;
- BF16 or FP16 model parameters with FP32 router computation.

This is not a generic Megatron converter. The argument preflight rejects
semantics that the custom Hugging Face model cannot reproduce exactly, such as
homogeneous layer-axis checkpoints, multi-latent attention, active MTP,
parameter biases, tied input/output embeddings, interleaved or partial RoPE,
expert capacity dropping, KEEL residual mode, and unsupported router modes. A
rejection happens before output tensors are written and names the offending
checkpoint option.

## Verification certificate

`conversion_info.json` records:

- the source checkpoint and iteration;
- exporter and dependency versions;
- every compatibility check that passed;
- ignored training-state and synthesized-key counts;
- the verification flags, tokenizer path, shard size, and exporter Git
  revision.

For a production conversion, require `verified: true`,
`settings.verify: true`, no `.export_incomplete` marker, and preferably
`settings.verify_load: true`. `cluster/validate_stage2_export.py` enforces that
contract for Slurm jobs.
