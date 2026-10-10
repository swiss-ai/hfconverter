# Apertus Checkpoint Converter

Convert checkpoints between Megatron and Hugging Face (HF).

## Megatron Bridge conversion

Use `convert_checkpoint.py` for both directions:

```bash
uv run --no-project python convert_checkpoint.py import <hf-model-or-directory> <new-megatron-output>
uv run --no-project python convert_checkpoint.py export <megatron-iteration> <hf-reference> <new-hf-output>
```

Use `convert_bridge.sbatch` to run either command on Slurm. Set `BRIDGE_ROOT` and
`MEGATRON_ROOT` to the compatible **source-code checkouts**, not checkpoint paths.
The launcher uses the container's Python and packages from `nemo.toml`. Check the
partition before submission with `sinfo -a -o "%P %a"`.

`examples/convert.sh` is the shared launcher. It selects the model settings and
submits `convert_bridge.sbatch`. The tested revisions below are its defaults.
Set `BRIDGE_COMMIT` and `MEGATRON_COMMIT` to use other revisions, tags, or branches.
Point the source roots at checkouts of those revisions; the launcher does not
change your checkouts. For example:

```bash
export BRIDGE_COMMIT=<bridge-revision>
export MEGATRON_COMMIT=<core-revision>
bash examples/convert.sh apertus2-megachonk
```

The export modes are:

- `config-only`: use a reference with config, tokenizer, and code, but no weights.
  The converter streams weights into shards with a default target of 4 GiB.
  Set `--max-shard-size-gib 16` for larger shards, or `MAX_SHARD_SIZE_GIB=16` with
  the shared launcher. Larger targets use more RAM and produce fewer files.
  This is not a total-memory limit; one tensor can exceed the target.
- `weighted`: use a matching local HF checkpoint. Bridge checks its expected keys
  and saves weights across the conversion ranks.

`check_apertus2_export_keys.py` is a key-validation wrapper, not another converter.
It supplies Apertus2's expected HF names to the generic converter for config-only
export. It does not load or transform weights. Model-specific checks stay in this
wrapper; model settings stay in the examples.

### Apertus2 Chonk

Run `bash examples/convert.sh apertus2-chonk` for the long-context
checkpoint at iteration 3576. It uses TP1/PP1/EP4/ETP1 on one GH200 node.

The script creates a config-only reference in scratch. It copies the config and
configuration code from
[`examples/apertus2-chonk-hf-reference`](examples/apertus2-chonk-hf-reference).
It downloads the tokenizer at a fixed revision. The included patch stops automatic
`</s>` insertion. It does not change token IDs or remove the EOS token.

Tested with these source versions:

- [Swiss AI Megatron-Bridge](https://github.com/swiss-ai/Megatron-Bridge), `apertus2/support`:
  `6ccdb3c369491374f1fef4320886e8bb1394bb3e`.
- [Megatron-LM-MoE PR #88](https://github.com/swiss-ai/Megatron-LM-MoE/pull/88):
  `42e7a63aaf49e78e33ccfdd0373a423d198b9fb0`.

The launcher checks the selected revisions. It sets the KDA fields with explicit
`--model-override NAME=JSON` options.

```bash
export BRIDGE_ROOT=/path/to/Megatron-Bridge
export MEGATRON_ROOT=/path/to/Megatron-LM-MoE
export CHECKPOINT=/path/to/megatron/iter_0003576
bash examples/convert.sh apertus2-chonk

# Optional: use a matching HF checkpoint with weights.
export HF_REFERENCE=/path/to/matching/hf-checkpoint
REFERENCE_MODE=weighted bash examples/convert.sh apertus2-chonk
```

### Apertus2 Megachonk

Run `bash examples/convert.sh apertus2-megachonk` for iteration
8500 of `megachonk-muonmd-lr5.50e-3-mlr2.74e-2-latmoe-e256-top8-kda31-nope-megachonk`.
It uses TP1/PP4/EP8/ETP1: 32 ranks across eight GH200 nodes. The pipeline
layout is `Et*16|t*15|t*15|t*15L`. The example sets per-channel KDA decay and disables
the legacy gate bias.

The default config-only mode needs no published HF model or reference weights.
The preparation script reuses the included Apertus2 configuration code and pinned
tokenizer recipe, with the included Apertus2 Megachonk config.

Tested with these source versions:

- Swiss AI Megatron-Bridge, `apertus2/support`:
  `6ccdb3c369491374f1fef4320886e8bb1394bb3e`.
- Megatron-LM-MoE PR #88:
  `42e7a63aaf49e78e33ccfdd0373a423d198b9fb0`.

```bash
export BRIDGE_ROOT=/path/to/Megatron-Bridge
export MEGATRON_ROOT=/path/to/Megatron-LM-MoE
# CHECKPOINT defaults to the iteration 8500 source in the profile.
export HF_OUTPUT=/capstor/store/cscs/swissai/infra01/apertus_checkpoints/v2/hf/megachonk/pretraining/iter_0008500
bash examples/convert.sh apertus2-megachonk

# Optional weighted-reference mode.
export HF_REFERENCE=/path/to/matching/hf-checkpoint
REFERENCE_MODE=weighted bash examples/convert.sh apertus2-megachonk
```

This checkpoint uses Marin histogram quantile balancing. HF uses `legacy` for
its raw-logit inference routing. To import it back for Marin training, set the
training estimator and bin count explicitly, along with the uneven pipeline:

```bash
uv run --no-project python convert_checkpoint.py import "$HF_OUTPUT" /new/megatron/output \
  --tp 1 --pp 4 --ep 8 --etp 1 --dtype bf16 --trust-remote-code \
  --pipeline-layout 'Et*16|t*15|t*15|t*15L' \
  --model-override 'moe_router_quantile_balancing_method="marin_histogram"' \
  --model-override moe_router_quantile_balancing_marin_num_bins=10000
```

Pass these arguments through `convert_bridge.sbatch` for a distributed run.

GH200 exposes HBM as host NUMA memory. CPU allocations must not consume the GPU's
conversion headroom. The Apertus2 examples set `CPU_MEMORY_NODES=0,1,2,3` to keep
CPU buffers in DDR on the tested Alps nodes. Check the NUMA layout before using
this setting on other hardware.

### Other models: GLM

`bash examples/convert.sh glm` shows how to convert a **non-Apertus model**
with Bridge. It imports GLM-4.5 from HF into a saved Megatron checkpoint. GLM
export back to HF is outside this example's scope.

The script uses the generic converter and launcher. No GLM import worker or
model-specific Bridge patch is needed. For another supported model, select its
compatible Bridge/Core revisions and change the HF input and parallelism.

GLM was tested with upstream NVIDIA Bridge
`bb46fee22aecc3ef0d8d083fff7eae79778d9fa8`, not the Apertus2 fork. This is the
profile's default revision. Initialize its
`3rdparty/Megatron-LM` submodule. TP8/PP1/EP16/ETP1 needs four GH200 nodes.

```bash
export BRIDGE_ROOT=/path/to/NVIDIA-Megatron-Bridge
bash examples/convert.sh glm

# Or reuse a downloaded HF checkpoint.
export HF_INPUT=/path/to/GLM-4.5-hf
export MEGATRON_OUTPUT=/path/to/new/megatron-checkpoint
bash examples/convert.sh glm
```

The default input is `zai-org/GLM-4.5`, pinned to revision
`cbb2c7cfb52fa128a9660cb1a7a78e017899e115`. Bridge downloads it through the HF
library. Set `HF_INPUT` and `HF_REVISION` to use another model or revision. The
shared HF cache stays on scratch, not in `$HOME`. GLM-4.5 needs about 668 GiB for
the downloaded weights and additional space for the Megatron checkpoint.

### Conversion rules

- Use `PP * lcm(TP, ETP * EP)` ranks. Set `--tp`, `--pp`, `--ep`, and `--etp`.
- Export needs a checkpoint iteration directory and a matching local HF reference.
- A config-only reference must contain no weights, indexes, or symlinks.
- Import accepts a local HF directory or an `owner/model` ID. Use `--hf-revision`
  to pin a download.
- Use a new or empty output directory. Keep it separate from the inputs.
- `--trust-remote-code` runs reference code. Use it only with trusted files.
- Generic config-only export can use `--expected-hf-index` or `--expected-hf-keys`
  for complete key validation. Without a manifest, Bridge mapping errors and
  duplicate keys are checked, but a missing HF output cannot be detected by name.
- Each example writes under `$SCRATCH/tmp`. Set `HF_OUTPUT` or `MEGATRON_OUTPUT`
  to use another durable output path.
- Use `RUN_ENV`, `SLURM_ACCOUNT`, and `SLURM_PARTITION` to change the launch setup.

For all options, run `uv run --no-project python convert_checkpoint.py --help`,
then `import --help` or `export --help`.

**Credit:** [Yixuan Xu (@Alvorecer721)](https://github.com/Alvorecer721)
for the idea to use Megatron Bridge for conversion.

## Run the tests

```bash
uv run --no-project --with pytest python -m pytest -q tests
```

These tests use small temporary files. They check input paths, CLI options,
reference modes, and HF weight-index consistency. They do not convert weights or
need GPUs. Use `tests/compare_hf_weights.py` separately to compare two real HF
checkpoints. That comparison needs PyTorch and safetensors.

## Legacy conversion and logit checks

The logit tools use the legacy setup. They need updates for the Bridge conversion
path and are not part of this workflow.

For legacy conversion, use `TP * PP` ranks:

```bash
sbatch --nodes=1 \
  --export=ALL,NPROC_PER_NODE=2,TENSOR_MODEL_PARALLEL_SIZE=2,PIPELINE_MODEL_PARALLEL_SIZE=1 \
  convert.sbatch <checkpoint-root> <iteration> <hf-output>

sbatch --nodes=1 --export=ALL,NPROC_PER_NODE=2,EXPECTED_TOTAL_RANKS=2 \
  logits_tools/get_native_dist_logits.sbatch <checkpoint-root> <iteration> "Test prompt."
sbatch logits_tools/get_hf_logits.sbatch <hf-output> "Test prompt."
uv run python logits_tools/compare_reports.py --pattern '*iter0000250*.report.json'
```

For larger checkpoints, set the node count, TP/PP/VPP, and `NPROC_PER_NODE` to
match the checkpoint. To run HF inference on multiple GPUs, set
`HF_DEVICE_MAP=auto`.
