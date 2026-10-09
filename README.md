# Apertus Checkpoint Converter

Convert Megatron checkpoints to Hugging Face format. Compare native and HF logits.

## Megatron Bridge conversion

Use `convert_bridge.sbatch` to export a Megatron checkpoint with Megatron Bridge.

Set the source-code paths before you submit a job. The launcher uses the
container's Python and packages. Use `sinfo -a -o "%P %a"` to check the partitions.

### Apertus2: Chonk

Use [`examples/apertus2-chonk.sh`](examples/apertus2-chonk.sh) for the long-context
checkpoint at iteration 3576. The script uses TP1/PP1/EP4/ETP1 on one GH200 node.

The script creates the HF reference in the run's scratch directory. It copies
the config and configuration code from
[`examples/apertus2-chonk-hf-reference`](examples/apertus2-chonk-hf-reference).
It downloads the tokenizer from a fixed upstream revision. Then it applies the
included patch to stop automatic `</s>` insertion. The patch does not change
token IDs or remove the EOS token.

The default `config-only` mode needs no published HF model or reference weights.
Use these source revisions:

- [Swiss AI Megatron-Bridge](https://github.com/swiss-ai/Megatron-Bridge):
  `f3bc8e7e6a722adde1176b2ba8980109e09493ef`.
- [Megatron-LM-MoE PR #88](https://github.com/swiss-ai/Megatron-LM-MoE/pull/88):
  `b0844260b2a0888c711dca230b88ec13b20f3229`.

The script checks both revisions. It sets Apertus2 Chonk's layout fields with
`--model-override NAME=JSON`.

`BRIDGE_ROOT` and `MEGATRON_ROOT` are **source-code paths**, not checkpoint paths.
Set the checkpoint, reference, and output paths separately:

- `CHECKPOINT`: the Megatron checkpoint iteration directory.
- `HF_REFERENCE`: a matching HF checkpoint for `weighted` mode. For `config-only`
  mode, use a directory with the config, tokenizer, and code, but no weights.
- `HF_OUTPUT`: a new HF output directory. If you do not set it, the script uses
  the run's scratch directory.

```bash
export BRIDGE_ROOT=/path/to/Megatron-Bridge
export MEGATRON_ROOT=/path/to/Megatron-LM-MoE
export CHECKPOINT=/path/to/megatron/iter_0003576
bash examples/apertus2-chonk.sh  # Builds the config-only reference in scratch.

# Optional: use a matching HF checkpoint that already has weights.
export HF_REFERENCE=/path/to/matching/hf-checkpoint
REFERENCE_MODE=weighted bash examples/apertus2-chonk.sh
```

Each run creates an output under `$SCRATCH/tmp/apertus2-chonk-<mode>-export-*/hf`.
Set `HF_OUTPUT` to use another output path. The default account is `csstaff`.
The default partition is `normal`. Use `SLURM_ACCOUNT` and `SLURM_PARTITION` to
change these settings.

### GLM

Use [`examples/glm.sh`](examples/glm.sh) to import GLM-4.5 from HF to a saved
Megatron checkpoint. This script does not export back to HF.

Use upstream NVIDIA Bridge and its specified Megatron Core revision.
Do not use the Apertus2 Chonk fork for GLM. Initialize the `3rdparty/Megatron-LM`
submodule first. TP8/PP1/EP16/ETP1 needs four GH200 nodes.

Set `HF_INPUT` to the local HF checkpoint. Set `MEGATRON_OUTPUT` to a new output
directory. `BRIDGE_ROOT` is the Bridge source-code path.

```bash
export BRIDGE_ROOT=/path/to/NVIDIA-Megatron-Bridge
export HF_INPUT=/path/to/GLM-4.5-hf
export MEGATRON_OUTPUT=/path/to/new/megatron-checkpoint
bash examples/glm.sh
```

### Conversion rules

For Megatron → HF export:

- Use a checkpoint iteration directory and a matching local HF reference.
- The default is `--reference-mode config-only`. Do not include weights or
  symlinks in this reference. The Apertus2 Chonk example creates this reference
  from the included files. For `weighted` mode, set `HF_REFERENCE` explicitly.
- Set `--tp`, `--pp`, `--ep`, and `--etp`. Use `PP * lcm(TP, ETP * EP)` ranks.
- Use a new or empty output directory. Keep it separate from both inputs.
- `--trust-remote-code` runs reference code. Use it only with trusted files.
- Use `RUN_ENV` to select another EDF. For options, run
  `uv run python convert_megatron_to_hf.py --help`.

### Test results

Apertus2 Chonk passed these tests on one GH200 node with EP4 (job `5014485`):

- Weighted HF export.
- Config-only HF export.
- HF → saved Megatron checkpoint → HF conversion.

All 30,978 tensors matched exactly in each comparison. Tokenizer behavior and
inference equivalence were not tested.

GLM-4.5 HF → Megatron import completed on 16 GPUs (job `5014909`).
The saved checkpoint contains the MTP parameters. GLM export back to HF, full
tensor comparison, tokenizer behavior, and inference were not validated.

### Tested versions

Apertus2 Chonk used these versions and the source revisions listed above:

| Component | Version |
| --- | --- |
| Swiss AI Megatron Bridge | `0.7.0+f3bc8e7e` |
| Megatron Core, MoE PR #88 | `0.17.0rc0` (`b0844260b2a0888c711dca230b88ec13b20f3229`) |
| Transformers | `5.17.0` |

The tests used Python `3.12.3` and the NeMo-RL image in `nemo.toml`.
These results apply only to the tested setup.

GLM used upstream Bridge `0.7.0+bb46fee22`, Megatron Core `0.20.0+441a987`
(`441a987409b8fdd2b90b463f7999bac970109a43`), and Transformers `5.17.0`.
GLM import does not need a model-specific Bridge patch.

**Credit:** [Yixuan Xu (@Alvorecer721)](https://github.com/Alvorecer721)
for the idea to use Megatron Bridge for conversion.

## Run the tests

From the repository root, run:

```bash
uv run --no-project --with pytest python -m pytest -q tests
```

These tests use small temporary files. They do not use an existing checkpoint,
load model weights, or need GPUs. They check input paths, reference modes, and
HF weight-index consistency.

To compare real HF checkpoints, use `tests/compare_hf_weights.py` separately.
This comparison needs two existing HF checkpoints, PyTorch, and safetensors.

## Legacy conversion and logit checks

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
