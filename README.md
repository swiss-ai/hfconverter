# Multimodal Checkpoint Converter to HuggingFace

Utilities for converting Apertus Megatron checkpoints to HuggingFace format and checking
native-vs-HF logits.

## Megatron -> HF

Set the topology explicitly to match the checkpoint. The conversion launcher checks:

```text
SLURM_NNODES * NPROC_PER_NODE == TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE
```

VPP does not add ranks. It only changes how layers are chunked within the physical PP
ranks, so it is passed separately through one of the virtual pipeline flags.

```bash
# 8B: TP=2, PP=1, 2 total ranks
sbatch --nodes=1 \
  --export=NPROC_PER_NODE=2,TENSOR_MODEL_PARALLEL_SIZE=2,PIPELINE_MODEL_PARALLEL_SIZE=1 \
  convert.sbatch <megatron_ckpt_dir> <iteration> <hf_output_dir>

# 70B: TP=4, PP=8, VPP=2 layers per virtual stage, 32 total ranks
sbatch --nodes=8 \
  --export=NPROC_PER_NODE=4,TENSOR_MODEL_PARALLEL_SIZE=4,PIPELINE_MODEL_PARALLEL_SIZE=8,NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE=2 \
  convert.sbatch <megatron_ckpt_dir> <iteration> <hf_output_dir>

# Custom topology
sbatch --nodes=<nodes> \
  --export=NPROC_PER_NODE=<n>,TENSOR_MODEL_PARALLEL_SIZE=<tp>,PIPELINE_MODEL_PARALLEL_SIZE=<pp>,NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE=<vpp> \
  convert.sbatch <megatron_ckpt_dir> <iteration> <hf_output_dir>
```

Useful overrides:

```bash
TOKENIZER=/path/to/tokenizer
RUN_ENV=/path/to/nemo.toml
LOCAL_MEGATRON_PATH=/path/to/Megatron-LM
NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE=2
```

If the checkpoint uses `--num-virtual-stages-per-pipeline-rank` instead, use:

```bash
NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK=<vpp>
```

## Compare logits

Use the same prompt for both jobs. Native logits must use the same distributed topology as
the source checkpoint. The native launcher reads TP/PP/VPP from checkpoint args, so you
only need to allocate the right number of torchrun ranks:

```text
SLURM_NNODES * NPROC_PER_NODE == checkpoint TP * checkpoint PP
```

HF logits can use `HF_DEVICE_MAP=auto` for large converted models.
For validation, `HF_DTYPE=fp16` can sometimes produce closer native-vs-HF logit
agreement than the default `HF_DTYPE=bf16`, depending on the checkpoint and kernels.

```bash
PROMPT="Sanity check prompt."

# 8B native logits
sbatch --nodes=1 \
  --export=NPROC_PER_NODE=2,EXPECTED_TOTAL_RANKS=2 \
  logits_tools/get_native_dist_logits.sbatch <megatron_ckpt_dir> <iteration> "$PROMPT"

# 70B native logits
sbatch --nodes=8 --gpus-per-node=4 \
  --export=NPROC_PER_NODE=4,EXPECTED_TOTAL_RANKS=32 \
  logits_tools/get_native_dist_logits.sbatch <megatron_ckpt_dir> <iteration> "$PROMPT"

# HF logits for converted output
sbatch logits_tools/get_hf_logits.sbatch <hf_ckpt_dir> "$PROMPT"

# HF logits for a large model sharded over visible GPUs
sbatch --gpus-per-node=4 \
  --export=HF_DEVICE_MAP=auto,HF_MAX_MEMORY=0:90GiB,1:90GiB,2:90GiB,3:90GiB,cpu:200GiB \
  logits_tools/get_hf_logits.sbatch <hf_ckpt_dir> "$PROMPT"

# Compare generated reports. Use --pattern to keep one model/iteration together.
python3 logits_tools/compare_reports.py --pattern '*iter0000250*.report.json'
```
