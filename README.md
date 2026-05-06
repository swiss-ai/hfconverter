# Multimodal Checkpoint Converter to HuggingFace

Utilities for converting Apertus Megatron checkpoints to HuggingFace format and checking
native-vs-HF logits.

## Megatron -> HF

Set the topology explicitly to match the checkpoint. The conversion launcher checks:

```text
SLURM_NNODES * NPROC_PER_NODE == TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE
```

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

# 8B HF logits, single GPU/default device map
sbatch logits_tools/get_hf_logits.sbatch <hf_ckpt_dir> "$PROMPT"

# 70B HF logits, sharded over visible GPUs
sbatch --gpus-per-node=4 \
  --export=HF_DEVICE_MAP=auto \
  logits_tools/get_hf_logits.sbatch <hf_ckpt_dir> "$PROMPT"

# Optional fp32 HF logits if bf16/fp16 behavior needs a stable reference
sbatch --gpus-per-node=4 \
  --export=HF_DEVICE_MAP=auto,HF_DTYPE=fp32 \
  logits_tools/get_hf_logits.sbatch <hf_ckpt_dir> "$PROMPT"

# Compare generated reports. Use --pattern to keep one model/iteration together.
python3 logits_tools/compare_reports.py --pattern '*iter0000250*.report.json'
```
