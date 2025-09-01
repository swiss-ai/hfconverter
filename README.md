# Checkpoint Converter to HuggingFace

## 🚀 Installation

No additional installation steps are required.  
All dependencies are bundled, simply run the `convert.sbatch` script.

## Usage

### Megatron -> HF

To convert a Megatron checkpoint to Hugging Face, use the following command:

```bash
sbatch convert.sbatch <ckpt-path> <iteration> <output-path>
```

Make sure that `TRANSFORMERS_BRANCH` and `MEGATRON_BRANCH` are set correctly in `convert.sbatch`.

70B Model Convert Example:

```bash
sbatch convert.sbatch /capstor/scratch/cscs/asolergi/main_run_70B_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-70b-512-nodes-1e-5lr/checkpoints-512-noOverlap/ 830000 /capstor/store/cscs/swissai/infra01/hf-checkpoints/Apertus70B-it830000
```

8B Model Convert Example:

```bash
sbatch convert.sbatch /iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-8b-128-nodes/checkpoints/ 1678000 /capstor/store/cscs/swissai/infra01/hf-checkpoints/Apertus8B-it1678000
```

If your tokenizer is outdated or you want to make sure you are using the updated tokenizer and chat template, please follow [SwissAI-->-Apertus] and use your converted HF checkpoint as `<swissai-model-path>` with `FORCE=1`. Note that `<swissai-model-path>` and `<apertus-output-path>` should be different.

### SwissAI -> Apertus

To update an old checkpoint with a deprecated model name, use the following command:

```bash
sbatch convert_swissai_to_apertus.sbatch <swissai-model-path> <apertus-output-path>
```

Make sure `PATH_TO_TOKENIZER`, `IS_BASE`, `PATH_TO_HFCONVERTER`, `TRANSFORMERS_BRANCH`, `IS_INSTRUCT` and `FORCE` are set correctly in `convert_swissai_to_apertus.sbatch`.

`PATH_TO_TOKENIZER` is the path to the correct tokenizer configs and chat template. It is not provided; the existing tokenizer in the old checkpoint path will be used.

Set `IS_BASE=1` for base models; use `IS_BASE=0` for long-context models or post-train checkpoints.

`IS_INSTRUCT=1` is used whenever the model is instruction-tuned, otherwise `IS_INSTRUCT=0`.

`FORCE=1` force-updates the model even if it is previously update to Apertus to make sure consistency, otherwise `FORCE=0` and the conversion will be skipped for already converted checkpoints.

Example: 

```bash
export IS_BASE=0
export IS_INSTRUCT=1
sbatch /iopsstor/scratch/cscs/ansaripo/hfconverter/convert_swissai_to_apertus.sbatch /capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/Apertus8B-tokens15T-longcontext64k/ /iopsstor/scratch/cscs/ansaripo/hf_checkpoints/test-swissai-apertus
```
