# Checkpoint Converter to HuggingFace

## 🚀 Installation

No additional installation steps are required.  
All dependencies are bundled, simply run the `convert.sbatch` script.

## Usage

To convert a checkpoint, use the following command:

```bash
sbatch convert.sbatch <ckpt-path> <iteration> <output-path>
```

### Example: Convert 70B Model

```bash
sbatch convert.sbatch /capstor/scratch/cscs/asolergi/main_run_70B_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-70b-512-nodes-1e-5lr/checkpoints-512-noOverlap/ 830000 /capstor/store/cscs/swissai/infra01/hf-checkpoints/Apertus70B-it830000
```

### Example: Convert 8B Model

```bash
sbatch convert.sbatch /iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-8b-128-nodes/checkpoints/ 1678000 /capstor/store/cscs/swissai/infra01/hf-checkpoints/Apertus8B-it1678000
```