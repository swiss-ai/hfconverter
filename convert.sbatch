#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --job-name=convert
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x_%j.log  # control where the stdout will be
#SBATCH --error=logs/%x_%j.err   # control where the error messages will be
#SBATCH --partition=normal       # partition name
#SBATCH --time=02:00:00

# Aux functions.
usage() {
        echo "Usage: sbatch convert.sbatch <ckpt-path> <iteration> <output-path>"
        echo "Example: sbatch convert.sbatch /capstor/scratch/cscs/asolergi/main_run_70B_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-70b-512-nodes-1e-5lr/checkpoints-512-noOverlap/ 830000 /capstor/store/cscs/swissai/infra01/hf-checkpoints/Apertus70B-it830000"
}
die() {
        echo "$*" >& 2
        exit 1
}

# Wake up.
set -e
echo "START TIME: $(date)"
echo "Using nodes: $SLURM_JOB_NODELIST"

# Config.
if (( $# != 3 )); then
        usage
        die "Invalid usage: Invalid argument count"
fi

TOKENIZER=${TOKENIZER:-alehc/swissai-tokenizer}
MODEL=$1
IT=$2
OUT_PATH=$3
TRANSFORMERS_BRANCH=${TRANSFORMERS_BRANCH:-""}
# make sure the output path exists
mkdir -p $OUT_PATH

# Setup temporary paths.
TEMP_PATH_ROOT=$SCRATCH/.tmp
TORCH_NODIST_PATH=$(mktemp -d -p $TEMP_PATH_ROOT)
function cleanup {
        rm -rf $TORCH_NODIST_PATH
}
trap cleanup EXIT

MEGATRON_PATH="/workspace/Megatron-LM"

if [[ $TRANSFORMERS_BRANCH != "" ]]; then
    INSTALL_CMD="pip install -v --no-cache-dir --no-build-isolation --no-deps --force-reinstall -U \
        \"transformers @ git+https://github.com/swiss-ai/transformers.git@${TRANSFORMERS_BRANCH}\""
else
    INSTALL_CMD=""
fi
echo "Installation command: $INSTALL_CMD"
srun -ul --environment=./env.toml bash -c " \
    $INSTALL_CMD
    echo 'Running torchdist->torch'
    cd ${MEGATRON_PATH}
    export PYTHONPATH=${MEGATRON_PATH}
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    MP=4  # We always use model_parallel=4 to ensure we never run out of cuda memory.
    torchrun --nproc-per-node=\${MP} scripts/conversion/torchdist_2_torch.py \
        --bf16 \
        --load=$MODEL \
        --ckpt-step=$IT \
        --ckpt-convert-save=$TORCH_NODIST_PATH \
        --pipeline-model-parallel-size=\${MP}
    unset CUDA_DEVICE_MAX_CONNECTIONS
    unset MP
    echo 'Running torch->hf'
    python tools/checkpoint/convert.py \
        --model-type=GPT \
        --loader=core \
        --saver=swissai_hf \
        --load-dir=$TORCH_NODIST_PATH/torch \
        --save-dir=$OUT_PATH \
        --hf-tokenizer=$TOKENIZER
"
# Goodbye.
echo "Goodbye."
echo "END TIME: $(date)"