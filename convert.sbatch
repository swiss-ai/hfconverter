#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --job-name=convert
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x_%j.log  # control where the stdout will be
#SBATCH --error=logs/%x_%j.err   # control where the error messages will be
#SBATCH --partition=normal       # partition name

export TMPDIR="${SLURM_TMPDIR:-/iopsstor/scratch/cscs/$USER/.tmp}"
mkdir -p "$TMPDIR"

# Aux functions.
usage() {
        echo "Usage: sbatch convert.sbatch <ckpt-path> <iteration> <output-path>"
        echo "Example: sbatch convert.sbatch /capstor/scratch/cscs/asolergi/main_run_70B_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-70b-512-nodes-1e-5lr/checkpoints-512-noOverlap/ 830000 /capstor/store/cscs/swissai/infra01/hf-checkpoints/Apertus70B-it830000"
}
die() {
        echo "$*" >& 2
        exit 1
}

# Keep tmp dirs if we pause for debugging
DEBUG=${DEBUG:-0}

pause_for_debug() {
    rc=$1
    if (( DEBUG != 1 )); then
        return "$rc"
    fi
    echo ""
    echo "=== FAILURE (rc=$rc) ==="
    echo "Job $SLURM_JOB_ID on nodes: $SLURM_JOB_NODELIST"
    echo "TORCH_NODIST_PATH=$TORCH_NODIST_PATH"
    echo "MEGATRON_PATH=$MEGATRON_PATH"
    echo ""
    echo "To get an interactive shell in THIS allocation (new terminal):"
    echo "  srun --jobid=$SLURM_JOB_ID --pty --environment=$PATH_TO_HFCONVERTER/env.toml bash -l"
    echo ""
    echo "When done, cancel the job:"
    echo "  scancel $SLURM_JOB_ID"
    echo "========================"
    echo ""
    DEBUG=1
    export DEBUG
    sleep infinity
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
TRANSFORMERS_BRANCH=${TRANSFORMERS_BRANCH:-"model/apertus"}
MEGATRON_BRANCH=${MEGATRON_BRANCH:-"multimodality/main"}
PATH_TO_HFCONVERTER=${PATH_TO_HFCONVERTER:-/iopsstor/scratch/cscs/${USER}/hfconverter}

# make sure the output path exists
mkdir -p $OUT_PATH

MEGATRON_TEMP_PATH_ROOT=$SCRATCH/.megatron_tmp/.megatron_tmp_$SLURM_JOB_ID
mkdir -p $MEGATRON_TEMP_PATH_ROOT
TEMP_PATH_ROOT=$SCRATCH/.tmp
TORCH_NODIST_PATH=$(mktemp -d -p $TEMP_PATH_ROOT)
function cleanup {
    if (( DEBUG == 0 )); then
        rm -rf "$TORCH_NODIST_PATH"
        rm -rf "$MEGATRON_TEMP_PATH_ROOT"
    else
        echo "DEBUG=1 -> keeping temp dirs:"
        echo "  $TORCH_NODIST_PATH"
        echo "  $MEGATRON_TEMP_PATH_ROOT"
    fi
}
trap cleanup EXIT
trap cleanup EXIT

cd $PATH_TO_HFCONVERTER

LOCAL_MEGATRON_PATH=${LOCAL_MEGATRON_PATH:-/capstor/store/cscs/swissai/infra01/users/mboronski/src/Megatron-LM}
MEGATRON_BRANCH=${MEGATRON_BRANCH:-multimodality/main}

if [[ -d "$LOCAL_MEGATRON_PATH/.git" ]]; then
    MEGATRON_PATH="$LOCAL_MEGATRON_PATH"
    echo "Using local Megatron-LM: $MEGATRON_PATH"
    echo "  commit: $(git -C "$MEGATRON_PATH" rev-parse HEAD)"
else
    echo "Local Megatron-LM not found at $LOCAL_MEGATRON_PATH; cloning into $MEGATRON_TEMP_PATH_ROOT"
    cd "$MEGATRON_TEMP_PATH_ROOT"
    git clone https://github.com/swiss-ai/Megatron-LM.git
    cd Megatron-LM
    git checkout "$MEGATRON_BRANCH"
    MEGATRON_PATH="$MEGATRON_TEMP_PATH_ROOT/Megatron-LM"
    echo "Using cloned Megatron-LM: $MEGATRON_PATH"
    echo "  commit: $(git -C "$MEGATRON_PATH" rev-parse HEAD)"
    cd "$PATH_TO_HFCONVERTER"
fi || pause_for_debug $?

if [[ $TRANSFORMERS_BRANCH != "" ]]; then
    INSTALL_CMD="pip install -v --no-cache-dir --no-build-isolation --no-deps --force-reinstall -U \
        \"transformers @ git+https://github.com/swiss-ai/transformers.git@${TRANSFORMERS_BRANCH}\" \
        && pip install -v --no-cache-dir -U 'huggingface-hub>=0.34.0,<1.0'"
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
    torchrun --nproc-per-node=2 scripts/conversion/torchdist_2_torch.py \
        --bf16 \
        --load=$MODEL \
        --ckpt-step=$IT \
        --ckpt-convert-save=$TORCH_NODIST_PATH \
        --pipeline-model-parallel-size=1 \
        --tensor-model-parallel-size=2
" || pause_for_debug $?

TORCH_SAVE_DIR="$TORCH_NODIST_PATH/torch"

srun -ul --environment=./env.toml bash -c "set -eux
  echo 'Stage1 output dir: $TORCH_SAVE_DIR'
  ls -lah '$TORCH_SAVE_DIR'
  echo 'Look for torch_dist artifacts (should be EMPTY for legacy torch):'
  find '$TORCH_SAVE_DIR' -maxdepth 3 \\( -name '*.distcp' -o -name 'metadata.json' \\) -print || true

  echo 'Look for legacy artifacts (should EXIST):'
  find '$TORCH_SAVE_DIR' -maxdepth 4 -path '*/mp_rank_*/*model_optim_rng.pt' -print | head
"

srun -ul --environment=./env.toml bash -c " \
    $INSTALL_CMD
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    echo 'Running torch->hf'
    cd ${MEGATRON_PATH}
    export PYTHONPATH=${MEGATRON_PATH}
    python tools/checkpoint/convert.py \
        --model-type=GPT \
        --loader=core \
        --saver=swissai_hf \
        --load-dir=$TORCH_NODIST_PATH/torch \
        --save-dir=$OUT_PATH \
        --hf-tokenizer=$TOKENIZER
" || pause_for_debug $?

# Goodbye.
echo "Goodbye."
echo "END TIME: $(date)"
