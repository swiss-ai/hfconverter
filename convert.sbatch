#!/bin/bash
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --job-name=convert
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x_%j.log  # control where the stdout will be
#SBATCH --error=logs/%x_%j.err   # control where the error messages will be
#SBATCH --partition=normal       # partition name

export TMPDIR="${SLURM_TMPDIR:-/iopsstor/scratch/cscs/$USER/.tmp}"
mkdir -p "$TMPDIR"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Important knobs (override via env).
DEBUG=${DEBUG:-0}

TOKENIZER=${TOKENIZER:-/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok}

MEGATRON_REPO=${MEGATRON_REPO:-https://github.com/swiss-ai/Megatron-LM.git}
MEGATRON_BRANCH=${MEGATRON_BRANCH:-feat/multimodal-convert-fix}

HFCONVERTER_REPO=${HFCONVERTER_REPO:-https://github.com/swiss-ai/hfconverter.git}
HFCONVERTER_BRANCH=${HFCONVERTER_BRANCH:-feat/multimodal-converter}

LOCAL_HFCONVERTER_PATH=${LOCAL_HFCONVERTER_PATH:-$SLURM_SUBMIT_DIR}
LOCAL_MEGATRON_PATH=${LOCAL_MEGATRON_PATH:-$SLURM_SUBMIT_DIR/Megatron-LM}

# Aux functions.
usage() {
        echo "Usage: sbatch convert.sbatch <ckpt-path> <iteration> <output-path>"
        echo "Example: sbatch convert.sbatch /capstor/scratch/cscs/asolergi/main_run_70B_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-70b-512-nodes-1e-5lr/checkpoints-512-noOverlap/ 830000 /capstor/store/cscs/swissai/infra01/hf-checkpoints/Apertus70B-it830000"
}
die() {
        echo "$*" >& 2
        exit 1
}

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

MODEL=$1
IT=$2
OUT_PATH=$3

# make sure the output path exists
mkdir -p "$OUT_PATH"

MEGATRON_TEMP_PATH_ROOT=$SCRATCH/.megatron_tmp/.megatron_tmp_$SLURM_JOB_ID
HFCONVERTER_TEMP_PATH_ROOT=$SCRATCH/.hfconverter_tmp/.hfconverter_tmp_$SLURM_JOB_ID
mkdir -p "$MEGATRON_TEMP_PATH_ROOT"
mkdir -p "$HFCONVERTER_TEMP_PATH_ROOT"
TEMP_PATH_ROOT=$SCRATCH/.tmp
TORCH_NODIST_PATH=$(mktemp -d -p "$TEMP_PATH_ROOT")
function cleanup {
    if (( DEBUG == 0 )); then
        rm -rf "$TORCH_NODIST_PATH"
        rm -rf "$MEGATRON_TEMP_PATH_ROOT"
        rm -rf "$HFCONVERTER_TEMP_PATH_ROOT"
    else
        echo "DEBUG=1 -> keeping temp dirs:"
        echo "  $TORCH_NODIST_PATH"
        echo "  $MEGATRON_TEMP_PATH_ROOT"
        echo "  $HFCONVERTER_TEMP_PATH_ROOT"
    fi
}
trap cleanup EXIT
trap cleanup EXIT

if [[ -d "$LOCAL_HFCONVERTER_PATH/.git" ]]; then
    PATH_TO_HFCONVERTER="$LOCAL_HFCONVERTER_PATH"
    echo "Using local hfconverter: $PATH_TO_HFCONVERTER"
    echo "  commit: $(git -C "$PATH_TO_HFCONVERTER" rev-parse HEAD)"
else
    echo "Local hfconverter not found at $LOCAL_HFCONVERTER_PATH; cloning into $HFCONVERTER_TEMP_PATH_ROOT"
    git clone "$HFCONVERTER_REPO" "$HFCONVERTER_TEMP_PATH_ROOT/hfconverter"
    if [[ -n "$HFCONVERTER_BRANCH" ]]; then
        git -C "$HFCONVERTER_TEMP_PATH_ROOT/hfconverter" checkout "$HFCONVERTER_BRANCH"
    fi
    PATH_TO_HFCONVERTER="$HFCONVERTER_TEMP_PATH_ROOT/hfconverter"
    echo "Using cloned hfconverter: $PATH_TO_HFCONVERTER"
    echo "  commit: $(git -C "$PATH_TO_HFCONVERTER" rev-parse HEAD)"
fi || pause_for_debug $?
export PATH_TO_HFCONVERTER

RUN_ENV=${RUN_ENV:-$PATH_TO_HFCONVERTER/nemo.toml}
echo "Using srun --environment=$RUN_ENV"

cd "$PATH_TO_HFCONVERTER"

if [[ -d "$LOCAL_MEGATRON_PATH/.git" ]]; then
    MEGATRON_PATH="$LOCAL_MEGATRON_PATH"
    echo "Using local Megatron-LM: $MEGATRON_PATH"
    echo "  commit: $(git -C "$MEGATRON_PATH" rev-parse HEAD)"
else
    echo "Local Megatron-LM not found at $LOCAL_MEGATRON_PATH; cloning into $MEGATRON_TEMP_PATH_ROOT"
    cd "$MEGATRON_TEMP_PATH_ROOT"
    git clone "$MEGATRON_REPO"
    cd Megatron-LM
    git checkout "$MEGATRON_BRANCH"
    MEGATRON_PATH="$MEGATRON_TEMP_PATH_ROOT/Megatron-LM"
    echo "Using cloned Megatron-LM: $MEGATRON_PATH"
    echo "  commit: $(git -C "$MEGATRON_PATH" rev-parse HEAD)"
    cd "$PATH_TO_HFCONVERTER"
fi || pause_for_debug $?

echo "Dependency installation disabled; using environment as-is."
srun -ul --environment="$RUN_ENV" bash -c " \
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

srun -ul --environment="$RUN_ENV" bash -c "set -eux
  echo 'Stage1 output dir: $TORCH_SAVE_DIR'
  ls -lah '$TORCH_SAVE_DIR'
  echo 'Look for torch_dist artifacts (should be EMPTY for legacy torch):'
  find '$TORCH_SAVE_DIR' -maxdepth 3 \\( -name '*.distcp' -o -name 'metadata.json' \\) -print || true

  echo 'Look for legacy artifacts (should EXIST):'
  find '$TORCH_SAVE_DIR' -maxdepth 4 -path '*/mp_rank_*/*model_optim_rng.pt' -print | head
"

srun -ul --environment="$RUN_ENV" bash -c " \
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
