#!/bin/bash
#SBATCH --account=infra01
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

NPROC_PER_NODE=${NPROC_PER_NODE:-2}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-2}
PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-1}
NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE=${NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE:-}
NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK=${NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK:-}
MASTER_PORT=${MASTER_PORT:-29500}
DISTRIBUTED_TIMEOUT_MINUTES=${DISTRIBUTED_TIMEOUT_MINUTES:-60}

MEGATRON_REPO=${MEGATRON_REPO:-https://github.com/swiss-ai/Megatron-LM.git}
MEGATRON_BRANCH=${MEGATRON_BRANCH:-multimodality/main}

HFCONVERTER_REPO=${HFCONVERTER_REPO:-https://github.com/swiss-ai/hfconverter.git}
HFCONVERTER_BRANCH=${HFCONVERTER_BRANCH:-main}

LOCAL_HFCONVERTER_PATH=${LOCAL_HFCONVERTER_PATH:-}
LOCAL_MEGATRON_PATH=${LOCAL_MEGATRON_PATH:-}

# Aux functions.
usage() {
        echo "Usage: sbatch [--nodes=N] --export=ALL,NPROC_PER_NODE=<n>,TENSOR_MODEL_PARALLEL_SIZE=<tp>,PIPELINE_MODEL_PARALLEL_SIZE=<pp>[,NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE=<vpp>] convert.sbatch <ckpt-path> <iteration> <output-path>"
        echo "Example: sbatch convert.sbatch /capstor/scratch/cscs/asolergi/main_run_70B_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-70b-512-nodes-1e-5lr/checkpoints-512-noOverlap/ 830000 /capstor/store/cscs/swissai/infra01/hf-checkpoints/Apertus70B-it830000"
        echo ""
        echo "The allocation must satisfy: SLURM_NNODES * NPROC_PER_NODE == TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE"
}
die() {
        echo "$*" >& 2
        exit 1
}

validate_topology() {
    VIRTUAL_PIPELINE_ARG=""
    if [[ -n "$NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE" && -n "$NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK" ]]; then
        die "Set only one VPP knob: NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE or NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK"
    fi
    if [[ -n "$NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE" && "$NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE" != "0" && "$NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE" != "none" ]]; then
        VIRTUAL_PIPELINE_ARG="--num-layers-per-virtual-pipeline-stage=$NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE"
    elif [[ -n "$NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK" && "$NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK" != "0" && "$NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK" != "none" ]]; then
        VIRTUAL_PIPELINE_ARG="--num-virtual-stages-per-pipeline-rank=$NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK"
    fi

    TOTAL_TORCHRUN_RANKS=$((SLURM_NNODES * NPROC_PER_NODE))
    EXPECTED_MODEL_RANKS=$((TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE))
    if (( TOTAL_TORCHRUN_RANKS != EXPECTED_MODEL_RANKS )); then
        die "Topology needs TP*PP=$EXPECTED_MODEL_RANKS ranks, but allocation gives SLURM_NNODES*NPROC_PER_NODE=$SLURM_NNODES*$NPROC_PER_NODE=$TOTAL_TORCHRUN_RANKS. Resubmit with matching --nodes or override NPROC_PER_NODE/TP/PP."
    fi

    echo "NPROC_PER_NODE=$NPROC_PER_NODE"
    echo "TENSOR_MODEL_PARALLEL_SIZE=$TENSOR_MODEL_PARALLEL_SIZE"
    echo "PIPELINE_MODEL_PARALLEL_SIZE=$PIPELINE_MODEL_PARALLEL_SIZE"
    echo "VIRTUAL_PIPELINE_ARG=${VIRTUAL_PIPELINE_ARG:-<none>}"
    echo "TOTAL_TORCHRUN_RANKS=$TOTAL_TORCHRUN_RANKS"
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
    echo "  srun --jobid=$SLURM_JOB_ID --pty --environment=$RUN_ENV bash -l"
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
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_ADDR MASTER_PORT
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

# Config.
if (( $# != 3 )); then
        usage
        die "Invalid usage: Invalid argument count"
fi

MODEL=$1
IT=$2
OUT_PATH=$3
validate_topology

# make sure the output path exists
mkdir -p "$OUT_PATH"

SCRATCH_ROOT="${SCRATCH:-/iopsstor/scratch/cscs/${USER}}"
MEGATRON_TEMP_PATH_ROOT=$SCRATCH_ROOT/.megatron_tmp/.megatron_tmp_$SLURM_JOB_ID
HFCONVERTER_TEMP_PATH_ROOT=$SCRATCH_ROOT/.hfconverter_tmp/.hfconverter_tmp_$SLURM_JOB_ID
mkdir -p "$MEGATRON_TEMP_PATH_ROOT"
mkdir -p "$HFCONVERTER_TEMP_PATH_ROOT"
TEMP_PATH_ROOT=$SCRATCH_ROOT/.tmp
mkdir -p "$TEMP_PATH_ROOT"
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

if [[ -e "$LOCAL_HFCONVERTER_PATH/.git" ]]; then
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
if [[ ! -f "$RUN_ENV" ]]; then
    die "RUN_ENV not found: $RUN_ENV"
fi
echo "Using srun --environment=$RUN_ENV"

cd "$PATH_TO_HFCONVERTER"

if [[ -e "$LOCAL_MEGATRON_PATH/.git" ]]; then
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
srun -ul --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --environment="$RUN_ENV" bash -c " \
    echo 'Running torchdist->torch'
    echo \"NODE_RANK=\${SLURM_NODEID}\"
    cd ${MEGATRON_PATH}
    export PYTHONPATH=${MEGATRON_PATH}
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc-per-node=$NPROC_PER_NODE \
        --node-rank=\${SLURM_NODEID} \
        --rdzv-backend=c10d \
        --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
        scripts/conversion/torchdist_2_torch.py \
        --bf16 \
        --load=$MODEL \
        --ckpt-step=$IT \
        --ckpt-convert-save=$TORCH_NODIST_PATH \
        --pipeline-model-parallel-size=$PIPELINE_MODEL_PARALLEL_SIZE \
        --tensor-model-parallel-size=$TENSOR_MODEL_PARALLEL_SIZE \
        $VIRTUAL_PIPELINE_ARG \
        --tokenizer-type=HuggingFaceTokenizer \
        --tokenizer-model=$TOKENIZER \
        --no-ckpt-fully-parallel-save \
        --distributed-timeout-minutes=$DISTRIBUTED_TIMEOUT_MINUTES \
        --no-use-tokenizer-model-from-checkpoint-args
" || pause_for_debug $?

TORCH_SAVE_DIR="$TORCH_NODIST_PATH/torch"

srun -ul --nodes=1 --ntasks=1 --environment="$RUN_ENV" bash -c "set -eux
  echo 'Stage1 output dir: $TORCH_SAVE_DIR'
  ls -lah '$TORCH_SAVE_DIR'
  echo 'Look for torch_dist artifacts (should be EMPTY for legacy torch):'
  find '$TORCH_SAVE_DIR' -maxdepth 3 \\( -name '*.distcp' -o -name 'metadata.json' \\) -print || true

  echo 'Look for legacy artifacts (should EXIST):'
  find '$TORCH_SAVE_DIR' -maxdepth 4 -path '*/mp_rank_*/*model_optim_rng.pt' -print | head
"

srun -ul --nodes=1 --ntasks=1 --environment="$RUN_ENV" bash -c " \
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
