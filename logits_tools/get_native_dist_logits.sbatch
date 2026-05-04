#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=native-logits
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=normal

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

usage() {
    echo "Usage: sbatch logits_tools/get_native_dist_logits.sbatch <ckpt-dir> <iteration> [out-pt] [out-report] [prompt]"
    echo "Example: sbatch logits_tools/get_native_dist_logits.sbatch /path/to/dist_ckpt 18000"
    echo "Example: sbatch logits_tools/get_native_dist_logits.sbatch /path/to/dist_ckpt 18000 /users/mboronski/hfconverter/logits_tools/results/native_custom.pt /users/mboronski/hfconverter/logits_tools/results/native_custom.report.json 'hello world'"
    echo ""
    echo "Default out-pt:"
    echo "  <hfconverter>/logits_tools/results/native_logits_iter<iteration>.pt"
    echo "Report is auto-saved next to .pt as <name>.report.json"
    echo ""
    echo "Optional env:"
    echo "  NPROC_PER_NODE=2                       # torchrun workers per node"
    echo "  NPROC=2                                # legacy alias for NPROC_PER_NODE"
    echo "  EXPECTED_TOTAL_RANKS=                  # optional guard: SLURM_NNODES*NPROC_PER_NODE must match"
    echo "  MASTER_PORT=29501"
    echo "  DISTRIBUTED_TIMEOUT_MINUTES=60"
    echo "  PATH_TO_HFCONVERTER=~/hfconverter      # where nemo.toml lives"
    echo "  RUN_ENV=~/hfconverter/nemo.toml        # same container env as convert.sbatch"
    echo "  LOCAL_MEGATRON_PATH=/path/to/Megatron-LM"
    echo "  MEGATRON_REPO=https://github.com/swiss-ai/Megatron-LM.git"
    echo "  MEGATRON_BRANCH=multimodality/main"
    echo "  TOKENIZER=/path/to/hf-tokenizer        # override checkpoint tokenizer_model"
}

die() {
    echo "$*" >&2
    exit 1
}

validate_torchrun_ranks() {
    TOTAL_TORCHRUN_RANKS=$((SLURM_NNODES * NPROC_PER_NODE))
    if [[ -n "${EXPECTED_TOTAL_RANKS:-}" && "$TOTAL_TORCHRUN_RANKS" != "$EXPECTED_TOTAL_RANKS" ]]; then
        die "Expected $EXPECTED_TOTAL_RANKS total torchrun ranks, got SLURM_NNODES*NPROC_PER_NODE=$SLURM_NNODES*$NPROC_PER_NODE=$TOTAL_TORCHRUN_RANKS. Override NPROC_PER_NODE or resubmit with matching --nodes."
    fi
}

if (( $# < 2 )); then
    usage
    exit 1
fi

CKPT_DIR="$1"
ITER="$2"
OUT_PT=""
OUT_REPORT=""
PROMPT="Sanity check prompt."
if (( $# >= 3 )); then
    if [[ "$3" == *.pt || "$3" == */* ]]; then
        OUT_PT="$3"
        if (( $# >= 4 )); then
            if [[ "$4" == *.json ]]; then
                OUT_REPORT="$4"
                PROMPT="${5:-$PROMPT}"
            else
                PROMPT="$4"
            fi
        fi
    else
        PROMPT="$3"
    fi
fi

EXPECTED_TOTAL_RANKS="${EXPECTED_TOTAL_RANKS:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${NPROC:-2}}"
MASTER_PORT="${MASTER_PORT:-29501}"
DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-60}"
TOKENIZER="${TOKENIZER:-/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok}"
if [[ -z "${PATH_TO_HFCONVERTER:-}" ]]; then
    if [[ -f "${SUBMIT_DIR}/nemo.toml" ]]; then
        PATH_TO_HFCONVERTER="${SUBMIT_DIR}"
    elif [[ -f "${SCRIPT_DIR}/../nemo.toml" ]]; then
        PATH_TO_HFCONVERTER="$(cd "${SCRIPT_DIR}/.." && pwd)"
    elif [[ -f "${SCRIPT_DIR}/nemo.toml" ]]; then
        PATH_TO_HFCONVERTER="$(cd "${SCRIPT_DIR}" && pwd)"
    else
        PATH_TO_HFCONVERTER="${HOME}/hfconverter"
    fi
fi

TOOLS_DIR="${PATH_TO_HFCONVERTER}/logits_tools"
if [[ ! -d "${TOOLS_DIR}" ]]; then
    TOOLS_DIR="${SCRIPT_DIR}"
fi
RESULTS_DIR="${TOOLS_DIR}/results"
mkdir -p "${RESULTS_DIR}"

if [[ -z "${OUT_PT}" ]]; then
    OUT_PT="${RESULTS_DIR}/native_logits_iter$(printf '%07d' "${ITER}").pt"
fi

RUN_ENV="${RUN_ENV:-${PATH_TO_HFCONVERTER}/nemo.toml}"
MEGATRON_REPO="${MEGATRON_REPO:-https://github.com/swiss-ai/Megatron-LM.git}"
MEGATRON_BRANCH="${MEGATRON_BRANCH:-multimodality/main}"
LOCAL_MEGATRON_PATH="${LOCAL_MEGATRON_PATH:-${PATH_TO_HFCONVERTER}/Megatron-LM}"
MEGATRON_TEMP_PATH_ROOT="${SCRATCH:-/iopsstor/scratch/cscs/${USER}}/.megatron_tmp/.logits_megatron_tmp_${SLURM_JOB_ID}"

cleanup() {
    if [[ "${DEBUG:-0}" != "1" && -n "${MEGATRON_TEMP_PATH_ROOT:-}" ]]; then
        rm -rf "$MEGATRON_TEMP_PATH_ROOT"
    fi
}
trap cleanup EXIT

mkdir -p "$(dirname "$OUT_PT")"
mkdir -p logs

if [[ "$CKPT_DIR" == *"your_user"* || "$CKPT_DIR" == *"your_dist_ckpt_dir"* ]]; then
    echo "ERROR: CKPT_DIR contains placeholder text: $CKPT_DIR"
    exit 1
fi
if [[ ! -d "$CKPT_DIR" ]]; then
    echo "ERROR: CKPT_DIR not found: $CKPT_DIR"
    exit 1
fi
ITER_DIR=$(printf "iter_%07d" "${ITER}")
if [[ ! -d "${CKPT_DIR}/${ITER_DIR}" ]]; then
    echo "ERROR: iteration directory not found: ${CKPT_DIR}/${ITER_DIR}"
    echo "Available iterations:"
    ls -1 "$CKPT_DIR" | grep '^iter_' | head -n 100 || true
    exit 1
fi

PY_SCRIPT="${TOOLS_DIR}/get_native_dist_logits.py"
if [[ ! -f "$PY_SCRIPT" ]]; then
    echo "ERROR: Python script not found: $PY_SCRIPT"
    echo "Set PATH_TO_HFCONVERTER to your hfconverter root (with nemo.toml and logits_tools/)."
    echo "SLURM_SUBMIT_DIR=$SUBMIT_DIR"
    exit 1
fi
if [[ ! -f "$RUN_ENV" ]]; then
    echo "ERROR: RUN_ENV not found: $RUN_ENV"
    echo "Set PATH_TO_HFCONVERTER to your hfconverter root (with nemo.toml and logits_tools/), or set RUN_ENV explicitly."
    echo "SLURM_SUBMIT_DIR=$SUBMIT_DIR"
    exit 1
fi

echo "START TIME: $(date)"
echo "CKPT_DIR=$CKPT_DIR"
echo "ITER=$ITER"
echo "ITER_DIR=$ITER_DIR"
echo "OUT_PT=$OUT_PT"
echo "OUT_REPORT=${OUT_REPORT:-<auto>}"
echo "RESULTS_DIR=$RESULTS_DIR"
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_ADDR MASTER_PORT
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "SLURM_NNODES=$SLURM_NNODES"
validate_torchrun_ranks
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "TOTAL_TORCHRUN_RANKS=$TOTAL_TORCHRUN_RANKS"
echo "DISTRIBUTED_TIMEOUT_MINUTES=$DISTRIBUTED_TIMEOUT_MINUTES"
echo "TOKENIZER=$TOKENIZER"
echo "SLURM_SUBMIT_DIR=$SUBMIT_DIR"
echo "PATH_TO_HFCONVERTER=$PATH_TO_HFCONVERTER"
echo "TOOLS_DIR=$TOOLS_DIR"
echo "PY_SCRIPT=$PY_SCRIPT"
echo "RUN_ENV=$RUN_ENV"

if [[ -e "$LOCAL_MEGATRON_PATH/.git" ]]; then
    MEGATRON_PATH="$LOCAL_MEGATRON_PATH"
    echo "Using local Megatron-LM: $MEGATRON_PATH"
    echo "  commit: $(git -C "$MEGATRON_PATH" rev-parse HEAD)"
else
    echo "Local Megatron-LM not found at $LOCAL_MEGATRON_PATH; cloning into $MEGATRON_TEMP_PATH_ROOT"
    mkdir -p "$MEGATRON_TEMP_PATH_ROOT"
    git clone "$MEGATRON_REPO" "$MEGATRON_TEMP_PATH_ROOT/Megatron-LM"
    if [[ -n "$MEGATRON_BRANCH" ]]; then
        git -C "$MEGATRON_TEMP_PATH_ROOT/Megatron-LM" checkout "$MEGATRON_BRANCH"
    fi
    MEGATRON_PATH="$MEGATRON_TEMP_PATH_ROOT/Megatron-LM"
    echo "Using cloned Megatron-LM: $MEGATRON_PATH"
    echo "  commit: $(git -C "$MEGATRON_PATH" rev-parse HEAD)"
fi

cd "$MEGATRON_PATH"
export PYTHONPATH="$MEGATRON_PATH"
export CUDA_DEVICE_MAX_CONNECTIONS=1

export MEGATRON_PATH NPROC_PER_NODE PY_SCRIPT CKPT_DIR ITER PROMPT OUT_PT OUT_REPORT TOKENIZER
export MASTER_ADDR MASTER_PORT DISTRIBUTED_TIMEOUT_MINUTES
srun -ul --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --environment="$RUN_ENV" bash -lc '
set -euo pipefail
echo "NODE_RANK=${SLURM_NODEID}"
cd "${MEGATRON_PATH}"
export PYTHONPATH="${MEGATRON_PATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
cmd=(
  torchrun
  --nnodes="${SLURM_NNODES}"
  --nproc-per-node="${NPROC_PER_NODE}"
  --node-rank="${SLURM_NODEID}"
  --rdzv-backend=c10d
  --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}"
  "${PY_SCRIPT}"
  --load "${CKPT_DIR}" \
  --ckpt-step "${ITER}" \
  --ckpt-format torch_dist \
  --auto-detect-ckpt-format \
  --use-checkpoint-args \
  --use-mp-args-from-checkpoint-args \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model "${TOKENIZER}" \
  --no-use-tokenizer-model-from-checkpoint-args \
  --distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES}" \
  --no-load-optim \
  --no-load-rng \
  --prompt "${PROMPT}" \
  --out-pt "${OUT_PT}"
)
if [[ -n "${OUT_REPORT}" ]]; then
  cmd+=(--out-report "${OUT_REPORT}")
fi
"${cmd[@]}"
'

echo "END TIME: $(date)"
