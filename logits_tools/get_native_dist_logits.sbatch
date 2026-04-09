#!/bin/bash
#SBATCH --account=a-infra01-1
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
    echo "  NPROC=2                                # must match runtime TP*PP for this job"
    echo "  PATH_TO_HFCONVERTER=~/hfconverter      # where env.toml lives"
    echo "  LOCAL_MEGATRON_PATH=/path/to/Megatron-LM"
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

NPROC="${NPROC:-2}"
if [[ -z "${PATH_TO_HFCONVERTER:-}" ]]; then
    if [[ -f "${SUBMIT_DIR}/env.toml" ]]; then
        PATH_TO_HFCONVERTER="${SUBMIT_DIR}"
    elif [[ -f "${SCRIPT_DIR}/../env.toml" ]]; then
        PATH_TO_HFCONVERTER="$(cd "${SCRIPT_DIR}/.." && pwd)"
    elif [[ -f "${SCRIPT_DIR}/env.toml" ]]; then
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

ENV_TOML="${PATH_TO_HFCONVERTER}/env.toml"
MEGATRON_PATH="${LOCAL_MEGATRON_PATH:-/capstor/store/cscs/swissai/infra01/users/mboronski/src/Megatron-LM}"

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
    echo "Set PATH_TO_HFCONVERTER to your hfconverter root (with env.toml and logits_tools/)."
    echo "SLURM_SUBMIT_DIR=$SUBMIT_DIR"
    exit 1
fi
if [[ ! -f "$ENV_TOML" ]]; then
    echo "ERROR: env.toml not found: $ENV_TOML"
    echo "Set PATH_TO_HFCONVERTER to your hfconverter root (with env.toml and logits_tools/)."
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
echo "NPROC=$NPROC"
echo "SLURM_SUBMIT_DIR=$SUBMIT_DIR"
echo "PATH_TO_HFCONVERTER=$PATH_TO_HFCONVERTER"
echo "TOOLS_DIR=$TOOLS_DIR"
echo "PY_SCRIPT=$PY_SCRIPT"

cd "$MEGATRON_PATH"
export PYTHONPATH="$MEGATRON_PATH"
export CUDA_DEVICE_MAX_CONNECTIONS=1

export NPROC PY_SCRIPT CKPT_DIR ITER PROMPT OUT_PT OUT_REPORT
srun -ul --environment="$ENV_TOML" bash -lc '
set -euo pipefail
cmd=(
  torchrun --nproc-per-node="${NPROC}" "${PY_SCRIPT}"
  --load "${CKPT_DIR}" \
  --ckpt-step "${ITER}" \
  --ckpt-format torch_dist \
  --auto-detect-ckpt-format \
  --use-checkpoint-args \
  --use-mp-args-from-checkpoint-args \
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
