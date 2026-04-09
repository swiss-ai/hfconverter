#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --job-name=hf-logits
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=normal

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

usage() {
    echo "Usage: sbatch logits_tools/get_hf_logits.sbatch <hf-ckpt-dir> [out-pt] [out-report] [prompt]"
    echo "Example: sbatch logits_tools/get_hf_logits.sbatch /path/to/hf_ckpt"
    echo "Example: sbatch logits_tools/get_hf_logits.sbatch /path/to/hf_ckpt /users/mboronski/hfconverter/logits_tools/results/hf_custom.pt /users/mboronski/hfconverter/logits_tools/results/hf_custom.report.json 'hello world'"
    echo ""
    echo "Default out-pt:"
    echo "  <hfconverter>/logits_tools/results/hf_logits_<ckpt_basename>.pt"
    echo "Report is auto-saved next to .pt as <name>.report.json"
    echo ""
    echo "Optional env:"
    echo "  PATH_TO_HFCONVERTER=~/hfconverter"
    echo "  HF_DTYPE=bf16                         # bf16|fp16|fp32|auto"
    echo "  HF_DEVICE=cuda                        # cuda|cpu"
    echo "  TRUST_REMOTE_CODE=1                   # pass --trust-remote-code"
    echo "  TRANSFORMERS_BRANCH=model/apertus     # swiss-ai transformers branch for apertus"
    echo "  SKIP_TRANSFORMERS_INSTALL=0           # set 1 to skip pip install"
}

if (( $# < 1 )); then
    usage
    exit 1
fi

HF_CKPT_DIR="$1"
OUT_PT=""
OUT_REPORT=""
PROMPT="Sanity check prompt."
if (( $# >= 2 )); then
    if [[ "$2" == *.pt || "$2" == */* ]]; then
        OUT_PT="$2"
        if (( $# >= 3 )); then
            if [[ "$3" == *.json ]]; then
                OUT_REPORT="$3"
                PROMPT="${4:-$PROMPT}"
            else
                PROMPT="$3"
            fi
        fi
    else
        PROMPT="$2"
    fi
fi

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
    CKPT_BASENAME="$(basename "${HF_CKPT_DIR%/}")"
    if [[ -z "${CKPT_BASENAME}" ]]; then
        CKPT_BASENAME="hf"
    fi
    OUT_PT="${RESULTS_DIR}/hf_logits_${CKPT_BASENAME}.pt"
fi

ENV_TOML="${PATH_TO_HFCONVERTER}/env.toml"
PY_SCRIPT="${TOOLS_DIR}/get_hf_logits.py"
HF_DTYPE="${HF_DTYPE:-bf16}"
HF_DEVICE="${HF_DEVICE:-cuda}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
TRANSFORMERS_BRANCH="${TRANSFORMERS_BRANCH:-model/apertus}"
SKIP_TRANSFORMERS_INSTALL="${SKIP_TRANSFORMERS_INSTALL:-0}"

mkdir -p "$(dirname "$OUT_PT")"
mkdir -p logs

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
if [[ ! -d "$HF_CKPT_DIR" ]]; then
    echo "ERROR: HF checkpoint dir not found: $HF_CKPT_DIR"
    exit 1
fi

echo "START TIME: $(date)"
echo "HF_CKPT_DIR=$HF_CKPT_DIR"
echo "OUT_PT=$OUT_PT"
echo "OUT_REPORT=${OUT_REPORT:-<auto>}"
echo "RESULTS_DIR=$RESULTS_DIR"
echo "HF_DTYPE=$HF_DTYPE"
echo "HF_DEVICE=$HF_DEVICE"
echo "TRANSFORMERS_BRANCH=$TRANSFORMERS_BRANCH"
echo "SKIP_TRANSFORMERS_INSTALL=$SKIP_TRANSFORMERS_INSTALL"
echo "SLURM_SUBMIT_DIR=$SUBMIT_DIR"
echo "PATH_TO_HFCONVERTER=$PATH_TO_HFCONVERTER"
echo "TOOLS_DIR=$TOOLS_DIR"
echo "PY_SCRIPT=$PY_SCRIPT"

export PY_SCRIPT HF_CKPT_DIR OUT_PT OUT_REPORT PROMPT HF_DTYPE HF_DEVICE TRUST_REMOTE_CODE
export TRANSFORMERS_BRANCH SKIP_TRANSFORMERS_INSTALL
srun -ul --environment="$ENV_TOML" bash -lc '
set -euo pipefail
if [[ "${SKIP_TRANSFORMERS_INSTALL}" != "1" && -n "${TRANSFORMERS_BRANCH}" ]]; then
  pip install -v --no-cache-dir --no-build-isolation --no-deps --force-reinstall -U \
    "transformers @ git+https://github.com/swiss-ai/transformers.git@${TRANSFORMERS_BRANCH}"
  pip install -v --no-cache-dir -U "huggingface-hub>=0.34.0,<1.0"
fi
cmd=(
  python "${PY_SCRIPT}"
  --hf-dir "${HF_CKPT_DIR}"
  --prompt "${PROMPT}"
  --out-pt "${OUT_PT}"
  --dtype "${HF_DTYPE}"
  --device "${HF_DEVICE}"
)
if [[ -n "${OUT_REPORT}" ]]; then
  cmd+=(--out-report "${OUT_REPORT}")
fi
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  cmd+=(--trust-remote-code)
fi
"${cmd[@]}"
'

echo "END TIME: $(date)"
