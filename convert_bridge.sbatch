#!/bin/bash
#SBATCH --account=csstaff
#SBATCH --job-name=bridge-convert
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/tmp/bridge-convert-%j.log

set -euo pipefail

if (( $# < 3 )); then
    echo "Usage: sbatch convert_bridge.sbatch <checkpoint-iteration-dir> <local-hf-reference> <output-dir> [converter options]" >&2
    exit 2
fi
: "${SCRATCH:?Set SCRATCH to shared scratch storage}"
: "${MEGATRON_ROOT:?Set MEGATRON_ROOT to a checkpoint-compatible checkout}"
: "${BRIDGE_ROOT:?Set BRIDGE_ROOT to the Swiss AI Megatron-Bridge checkout}"

HFCONVERTER_ROOT="${LOCAL_HFCONVERTER_PATH:-${SLURM_SUBMIT_DIR}}"
RUN_ENV="${RUN_ENV:-$HFCONVERTER_ROOT/nemo.toml}"
for path in \
    "$RUN_ENV" \
    "$HFCONVERTER_ROOT/convert_megatron_to_hf.py" \
    "$MEGATRON_ROOT/megatron/core/__init__.py" \
    "$BRIDGE_ROOT/src/megatron/bridge/__init__.py"
do
    if [[ ! -f "$path" ]]; then
        echo "Missing required file: $path" >&2
        exit 2
    fi
done

SESSION_DIR="${SCRATCH}/tmp/bridge-convert-${SLURM_JOB_ID}"
mkdir -p "$SESSION_DIR"
export HFCONVERTER_ROOT MEGATRON_ROOT BRIDGE_ROOT SESSION_DIR
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
export MASTER_ADDR MASTER_PORT
for root in "$HFCONVERTER_ROOT" "$MEGATRON_ROOT" "$BRIDGE_ROOT"; do
    echo "Source: $root commit=$(git -C "$root" rev-parse HEAD)"
done

# Slurm provides one process per GPU. Do not nest torchrun in this step.
srun --mpi=pmix --network=disable_rdzv_get --gpu-bind=none \
    --environment="$RUN_ENV" \
    --container-mounts="${SCRATCH}:${SCRATCH},${HOME}:${HOME},/capstor:/capstor,/iopsstor:/iopsstor,/ritom:/ritom" \
    -l -u bash -c '
        set -euo pipefail
        export RANK="$SLURM_PROCID"
        export LOCAL_RANK="$SLURM_LOCALID"
        export WORLD_SIZE="$SLURM_NTASKS"
        export PYTHONPATH="$MEGATRON_ROOT:$BRIDGE_ROOT/src:$BRIDGE_ROOT/scripts/conversion${PYTHONPATH:+:$PYTHONPATH}"
        unset VIRTUAL_ENV UV_PYTHON PYTHONHOME
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        export CUDA_DEVICE_MAX_CONNECTIONS=1

        cache_root="/ritom/scratch/cscs/$USER/cache"
        [[ -d /ritom/scratch/cscs/$USER ]] || cache_root="$SESSION_DIR/cache"
        export XDG_CACHE_HOME="$cache_root"
        export UV_CACHE_DIR="$cache_root/uv"
        export PIP_CACHE_DIR="$cache_root/pip"
        export HF_HOME="$cache_root/hf"
        export HF_HUB_CACHE="$cache_root/hf/hub"
        export TORCH_HOME="$cache_root/torch"

        runtime="/tmp/bridge-convert-$SLURM_JOB_ID-$RANK"
        export TMPDIR="$runtime/tmp"
        export HF_MODULES_CACHE="$runtime/hf_modules"
        export TRITON_CACHE_DIR="$runtime/triton"
        export TORCHINDUCTOR_CACHE_DIR="$runtime/inductor"
        export TORCH_EXTENSIONS_DIR="$runtime/extensions"
        export CUDA_CACHE_PATH="$runtime/cuda"
        export FLASHINFER_WORKSPACE_BASE="$runtime/flashinfer"
        mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$HF_HOME"
        exec uv run --no-project python \
            "$HFCONVERTER_ROOT/convert_megatron_to_hf.py" "$@"
    ' bash "$@"
