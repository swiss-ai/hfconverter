#!/bin/bash
# One Slurm task per GPU. Bridge owns conversion and process-group cleanup.
set -euo pipefail

: "${BRIDGE_ROOT:?Set BRIDGE_ROOT}"
: "${MEGATRON_ROOT:?Set MEGATRON_ROOT}"
: "${HF_INPUT:?Set HF_INPUT}"
: "${MEGATRON_OUTPUT:?Set MEGATRON_OUTPUT}"

export PYTHONPATH="$MEGATRON_ROOT:$BRIDGE_ROOT/src:$BRIDGE_ROOT/scripts/conversion${PYTHONPATH:+:$PYTHONPATH}"
export RANK="$SLURM_PROCID"
export LOCAL_RANK="$SLURM_LOCALID"
export WORLD_SIZE="$SLURM_NTASKS"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=29675

export XDG_CACHE_HOME="/ritom/scratch/cscs/$USER/cache"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export HF_HOME="$XDG_CACHE_HOME/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export UV_PYTHON_PREFERENCE=only-system
export UV_PYTHON_DOWNLOADS=never
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
unset UV_PYTHON VIRTUAL_ENV PYTHONHOME

runtime="/tmp/glm45-import-$SLURM_JOB_ID-$RANK"
export TMPDIR="$runtime/tmp"
export TRITON_CACHE_DIR="$runtime/triton"
export TORCHINDUCTOR_CACHE_DIR="$runtime/inductor"
export TORCH_EXTENSIONS_DIR="$runtime/extensions"
export CUDA_CACHE_PATH="$runtime/cuda"
export HF_MODULES_CACHE="$runtime/hf_modules"
export FLASHINFER_WORKSPACE_BASE="$runtime/flashinfer"
mkdir -p "$TMPDIR" "$UV_CACHE_DIR"

exec uv run --no-project python \
    "$BRIDGE_ROOT/scripts/conversion/run_conversion.py" \
    import \
    --device gpu \
    --hf-model "$HF_INPUT" \
    --megatron-path "$MEGATRON_OUTPUT" \
    --torch-dtype bfloat16 \
    --tp 8 \
    --pp 1 \
    --ep 16 \
    --etp 1 \
    --low-memory-save \
    --distributed-timeout-minutes 45
