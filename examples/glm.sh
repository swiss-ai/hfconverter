#!/bin/bash
# GLM-4.5: HF to saved Megatron, using upstream Bridge and its pinned Core.
set -euo pipefail

: "${SCRATCH:?Set SCRATCH}"
: "${BRIDGE_ROOT:?Set BRIDGE_ROOT to the NVIDIA Megatron Bridge checkout}"
: "${HF_INPUT:?Set HF_INPUT to the local GLM-4.5 HF checkpoint}"

BRIDGE_COMMIT=bb46fee22aecc3ef0d8d083fff7eae79778d9fa8
[[ "$(git -C "$BRIDGE_ROOT" rev-parse HEAD)" == "$BRIDGE_COMMIT" ]] || {
    echo "Use NVIDIA-NeMo/Megatron-Bridge at $BRIDGE_COMMIT" >&2
    exit 2
}
export MEGATRON_ROOT="$BRIDGE_ROOT/3rdparty/Megatron-LM"
expected="$(git -C "$BRIDGE_ROOT" rev-parse HEAD:3rdparty/Megatron-LM)"
[[ "$(git -C "$MEGATRON_ROOT" rev-parse HEAD)" == "$expected" ]] || {
    echo "Run git submodule update --init 3rdparty/Megatron-LM" >&2
    exit 2
}

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BRIDGE_ROOT HF_INPUT
mkdir -p "$SCRATCH/tmp"
SESSION="$(mktemp -d "$SCRATCH/tmp/glm45-hf-to-megatron-XXXXXX")"
export MEGATRON_OUTPUT="${MEGATRON_OUTPUT:-$SESSION/megatron}"
export RUN_ENV="${RUN_ENV:-$REPO/nemo.toml}"

echo "HF input: $HF_INPUT"
echo "Megatron output: $MEGATRON_OUTPUT"
echo "Log: $SESSION/import.log"

# TP8/PP1/EP16/ETP1: four GH200 nodes, with conversion memory headroom.
srun \
    --job-name=glm45-hf-to-megatron \
    --account="${SLURM_ACCOUNT:-csstaff}" \
    --partition="${SLURM_PARTITION:-normal}" \
    --nodes=4 \
    --ntasks=16 \
    --ntasks-per-node=4 \
    --gpus-per-node=4 \
    --cpus-per-task=72 \
    --time=02:00:00 \
    --exclusive \
    --mpi=pmix \
    --network=disable_rdzv_get \
    --gpu-bind=none \
    --kill-on-bad-exit=1 \
    --environment="$RUN_ENV" \
    --container-mounts="${SCRATCH}:${SCRATCH},${HOME}:${HOME},/capstor:/capstor,/iopsstor:/iopsstor,/ritom:/ritom,/users:/users" \
    -l \
    -u bash "$REPO/examples/glm_import_worker.sh" \
    2>&1 | tee "$SESSION/import.log"
