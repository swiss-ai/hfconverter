#!/bin/bash
# Apertus2: Chonk long-context export. Idea: Yixuan Xu (@Alvorecer721).
# Source recipe: swiss-ai/apertus2_experiments, eval-kda-120b-chonk.
#
set -euo pipefail

: "${SCRATCH:?Set SCRATCH}"
: "${BRIDGE_ROOT:?Set BRIDGE_ROOT to the Swiss AI Bridge checkout}"
: "${MEGATRON_ROOT:?Set MEGATRON_ROOT to the Apertus2 Chonk Megatron checkout}"
# https://github.com/swiss-ai/Megatron-Bridge (main)
BRIDGE_COMMIT=f3bc8e7e6a722adde1176b2ba8980109e09493ef
# https://github.com/swiss-ai/Megatron-LM-MoE/pull/88 (PR head)
MEGATRON_COMMIT=b0844260b2a0888c711dca230b88ec13b20f3229
for pair in "$BRIDGE_ROOT:$BRIDGE_COMMIT" "$MEGATRON_ROOT:$MEGATRON_COMMIT"; do
    [[ "$(git -C "${pair%:*}" rev-parse HEAD)" == "${pair##*:}" ]] || {
        echo "Use the pinned source revision: $pair" >&2
        exit 2
    }
done

export LOCAL_HFCONVERTER_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BRIDGE_ROOT MEGATRON_ROOT
CHECKPOINT="${CHECKPOINT:-/capstor/store/cscs/swissai/infra01/apertus_checkpoints/v2/megatron/chonk/pretraining/long_context/iter_0003576}"
REFERENCE_MODE="${REFERENCE_MODE:-config-only}"
mkdir -p "$SCRATCH/tmp"
SESSION="$(mktemp -d "$SCRATCH/tmp/apertus2-chonk-${REFERENCE_MODE}-export-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
HF_OUTPUT="${HF_OUTPUT:-$SESSION/hf}"

case "$REFERENCE_MODE" in
    config-only)
        if [[ -z "${HF_REFERENCE:-}" ]]; then
            HF_REFERENCE="$SESSION/reference"
            bash "$LOCAL_HFCONVERTER_PATH/examples/prepare_apertus2_chonk_reference.sh" "$HF_REFERENCE"
        fi
        ;;
    weighted)
        : "${HF_REFERENCE:?Set HF_REFERENCE to a matching HF checkpoint with weights}"
        ;;
    *)
        echo "REFERENCE_MODE must be weighted or config-only" >&2
        exit 2
        ;;
esac

echo "Megatron checkpoint: $CHECKPOINT"
echo "HF reference ($REFERENCE_MODE): $HF_REFERENCE"
echo "HF output: $HF_OUTPUT"
echo "Log: $SESSION/export-%j.log"

# TP1/PP1/EP4/ETP1: one GH200 node.
sbatch \
    --account="${SLURM_ACCOUNT:-csstaff}" \
    --partition="${SLURM_PARTITION:-normal}" \
    --nodes=1 \
    --ntasks=4 \
    --ntasks-per-node=4 \
    --gpus-per-node=4 \
    --output="$SESSION/export-%j.log" \
    "$LOCAL_HFCONVERTER_PATH/convert_bridge.sbatch" \
    "$CHECKPOINT" \
    "$HF_REFERENCE" \
    "$HF_OUTPUT" \
    --tp 1 \
    --pp 1 \
    --ep 4 \
    --etp 1 \
    --reference-mode "$REFERENCE_MODE" \
    --model-override kda_legacy_gate_out_proj_bias=true \
    --model-override linear_attn_a_log_per_channel=false \
    --trust-remote-code
