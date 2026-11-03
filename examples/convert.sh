#!/usr/bin/env bash
# Shared Slurm launcher for the supported Megatron Bridge conversion profiles.
set -euo pipefail

usage() {
    echo "Usage: $0 <apertus2-chonk|apertus2-megachonk|glm>" >&2
}

if (( $# != 1 )); then usage; exit 2; fi
PROFILE="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "$PROFILE" in
    apertus2-chonk)
        KIND=export
        BRIDGE_COMMIT="${BRIDGE_COMMIT:-6ccdb3c369491374f1fef4320886e8bb1394bb3e}"
        MEGATRON_COMMIT="${MEGATRON_COMMIT:-42e7a63aaf49e78e33ccfdd0373a423d198b9fb0}"
        : "${BRIDGE_ROOT:?Set BRIDGE_ROOT to the Swiss AI Megatron-Bridge checkout}"
        : "${MEGATRON_ROOT:?Set MEGATRON_ROOT to the Megatron-LM-MoE checkout}"
        CHECKPOINT="${CHECKPOINT:-/capstor/store/cscs/swissai/infra01/apertus_checkpoints/v2/megatron/chonk/pretraining/long_context/iter_0003576}"
        REFERENCE_MODE="${REFERENCE_MODE:-config-only}"
        MODEL_ARGS=(--tp 1 --pp 1 --ep 4 --etp 1 --dtype bf16 --reference-mode "$REFERENCE_MODE"
            --model-override kda_legacy_gate_out_proj_bias=true
            --model-override linear_attn_a_log_per_channel=false --trust-remote-code)
        NODES=1; WORLD=4; JOB_TIME="${SLURM_TIME:-03:00:00}"
        ;;
    apertus2-megachonk)
        KIND=export
        BRIDGE_COMMIT="${BRIDGE_COMMIT:-6ccdb3c369491374f1fef4320886e8bb1394bb3e}"
        MEGATRON_COMMIT="${MEGATRON_COMMIT:-42e7a63aaf49e78e33ccfdd0373a423d198b9fb0}"
        : "${BRIDGE_ROOT:?Set BRIDGE_ROOT to the Swiss AI Megatron-Bridge checkout}"
        : "${MEGATRON_ROOT:?Set MEGATRON_ROOT to the Megatron-LM-MoE checkout}"
        CHECKPOINT="${CHECKPOINT:-/ritom/scratch/cscs/ahuang/megatron-apertus-moe/_research/results/ckpts/megachonk/megachonk-muonmd-lr5.50e-3-mlr2.74e-2-latmoe-e256-top8-kda31-nope-megachonk/iter_0008500}"
        REFERENCE_MODE="${REFERENCE_MODE:-config-only}"
        MODEL_ARGS=(--tp 1 --pp 4 --ep 8 --etp 1 --dtype bf16 --reference-mode "$REFERENCE_MODE"
            --pipeline-layout 'Et*16|t*15|t*15|t*15L'
            --model-override kda_legacy_gate_out_proj_bias=false
            --model-override linear_attn_a_log_per_channel=true --trust-remote-code)
        NODES=8; WORLD=32; JOB_TIME="${SLURM_TIME:-03:00:00}"
        ;;
    glm)
        KIND=import
        BRIDGE_COMMIT="${BRIDGE_COMMIT:-bb46fee22aecc3ef0d8d083fff7eae79778d9fa8}"
        : "${BRIDGE_ROOT:?Set BRIDGE_ROOT to the NVIDIA Megatron-Bridge checkout}"
        MEGATRON_ROOT="${MEGATRON_ROOT:-$BRIDGE_ROOT/3rdparty/Megatron-LM}"
        MEGATRON_COMMIT="${MEGATRON_COMMIT:-$(git -C "$BRIDGE_ROOT" rev-parse HEAD:3rdparty/Megatron-LM)}"
        HF_INPUT="${HF_INPUT:-zai-org/GLM-4.5}"
        HF_REVISION="${HF_REVISION:-cbb2c7cfb52fa128a9660cb1a7a78e017899e115}"
        NODES=4; WORLD=16; JOB_TIME="${SLURM_TIME:-02:00:00}"
        MODEL_ARGS=(--tp 8 --pp 1 --ep 16 --etp 1 --dtype bf16)
        ;;
    *) usage; echo "Unknown profile: $PROFILE" >&2; exit 2 ;;
esac

: "${SCRATCH:?Set SCRATCH}"
: "${SLURM_ACCOUNT:=csstaff}"
: "${SLURM_PARTITION:=normal}"
export BRIDGE_ROOT MEGATRON_ROOT
export LOCAL_HFCONVERTER_PATH="$ROOT"
export CONVERTER_SCRIPT="${CONVERTER_SCRIPT:-$ROOT/convert_checkpoint.py}"
if [[ "$PROFILE" == apertus2-* ]]; then
    export CONVERTER_SCRIPT="$ROOT/check_apertus2_export_keys.py"
    export CPU_MEMORY_NODES="${CPU_MEMORY_NODES:-0,1,2,3}"
fi

check_revision() {
    local path="$1" expected="$2" actual
    actual="$(git -C "$path" rev-parse HEAD)"
    expected="$(git -C "$path" rev-parse "$expected^{commit}")"
    [[ "$actual" == "$expected" ]] || {
        echo "Source revision mismatch: $path has $actual; expected $expected" >&2
        exit 2
    }
}
check_revision "$BRIDGE_ROOT" "$BRIDGE_COMMIT"
check_revision "$MEGATRON_ROOT" "$MEGATRON_COMMIT"

mkdir -p "$SCRATCH/tmp"
SESSION="$(mktemp -d "$SCRATCH/tmp/${PROFILE}-${KIND}-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
if [[ "$KIND" == export ]]; then
    case "$REFERENCE_MODE" in
        config-only)
            if [[ -z "${HF_REFERENCE:-}" ]]; then
                HF_REFERENCE="$SESSION/reference"
                bash "$ROOT/examples/prepare_apertus2_reference.sh" \
                    "$HF_REFERENCE" "$ROOT/examples/${PROFILE}-hf-reference"
            fi
            ;;
        weighted) : "${HF_REFERENCE:?Set HF_REFERENCE to a matching weighted Apertus2 HF checkpoint}" ;;
        *) echo "REFERENCE_MODE must be config-only or weighted" >&2; exit 2 ;;
    esac
    HF_OUTPUT="${HF_OUTPUT:-$SESSION/hf}"
    args=(export "$CHECKPOINT" "$HF_REFERENCE" "$HF_OUTPUT" "${MODEL_ARGS[@]}"
        --max-shard-size-gib "${MAX_SHARD_SIZE_GIB:-4}")
    printf '%s checkpoint: %s\nHF reference (%s): %s\nHF output: %s\n' \
        "$PROFILE" "$CHECKPOINT" "$REFERENCE_MODE" "$HF_REFERENCE" "$HF_OUTPUT"
else
    MEGATRON_OUTPUT="${MEGATRON_OUTPUT:-$SESSION/megatron}"
    args=(import "$HF_INPUT" "$MEGATRON_OUTPUT" "${MODEL_ARGS[@]}")
    if [[ ! -d "$HF_INPUT" ]]; then args+=(--hf-revision "$HF_REVISION"); fi
    printf 'HF input: %s\nMegatron output: %s\n' "$HF_INPUT" "$MEGATRON_OUTPUT"
fi
printf 'Source profile: %s\nLog: %s/%s-%%j.log\n' "$PROFILE" "$SESSION" "$KIND"

sbatch \
    --account="$SLURM_ACCOUNT" \
    --partition="$SLURM_PARTITION" \
    --nodes="$NODES" --ntasks="$WORLD" --ntasks-per-node=4 --gpus-per-node=4 \
    --cpus-per-task=72 --time="$JOB_TIME" \
    --output="$SESSION/$KIND-%j.log" \
    "$ROOT/convert_bridge.sbatch" "${args[@]}"
