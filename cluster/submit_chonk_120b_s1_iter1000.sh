#!/usr/bin/env bash
# Submit the verified Stage-2 conversion for Ahuang's Chonk 120B S1 iter_0001000.
#
# The source is already a weight-only torch_dist checkpoint, so Stage 1 must not
# be run. This wrapper pins the checkpoint, tokenizer, resources, and exporter
# settings that passed metadata preflight.
#
# Usage:
#   cluster/submit_chonk_120b_s1_iter1000.sh
#   cluster/submit_chonk_120b_s1_iter1000.sh --test-only
#
# Optional environment overrides:
#   TD_ITER_DIR, TOKENIZER_DIR, HF_OUT_DIR, HF_ENV, LOG_DIR
#   PARTITION, TIME_LIMIT, MEM_MIB

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd -- "$SCRIPT_DIR/.." && pwd)}

TD_ITER_DIR=${TD_ITER_DIR:-/iopsstor/scratch/cscs/ahuang/megatron-apertus-moe/_research/results/ckpts/chonk/120b-moe-256e-latent-swa15-nope-s1-muonmd-lr5.889e-3-mlr5.889e-2-latmoe-e256-top8-swa15-w512-nope-s1/iter_0001000}
TOKENIZER_DIR=${TOKENIZER_DIR:-/iopsstor/scratch/cscs/mvasilev/Megatron-LM-MoE/_research/data/apertus-mul-200k-tokenizer}
HF_OUT_DIR=${HF_OUT_DIR:-/iopsstor/scratch/cscs/mvasilev/hf-export/chonk-120b-s1-iter1000}
HF_ENV=${HF_ENV:-$REPO/cluster/edf/apertus2-hf.toml}
LOG_DIR=${LOG_DIR:-$REPO/cluster/logs/chonk-120b-s1-iter1000}

PARTITION=${PARTITION:-normal}
TIME_LIMIT=${TIME_LIMIT:-01:30:00}
MEM_MIB=${MEM_MIB:-460000}
VERIFY_LOAD=0
EXTRA_EXPORT_ARGS="--max-shard-size 5GB"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

usage() {
    echo "usage: cluster/submit_chonk_120b_s1_iter1000.sh [--test-only]"
}

TEST_ONLY=0
case "${1:-}" in
    "")
        ;;
    --test-only)
        TEST_ONLY=1
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        die "unsupported argument: $1"
        ;;
esac
[ "$#" -le 1 ] || die "expected at most one argument"

[ -f "$REPO/cluster/stage2_export.sbatch" ] \
    || die "no Stage-2 wrapper at $REPO/cluster/stage2_export.sbatch"
[ -f "$TD_ITER_DIR/common.pt" ] \
    || die "source is not a torch_dist iteration directory: $TD_ITER_DIR"
[ -f "$TOKENIZER_DIR/tokenizer.json" ] \
    || die "no tokenizer.json in $TOKENIZER_DIR"
[ -f "$HF_ENV" ] || die "no HF environment file at $HF_ENV"

case "$HF_OUT_DIR" in
    /*) ;;
    *) die "HF_OUT_DIR must be absolute: $HF_OUT_DIR" ;;
esac
HF_OUT_DIR=$(realpath -m -- "$HF_OUT_DIR") \
    || die "cannot resolve HF_OUT_DIR: $HF_OUT_DIR"
if [ -e "$HF_OUT_DIR" ]; then
    [ -d "$HF_OUT_DIR" ] || die "HF_OUT_DIR exists and is not a directory: $HF_OUT_DIR"
    [ -z "$(find "$HF_OUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ] \
        || die "HF_OUT_DIR is non-empty; use a fresh output directory: $HF_OUT_DIR"
fi

[[ "$MEM_MIB" =~ ^[1-9][0-9]*$ ]] \
    || die "MEM_MIB must be a positive integer (got $MEM_MIB)"
mkdir -p -- "$LOG_DIR"

if [ -n "$(git -C "$REPO" status --porcelain -- \
    exporter configuration_apertus2.py modeling_apertus2.py 2>/dev/null)" ]; then
    echo "WARNING: exporter/model sources are dirty; conversion_info.json will record a -dirty revision." >&2
fi

echo "Chonk 120B S1 conversion"
echo "  source      : $TD_ITER_DIR"
echo "  output      : $HF_OUT_DIR"
echo "  tokenizer   : $TOKENIZER_DIR"
echo "  allocation  : partition=$PARTITION time=$TIME_LIMIT mem=$MEM_MIB MiB"
echo "  verification: shard bit-identity; full from_pretrained() reload disabled"
echo "  shard size  : 5GB"

SBATCH_ARGS=(
    "--partition=$PARTITION"
    "--time=$TIME_LIMIT"
    "--mem=$MEM_MIB"
    "--job-name=chonk-120b-s1-export"
    "--chdir=$REPO"
    "--output=$LOG_DIR/%x-%j.log"
    "--error=$LOG_DIR/%x-%j.log"
    "--export=ALL"
)

SUBMIT_ENV=(
    env
    -u TORCH_FORCE_WEIGHTS_ONLY_LOAD
    -u TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
    "REPO=$REPO"
    "TD_ITER_DIR=$TD_ITER_DIR"
    "HF_OUT_DIR=$HF_OUT_DIR"
    "TOKENIZER_DIR=$TOKENIZER_DIR"
    "HF_ENV=$HF_ENV"
    "VERIFY_LOAD=$VERIFY_LOAD"
    "SKIP_INSPECT="
    "EXTRA_EXPORT_ARGS=$EXTRA_EXPORT_ARGS"
)

if [ "$TEST_ONLY" -eq 1 ]; then
    echo "  mode        : Slurm validation only; no job will be submitted"
    "${SUBMIT_ENV[@]}" sbatch --test-only "${SBATCH_ARGS[@]}" \
        "$REPO/cluster/stage2_export.sbatch"
    echo "Slurm accepted the request; no job was submitted."
    exit 0
fi

JOB_ID=$("${SUBMIT_ENV[@]}" sbatch --parsable "${SBATCH_ARGS[@]}" \
    "$REPO/cluster/stage2_export.sbatch")
JOB_ID=${JOB_ID%%;*}
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || die "sbatch returned an unexpected job id: $JOB_ID"

echo "Submitted job $JOB_ID"
echo "  status: squeue -j $JOB_ID"
echo "  log   : $LOG_DIR/chonk-120b-s1-export-$JOB_ID.log"
echo "  model : $HF_OUT_DIR"
