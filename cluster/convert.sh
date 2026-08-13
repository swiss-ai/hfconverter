#!/bin/bash
# Submit a Stage 2 export of a Megatron torch_dist checkpoint to Slurm.
#
# Run from the repository root:
#
#   cluster/convert.sh /path/to/torch_dist/iter_XXXXXXX /path/to/fresh-hf-output
#
# Everything after the two paths is passed to sbatch, e.g.:
#
#   cluster/convert.sh <ckpt> <out> --reservation=<name> --partition=normal
#
# Environment knobs are forwarded to cluster/stage2_export.sbatch:
# TOKENIZER_DIR, VERIFY_LOAD (default 0), EXTRA_EXPORT_ARGS, HF_ENV.
set -euo pipefail

USAGE="usage: cluster/convert.sh <torch_dist iter dir> <fresh output dir> [sbatch args...]"
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    echo "$USAGE"
    exit 0
fi
CKPT=${1:?$USAGE}
OUT=${2:?$USAGE}
shift 2

exec sbatch "$@" \
    --export=ALL,REPO="${REPO:-$PWD}",TD_ITER_DIR="$CKPT",HF_OUT_DIR="$OUT",VERIFY_LOAD="${VERIFY_LOAD:-0}" \
    cluster/stage2_export.sbatch
