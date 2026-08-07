#!/bin/bash
# =============================================================================
# torch_dist key inspection inside the HF-side container
# =============================================================================
#
# Replaces the login-node `$PY cluster/inspect_torchdist_keys.py <iter-dir>`.
# Stage 1 keeps conversion deliberately small and does not inspect the result.
# Stage 2 runs this check as a fatal gate before exporting; this standalone
# wrapper lets you inspect the result before Stage 2 or repeat the inspection.
#
# USAGE
#   sbatch --export=ALL,TD_ITER_DIR=<.../torch_dist/iter_XXXXXXX> \
#       $REPO/cluster/inspect_keys.sbatch
#   # knobs: ALL_KEYS=1 (full key/shape/dtype dump), REPO=..., HF_ENV=...
# =============================================================================

#SBATCH --job-name=apertus2-inspect
#SBATCH --time=00:15:00
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH --account=infra01
#SBATCH --cpus-per-task=72
#SBATCH --mem=120000
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.log

set -euo pipefail
echo "START TIME: $(date)"

REPO=${REPO:-/iopsstor/scratch/cscs/$USER/hfconverter}
HF_ENV=${HF_ENV:-$REPO/cluster/edf/apertus2-hf.toml}
ALL_KEYS=${ALL_KEYS:-}

die() { echo "FATAL: $*" >&2; exit 1; }
: "${TD_ITER_DIR:?TD_ITER_DIR is required: the torch_dist iter_XXXXXXX directory}"
[ -d "$TD_ITER_DIR" ] || die "no dir at $TD_ITER_DIR"
[ -f "$HF_ENV" ] || die "no EDF at $HF_ENV"
IMG=$(sed -n 's/^image *= *"\(.*\)"/\1/p' "$HF_ENV"); IMG=${IMG//'${USER}'/$USER}
case "$IMG" in /*) [ -f "$IMG" ] || die "container image $IMG missing — build it first: sbatch $REPO/cluster/container/build_container.sbatch";; esac

mkdir -p "$REPO/cluster/cache/tmp"
ENV_SETUP="export PYTHONPATH=$REPO; \
    export TMPDIR=$REPO/cluster/cache/tmp; \
    export TOKENIZERS_PARALLELISM=false; \
    cd $REPO"

srun -lu --environment="$HF_ENV" --nodes=1 --ntasks=1 \
    --cpus-per-task "${SLURM_CPUS_PER_TASK:-72}" \
    bash -c "$ENV_SETUP; python cluster/inspect_torchdist_keys.py '$TD_ITER_DIR' ${ALL_KEYS:+--all}"

echo "END TIME: $(date)"
