#!/bin/bash
# =============================================================================
# Full pytest suite (modeling + exporter) — inside the HF-side container
# =============================================================================
#
# Runs the suite on a compute node in the same environment used by Stage 2.
#
# THE SKIP GATE: a wrong environment (no megatron.core) makes every exporter
# test module skip at collection and the run stays green while testing nothing.
# This job therefore fails
# if the summary line contains any skips, not just on a nonzero pytest exit.
# The gate is exit 0 plus zero skipped; the exact test count may change.
#
# USAGE
#   sbatch $REPO/cluster/tests.sbatch
#   # knobs: REPO=... HF_ENV=... PYTEST_ARGS='tests/exporter -q'
# =============================================================================

#SBATCH --job-name=apertus-moe-tests
#SBATCH --time=01:00:00
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
HF_ENV=${HF_ENV:-$REPO/cluster/edf/apertus-moe-hf.toml}
PYTEST_ARGS=${PYTEST_ARGS:-tests/ -q -rs}

die() { echo "FATAL: $*" >&2; exit 1; }
[ -d "$REPO" ] || die "no repo at $REPO"
[ -f "$HF_ENV" ] || die "no EDF at $HF_ENV"
IMG=$(sed -n 's/^image *= *"\(.*\)"/\1/p' "$HF_ENV"); IMG=${IMG//'${USER}'/$USER}
case "$IMG" in /*) [ -f "$IMG" ] || die "container image $IMG missing — build it first: sbatch $REPO/cluster/container/build_container.sbatch";; esac

RES=$REPO/cluster/results
mkdir -p "$RES" "$REPO/cluster/cache/tmp"
SUMMARY=$RES/tests-${SLURM_JOB_ID}.txt

echo "REPO    : $REPO"
echo "HF_ENV  : $HF_ENV ($IMG)"
echo "PYTEST  : $PYTEST_ARGS"

# Keep TMPDIR off /tmp (tmpfs eats --mem), use the HF hub offline (every input
# is local, so a network lookup is a bug), and put the repository on PYTHONPATH.
# conftest also inserts it; this covers direct `python -m` runs.
ENV_SETUP="export PYTHONPATH=$REPO; \
    export TMPDIR=$REPO/cluster/cache/tmp; \
    export TOKENIZERS_PARALLELISM=false; \
    export HF_HUB_OFFLINE=1; \
    export TRANSFORMERS_OFFLINE=1; \
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-72}; \
    cd $REPO"

set +e
srun -lu --environment="$HF_ENV" --nodes=1 --ntasks=1 \
    --cpus-per-task "${SLURM_CPUS_PER_TASK:-72}" \
    bash -c "$ENV_SETUP; python -m pytest $PYTEST_ARGS" 2>&1 | tee "$SUMMARY"
RC=${PIPESTATUS[0]}
set -e

if [ "$RC" -ne 0 ]; then
    echo "FATAL: pytest exited $RC" >&2
    exit "$RC"
fi
if grep -qE '[0-9]+ skipped' "$SUMMARY"; then
    echo "FATAL: tests were SKIPPED — the container env is broken (megatron.core" >&2
    echo "       or transformers missing?). A green run with skips tests nothing" >&2
    echo "       Summary: $SUMMARY" >&2
    exit 1
fi

echo "TESTS OK (0 skipped). Summary kept at $SUMMARY"
echo "END TIME: $(date)"
