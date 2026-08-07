#!/bin/bash
# =============================================================================
# STAGE 2 — torch_dist checkpoint -> HF safetensors
# =============================================================================
#
# Replaces the login-node `$PY -m exporter.export ...`. Two steps in one job:
#
#   1. inspect_torchdist_keys.py on the input.
#      Its EXIT CODE gates only "is this a torch_dist checkpoint" (RED FLAG
#      unmapped keys are printed but return 0 — the exporter's key bijection is
#      what refuses those); its stdout is the eyeball record in the job log.
#   2. python -m exporter.export with default bitwise verification and, when
#      VERIFY_LOAD=1, --verify-load (from_pretrained, zero missing/unexpected/
#      mismatched keys, fused 3D expert params re-checked bitwise).
#
# Exit 0 == every exporter assert incl. the key bijection passed.
# A finished export has conversion_info.json and no .export_incomplete marker.
#
# CPU-only, single process. The exporter refuses a non-empty --output-dir
# (re-runs need a fresh HF_OUT_DIR); this wrapper pre-checks for a clearer
# error and to fail before the srun.
#
# USAGE
#   sbatch --export=ALL,TD_ITER_DIR=<.../torch_dist/iter_X>,HF_OUT_DIR=<fresh dir> \
#       $REPO/cluster/stage2_export.sbatch
#   # knobs: TOKENIZER_DIR (default: the 1.5b-era 200064-vocab tokenizer; set empty
#   #        for a test fixture with no tokenizer), VERIFY_LOAD=1 (or 0), SKIP_INSPECT=,
#   #        EXTRA_EXPORT_ARGS='', REPO=..., HF_ENV=...
#
# TOKENIZER_DIR IS NOT OPTIONAL FOR A NEW MODEL. The default belongs to the 1.5b
# checkpoints (200064 entries). The exporter now refuses a tokenizer whose ids exceed
# the checkpoint's vocab_size, so a forgotten knob fails loudly rather than welding the
# wrong vocabulary onto the model — but you still have to supply the right one.
# args.tokenizer_model in common.pt names the tokenizer the checkpoint trained with.
#
# The exporter streams from torch_dist: it plans the
# shard layout from metadata, then loads/writes/verifies one output shard at a time.
# --max-shard-size is therefore a MEMORY knob, not just an output-format one — peak RAM
# is ~ one shard + the sources straddling its boundary. Lower it for very large models:
#   EXTRA_EXPORT_ARGS='--max-shard-size 2GB'
#
# THE DEFAULTS BELOW SUIT THE 1.5b MODELS (debug partition, 45 min, 120 GB). They are far
# too small for a 100B-class checkpoint. The inspection step prints the exported size;
# override the allocation on the command line, which beats the #SBATCH lines:
#
#   sbatch --partition=normal --time=08:00:00 --mem=460000 \
#       --export=ALL,TD_ITER_DIR=...,HF_OUT_DIR=...,TOKENIZER_DIR=...,VERIFY_LOAD=0 \
#       $REPO/cluster/stage2_export.sbatch
#
# --verify-load reloads the ENTIRE finished model into CPU RAM, so with it on you need
# roughly (model size + headroom) of --mem; the wrapper checks that below and refuses to
# start an export that cannot finish. VERIFY_LOAD=0 skips the reload and produces a
# weaker completion certificate (validate_stage2_export.py then needs
# --allow-no-verify-load, which it is given automatically).
# =============================================================================

#SBATCH --job-name=apertus2-stage2
#SBATCH --time=00:45:00
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
FORK=${FORK:-/iopsstor/scratch/cscs/$USER/Megatron-LM-MoE}
HF_ENV=${HF_ENV:-$REPO/cluster/edf/apertus2-hf.toml}
# TOKENIZER_DIR may be set empty on purpose (a tiny test fixture has no HF
# tokenizer) — distinguish unset (default to the real one) from set-empty.
TOKENIZER_DIR=${TOKENIZER_DIR-$FORK/_research/data/apertus-mul-200k-tokenizer}
VERIFY_LOAD=${VERIFY_LOAD:-1}
SKIP_INSPECT=${SKIP_INSPECT:-}
EXTRA_EXPORT_ARGS=${EXTRA_EXPORT_ARGS:-}

die() { echo "FATAL: $*" >&2; exit 1; }

case "$VERIFY_LOAD" in
    1|true|TRUE|yes|YES) VERIFY_LOAD_ARG=--verify-load; VALIDATE_LOAD_ARG= ;;
    0|false|FALSE|no|NO) VERIFY_LOAD_ARG=; VALIDATE_LOAD_ARG=--allow-no-verify-load ;;
    *) die "VERIFY_LOAD must be 1/0, true/false, or yes/no (got '$VERIFY_LOAD')" ;;
esac

: "${TD_ITER_DIR:?TD_ITER_DIR is required: the torch_dist iter_XXXXXXX directory (stage-1 output)}"
: "${HF_OUT_DIR:?HF_OUT_DIR is required: a fresh dir for the HF export}"
[ -f "$TD_ITER_DIR/common.pt" ] || die "$TD_ITER_DIR has no common.pt — not a torch_dist iter dir (did you pass DST_ROOT instead of .../torch_dist/iter_XXXXXXX?)"
if [ -e "$HF_OUT_DIR" ] && [ -n "$(ls -A "$HF_OUT_DIR" 2>/dev/null)" ]; then
    die "HF_OUT_DIR $HF_OUT_DIR is non-empty — the exporter refuses it; use a fresh dir"
fi
if [ -n "$TOKENIZER_DIR" ]; then
    [ -f "$TOKENIZER_DIR/tokenizer.json" ] || die "no tokenizer.json in TOKENIZER_DIR=$TOKENIZER_DIR"
fi
[ -f "$HF_ENV" ] || die "no EDF at $HF_ENV"
IMG=$(sed -n 's/^image *= *"\(.*\)"/\1/p' "$HF_ENV"); IMG=${IMG//'${USER}'/$USER}
case "$IMG" in /*) [ -f "$IMG" ] || die "container image $IMG missing — build it first: sbatch $REPO/cluster/container/build_container.sbatch";; esac

mkdir -p "$REPO/cluster/cache/tmp"

echo "TD_ITER_DIR   : $TD_ITER_DIR"
echo "HF_OUT_DIR    : $HF_OUT_DIR"
echo "TOKENIZER_DIR : ${TOKENIZER_DIR:-<none — tokenizer files skipped>}"
echo "VERIFY_LOAD   : $VERIFY_LOAD"
echo "HF_ENV        : $HF_ENV ($IMG)"

ENV_SETUP="export PYTHONPATH=$REPO; \
    export TMPDIR=$REPO/cluster/cache/tmp; \
    export TOKENIZERS_PARALLELISM=false; \
    export HF_HUB_OFFLINE=1; \
    export TRANSFORMERS_OFFLINE=1; \
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-72}; \
    cd $REPO"

# ---- 1. key inspection gate --------------------------------------------------
if [ -z "$SKIP_INSPECT" ]; then
    echo; echo "=== torch_dist key inspection: $TD_ITER_DIR ==="
    srun -lu --environment="$HF_ENV" --nodes=1 --ntasks=1 \
        --cpus-per-task "${SLURM_CPUS_PER_TASK:-72}" \
        bash -c "$ENV_SETUP; python cluster/inspect_torchdist_keys.py '$TD_ITER_DIR'"
fi

# ---- 1b. can this allocation finish the job? --------------------------------
# Metadata only: no tensor bytes are read. --verify-load reloads the whole finished model
# into CPU RAM (exporter/writer.py verify_from_pretrained), so an allocation smaller than
# the model is an OOM hours into the run — cheaper to refuse now.
# `srun -l` prefixes every line with the task id ("0: "), so strip that before matching a
# bare integer — otherwise the check silently degrades to the skip branch below.
MODEL_MIB=$(srun -lu --environment="$HF_ENV" --nodes=1 --ntasks=1 \
    --cpus-per-task "${SLURM_CPUS_PER_TASK:-72}" \
    bash -c "$ENV_SETUP; python - '$TD_ITER_DIR'" <<'PYEOF' | sed -n 's/^\([0-9]\+: \)\?\([0-9]\+\)$/\2/p' | tail -1
import math, sys, warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from megatron.core import dist_checkpointing
meta = dist_checkpointing.load_tensors_metadata(sys.argv[1])
model = [v for k, v in meta.items() if not k.startswith(("optimizer.", "opt_param_scheduler"))]
print(round(sum(math.prod(tuple(t.global_shape)) * t.dtype.itemsize for t in model) / 1024**2))
PYEOF
)
if [[ "$MODEL_MIB" =~ ^[0-9]+$ ]]; then
    echo "MODEL SIZE    : $((MODEL_MIB / 1024)) GiB of bf16 weights to export"
    MEM_MIB=${SLURM_MEM_PER_NODE:-0}
    if [ -n "$VERIFY_LOAD_ARG" ] && [ "$MEM_MIB" -gt 0 ] && [ "$MEM_MIB" -lt $((MODEL_MIB * 5 / 4)) ]; then
        die "VERIFY_LOAD=1 reloads the whole $((MODEL_MIB / 1024)) GiB model into RAM but this job has only $((MEM_MIB / 1024)) GiB.
       Either raise the allocation:  sbatch --mem=$(( (MODEL_MIB * 5 / 4 + 1023) / 1024 * 1024 )) --partition=normal --time=08:00:00 ...
       or skip the reload:          VERIFY_LOAD=0  (weaker certificate; shard-level bit-identity is still verified)"
    fi
else
    echo "WARNING: could not size the model from metadata (got '$MODEL_MIB'); skipping the memory check" >&2
fi

# ---- 2. export ---------------------------------------------------------------
EXPORT_CMD="python -m exporter.export \
    --checkpoint-dir '$TD_ITER_DIR' \
    --output-dir '$HF_OUT_DIR' \
    ${TOKENIZER_DIR:+--tokenizer-dir '$TOKENIZER_DIR'} \
    $VERIFY_LOAD_ARG \
    --log-level INFO \
    $EXTRA_EXPORT_ARGS"

echo; echo "=== exporter.export ==="
echo "CMD: $EXPORT_CMD"
srun -lu --environment="$HF_ENV" --nodes=1 --ntasks=1 \
    --cpus-per-task "${SLURM_CPUS_PER_TASK:-72}" \
    bash -c "$ENV_SETUP; $EXPORT_CMD"

[ -f "$HF_OUT_DIR/conversion_info.json" ] || die "export srun returned 0 but no conversion_info.json at $HF_OUT_DIR"
[ ! -e "$HF_OUT_DIR/.export_incomplete" ] || die "export returned with $HF_OUT_DIR/.export_incomplete still present"
python3 "$REPO/cluster/validate_stage2_export.py" \
    "$HF_OUT_DIR" "$TD_ITER_DIR" $VALIDATE_LOAD_ARG \
    || die "export provenance or verification settings are not pipeline-ready"
echo; echo "=== conversion_info.json ==="
cat "$HF_OUT_DIR/conversion_info.json"
echo
echo "STAGE 2 DONE: $HF_OUT_DIR"
echo "END TIME: $(date)"
