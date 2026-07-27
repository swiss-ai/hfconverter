#!/usr/bin/env bash
# Submit the two jobs that turn one legacy Megatron checkpoint into one HF model.
#
# This is the normal conversion entry point for a legacy checkpoint.
#
# Usage:
#   STAGE1_ENV=/absolute/path/to/environment.toml \
#   TRUST_LEGACY_CHECKPOINT=1 \
#   SRC_TP=1 SRC_PP=1 SRC_EP=4 SRC_ETP=1 SRC_CP=1 PRECISION=bf16 \
#   ROUTING_TYPE=seq_aux_loss MOE_AUX_LOSS_COEFF=1e-4 \
#   INIT_METHOD_STD=0.0360844 NORM_EPSILON=1e-5 SANDWICH_NORM=0 \
#     cluster/submit_conversion.sh SOURCE_CHECKPOINT OUTPUT_ROOT
#
# Result:
#   OUTPUT_ROOT/torch_dist/iter_XXXXXXX/   normalized Megatron checkpoint
#   OUTPUT_ROOT/hf/                        final Hugging Face model
#
# The script only validates inputs and calls sbatch.  Model weights are read later
# by the two dependent SLURM jobs.
#
# Composability hook:
#   JOB_ID_FILE=/existing/temporary/file
# writes the Stage 1 and Stage 2 job IDs, one per line, after both submissions succeed.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
MEGATRON_PATH=${MEGATRON_PATH:-/iopsstor/scratch/cscs/${USER}/Megatron-LM-MoE}
TOKENIZER_DIR=${TOKENIZER_DIR:-$MEGATRON_PATH/_research/data/apertus-mul-200k-tokenizer}
STAGE1_ENV=${STAGE1_ENV:-$REPO/cluster/edf/apertus-moe-hf.toml}
TRUST_LEGACY_CHECKPOINT=${TRUST_LEGACY_CHECKPOINT:-0}
HF_ENV=${HF_ENV:-$REPO/cluster/edf/apertus-moe-hf.toml}
EXTRA_EXPORT_ARGS=${EXTRA_EXPORT_ARGS:-}
LOG_DIR=${LOG_DIR:-$REPO/cluster/logs}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

SOURCE_CHECKPOINT=${1:-}
OUTPUT_ROOT=${2:-}
[ -n "$SOURCE_CHECKPOINT" ] && [ -n "$OUTPUT_ROOT" ] \
    || die "usage: cluster/submit_conversion.sh SOURCE_CHECKPOINT OUTPUT_ROOT"
[ "$#" -eq 2 ] || die "expected exactly two paths: SOURCE_CHECKPOINT OUTPUT_ROOT"
if [ -n "${JOB_ID_FILE:-}" ]; then
    [ -f "$JOB_ID_FILE" ] && [ -w "$JOB_ID_FILE" ] \
        || die "JOB_ID_FILE must name a writable existing temporary file: $JOB_ID_FILE"
fi
case "$SOURCE_CHECKPOINT" in
    /*) ;;
    *) die "SOURCE_CHECKPOINT must be an absolute path: $SOURCE_CHECKPOINT" ;;
esac
case "$OUTPUT_ROOT" in
    /*) ;;
    *) die "OUTPUT_ROOT must be an absolute path: $OUTPUT_ROOT" ;;
esac
# Canonicalize before the containment check. `realpath -m` resolves existing symlink prefixes
# while allowing the fresh output leaf not to exist yet.
SOURCE_CHECKPOINT=$(realpath -e -- "$SOURCE_CHECKPOINT") \
    || die "cannot resolve source checkpoint: $SOURCE_CHECKPOINT"
OUTPUT_ROOT=$(realpath -m -- "$OUTPUT_ROOT") \
    || die "cannot resolve output root: $OUTPUT_ROOT"
case "${OUTPUT_ROOT%/}/" in
    "${SOURCE_CHECKPOINT%/}/"*) \
        die "OUTPUT_ROOT must not be inside the source checkpoint: $OUTPUT_ROOT" ;;
esac

# A generic conversion must not silently borrow the plain checkpoint's profile.
# These are the values Megatron cannot always recover safely from a legacy checkpoint.
required_profile=(
    SRC_TP SRC_PP SRC_EP SRC_ETP SRC_CP PRECISION
    ROUTING_TYPE MOE_AUX_LOSS_COEFF
    INIT_METHOD_STD NORM_EPSILON SANDWICH_NORM
)
for name in "${required_profile[@]}"; do
    [ -n "${!name:-}" ] || die "$name is required; copy it from the source checkpoint profile"
done
[ "$TRUST_LEGACY_CHECKPOINT" = 1 ] \
    || die "set TRUST_LEGACY_CHECKPOINT=1 only after confirming the legacy checkpoint is trusted"

[ -d "$SOURCE_CHECKPOINT" ] || die "source checkpoint does not exist: $SOURCE_CHECKPOINT"
[ -f "$SOURCE_CHECKPOINT/latest_checkpointed_iteration.txt" ] \
    || die "source checkpoint has no latest_checkpointed_iteration.txt: $SOURCE_CHECKPOINT"
SOURCE_ITERATION=$(<"$SOURCE_CHECKPOINT/latest_checkpointed_iteration.txt")
[[ "$SOURCE_ITERATION" =~ ^[0-9]+$ ]] \
    || die "source checkpoint iteration must be numeric, got: $SOURCE_ITERATION"
printf -v ITERATION_DIR 'iter_%07d' "$((10#$SOURCE_ITERATION))"

[ -f "$MEGATRON_PATH/pretrain_gpt.py" ] \
    || die "Megatron fork is not readable at MEGATRON_PATH=$MEGATRON_PATH"
[ -f "$TOKENIZER_DIR/tokenizer.json" ] \
    || die "tokenizer.json is missing from TOKENIZER_DIR=$TOKENIZER_DIR"

for environment_name in STAGE1_ENV HF_ENV; do
    environment_file=${!environment_name}
    case "$environment_file" in
        /*) ;;
        *) die "$environment_name must be an absolute EDF path: $environment_file" ;;
    esac
    case "$environment_name" in
        STAGE1_ENV) environment_label="Stage 1" ;;
        HF_ENV) environment_label="HF" ;;
    esac
    [ -f "$environment_file" ] && [ -r "$environment_file" ] \
        || die "$environment_label environment file is not readable: $environment_file"
    environment_file=$(realpath -e -- "$environment_file") \
        || die "cannot resolve $environment_name=$environment_file"
    printf -v "$environment_name" '%s' "$environment_file"
done

validate_environment() {
    local label=$1 environment_file=$2 image
    local -a image_entries=()
    [ -r "$environment_file" ] \
        || die "$label environment file is not readable: $environment_file"
    mapfile -t image_entries < <(sed -n 's/^image *= *"\(.*\)"/\1/p' "$environment_file")
    [ "${#image_entries[@]}" -eq 1 ] \
        || die "$label environment must contain exactly one image entry: $environment_file"
    image=${image_entries[0]}
    image=${image//'${USER}'/$USER}
    [[ "$image" != *'${'* ]] \
        || die "$label image contains an unsupported placeholder: $image"
    case "$image" in
        /*) ;;
        *) die "$label image must be an absolute local path, got: $image" ;;
    esac
    [ -f "$image" ] && [ -r "$image" ] \
        || die "$label image is not readable: $image (build it with cluster/container/build_container.sbatch)"
}

validate_environment "Stage 1" "$STAGE1_ENV"
validate_environment "HF" "$HF_ENV"

if [ -e "$OUTPUT_ROOT" ] && [ ! -d "$OUTPUT_ROOT" ]; then
    die "output root exists and is not a directory: $OUTPUT_ROOT"
fi
if [ -d "$OUTPUT_ROOT" ] && [ -n "$(ls -A "$OUTPUT_ROOT" 2>/dev/null)" ]; then
    die "output root is not empty; preserve it and choose a fresh path: $OUTPUT_ROOT"
fi

for name in SRC_TP SRC_PP SRC_EP SRC_ETP SRC_CP; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] \
        || die "$name must be a positive integer, got: ${!name}"
done
[ "$SRC_CP" -eq 1 ] \
    || die "this Stage 1 implementation supports only SRC_CP=1, got: $SRC_CP"
((SRC_TP % SRC_ETP == 0)) \
    || die "SRC_ETP must divide SRC_TP; got SRC_TP=$SRC_TP and SRC_ETP=$SRC_ETP"
WORLD_SIZE=$((SRC_TP * SRC_PP * SRC_EP))
((WORLD_SIZE <= 4)) \
    || die "source topology needs $WORLD_SIZE ranks, but Stage 1 requests four GPUs"
case "$PRECISION" in
    bf16|fp16) ;;
    *) die "PRECISION must be bf16 or fp16, got: $PRECISION" ;;
esac
case "$SANDWICH_NORM" in
    0|false|FALSE|no|NO|1|true|TRUE|yes|YES) ;;
    *) die "SANDWICH_NORM must be a boolean value, got: $SANDWICH_NORM" ;;
esac
number_re='^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$'
[[ "$INIT_METHOD_STD" =~ $number_re ]] \
    || die "INIT_METHOD_STD must be a non-negative number, got: $INIT_METHOD_STD"
[[ "$NORM_EPSILON" =~ $number_re ]] \
    || die "NORM_EPSILON must be a non-negative number, got: $NORM_EPSILON"
read -r -a routing_values <<< "$ROUTING_TYPE"
read -r -a aux_values <<< "$MOE_AUX_LOSS_COEFF"
[ "${#routing_values[@]}" -eq "${#aux_values[@]}" ] \
    || die "ROUTING_TYPE and MOE_AUX_LOSS_COEFF must have the same number of values"
for value in "${aux_values[@]}"; do
    [[ "$value" =~ $number_re ]] \
        || die "every MOE_AUX_LOSS_COEFF value must be non-negative, got: $value"
done

# Verify the human-supplied values against the legacy checkpoint's saved Namespace.  The
# restricted reader skips tensor storages, so this does not load model weights on the login node.
python3 "$SCRIPT_DIR/validate_source_profile.py" "$SOURCE_CHECKPOINT" \
    --src-tp "$SRC_TP" --src-pp "$SRC_PP" --src-ep "$SRC_EP" \
    --src-etp "$SRC_ETP" --src-cp "$SRC_CP" --precision "$PRECISION" \
    --routing-type "$ROUTING_TYPE" --moe-aux-loss-coeff "$MOE_AUX_LOSS_COEFF" \
    --init-method-std "$INIT_METHOD_STD" --norm-epsilon "$NORM_EPSILON" \
    --sandwich-norm "$SANDWICH_NORM" \
    || die "the supplied parameters do not describe the source checkpoint"

TD_ITER_DIR=$OUTPUT_ROOT/torch_dist/$ITERATION_DIR
HF_OUT_DIR=$OUTPUT_ROOT/hf
mkdir -p "$LOG_DIR"

echo "Conversion plan"
echo "  source checkpoint : $SOURCE_CHECKPOINT"
echo "  source iteration  : $SOURCE_ITERATION"
echo "  source topology   : TP=$SRC_TP PP=$SRC_PP EP=$SRC_EP ETP=$SRC_ETP CP=$SRC_CP"
echo "  precision         : $PRECISION (router fp32 is enforced by Stage 1)"
echo "  routing           : $ROUTING_TYPE; aux coefficient: $MOE_AUX_LOSS_COEFF"
echo "  sandwich norm     : $SANDWICH_NORM"
echo "  Stage 1 environment: $STAGE1_ENV"
echo "  legacy pickle     : trusted by explicit operator assertion"
echo "  HF environment    : $HF_ENV"
echo "  torch_dist result : $TD_ITER_DIR"
echo "  HF result         : $HF_OUT_DIR"

stage1_job=$(env -u JOB_ID_FILE \
    SRC_TP="$SRC_TP" SRC_PP="$SRC_PP" SRC_EP="$SRC_EP" SRC_ETP="$SRC_ETP" \
    SRC_CP="$SRC_CP" PRECISION="$PRECISION" ROUTING_TYPE="$ROUTING_TYPE" \
    MOE_AUX_LOSS_COEFF="$MOE_AUX_LOSS_COEFF" INIT_METHOD_STD="$INIT_METHOD_STD" \
    NORM_EPSILON="$NORM_EPSILON" SANDWICH_NORM="$SANDWICH_NORM" \
    MEGATRON_PATH="$MEGATRON_PATH" STAGE1_ENV="$STAGE1_ENV" \
    TRUST_LEGACY_CHECKPOINT="$TRUST_LEGACY_CHECKPOINT" \
  sbatch --parsable --kill-on-invalid-dep=yes \
    --output="$LOG_DIR/%x-%j.log" --error="$LOG_DIR/%x-%j.log" \
    --export=ALL \
    "$SCRIPT_DIR/stage1_torchdist.sbatch" "$SOURCE_CHECKPOINT" "$OUTPUT_ROOT") \
    || die "could not submit Stage 1"
stage1_job=${stage1_job%%;*}

stage2_job=$(env -u JOB_ID_FILE \
    -u TORCH_FORCE_WEIGHTS_ONLY_LOAD -u TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD \
    REPO="$REPO" FORK="$MEGATRON_PATH" HF_ENV="$HF_ENV" \
    TD_ITER_DIR="$TD_ITER_DIR" HF_OUT_DIR="$HF_OUT_DIR" \
    TOKENIZER_DIR="$TOKENIZER_DIR" VERIFY_LOAD=1 SKIP_INSPECT= \
    EXTRA_EXPORT_ARGS="$EXTRA_EXPORT_ARGS" \
  sbatch --parsable --kill-on-invalid-dep=yes \
    --dependency="afterok:$stage1_job" \
    --output="$LOG_DIR/%x-%j.log" --error="$LOG_DIR/%x-%j.log" \
    --export=ALL \
    "$SCRIPT_DIR/stage2_export.sbatch") \
    || die "Stage 1 was submitted as $stage1_job, but Stage 2 submission failed"
stage2_job=${stage2_job%%;*}

if [ -n "${JOB_ID_FILE:-}" ]; then
    printf '%s\n%s\n' "$stage1_job" "$stage2_job" > "$JOB_ID_FILE"
fi

echo
echo "Submitted"
echo "  Stage 1 job: $stage1_job"
echo "  Stage 2 job: $stage2_job (starts only after Stage 1 succeeds)"
echo "  Watch       : squeue -u \$USER"
echo "  Logs        : $LOG_DIR/"
echo
echo "After job $stage2_job finishes successfully, the HF model is:"
echo "  $HF_OUT_DIR"
