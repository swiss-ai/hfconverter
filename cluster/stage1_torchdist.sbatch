#!/bin/bash
#SBATCH --job-name=convert-checkpoint
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=convert-checkpoint-%j.out
#SBATCH --error=convert-checkpoint-%j.err

set -euo pipefail

# Usage:
#   TRUST_LEGACY_CHECKPOINT=1 \
#     sbatch --export=ALL cluster/stage1_torchdist.sbatch SRC_CKPT DST_ROOT
#
# Optional source-profile overrides:
#   STAGE1_ENV=/absolute/path/to/environment.toml \
#   TRUST_LEGACY_CHECKPOINT=1 \
#   SRC_TP=1 SRC_PP=1 SRC_EP=4 SRC_ETP=1 SRC_CP=1 PRECISION=bf16 \
#   ROUTING_TYPE=seq_aux_loss MOE_AUX_LOSS_COEFF=1e-4 \
#   INIT_METHOD_STD=0.0360844 NORM_EPSILON=1e-5 SANDWICH_NORM=0 \
#   WINDOW_SIZE='(1024,0)' WINDOW_ATTN_SKIP_FREQ='([1,1,1,1,1,0]*2+[1,1,1,1,0])' \
#   NO_ROPE_FREQ='([0,0,0,0,0,1]*2+[0,0,0,0,1])' \
#     sbatch --export=ALL \
#     cluster/stage1_torchdist.sbatch SRC_CKPT DST_ROOT
#
# The defaults describe the plain 1.5B checkpoint. The sandwich checkpoint needs
# SANDWICH_NORM=1 and MOE_AUX_LOSS_COEFF=1e-3. The tiny QB fixture needs:
#   ROUTING_TYPE='seq_aux_loss quantile_balancing'
#   MOE_AUX_LOSS_COEFF='1e-4 0.0'
#   INIT_METHOD_STD=0.02 SANDWICH_NORM=1

# Read the two paths, then describe the source checkpoint's shard topology.
SRC_CKPT=${1:?"usage: sbatch cluster/stage1_torchdist.sbatch SRC_CKPT DST_ROOT"}
DST_ROOT=${2:?"usage: sbatch cluster/stage1_torchdist.sbatch SRC_CKPT DST_ROOT"}
SRC_TP=${SRC_TP:-1}
SRC_PP=${SRC_PP:-1}
SRC_EP=${SRC_EP:-4}
SRC_ETP=${SRC_ETP:-1}
SRC_CP=${SRC_CP:-1}
PRECISION=${PRECISION:-bf16}
# These explicit values cover native checkpoint-restore gaps and make the source profile visible.
# The defaults match the plain 1.5B checkpoint; space-separated values support the tiny QB fixture.
ROUTING_TYPE=${ROUTING_TYPE:-seq_aux_loss}
MOE_AUX_LOSS_COEFF=${MOE_AUX_LOSS_COEFF:-1e-4}
INIT_METHOD_STD=${INIT_METHOD_STD:-0.0360844}
NORM_EPSILON=${NORM_EPSILON:-1e-5}
SANDWICH_NORM=${SANDWICH_NORM:-0}
# Sliding-window and NoPE patterns, in the fork's own CLI spelling. These are NOT optional
# polish: megatron/training/checkpointing.py:set_args_from_checkpoint restores ~90 arguments and
# window_size / window_attn_skip_freq / no_rope_freq are in none of them, so --use-checkpoint-args
# leaves all three at None. Neither has a parameter footprint, so the strict tensor load below
# still succeeds and the resulting torch_dist checkpoint records "full attention on every layer,
# RoPE on every layer". Stage 2 derives config.json from exactly those saved arguments, so an
# omission here is not a crash — it is a model that loads, generates fluent text, and computes
# the wrong attention on every layer. Empty means "the source really had none".
WINDOW_SIZE=${WINDOW_SIZE:-}
WINDOW_ATTN_SKIP_FREQ=${WINDOW_ATTN_SKIP_FREQ:-}
NO_ROPE_FREQ=${NO_ROPE_FREQ:-}
MEGATRON_PATH=${MEGATRON_PATH:-/iopsstor/scratch/cscs/${USER}/Megatron-LM-MoE}
# Pyxis consumes an EDF file, not the cluster/container source directory.  The
# repository EDF points at the already-built images/apertus2-hf.sqsh image.
STAGE1_ENV=${STAGE1_ENV:-/iopsstor/scratch/cscs/${USER}/hfconverter/cluster/edf/apertus2-hf.toml}
TRUST_LEGACY_CHECKPOINT=${TRUST_LEGACY_CHECKPOINT:-0}

for topology_name in SRC_TP SRC_PP SRC_EP SRC_ETP SRC_CP; do
  topology_value=${!topology_name}
  if [[ ! ${topology_value} =~ ^[1-9][0-9]*$ ]]; then
    echo "${topology_name} must be a positive integer, got: ${topology_value}" >&2
    exit 1
  fi
done
WORLD_SIZE=$((SRC_TP * SRC_PP * SRC_EP))
GLOBAL_BATCH_SIZE=$((WORLD_SIZE / (SRC_TP * SRC_PP)))

# Fail before requesting work from Megatron when the small job profile cannot handle it.
if ((SRC_TP % SRC_ETP != 0)); then
  echo "SRC_ETP must divide SRC_TP; got SRC_TP=${SRC_TP}, SRC_ETP=${SRC_ETP}." >&2
  exit 1
fi
# SRC_CP describes the SOURCE run, not this job. Context parallelism splits the sequence, never
# the parameters: a CP=4 run writes exactly the same mp_rank_<tp>_<ep> directories as a CP=1 run
# (get_checkpoint_name takes no CP rank), and check_checkpoint_args never compares it. So Stage 1
# reads a CP>1 checkpoint at CP=1 — which it must, since WORLD_SIZE below is TP*PP*EP and this job
# has four GPUs. The value is still validated and echoed because validate_source_profile.py
# compares it against the saved args, where a mismatch means the operator described a different run.
if ((WORLD_SIZE > 4)); then
  echo "Source topology requires ${WORLD_SIZE} ranks, but this job has 4 GPUs." >&2
  exit 1
fi
if [[ ! -d ${SRC_CKPT} ]]; then
  echo "Source checkpoint does not exist: ${SRC_CKPT}" >&2
  exit 1
fi
if [[ ${TRUST_LEGACY_CHECKPOINT} != 1 ]]; then
  echo "Legacy Megatron checkpoints contain Python pickle metadata." >&2
  echo "Set TRUST_LEGACY_CHECKPOINT=1 only when you trust every file in: ${SRC_CKPT}" >&2
  exit 1
fi
case ${STAGE1_ENV} in
  /*) ;;
  *) echo "STAGE1_ENV must be an absolute EDF path, got: ${STAGE1_ENV}" >&2; exit 1 ;;
esac
if [[ ! -f ${STAGE1_ENV} || ! -r ${STAGE1_ENV} ]]; then
  echo "Stage 1 environment file is not readable: ${STAGE1_ENV}" >&2
  exit 1
fi
STAGE1_ENV=$(realpath -e -- "${STAGE1_ENV}")
mapfile -t STAGE1_IMAGE_ENTRIES < <(sed -n 's/^image *= *"\(.*\)"/\1/p' "${STAGE1_ENV}")
if ((${#STAGE1_IMAGE_ENTRIES[@]} != 1)); then
  echo "Stage 1 environment must contain exactly one image entry: ${STAGE1_ENV}" >&2
  exit 1
fi
STAGE1_IMAGE=${STAGE1_IMAGE_ENTRIES[0]//'${USER}'/${USER}}
if [[ ${STAGE1_IMAGE} == *'${'* ]]; then
  echo "Stage 1 image contains an unsupported placeholder: ${STAGE1_IMAGE}" >&2
  exit 1
fi
case ${STAGE1_IMAGE} in
  /*) ;;
  *) echo "Stage 1 image must be an absolute local path, got: ${STAGE1_IMAGE}" >&2; exit 1 ;;
esac
if [[ ! -f ${STAGE1_IMAGE} || ! -r ${STAGE1_IMAGE} ]]; then
  echo "Stage 1 image is not readable: ${STAGE1_IMAGE}" >&2
  exit 1
fi
STAGE1_IMAGE=$(realpath -e -- "${STAGE1_IMAGE}")
if [[ ${PRECISION} != bf16 && ${PRECISION} != fp16 ]]; then
  echo "PRECISION must be bf16 or fp16, got: ${PRECISION}" >&2
  exit 1
fi
if [[ ! ${INIT_METHOD_STD} =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
  echo "INIT_METHOD_STD must be a non-negative number, got: ${INIT_METHOD_STD}" >&2
  exit 1
fi
if [[ ! ${NORM_EPSILON} =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
  echo "NORM_EPSILON must be a non-negative number, got: ${NORM_EPSILON}" >&2
  exit 1
fi
case ${SANDWICH_NORM} in
  0|false|FALSE|no|NO|1|true|TRUE|yes|YES) ;;
  *) echo "SANDWICH_NORM must be 1/0, true/false, or yes/no; got: ${SANDWICH_NORM}" >&2; exit 1 ;;
esac

# The three pattern arguments are forwarded verbatim to Megatron, which evaluates them with
# arguments.py:_eval_pattern -- a bare eval() behind exactly this character whitelist. Reject
# anything else here so a typo fails at submission instead of inside the container.
# Held in a variable because =~ treats a quoted pattern as a literal string; and written with a
# leading ']' because a bracket expression has no escape character -- '\]' would close the set.
PATTERN_CHARSET='^[][0-9,()+*]+$'
for pattern_name in WINDOW_SIZE WINDOW_ATTN_SKIP_FREQ NO_ROPE_FREQ; do
  pattern_value=${!pattern_name}
  if [[ -n ${pattern_value} && ! ${pattern_value} =~ $PATTERN_CHARSET ]]; then
    echo "${pattern_name} may contain only digits and , [ ] ( ) + * ; got: ${pattern_value}" >&2
    exit 1
  fi
done
# is_layer_window_attention treats a missing skip frequency as "every layer slides", so a
# per-layer pattern without a window is not a milder request -- it is a different model.
if [[ -n ${WINDOW_ATTN_SKIP_FREQ} && -z ${WINDOW_SIZE} ]]; then
  echo "WINDOW_ATTN_SKIP_FREQ needs WINDOW_SIZE; without it no layer slides at all." >&2
  exit 1
fi

TRACKER=${SRC_CKPT}/latest_checkpointed_iteration.txt
if [[ ! -f ${TRACKER} ]]; then
  echo "Source checkpoint has no iteration tracker: ${TRACKER}" >&2
  exit 1
fi
SOURCE_ITERATION=$(<"${TRACKER}")
if [[ ! ${SOURCE_ITERATION} =~ ^[0-9]+$ ]]; then
  echo "Source iteration must be numeric, got: ${SOURCE_ITERATION}" >&2
  exit 1
fi
printf -v SOURCE_ITER_DIR 'iter_%07d' "$((10#${SOURCE_ITERATION}))"
SUCCESS_MARKER=${DST_ROOT}/stage1-complete
if [[ -e ${DST_ROOT} && ! -d ${DST_ROOT} ]]; then
  echo "Destination exists and is not a directory: ${DST_ROOT}" >&2
  exit 1
fi
if [[ -d ${DST_ROOT} && -n $(ls -A "${DST_ROOT}") ]]; then
  echo "Destination is not empty; preserve it for diagnosis and use a fresh root: ${DST_ROOT}" >&2
  exit 1
fi

# Make the two paths and topology visible inside the container.
mkdir -p "${DST_ROOT}"
export SRC_CKPT DST_ROOT SRC_TP SRC_PP SRC_EP SRC_ETP SRC_CP PRECISION TRUST_LEGACY_CHECKPOINT
export ROUTING_TYPE MOE_AUX_LOSS_COEFF INIT_METHOD_STD NORM_EPSILON SANDWICH_NORM
export WINDOW_SIZE WINDOW_ATTN_SKIP_FREQ NO_ROPE_FREQ
export MEGATRON_PATH WORLD_SIZE GLOBAL_BATCH_SIZE SOURCE_ITER_DIR
MEGATRON_LOG=${DST_ROOT}/stage1-megatron.log
echo "Stage 1 environment: ${STAGE1_ENV}"
echo "Stage 1 image      : ${STAGE1_IMAGE}"
echo "Source context parallel size: ${SRC_CP} (Stage 1 always runs at 1; CP shards no parameters)"
echo "Sliding window     : ${WINDOW_SIZE:-<none — every layer uses full attention>}"
echo "Window skip pattern: ${WINDOW_ATTN_SKIP_FREQ:-<none — every layer slides if a window is set>}"
echo "NoPE pattern       : ${NO_ROPE_FREQ:-<none — every layer applies RoPE>}"

# Start one container task. torchrun below creates the individual Megatron ranks.
srun \
  --mpi=pmix \
  --network=disable_rdzv_get \
  --environment="${STAGE1_ENV}" \
  --export=ALL,SRC_CKPT,DST_ROOT,SRC_TP,SRC_PP,SRC_EP,SRC_ETP,SRC_CP,PRECISION,TRUST_LEGACY_CHECKPOINT,ROUTING_TYPE,MOE_AUX_LOSS_COEFF,INIT_METHOD_STD,NORM_EPSILON,SANDWICH_NORM,WINDOW_SIZE,WINDOW_ATTN_SKIP_FREQ,NO_ROPE_FREQ,MEGATRON_PATH,WORLD_SIZE,GLOBAL_BATCH_SIZE,SOURCE_ITER_DIR \
  -u bash -lc '
set -euo pipefail
if [[ ${TRUST_LEGACY_CHECKPOINT} != 1 ]]; then
  echo "Stage 1 lost its trusted-checkpoint assertion before entering the container." >&2
  exit 1
fi
# PyTorch 2.6+ defaults torch.load() to its restricted weights-only unpickler.
# This trusted legacy Megatron checkpoint also stores arguments and NumPy RNG metadata,
# so the old fork torch.load() call sites require normal pickle loading. Scope the
# compatibility switch to this container process; Stage 2 never inherits it.
unset TORCH_FORCE_WEIGHTS_ONLY_LOAD
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
echo "WARNING: unrestricted Python pickle loading is enabled for ${SRC_CKPT}" >&2
echo "WARNING: checkpoint code can execute with access to the job mounts and user permissions." >&2
export PYTHONPATH="${MEGATRON_PATH}:${PYTHONPATH:-}"
if [[ ! -f ${MEGATRON_PATH}/pretrain_gpt.py ]]; then
  echo "Megatron entry point is not visible: ${MEGATRON_PATH}/pretrain_gpt.py" >&2
  exit 1
fi
if [[ ! -d ${SRC_CKPT} ]]; then
  echo "Source checkpoint is not visible inside the container: ${SRC_CKPT}" >&2
  exit 1
fi
if [[ ! -d ${DST_ROOT} ]]; then
  echo "Destination is not visible inside the container: ${DST_ROOT}" >&2
  exit 1
fi

# Convert the simple PRECISION value into the Megatron boolean precision flag.
if [[ ${PRECISION} == bf16 ]]; then
  precision_flag=--bf16
else
  precision_flag=--fp16
fi

# Both Megatron flags accept one or more values. Arrays preserve the QB pair as two arguments.
read -r -a routing_type_args <<< "${ROUTING_TYPE}"
read -r -a aux_loss_coeff_args <<< "${MOE_AUX_LOSS_COEFF}"
if ((${#routing_type_args[@]} != ${#aux_loss_coeff_args[@]})); then
  echo "ROUTING_TYPE and MOE_AUX_LOSS_COEFF must contain the same number of values." >&2
  exit 1
fi
case ${SANDWICH_NORM} in
  1|true|TRUE|yes|YES) sandwich_args=(--sandwich-norm) ;;
  *) sandwich_args=() ;;
esac

# An unset pattern must contribute NO flag: --window-size "" would parse as an empty tuple
# rather than "no sliding window".
attention_pattern_args=()
if [[ -n ${WINDOW_SIZE} ]]; then
  attention_pattern_args+=(--window-size "${WINDOW_SIZE}")
fi
if [[ -n ${WINDOW_ATTN_SKIP_FREQ} ]]; then
  attention_pattern_args+=(--window-attn-skip-freq "${WINDOW_ATTN_SKIP_FREQ}")
fi
if [[ -n ${NO_ROPE_FREQ} ]]; then
  attention_pattern_args+=(--no-rope-freq "${NO_ROPE_FREQ}")
fi

# Build the model, load legacy weights, save torch_dist, and exit without training.
# Router fp32 is a supported-source invariant required by the Stage 2/HF contract.
# The source-profile preflight protects arguments the fork cannot restore safely.
python -m torch.distributed.run \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=29573 \
  --nproc_per_node="${WORLD_SIZE}" \
  "${MEGATRON_PATH}/pretrain_gpt.py" \
  --load "${SRC_CKPT}" \
  --ckpt-format torch \
  --ckpt-convert-format torch_dist \
  --ckpt-convert-save "${DST_ROOT}" \
  --use-checkpoint-args \
  --use-mp-args-from-checkpoint-args \
  --tensor-model-parallel-size "${SRC_TP}" \
  --pipeline-model-parallel-size "${SRC_PP}" \
  --expert-model-parallel-size "${SRC_EP}" \
  --expert-tensor-parallel-size "${SRC_ETP}" \
  --context-parallel-size 1 \
  --transformer-impl transformer_engine \
  "${precision_flag}" \
  --init-method-std "${INIT_METHOD_STD}" \
  --norm-epsilon "${NORM_EPSILON}" \
  "${sandwich_args[@]}" \
  "${attention_pattern_args[@]}" \
  --moe-router-dtype fp32 \
  --moe-router-load-balancing-type "${routing_type_args[@]}" \
  --moe-aux-loss-coeff "${aux_loss_coeff_args[@]}" \
  --sequence-parallel \
  --skip-train \
  --no-load-optim \
  --no-save-optim \
  --no-load-rng \
  --no-save-rng \
  --no-ckpt-fully-parallel-save \
  --micro-batch-size 1 \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --no-one-logger
' 2>&1 | tee "${MEGATRON_LOG}"

if grep -qF 'load_return:' "${MEGATRON_LOG}"; then
  echo "Megatron retried a strict model load non-strictly; refusing this conversion." >&2
  exit 1
fi
EXPECTED_ITER_DIR=${DST_ROOT}/torch_dist/${SOURCE_ITER_DIR}
OUTPUT_TRACKER=${DST_ROOT}/torch_dist/latest_checkpointed_iteration.txt
if [[ ! -f ${EXPECTED_ITER_DIR}/common.pt || ! -f ${EXPECTED_ITER_DIR}/.metadata || ! -f ${EXPECTED_ITER_DIR}/metadata.json ]]; then
  echo "Megatron returned without a complete distributed checkpoint in ${EXPECTED_ITER_DIR}." >&2
  exit 1
fi
if [[ ! -f ${OUTPUT_TRACKER} ]]; then
  echo "Megatron returned without creating ${OUTPUT_TRACKER}." >&2
  exit 1
fi
OUTPUT_ITERATION=$(<"${OUTPUT_TRACKER}")
if [[ ${OUTPUT_ITERATION} != "${SOURCE_ITERATION}" ]]; then
  echo "Output tracker says ${OUTPUT_ITERATION}, expected ${SOURCE_ITERATION}." >&2
  exit 1
fi

# The pipeline trusts only this wrapper-owned marker. Write it last and rename it atomically.
printf '%s\n' "${SOURCE_ITER_DIR}" > "${SUCCESS_MARKER}.tmp"
mv -- "${SUCCESS_MARKER}.tmp" "${SUCCESS_MARKER}"
echo "Converted checkpoint: ${EXPECTED_ITER_DIR}"
