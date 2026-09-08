#!/bin/bash

set -euo pipefail

echo "NODE_RANK=${SLURM_NODEID}"
cd "${MEGATRON_PATH}"
export PYTHONPATH="${MEGATRON_PATH}"
"${HFCONVERTER_PYTHON:-python}" "$(dirname "${BASH_SOURCE[0]}")/../check_megatron_source.py" "${MEGATRON_PATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

cmd=(
  "${HFCONVERTER_PYTHON:-python}" -m torch.distributed.run
  --nnodes="${SLURM_NNODES}"
  --nproc-per-node="${NPROC_PER_NODE}"
  --node-rank="${SLURM_NODEID}"
  --rdzv-backend=c10d
  --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}"
  "${PY_SCRIPT}"
  --load "${CKPT_DIR}"
  --ckpt-step "${ITER}"
  --ckpt-format torch_dist
  --auto-detect-ckpt-format
  --use-checkpoint-args
  --use-mp-args-from-checkpoint-args
  --tokenizer-type HuggingFaceTokenizer
  --tokenizer-model "${TOKENIZER}"
  --no-use-tokenizer-model-from-checkpoint-args
  --distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES}"
  --no-load-optim
  --no-load-rng
  --prompt "${PROMPT}"
  --out-pt "${OUT_PT}"
)
if [[ -n "${OUT_REPORT}" ]]; then
  cmd+=(--out-report "${OUT_REPORT}")
fi

"${cmd[@]}"
