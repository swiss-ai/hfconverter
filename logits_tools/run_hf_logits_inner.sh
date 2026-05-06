#!/bin/bash

set -euo pipefail

if [[ "${SKIP_TRANSFORMERS_INSTALL}" != "1" && -n "${TRANSFORMERS_BRANCH}" ]]; then
  pip install -v --no-cache-dir --no-build-isolation --no-deps --force-reinstall -U \
    "transformers @ git+https://github.com/swiss-ai/transformers.git@${TRANSFORMERS_BRANCH}"
  pip install -v --no-cache-dir -U "huggingface-hub>=0.34.0,<1.0"
fi

cmd=(
  python "${PY_SCRIPT}"
  --hf-dir "${HF_CKPT_DIR}"
  --prompt "${PROMPT}"
  --out-pt "${OUT_PT}"
  --dtype "${HF_DTYPE}"
  --device "${HF_DEVICE}"
  --device-map "${HF_DEVICE_MAP}"
)
if [[ -n "${OUT_REPORT}" ]]; then
  cmd+=(--out-report "${OUT_REPORT}")
fi
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  cmd+=(--trust-remote-code)
fi

"${cmd[@]}"
