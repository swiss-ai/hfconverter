#!/bin/bash
# Overlay the live vLLM checkout's apertus2 support onto the image's stock vllm 0.28.0.
#
# The checkout (apertus2/k3, based on v0.27.1) adds exactly six pure-Python production
# files on top of upstream: the apertus2 model, its registry line, the SSSGLU activation,
# and three MoE-activation allowlist touches. Everything compiled that apertus2 needs —
# kimi_k3 kernels, FlashKDA (vllm/_flashkda_C.abi3.so), MoE kernels — already ships in the
# pip wheel, so patching the installed package is the whole port. --fuzz=3 absorbs upstream
# context drift between 0.27.1 and 0.28.0 (verified against the 0.28.0 wheel: the one
# drifted hunk lands SSSGLU in triton_moe's _supports_activation list, where it belongs).
#
# Run INSIDE the container at job start (the EDF is writable; nothing persists across
# jobs, so every job starts from the pristine venv):
#   bash /iopsstor/scratch/cscs/$USER/hfconverter/cluster/apply_apertus2_to_vllm.sh
set -euo pipefail

VLLM_CHECKOUT=${VLLM_CHECKOUT:-/iopsstor/scratch/cscs/$USER/vllm}
VLLM_PY=${VLLM_PY:-/opt/vllm/bin/python}
BASE_TAG=${BASE_TAG:-v0.27.1}

SITE=$("$VLLM_PY" -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))")
echo "checkout : $VLLM_CHECKOUT ($(git -C "$VLLM_CHECKOUT" describe --tags))"
echo "target   : $SITE ($("$VLLM_PY" -c 'import vllm; print(vllm.__version__)'))"

git -C "$VLLM_CHECKOUT" diff "$BASE_TAG"..HEAD -- \
    vllm/model_executor/models/apertus2.py \
    vllm/model_executor/models/registry.py \
    vllm/model_executor/layers/activation.py \
    vllm/model_executor/layers/fused_moe/activation.py \
    vllm/model_executor/layers/fused_moe/experts/fused_batched_moe.py \
    vllm/model_executor/layers/fused_moe/experts/triton_moe.py \
  | patch -p1 --fuzz=3 -d "$SITE"

"$VLLM_PY" - <<'EOF'
from vllm.model_executor.models.registry import ModelRegistry
assert "Apertus2ForCausalLM" in ModelRegistry.get_supported_archs(), "apertus2 not registered"
print("vllm overlay OK: Apertus2ForCausalLM registered")
EOF
