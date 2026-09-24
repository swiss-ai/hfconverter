#!/bin/bash
# Overlay the live vLLM checkout's apertus2 support onto the image's stock vllm 0.28.0.
#
# The checkout (apertus2/v0.28.0, based exactly on tag v0.28.0) adds pure-Python files on
# top of upstream: the apertus2 model (softmax + KDA layers), its two registry lines, the
# KDA splitting-op entry in the compilation config, the SSSGLU activation, and three
# MoE-activation allowlist touches. Everything compiled that apertus2 needs — kimi_k3 KDA
# kernels, FlashKDA (vllm/_flashkda_C.abi3.so), MoE kernels — already ships in the pip
# wheel, so patching the installed package is the whole port. BASE_TAG must be the tag the
# checkout is based on (v0.28.0 == the wheel, so the patch applies without drift; --fuzz=3
# stays for the day the two diverge again).
#
# Run INSIDE the container at job start (the EDF is writable; nothing persists across
# jobs, so every job starts from the pristine venv):
#   bash /iopsstor/scratch/cscs/$USER/hfconverter/cluster/apply_apertus2_to_vllm.sh
set -euo pipefail

VLLM_CHECKOUT=${VLLM_CHECKOUT:-/iopsstor/scratch/cscs/$USER/vllm-v0280}
VLLM_PY=${VLLM_PY:-/opt/vllm/bin/python}
BASE_TAG=${BASE_TAG:-v0.28.0}
# VLLM_REF=HEAD overlays the committed branch tip; VLLM_REF=WORKTREE overlays the checkout's
# working tree (tracked files, uncommitted edits included) for testing before a commit.
VLLM_REF=${VLLM_REF:-HEAD}

SITE=$("$VLLM_PY" -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))")
echo "checkout : $VLLM_CHECKOUT ($(git -C "$VLLM_CHECKOUT" describe --tags))"
echo "target   : $SITE ($("$VLLM_PY" -c 'import vllm; print(vllm.__version__)'))"

if [ "$VLLM_REF" = WORKTREE ]; then DIFF_RANGE="$BASE_TAG"; else DIFF_RANGE="$BASE_TAG..$VLLM_REF"; fi
echo "overlay  : git diff $DIFF_RANGE"
git -C "$VLLM_CHECKOUT" diff "$DIFF_RANGE" -- \
    vllm/model_executor/models/apertus2.py \
    vllm/models/kimi_k3/nvidia/ops/third_party/kda/chunk.py \
    vllm/models/kimi_k3/nvidia/ops/third_party/kda/fused_recurrent.py \
    vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py \
    vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py \
    vllm/model_executor/models/registry.py \
    vllm/config/compilation.py \
    vllm/model_executor/layers/activation.py \
    vllm/model_executor/layers/fused_moe/activation.py \
    vllm/model_executor/layers/fused_moe/experts/fused_batched_moe.py \
    vllm/model_executor/layers/fused_moe/experts/triton_moe.py \
  | patch -p1 --fuzz=3 -d "$SITE"

"$VLLM_PY" - <<'PYEOF'
from vllm.config import CompilationConfig
from vllm.model_executor.models.registry import ModelRegistry

archs = ModelRegistry.get_supported_archs()
for arch in ("Apertus2ForCausalLM", "Apertus2KDAForCausalLM"):
    assert arch in archs, f"{arch} not registered"
assert "vllm::apertus2_kda_attention_core" in CompilationConfig._attention_ops, (
    "KDA core is not a splitting op: vllm/config/compilation.py hunk did not apply"
)
print("vllm overlay OK: Apertus2ForCausalLM + Apertus2KDAForCausalLM registered")
PYEOF
