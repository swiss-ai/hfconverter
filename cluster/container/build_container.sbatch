#!/bin/bash
# =============================================================================
# Build the default Stage 1, Stage 2, and test container image as a job.
# =============================================================================
#
# WHY A JOB: per docs.cscs.ch "Building container images on Alps", podman builds
#   run on a COMPUTE node with image storage in /dev/shm (tmpfs) — which is wiped
#   when the allocation ends. So `podman build` and `enroot import` MUST happen
#   inside the SAME allocation. This sbatch does both, then smoke-tests the
#   resulting squashfs via srun --environment before the allocation is released.
#   The fork smoke proves imports and optimizer compatibility; a real Stage 1
#   checkpoint load/save remains the end-to-end conversion check.
#   (The Apertus handbook page on custom containers is 403 from the login node —
#   network-ACL'd; the CSCS docs procedure above is the same workflow.)
#   NOTHING builds on the login node — there you only edit these files and
#   `sbatch` this script. Equivalent interactive route (the handbook flow):
#   `srun --container-writable -p debug --pty bash`, then run this script's
#   steps by hand in the pty.
#
# PREREQ (one-time, already present for $USER): rootless-podman storage config at
#   $HOME/.config/containers/storage.conf:
#       [storage]
#         driver = "overlay"
#         runroot = "/dev/shm/$USER/runroot"
#       graphroot = "/dev/shm/$USER/root"
#       [storage.options.overlay]
#         mount_program = "/usr/bin/fuse-overlayfs-1.13"
#
# MEMORY: /dev/shm pages count against --mem (unreclaimable tmpfs). The NGC base
#   unpacks to ~25 GB + pip layers, so the
#   230 GB request leaves ample headroom.
#
# NETWORK: the FROM pull hits jfrog.svc.cscs.ch (internal) and pip hits PyPI —
#   both reachable from compute nodes. If the jfrog pull is denied:
#   `podman login jfrog.svc.cscs.ch` once (interactively on a compute node).
#
# USAGE
#   sbatch $REPO/cluster/container/build_container.sbatch
#   # knobs: SQSH_OUT=... REPO=... IMAGE_TAG=... MEGATRON_PATH=...
# =============================================================================

#SBATCH --job-name=apertus-moe-hf-container
#SBATCH --time=01:20:00
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --no-requeue
#SBATCH --account=infra01
#SBATCH --cpus-per-task=72
#SBATCH --mem=230000
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.log

set -euo pipefail
echo "START TIME: $(date)"

# ---- knobs ------------------------------------------------------------------
REPO=${REPO:-/iopsstor/scratch/cscs/$USER/hfconverter}
IMAGE_TAG=${IMAGE_TAG:-apertus-moe-hf:latest}
SQSH_OUT=${SQSH_OUT:-/iopsstor/scratch/cscs/$USER/images/apertus-moe-hf.sqsh}
CONTAINERFILE=$REPO/cluster/container/Containerfile
# PYTHONPATH the fork-personality smoke (step 4) tests: the live fork checkout
# on scratch (no fork copy is baked into the image — jobs shadow pip mcore with
# this path at runtime, exactly like on a139).
MEGATRON_PATH=${MEGATRON_PATH:-/iopsstor/scratch/cscs/$USER/Megatron-LM-MoE}

die() { echo "FATAL: $*" >&2; exit 1; }

[ -f "$CONTAINERFILE" ] || die "no Containerfile at $CONTAINERFILE"

# storage.conf gate — without it podman writes to $HOME (quota + no overlay).
STORAGE_CONF=${XDG_CONFIG_HOME:-$HOME/.config}/containers/storage.conf
if [ ! -f "$STORAGE_CONF" ]; then
    die "no $STORAGE_CONF — create it first (see the PREREQ block in this script's header)"
fi

# TMPDIR MUST NOT BE ON LUSTRE — and must not be "fixed" back to $REPO/cluster/
# cache/tmp to match the sibling sbatches. Job 2778292 died at STEP 2/3 with
#   mkdir <old TMPDIR>/buildah3552061358/mnt/rootfs: permission denied
# buildah mkdtemps an OCI bundle ("buildah<rand>") under $TMPDIR per RUN step and
# creates mnt/rootfs in it — the mountpoint the store's overlay is bind-mounted
# onto — from its re-exec'd user-namespace child, running as container-uid-0.
# Lustre does not translate userns uid mappings, so that mkdir is denied, while
# the non-userns parent mkdirs the very same Lustre dir fine (that job left a
# storage-run-$UID tree there proving it). The sibling convention
# (tests.sbatch:54) is right for jobs that run
# python inside an enroot container, uid-mapped 1:1 — no uid-0 hop for Lustre to
# fumble — and their TMPDIR holds real bulk (triton/inductor caches). Not here.
#
# The --mem reasoning this line used to carry (tmpfs pages are unreclaimable and
# /tmp is not sized into the request) is sound and still holds — see the MEMORY
# header. It just was not guarding anything here, so Lustre was paid for nothing:
#  - buildah's bundle in $TMPDIR is config.json + an empty mountpoint (KBs). The
#    ~25 GB NGC rootfs lives in graphroot=/dev/shm/$USER/root per storage.conf —
#    already sized into --mem above, exactly as the MEMORY header says.
#  - podman's runtime dir falls back to $TMPDIR/storage-run-$UID when
#    XDG_RUNTIME_DIR is dead on compute nodes (see the /run/user/$UID/bus warning
#    in that log): pid/lock/event files, also KBs.
#  - step 2's `enroot import` never reads TMPDIR at all: /etc/enroot/enroot.conf
#    pins ENROOT_TEMP_PATH=/dev/shm, and enroot only defaults that from TMPDIR
#    when it is unset (/usr/bin/enroot:83). `-x mount` mksquashfs's straight from
#    the podman mountpoint into $SQSH_OUT — it never stages a rootfs copy at all.
# So: keep TMPDIR on /dev/shm — a local fs buildah can mount in a userns, in the
# one tmpfs pool --mem above already sizes, wiped with the allocation like
# runroot/graphroot. (CSCS's "Building container images on Alps" recipe puts every
# container path on /dev/shm and never sets TMPDIR at all.)
export TMPDIR=/dev/shm/$USER/tmp
mkdir -p "$TMPDIR" "$(dirname "$SQSH_OUT")"

echo "CONTAINERFILE : $CONTAINERFILE"
echo "IMAGE_TAG     : $IMAGE_TAG"
echo "SQSH_OUT      : $SQSH_OUT"
echo "podman        : $(podman --version)"
echo "enroot        : $(enroot version)"
df -h /dev/shm | sed 's/^/  /'

# ---- 1. build ---------------------------------------------------------------
cd "$REPO/cluster/container"
podman build -t "$IMAGE_TAG" -f "$CONTAINERFILE" .

# ---- 2. export to squashfs (docs.cscs.ch verbatim command shape) ------------
# Import to a temp name first so a mid-write failure never leaves a truncated
# file at the path the EDF points to.
TMP_SQSH=$SQSH_OUT.tmp.$SLURM_JOB_ID
rm -f "$TMP_SQSH"
enroot import -x mount -o "$TMP_SQSH" "podman://$IMAGE_TAG"
if [ -f "$SQSH_OUT" ]; then
    echo "NOTE: replacing existing $SQSH_OUT (jobs already RUNNING against it keep"
    echo "      their mounted copy only until the fs notices; avoid rebuilding while"
    echo "      HF-side jobs are active)."
fi
mv -f "$TMP_SQSH" "$SQSH_OUT"
ls -lh "$SQSH_OUT"

# ---- 3. smoke test the HF personality through the real container engine -----
# A throwaway EDF pointing at exactly what we just wrote (the repo EDF may point
# elsewhere if SQSH_OUT was overridden). 1 GPU is allocated, so CUDA visibility
# is asserted — this validates the full runtime path the HF-side jobs will use.
# $TMPDIR is /dev/shm, i.e. node-local: fine, because --nodes=1 and this srun
# lands on the same node that just wrote the file. $SQSH_OUT it points at is on
# Lustre and is already moved into place above, so a smoke failure never costs
# the image — only the verdict on it.
SMOKE_EDF=$TMPDIR/apertus-moe-hf-smoke-$SLURM_JOB_ID.toml
cat > "$SMOKE_EDF" <<EOF
image = "$SQSH_OUT"
mounts = ["/capstor:/capstor", "/users:/users", "/iopsstor:/iopsstor"]
workdir = "$REPO"
writable = true
EOF

srun -lu --nodes=1 --ntasks=1 --environment="$SMOKE_EDF" \
    python -c "\
import sys, torch, transformers, megatron.core, safetensors, pytest; \
assert torch.cuda.is_available(), 'no CUDA inside the container'; \
assert transformers.__version__ == '5.8.1', transformers.__version__; \
print('SMOKE OK: python', sys.version.split()[0], \
      '| torch', torch.__version__, 'cuda', torch.version.cuda, \
      '| transformers', transformers.__version__, \
      '| device', torch.cuda.get_device_name(0))"

# ---- 4. smoke the FORK personality used by Stage 1 ---------------------------
# PYTHONPATH puts the 0.17-era fork checkout (on scratch, mounted via /iopsstor)
# in front: `megatron` must resolve to the
# fork, and the fork's optimizer factory must see emerging-optimizers (>=0.2;
# 0.3.0 from PyPI) via importlib.metadata — the probe optimizer/__init__.py:36-45 runs,
# and the dependency old checkpoints with optimizer='md_decoupling' need
# beyond the fallback base. Importing md_decoupling exercises the EO
# orthogonalized-optimizer chain (and thus absl + triton) for real.
# Written to a file to keep the quoting sane through srun -> bash -c -> python;
# $TMPDIR is node-local /dev/shm, same node as the srun (--nodes=1).
FORK_SMOKE_PY=$TMPDIR/fork_smoke_$SLURM_JOB_ID.py
cat > "$FORK_SMOKE_PY" <<EOF
# NOTE: check megatron.core, NOT megatron — both the fork's and pip mcore's
# top-level megatron/ lack __init__.py, so they are NAMESPACE packages whose
# portions MERGE across sys.path (megatron.__file__ is None). The submodule
# megatron.core then resolves by sys.path ORDER, which is exactly the
# PYTHONPATH-first shadowing this smoke exists to prove.
import megatron.core
assert megatron.core.__file__.startswith("$MEGATRON_PATH"), \
    "megatron.core resolved to %s, not the fork" % megatron.core.__file__
import megatron.core.optimizer as mo
assert mo.HAVE_EMERGING_OPTIMIZERS and mo.HAVE_EO_V02, (mo.HAVE_EMERGING_OPTIMIZERS, mo.HAVE_EO_V02)
from megatron.core.optimizer import md_decoupling
print("FORK SMOKE OK:", megatron.core.__file__)
EOF
srun -lu --nodes=1 --ntasks=1 --environment="$SMOKE_EDF" \
    bash -c "export PYTHONPATH=$MEGATRON_PATH && python $FORK_SMOKE_PY"
rm -f "$SMOKE_EDF" "$FORK_SMOKE_PY"

echo
echo "CONTAINER READY : $SQSH_OUT"
echo "Referenced by   : $REPO/cluster/edf/apertus-moe-hf.toml"
echo "HF side next    : sbatch $REPO/cluster/tests.sbatch   # full suite, expect 0 skipped"
echo "fork-side smoke : ready for Stage 1"
echo "END TIME: $(date)"
