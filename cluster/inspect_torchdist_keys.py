# Copyright 2026 the Swiss AI Initiative. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Print the key/shape/dtype inventory of a Megatron torch_dist checkpoint.

Use this after ``stage1_torchdist.sbatch`` or as a standalone triage tool.

VERIFIED: ``dist_checkpointing.load_tensors_metadata`` needs NO process group and NO
GPU, so this runs equally well inside the
training container (where ``megatron`` is the fork) and on the login node under
/iopsstor/scratch/cscs/mvasilev/venvs/hfdev312/bin/python (pip megatron-core 0.18).

It reports optional fork-feature keys separately so the selected mapping layout is visible
before conversion.

    python inspect_torchdist_keys.py <iter_XXXXXXX dir> [--all]
"""

import argparse
import math
import sys
import warnings
from pathlib import Path

with warnings.catch_warnings():
    # pip mcore without TransformerEngine/Apex warns at import; harmless.
    warnings.simplefilter("ignore")
    from megatron.core import dist_checkpointing

# (label, predicate on the key) — the three fork features the exporter branches on,
# plus the stacked-expert layout that design pin D1 depends on.
FEATURE_PROBES = [
    ("sandwich norm  (post_self_attn_layernorm)", lambda k: k.endswith("post_self_attn_layernorm.weight")),
    ("sandwich norm  (post_mlp_layernorm)", lambda k: k.endswith("post_mlp_layernorm.weight")),
    ("latent MoE     (fc1_latent_proj)", lambda k: k.endswith("mlp.fc1_latent_proj.weight")),
    ("latent MoE     (fc2_latent_proj)", lambda k: k.endswith("mlp.fc2_latent_proj.weight")),
    ("QB routing     (router.qb_beta)", lambda k: k.endswith("mlp.router.qb_beta")),
    ("expert bias    (router.expert_bias)", lambda k: k.endswith("mlp.router.expert_bias")),
    ("router weight", lambda k: k.endswith("mlp.router.weight")),
    # NOTE the DOUBLED 'experts.': that is what torch_dist persists (ShardedTensor.key), see
    # A single-'experts.' spelling here would report the real
    # checkpoint as having no expert weights.
    ("STACKED experts fc1", lambda k: k.endswith("mlp.experts.experts.linear_fc1.weight")),
    ("STACKED experts fc2", lambda k: k.endswith("mlp.experts.experts.linear_fc2.weight")),
    # The offloaded alternative to the two rows above: fused, and transposed to (E, in, out).
    # Exactly one of the two pairs is present, selected by args.moe_use_offloading_experts.
    ("OFFLOADED experts w1", lambda k: k.endswith("mlp.experts.experts.weight1")),
    ("OFFLOADED experts w2", lambda k: k.endswith("mlp.experts.experts.weight2")),
    ("shared expert fc1", lambda k: k.endswith("mlp.shared_experts.linear_fc1.weight")),
]

# Keys that mean "a fork feature the exporter does not map".
RED_FLAGS = [
    ("shared expert gate", lambda k: k.endswith("shared_experts.gate_weight")),
    ("router bias", lambda k: k.endswith("router.bias")),
    ("multi-token prediction", lambda k: k.startswith("mtp")),
    ("PNGLU", lambda k: "polynorm_glu" in k),
    ("learned position embeddings", lambda k: k.startswith("embedding.position_embeddings.")),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint_dir", help="the torch_dist iter_XXXXXXX directory")
    ap.add_argument("--all", action="store_true", help="print every key, not just a summary")
    ns = ap.parse_args(argv)

    ckpt = Path(ns.checkpoint_dir)
    if not dist_checkpointing.check_is_distributed_checkpoint(str(ckpt)):
        print(f"FATAL: not a torch_dist checkpoint: {ckpt}", file=sys.stderr)
        return 1

    meta = dist_checkpointing.load_tensors_metadata(str(ckpt))
    keys = sorted(meta)
    print(f"{len(keys)} tensor keys in {ckpt}\n")

    model_keys = [k for k in keys if not k.startswith(("optimizer.", "opt_param_scheduler"))]
    optim_keys = [k for k in keys if k.startswith(("optimizer.", "opt_param_scheduler"))]

    # Size the export job from this: the HF output is the model bytes, and --verify-load
    # reloads the whole model into CPU RAM afterwards.
    parameters = sum(math.prod(tuple(meta[k].global_shape)) for k in model_keys)
    model_bytes = sum(
        math.prod(tuple(meta[k].global_shape)) * meta[k].dtype.itemsize for k in model_keys
    )
    print(f"model tensors: {len(model_keys)} keys, {parameters / 1e9:.1f}B parameters, "
          f"{model_bytes / 1024**3:.0f} GiB on disk as exported")
    print(f"  -> size the export job for {model_bytes / 1024**3:.0f} GiB of output; with "
          f"--verify-load the whole model is also reloaded into RAM\n")
    if optim_keys:
        # Expected to be EMPTY: stage 1 passes --no-save-optim.
        print(f"WARNING: {len(optim_keys)} optimizer/scheduler keys present — "
              f"was --no-save-optim passed? (the exporter drops them, so this is not fatal)\n")

    print("--- checkpoint feature probes ---")
    for label, pred in FEATURE_PROBES:
        hits = [k for k in model_keys if pred(k)]
        if hits:
            shapes = sorted({tuple(meta[k].global_shape) for k in hits})
            dtypes = sorted({str(meta[k].dtype) for k in hits})
            print(f"  PRESENT  {label:44s} x{len(hits):<4d} shapes={shapes} dtypes={dtypes}")
        else:
            print(f"  absent   {label:44s}")

    findings = [(label, [k for k in model_keys if pred(k)]) for label, pred in RED_FLAGS]
    findings = [(label, hits) for label, hits in findings if hits]
    if findings:
        print("\n--- RED FLAGS: unmapped fork features (the exporter will refuse these) ---")
        for label, hits in findings:
            print(f"  {label}: {len(hits)} keys, e.g. {hits[0]}")
    else:
        print("\nno unmapped-fork-feature keys found.")

    if ns.all:
        print(f"\n--- all {len(model_keys)} model keys ---")
        width = max(len(k) for k in model_keys)
        for k in model_keys:
            print(f"  {k:{width}s}  {tuple(meta[k].global_shape)!s:24s} {meta[k].dtype}")
    else:
        print("\n(re-run with --all for the full key list)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
