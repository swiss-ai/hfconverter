# Apertus 2 MoE Hugging Face export

This repository converts supported Apertus Megatron checkpoints into a
Hugging Face model directory. It contains the converter, the custom
Transformers implementation copied into every export, and the Slurm wrappers
used on Clariden. It contains no model weights.

## Repository structure

```text
.
├── configuration_apertus2.py       # Hugging Face configuration
├── modeling_apertus2.py            # Hugging Face model and expert parallel plan
├── exporter/                      # torch_dist reader, mapping, writer, verification
├── cluster/                       # Slurm submission and validation scripts
└── tests/                         # conversion and exported-model tests
```

The exporter internals are described in
[`exporter/README.md`](exporter/README.md). Cluster workflows and resource
requirements are in [`cluster/README.md`](cluster/README.md).

## Convert a checkpoint

The input must be a Megatron Core `torch_dist/iter_XXXXXXX` directory. From
the repository root:

```bash
cluster/convert.sh /path/to/torch_dist/iter_XXXXXXX /path/to/fresh-hf-output
```

Everything after the two paths is passed to sbatch, e.g.
`--reservation=<name> --partition=normal`. The output directory must be
absent or empty. `VERIFY_LOAD=1` additionally reloads the complete export
through `from_pretrained()`, which materializes a second model-sized copy in
CPU memory; every written shard is reloaded and compared bitwise either way.
`TOKENIZER_DIR` defaults to the 200,064-token Apertus tokenizer —
`args.tokenizer_model` in the checkpoint's `common.pt` names the right one,
and the exporter refuses a tokenizer larger than the checkpoint's vocabulary.

A legacy Megatron `torch` checkpoint must first be normalized to `torch_dist`
by Stage 1; `cluster/submit_conversion.sh` submits both stages with a job
dependency. See [`cluster/README.md`](cluster/README.md).

In an environment with Megatron Core 0.18 and Transformers 5.8.1, Stage 2 can
also run directly:

```bash
python -m exporter.export \
  --checkpoint-dir /path/to/torch_dist/iter_XXXXXXX \
  --output-dir /path/to/fresh-hf-output \
  --tokenizer-dir /path/to/tokenizer \
  --max-shard-size 5GB
```

`--max-shard-size` is also a memory knob: the exporter streams one output
shard at a time, so peak RAM tracks the shard plus the source tensors
straddling its boundary.

## What Stage 2 does

Stage 2 derives the Hugging Face architecture from the checkpoint's saved
arguments and refuses any feature it cannot represent exactly. Supported
checkpoints cover dense/MoE layer schedules, sigmoid routing with quantile
balancing and expert bias, routed-expert latent spaces, shared experts,
SSSGLU, sandwich and QK normalization, sliding-window/NoPE attention
schedules, and the attention output gate. The exporter plans every output
tensor up front, then streams the checkpoint one safetensors shard at a
time — splitting fused QKV (including packed attention-gate heads) and
gate/up projections, rearranging expert tensors, never casting dtypes — and
reloads each written shard for a bitwise comparison. The export ends with the
tokenizer, `config.json`, the custom model code, and a
`conversion_info.json` certificate; a failed run deliberately leaves an
`.export_incomplete` marker for diagnosis.

## Load the exported model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/path/to/hf-export/model"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
```

Tensor parallel loading is not supported. Expert parallelism shards only the
routed experts; a distributed launcher enables it with
`distributed_config=DistributedConfig(enable_expert_parallel=True)` from
`transformers.distributed`.

## Container

Conversion jobs run in a pinned squashfs image at
`/iopsstor/scratch/cscs/$USER/images/apertus2-hf.sqsh`, referenced by
`cluster/edf/apertus2-hf.toml`. Build it with
`sbatch cluster/container/build_container.sbatch` from the repository root —
the build must run on a compute node, not the login node. The image pins
Transformers 5.8.1 and Megatron Core 0.18. Storage prerequisites and
overrides are in [`cluster/README.md`](cluster/README.md).

## Tests

The repository tests only the conversion path and the model implementation
included in an export:

```bash
HF_HOME=/tmp/apertus2-hf-hf-home \
HF_MODULES_CACHE=/tmp/apertus2-hf-modules \
  python -m pytest tests/ -q
```

On Clariden, the same suite can run in the pinned container:

```bash
sbatch cluster/tests.sbatch
```
