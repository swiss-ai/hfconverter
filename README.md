# Apertus 2 MoE Hugging Face export

This repository converts supported Apertus Megatron checkpoints into a
Hugging Face model directory. It contains the converter, the custom
Transformers implementation copied into every export, and the Slurm wrappers
used on Clariden.

The repository does not contain model weights.

## Repository structure

```text
.
├── configuration_apertus_moe.py   # Hugging Face configuration
├── modeling_apertus_moe.py        # Hugging Face model and expert parallel plan
├── exporter/                      # torch_dist reader, mapping, writer, verification
├── cluster/                       # Slurm submission and validation scripts
└── tests/                         # conversion and exported-model tests
```

The exporter internals are described in
[`exporter/README.md`](exporter/README.md). Cluster workflows and resource
requirements are in [`cluster/README.md`](cluster/README.md).

The Clariden conversion container is located at:

```text
/iopsstor/scratch/cscs/mvasilev/images/apertus-moe-hf.sqsh
```

It is referenced by `cluster/edf/apertus-moe-hf.toml` and can be rebuilt with
`sbatch cluster/container/build_container.sbatch`.

## Chonk 120B export

The completed Hugging Face model is stored at:

```text
/iopsstor/scratch/cscs/mvasilev/hf-export/chonk-120b-s1-iter1000
```

It was exported from:

```text
/iopsstor/scratch/cscs/ahuang/megatron-apertus-moe/_research/results/ckpts/chonk/120b-moe-256e-latent-swa15-nope-s1-muonmd-lr5.889e-3-mlr5.889e-2-latmoe-e256-top8-swa15-w512-nope-s1/iter_0001000
```

The export contains 46 safetensors shards and 228,784,602,496 tensor
bytes (about 213.1 GiB). Conversion and a full
`AutoModelForCausalLM.from_pretrained()` reload both passed.

The model supports:

- `AutoConfig`, `AutoModel`, and `AutoModelForCausalLM` with
  `trust_remote_code=True`;
- 42 decoder layers, hidden size 3584, 32 attention heads, 8 KV heads,
  and a 200,064-token vocabulary;
- 39 MoE layers with 256 routed experts, top-8 routing, one shared expert,
  and a 1792-dimensional routed-expert latent space;
- SSSGLU, sandwich normalization, sigmoid routing, and quantile balancing;
- causal sliding-window attention with a 513-token window, full-attention
  layers every sixth layer, and the matching NoPE schedule;
- sequences up to 8192 tokens;
- native Transformers expert parallelism. EP=4 with SDPA was used for the
  Hugging Face evaluation.

Model parameters, including expert weights, are stored as BF16. The 78 router
state buffers (`expert_bias` and `qb_beta`) remain FP32. Training used a
transient FP8 expert cache, but the Megatron checkpoint persisted BF16 master
weights rather than FP8 tensors; the exporter preserves those BF16 values and
does not quantize them.

Tensor parallel loading is not supported by the custom model. Expert
parallelism shards only routed experts; attention, embeddings, router weights,
the shared expert, and latent projections remain replicated.

## Run the Chonk conversion

Run from the repository root. The existing model directory is non-empty, so
choose a fresh output path for another conversion.

First ask Slurm to validate the request without submitting a job:

```bash
HF_OUT_DIR=/iopsstor/scratch/cscs/mvasilev/hf-export/chonk-120b-s1-iter1000-new \
  cluster/submit_chonk_120b_s1_iter1000.sh --test-only
```

Then submit it:

```bash
HF_OUT_DIR=/iopsstor/scratch/cscs/mvasilev/hf-export/chonk-120b-s1-iter1000-new \
  cluster/submit_chonk_120b_s1_iter1000.sh
```

This checkpoint is already in Megatron `torch_dist` format, so the wrapper
runs Stage 2 only. It requests 460,000 MiB of CPU memory, writes 5 GB shards,
checks every written tensor bitwise, skips the full `from_pretrained()` reload,
and writes `conversion_info.json` last. The completed reference export listed
above was produced earlier with the full reload enabled; it passed in 17
minutes 35 seconds.

The pinned wrapper uses `VERIFY_LOAD=0` because a full reload materializes the
entire 213.1 GiB model in CPU memory; the reference run peaked around 242 GiB
while performing that check. New exports still reload each safetensors shard
and compare every tensor bitwise. Their certificate records
`"verified": true` and `"settings.verify_load": false`.

For a compatible `torch_dist` checkpoint in an environment with Megatron Core
0.18 and Transformers 5.8.1, the converter can also be invoked directly:

```bash
python -m exporter.export \
  --checkpoint-dir /path/to/torch_dist/iter_0001000 \
  --output-dir /path/to/fresh-hf-output \
  --tokenizer-dir /path/to/tokenizer \
  --max-shard-size 5GB \
  --verify-load
```

The output directory must be absent or empty. `--verify-load` materializes the
entire exported model in CPU memory; omit it only when the allocation cannot
hold a second model-sized copy. Per-shard bitwise verification remains enabled
unless `--no-verify` is explicitly supplied.

## How `exporter/` converts a checkpoint

`exporter/` implements Stage 2 of the conversion. Its input must already be a
Megatron Core `torch_dist/iter_XXXXXXX` directory. A legacy Megatron `torch`
checkpoint must first be normalized by `cluster/stage1_torchdist.sbatch`.

```text
Megatron torch_dist checkpoint
        │
        ├─ saved arguments + global tensor metadata
        v
validate architecture and build an exact tensor/shard plan
        │
        ├─ load only the source tensors needed by the next shard
        v
split / transpose / rename tensors into Hugging Face layout
        │
        ├─ write and bitwise-verify one safetensors shard at a time
        v
copy tokenizer + custom Apertus MoE configuration/model code
        │
        ├─ optional full from_pretrained() verification
        v
conversion_info.json + completed Hugging Face directory
```

The pipeline is:

1. `reader.py` reads the saved Megatron arguments and global tensor metadata.
   `config_from_args.py` derives the Hugging Face architecture and rejects
   checkpoint features that cannot be represented exactly. No model tensor
   bytes are loaded at this stage.
2. `mapping.py` separates model tensors from optimizer and scheduler state,
   checks that every source model key is consumed exactly once, and plans every
   Hugging Face key, shape, and dtype. `writer.py` computes the safetensors
   shard layout from that metadata before allocating the tensors.
3. `output_claim.py` claims the fresh output directory with
   `.export_incomplete`. A failed or interrupted conversion deliberately leaves
   this marker and any completed shards for diagnosis.
4. `export.py` processes the planned output shards in order. For each shard,
   Megatron Core reconstructs only the required source tensors. `mapping.py`
   and `transforms.py` then split fused QKV and gate/up projections, rearrange
   expert tensors, transpose layouts where required, and assign the Hugging
   Face names. Parameters are not cast.
5. `writer.py` writes the shard, reloads it, and checks every tensor's dtype and
   value bitwise. Source tensors are released after their final output is
   written, which keeps normal conversion memory bounded by the active sources
   and shard instead of the complete model.
6. The exporter writes the safetensors index, validates and copies the
   tokenizer, creates `config.json`, and copies
   `configuration_apertus_moe.py` and `modeling_apertus_moe.py`. The
   `auto_map` entries in `config.json` let Transformers load these classes with
   `trust_remote_code=True`.
7. If `--verify-load` is enabled, the complete exported model is loaded through
   `AutoModelForCausalLM.from_pretrained()` and compared with disk-backed
   references. Finally, `conversion_info.json` records provenance and
   verification settings, and `.export_incomplete` is removed.

The pinned Chonk wrapper uses `VERIFY_LOAD=0`, so step 7 skips only the
full-model reload. The bitwise verification in step 5 still runs for every
safetensors shard.

## Load the Hugging Face model

The model uses custom code from its own directory:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = (
    "/iopsstor/scratch/cscs/mvasilev/hf-export/"
    "chonk-120b-s1-iter1000"
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
```

The complete model is too large for an ordinary single-GPU load. A distributed
launcher can enable the exported expert-parallel plan with:

```python
from transformers.distributed import DistributedConfig

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    distributed_config=DistributedConfig(enable_expert_parallel=True),
)
```

## Conversion scripts

| Path | Purpose |
| --- | --- |
| `exporter/export.py` | Direct `torch_dist` to Hugging Face conversion |
| `cluster/submit_chonk_120b_s1_iter1000.sh` | Pinned, verified Chonk 120B Stage 2 submission |
| `cluster/stage2_export.sbatch` | Generic `torch_dist` export job |
| `cluster/submit_conversion.sh` | Submit legacy Stage 1 and dependent Stage 2 jobs |
| `cluster/stage1_torchdist.sbatch` | Normalize a trusted legacy Megatron `torch` checkpoint to `torch_dist` |
| `cluster/inspect_torchdist_keys.py` | Inspect tensors, dtypes, and unsupported source keys |
| `cluster/validate_source_profile.py` | Check legacy checkpoint arguments before Stage 1 |
| `cluster/validate_stage2_export.py` | Validate the final conversion certificate |

## Tests

The repository tests only the conversion path and the model implementation
included in an export:

```bash
HF_HOME=/tmp/apertus-moe-hf-hf-home \
HF_MODULES_CACHE=/tmp/apertus-moe-hf-modules \
  python -m pytest tests/ -q
```

On Clariden, the same suite can run in the pinned container:

```bash
sbatch cluster/tests.sbatch
```
