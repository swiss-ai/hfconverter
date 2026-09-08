# NeMo 26.08 HF conversion runtime

The HF converter uses the unmodified NeMo Framework 26.08.00 image for
Transformers 5.12.1 and tokenizers 0.22.2. It preserves the base image's
dependencies; no additional Transformers, XIELU, ModelOpt or Megatron package
is installed. This is a conversion image, not a change to the training image.

The shared CSCS ARM64 artifact is
`/capstor/store/cscs/swissai/infra01/MLLM/containers/nemo-26.08.00-aarch64.sqsh`.
The repository's `nemo.toml` selects that artifact. A copy is also stored
alongside the shared image as `nemo-26.08.00-aarch64.toml`. Slurm reads this
TOML from the host filesystem; it does not need to be baked into the image.
Use `srun --environment=nemo.toml`; keep the environment on `srun`, not `sbatch`.
The TOML sets `HFCONVERTER_PYTHON=/opt/venv/bin/python` and disables Python
user-site packages. Conversion and logits launchers invoke that interpreter
explicitly, including `python -m torch.distributed.run`, because mounted user
profiles can rewrite PATH and select a host Conda installation. When using a
custom TOML, set `HFCONVERTER_PYTHON` to its interpreter; without that setting,
the launchers retain their usual `python` lookup.

The default tokenizer is the released `swiss-ai/Apertus-v1.5-8B` revision
`a411d838600baf0e3635a3daf66fb7c55fc97bb6`, stored at
`/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/Apertus-v1.5-8B-a411d838`.
Its six tokenizer/config files are byte-identical to the release and to the
files used in validation. The shared folder includes `source.json` with
per-file SHA256 checksums. Set `TOKENIZER` for other checkpoint tokenizers.

## Select the training source

| Component | Validated Apertus 1.5 SFT environment | NeMo 26.08 conversion runtime |
| --- | --- | --- |
| Megatron Core | Our checkout: 0.16.0rc0 | Bundled distribution: 0.19.0; our checkout is selected instead |
| Transformers | 4.57.6 | 5.12.1 |
| tokenizers | 0.22.1 | 0.22.2 |
| Transformer Engine | 2.10.0+769ed778 | 2.17.1+4329ff84 |
| PyTorch | 2.10.0a0+b4e4ee81d3.nv25.12 | 2.13.0a0+8145d630e8.nv26.6.54250401 |

The training checkout version is **0.16**, not 1.6. The tested source commit
is `430aa60676c4f9f832ad18b8ce30e6740fe7ad2f` in `swiss-ai/Megatron-LM`.
The converter's clone fallback defaults to this exact commit. To use a
different checkpoint's source, set `LOCAL_MEGATRON_PATH` explicitly (or
override `MEGATRON_BRANCH`). The selected source must contain the Swiss AI
checkpoint-conversion scripts and custom model implementation.

Inside every conversion process:

```bash
export PYTHONPATH="$MEGATRON_PATH"
"${HFCONVERTER_PYTHON:-python}" /path/to/hfconverter/check_megatron_source.py "$MEGATRON_PATH"
```

The guard fails if Python resolves Megatron Core, training code or model
builders outside the selected checkout. It reports both the selected source
version and installed distribution version. `pip show megatron-core` alone
reports the bundled 0.19.0 package and does not establish which source is used.

## Completed conversion and inference validation

The measurements used the 8B text-SFT checkpoint at iteration 911, TP2/PP1,
on one GH200 node. The repository's `convert.sbatch` completed both conversion
stages on the shared image in job 3324022. Its initial audit then failed
because the old audit assumed sharded safetensors; Transformers 5 saved one
`model.safetensors` file. After teaching the audit both formats, job 3324146
validated that same untouched export and completed with exit code 0.

- All 451 exported BF16 tensors matched the original native TP checkpoint
  shards exactly, including both `[266752, 4096]` vocabulary matrices.
- HF loaded all tensors without missing, unexpected or mismatched keys.
- HF/native logits were close: top1 agreed (token 10518), KL(native || HF)
  was 0.0008817708, and raw-logit relative L2 was 0.016607821.
- Effective RoPE matched factor 8 and theta 4000000; maximum inverse-frequency
  absolute error was 2.3283064e-10. The raw tokenizer backend JSON matched the
  release completely, the chat template was byte-identical, and all 266440
  token IDs and special-token settings matched. Three chat prompts with
  thinking disabled encoded identically and contained one BOS. Greedy
  `2 + 2` generation answered `4`.
- The guard accepted the selected checkout and rejected the image-installed
  MCore. Actual module imports also came from the selected source. Four
  source-guard unit tests and shell syntax checks passed locally.

The shared directory contains `nemo-26.08.00-aarch64.validation.json` with
versions, checks and measured logits. No tokenizer/config repair was applied
to the export before this validation.

Earlier native-runtime comparison job 3323570 and source audit job 3323637
also completed with exit code 0. All entries of the 266752-element
**last-token logit vector** were identical between the training image and
NeMo 26.08 for the six-token prompt `The capital of Switzerland is`:
relative L2 = 0 and KL = 0. HF versus native logits are not identical, as the
measurements above show.

These are specific smoke tests, not a claim that all prompts, sequence
lengths, training losses or benchmarks are identical. TE 2.17.1 and Energon
7.4.0 are outside our checkout's declared development ranges (<2.11 and 6.x).
Training/backward, optimizer resume, 70B and multi-node communication have
not been validated by these tests.

Stock Transformers 5.12.1 supports this export's `model_type=apertus`, but
does not register the released `apertus1p5` multimodal wrapper. Full wrapper
support requires the separately verified Swiss AI Transformers implementation.

## Reuse the official image

The ARM64 base manifest is pinned to
`sha256:ac012c8d5b7b72fe60ca53e2519175fa8c27966b2d2f8efe6d1ed0559aff4ba0`.
The shared SquashFS is a byte-for-byte copy of the tested NeMo import, with
SHA256 `d92b7f81ea5f51fb0539d5319184ad78bc8ace4dec1aacff763b82f3da4a11ba`.
No image rebuild or package installation is needed on CSCS. Converter code,
source guards and the TOML remain in this repository.

The Dockerfile selects the same official base for workflows that use Docker.
It no longer installs floating Transformers/XIELU/Megatron revisions or
replaces NeMo's ModelOpt package.
