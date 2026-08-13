# Cluster conversion

These scripts run checkpoint conversion on Clariden. There are two supported
workflows:

- a Megatron `torch_dist` checkpoint goes directly through Stage 2;
- a trusted legacy Megatron `torch` checkpoint first goes through Stage 1,
  then Stage 2 starts only if Stage 1 succeeds.

```text
legacy torch checkpoint ── Stage 1 ──> torch_dist/iter_XXXXXXX
                                             │
torch_dist checkpoint ───────────────────────┘
                                             │
                                         Stage 2
                                             │
                                             v
                                  Hugging Face safetensors
```

## Files

| Path | Role |
| --- | --- |
| `convert.sh` | Submit Stage 2 for a `torch_dist` checkpoint |
| `stage2_export.sbatch` | Inspect, size, export, verify, and certify a `torch_dist` checkpoint |
| `submit_conversion.sh` | Validate and submit the two-job legacy conversion |
| `stage1_torchdist.sbatch` | Load a legacy checkpoint with Megatron and save normalized `torch_dist` |
| `ckpt_args.py` | Read saved legacy arguments without materializing tensor storage |
| `validate_source_profile.py` | Compare supplied Stage 1 settings with saved checkpoint arguments |
| `inspect_torchdist_keys.py` | Report tensor keys, dtypes, sizes, and unsupported features |
| `validate_stage2_export.py` | Check source provenance and verification settings |
| `tests.sbatch` | Run the repository tests in the cluster image |
| `container/build_container.sbatch` | Build the squashfs image referenced by the EDF |
| `edf/apertus2-hf.toml` | Pyxis environment used by conversion and tests |

## Stage 2

For a `torch_dist` checkpoint, from the repository root:

```bash
cluster/convert.sh /path/to/torch_dist/iter_XXXXXXX /path/to/fresh-hf-output
```

Arguments after the two paths go to sbatch (for example
`--reservation=<name> --partition=normal`; a reservation belongs to exactly
one partition, and `stage2_export.sbatch` defaults to `--partition=debug`, so
set both together — read the partition off `scontrol show res <name>`).
`TOKENIZER_DIR`, `VERIFY_LOAD` (default 0), `EXTRA_EXPORT_ARGS`, and `HF_ENV`
are taken from the environment.

With every knob explicit:

```bash
TD_ITER_DIR=/path/to/torch_dist/iter_XXXXXXX \
HF_OUT_DIR=/path/to/fresh-hf-output \
TOKENIZER_DIR=/path/to/tokenizer \
VERIFY_LOAD=1 \
EXTRA_EXPORT_ARGS='--max-shard-size 5GB' \
  sbatch \
    --partition=normal \
    --time=08:00:00 \
    --mem=460000 \
    --export=ALL \
    cluster/stage2_export.sbatch
```

Stage 2 performs these operations in order:

1. inspect the checkpoint tensor inventory;
2. estimate model bytes from metadata and check the allocation when a full
   reload was requested;
3. run `python -m exporter.export`;
4. require `conversion_info.json` and remove no failure evidence;
5. validate the certificate against the source iteration.

`VERIFY_LOAD=0` skips only the full-model reload. Shard-level bitwise
verification still runs. Set `SKIP_INSPECT=1` only if the same checkpoint has
already passed the key inspection.

## Legacy checkpoint conversion

Legacy checkpoints contain pickle metadata and must be explicitly trusted.
Their topology and several forward-semantic values must be supplied; the
submitter verifies them against the saved checkpoint before creating a Slurm
job.

```bash
STAGE1_ENV=/iopsstor/scratch/cscs/mvasilev/hfconverter/cluster/edf/apertus2-hf.toml \
HF_ENV=/iopsstor/scratch/cscs/mvasilev/hfconverter/cluster/edf/apertus2-hf.toml \
TRUST_LEGACY_CHECKPOINT=1 \
SRC_TP=1 \
SRC_PP=1 \
SRC_EP=4 \
SRC_ETP=1 \
SRC_CP=1 \
PRECISION=bf16 \
ROUTING_TYPE=seq_aux_loss \
MOE_AUX_LOSS_COEFF=1e-4 \
INIT_METHOD_STD=0.0360844 \
NORM_EPSILON=1e-5 \
SANDWICH_NORM=0 \
WINDOW_SIZE= \
WINDOW_ATTN_SKIP_FREQ= \
NO_ROPE_FREQ= \
  cluster/submit_conversion.sh \
    /path/to/legacy-checkpoint-root \
    /path/to/fresh-conversion-root
```

`WINDOW_SIZE`, `WINDOW_ATTN_SKIP_FREQ` and `NO_ROPE_FREQ` are required to be
*set*, though empty is a valid answer for a checkpoint that has none:
Megatron's `--use-checkpoint-args` does not restore them, and neither
sliding-window attention nor NoPE has a parameter footprint, so an omission
would produce a model that loads, generates fluent text, and attends over the
wrong span on every layer. `validate_source_profile.py` compares the supplied
values against the checkpoint's own saved arguments before any job is
submitted, so a forgotten flag is a refused submission rather than a wrong
model. (Megatron's `window_size` is inclusive at both ends, so `(512, 0)`
becomes `sliding_window` 513; Megatron's `no_rope_freq` marks the layers that
*skip* the rotation while Hugging Face's `no_rope_layers` marks the layers
that *keep* it — `exporter/config_from_args.py` performs both conversions.)

`submit_conversion.sh` also takes `RESERVATION` and `STAGE2_PARTITION`, which
must be set together; leaving both empty keeps each stage's own `#SBATCH`
defaults.

Do not copy those example values blindly. Inspect the source first:

```bash
python3 cluster/ckpt_args.py /path/to/legacy-checkpoint-root
python3 cluster/ckpt_args.py /path/to/legacy-checkpoint-root --grep 'moe_*'
```

The resulting layout is:

```text
/path/to/fresh-conversion-root/
├── stage1-complete
├── torch_dist/
│   └── iter_XXXXXXX/
└── hf/
```

Stage 1 uses four GPUs and currently accepts source topologies whose
`TP × PP × EP` is at most four, with context parallel size 1. It refuses a
non-empty destination, a non-strict Megatron load fallback, missing distributed
checkpoint metadata, or an iteration mismatch. The marker `stage1-complete` is
written last.

Stage 2 is submitted with an `afterok` dependency and defaults to
`VERIFY_LOAD=1`. Its Hugging Face output is written under `OUTPUT_ROOT/hf`.

## Container

The EDF expects the squashfs image at:

```text
/iopsstor/scratch/cscs/$USER/images/apertus2-hf.sqsh
```

The build runs through Slurm because Podman and Enroot must operate in the same
compute-node allocation. Do not run `podman build` on the login node.

Before the first build, configure rootless Podman storage in
`$HOME/.config/containers/storage.conf`. Replace `<user>` with your username:

```toml
[storage]
driver = "overlay"
runroot = "/dev/shm/<user>/runroot"
graphroot = "/dev/shm/<user>/root"

[storage.options.overlay]
mount_program = "/usr/bin/fuse-overlayfs-1.13"
```

Then submit the build from the repository root:

```bash
cd /iopsstor/scratch/cscs/$USER/hfconverter
sbatch cluster/container/build_container.sbatch
```

The default job uses the `debug` partition, one GPU, 72 CPU cores, 230,000 MiB
of memory, and a 1 hour 20 minute time limit. It:

1. builds `cluster/container/Containerfile` with Podman;
2. imports the image into a temporary squashfs with Enroot;
3. atomically moves it to
   `/iopsstor/scratch/cscs/$USER/images/apertus2-hf.sqsh`;
4. launches the resulting image and checks CUDA, PyTorch, Transformers,
   Megatron Core, safetensors, and pytest;
5. checks that Stage 1 can import the live Megatron fork and its optimizer
   dependencies.

The Stage 1 smoke test means `MEGATRON_PATH` must point to a readable
`Megatron-LM-MoE` checkout while building, even though Stage 2 itself does not
need that checkout. Override paths and the local Podman tag when necessary:

```bash
REPO=/iopsstor/scratch/cscs/$USER/hfconverter \
SQSH_OUT=/iopsstor/scratch/cscs/$USER/images/apertus2-hf.sqsh \
MEGATRON_PATH=/iopsstor/scratch/cscs/$USER/Megatron-LM-MoE \
IMAGE_TAG=apertus2-hf:latest \
  sbatch --export=ALL cluster/container/build_container.sbatch
```

The build pulls the NGC base image from the internal CSCS JFrog mirror and
installs Python packages from PyPI. If the base-image pull is denied, run
`podman login jfrog.svc.cscs.ch` interactively on a compute node, then resubmit.
Do not replace the squashfs while conversion or evaluation jobs are using it.

The image pins Transformers 5.8.1 and Megatron Core 0.18.0. Stage 1 puts the
live Megatron fork checkout first on `PYTHONPATH`; Stage 2 uses the installed
Megatron Core package.

## Completion and logs

All wrappers preserve failed output for diagnosis and require a fresh directory
for the next attempt. A valid Stage 2 result has:

- `conversion_info.json`;
- `"verified": true` and `"settings.verify": true`;
- `"settings.verify_load": true` when full verification was requested;
- no `.export_incomplete` marker;
- source path and iteration matching the submitted checkpoint.

Jobs submitted through `submit_conversion.sh` default their logs to
`cluster/logs/`.

Run the conversion-focused test suite with:

```bash
sbatch cluster/tests.sbatch
```
