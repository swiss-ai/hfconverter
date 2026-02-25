# Multimodal Checkpoint Converter to HuggingFace

This is a WIP multimodal converter that assumes many things (ie. TP, PP, World size etc).

## Megatron -> HF

```
sbatch convert.sbatch <dir path to megatron checkpoint> <checkpoint iteration> <dir to path to write to>
```

For example:

```
sbatch ~/hfconverter/convert.sbatch /capstor/store/cscs/swissai/infra01/MLLM/apertus-8b/extended_model_vocab_266440 1 /capstor/store/cscs/swissai/infra01/hf-checkpoints/Apertus-audio-ablations-base-extended-1
```

## Compare logits

Minimal script to compare logits between original Megatron checkpoint and converted HF checkpoint

```bash
# 1) Generate native Megatron logits report
sbatch logits_tools/get_native_dist_logits.sbatch <megatron_ckpt_dir> <iteration> "Sanity check prompt."

# 2) Generate converted HF logits report (use the same prompt)
sbatch logits_tools/get_hf_logits.sbatch <hf_ckpt_dir> "Sanity check prompt."

# 3) Compare generated *.report.json files
python logits_tools/compare_reports.py
```
