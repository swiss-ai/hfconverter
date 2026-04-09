#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Extract last-token logits from an HF checkpoint.")
    parser.add_argument("--hf-dir", type=str, required=True, help="Path to HF checkpoint directory")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    parser.add_argument("--out-pt", type=str, default="", help="Output .pt file path")
    parser.add_argument("--out-report", type=str, default="", help="Output report .json file path")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32", "auto"],
        help="Model loading dtype",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to HF loaders",
    )
    return parser.parse_args()


def resolve_output_paths(out_pt_arg: str, out_report_arg: str, file_prefix: str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_results_dir = os.path.join(script_dir, "results")
    os.makedirs(default_results_dir, exist_ok=True)

    if out_pt_arg:
        out_pt = out_pt_arg
    else:
        out_pt = os.path.join(default_results_dir, f"{file_prefix}.pt")

    if out_report_arg:
        out_report = out_report_arg
    else:
        if out_pt.endswith(".pt"):
            out_report = f"{out_pt[:-3]}.report.json"
        else:
            out_report = f"{out_pt}.report.json"

    return out_pt, out_report


def resolve_dtype(dtype_name: str):
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    return "auto"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_comparison_report(prompt: str, token_ids, last_token_logits: torch.Tensor):
    logits_fp32 = last_token_logits.detach().float().contiguous().cpu()
    prompt_sha256 = _sha256(prompt.encode("utf-8"))
    token_ids_sha256 = _sha256(",".join(str(x) for x in token_ids).encode("utf-8"))
    logits_sha256 = _sha256(logits_fp32.numpy().tobytes())

    top1_id = int(torch.argmax(logits_fp32[0]).item())
    top1_logit = float(logits_fp32[0, top1_id].item())
    top5_logits, top5_token_ids = torch.topk(logits_fp32[0], k=5)

    comparison_key = f"{prompt_sha256}:{token_ids_sha256}:{logits_sha256}"
    return {
        "prompt_sha256": prompt_sha256,
        "token_ids_sha256": token_ids_sha256,
        "logits_sha256": logits_sha256,
        "comparison_key": comparison_key,
        "num_prompt_tokens": int(len(token_ids)),
        "vocab_size": int(logits_fp32.shape[-1]),
        "top1_token_id": top1_id,
        "top1_logit": top1_logit,
        "top5_token_ids": [int(x) for x in top5_token_ids.tolist()],
        "top5_logits": [float(x) for x in top5_logits.tolist()],
    }


@torch.inference_mode()
def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    model_dtype = resolve_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_dir, trust_remote_code=args.trust_remote_code, use_fast=False
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=model_dtype,
    ).eval()
    model.to(args.device)

    encoded = tokenizer(args.prompt, return_tensors="pt")
    encoded = {k: v.to(args.device) for k, v in encoded.items()}
    token_ids = encoded["input_ids"][0].tolist()
    if len(token_ids) == 0:
        raise RuntimeError("Prompt tokenized to an empty sequence.")

    logits = model(**encoded).logits
    last_token_logits = logits[:, -1, :].float().cpu()
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    ckpt_name = os.path.basename(os.path.normpath(args.hf_dir)) or "hf"
    default_prefix = f"hf_logits_{ckpt_name}"
    out_pt, out_report = resolve_output_paths(args.out_pt, args.out_report, default_prefix)
    report = build_comparison_report(args.prompt, token_ids, last_token_logits)
    report["output_pt_path"] = out_pt
    report["output_report_path"] = out_report
    report["mode"] = "hf"
    report["timestamp_utc"] = timestamp

    out_dir = os.path.dirname(out_pt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.save(
        {
            "prompt": args.prompt,
            "token_ids": token_ids,
            "last_token_logits": last_token_logits,
            "hf_dir": args.hf_dir,
            "dtype": args.dtype,
            "device": args.device,
            "comparison_report": report,
        },
        out_pt,
    )

    report_dir = os.path.dirname(out_report)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"saved_pt {out_pt} shape={tuple(last_token_logits.shape)}")
    print(f"saved_report {out_report}")
    print("top5_token_ids:", report["top5_token_ids"])
    print("top5_logits:", report["top5_logits"])
    print("comparison_report_start")
    print(f"prompt_sha256={report['prompt_sha256']}")
    print(f"token_ids_sha256={report['token_ids_sha256']}")
    print(f"logits_sha256={report['logits_sha256']}")
    print(f"comparison_key={report['comparison_key']}")
    print(f"top1_token_id={report['top1_token_id']}")
    print(f"top1_logit={report['top1_logit']}")
    print("comparison_report_end")


if __name__ == "__main__":
    main()
