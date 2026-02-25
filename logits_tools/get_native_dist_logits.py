#!/usr/bin/env python3
import hashlib
import json
import os
from datetime import datetime
from functools import partial

import torch

from gpt_builders import gpt_builder
from model_provider import model_provider
from megatron.core import mpu
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
from megatron.training import get_args, get_model, get_tokenizer
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training.utils import get_ltor_masks_and_position_ids


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def patch_te_set_extra_state_eof():
    """Ignore corrupt/empty TE extra_state blobs when loading older checkpoints."""
    try:
        from transformer_engine.pytorch.module import base as te_base
    except Exception:
        return

    cls = te_base.TransformerEngineBaseModule
    if getattr(cls, "_patched_ignore_eof_extra_state", False):
        return

    original_set_extra_state = cls.set_extra_state

    def safe_set_extra_state(self, state):
        try:
            return original_set_extra_state(self, state)
        except EOFError:
            # Some checkpoints carry empty/legacy TE extra_state payloads.
            # Skipping this preserves weights and lets inference proceed.
            return None

    cls.set_extra_state = safe_set_extra_state
    cls._patched_ignore_eof_extra_state = True


def extra_args(parser):
    group = parser.add_argument_group("native-logits")
    group.add_argument("--prompt", type=str, required=True)
    group.add_argument("--out-pt", type=str, default="")
    group.add_argument("--out-report", type=str, default="")
    return parser


@torch.inference_mode()
def main():
    initialize_megatron(
        extra_args_provider=extra_args,
        args_defaults={
            "no_load_rng": True,
            "no_load_optim": True,
            "micro_batch_size": 1,
            "exit_on_missing_checkpoint": True,
        },
    )

    args = get_args()
    if args.pipeline_model_parallel_size != 1:
        raise RuntimeError(
            f"This minimal script assumes PP=1, got PP={args.pipeline_model_parallel_size}."
        )

    patch_te_set_extra_state_eof()

    model = get_model(partial(model_provider, gpt_builder), wrap_with_ddp=False)
    load_checkpoint(model, None, None, strict=True)
    model = model[0]
    model.eval()

    tokenizer = get_tokenizer()
    token_ids = [int(x) for x in tokenizer.tokenize(args.prompt)]
    if len(token_ids) == 0:
        raise RuntimeError("Prompt tokenized to an empty sequence.")

    tokens = torch.tensor(token_ids, dtype=torch.long, device="cuda").unsqueeze(0)

    eod_token = getattr(tokenizer, "eod", None)
    if eod_token is None:
        eod_token = getattr(tokenizer, "eos", 0)
    pad_token = getattr(tokenizer, "pad", 0)

    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=eod_token,
        pad_token=pad_token,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        pad_mask_loss=False,
    )

    logits = model(tokens, position_ids, attention_mask)
    if args.tensor_model_parallel_size > 1:
        logits = gather_from_tensor_model_parallel_region(logits)
    last_token_logits = logits[:, -1, :].float().cpu()

    if mpu.get_tensor_model_parallel_rank() == 0:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        iter_tag_raw = getattr(args, "ckpt_step", None)
        if iter_tag_raw is None:
            iter_tag = "iter_unknown"
        else:
            try:
                iter_tag = f"iter{int(iter_tag_raw):07d}"
            except Exception:
                iter_tag = f"iter_{iter_tag_raw}"
        default_prefix = f"native_logits_{iter_tag}"
        out_pt, out_report = resolve_output_paths(args.out_pt, args.out_report, default_prefix)

        report = build_comparison_report(args.prompt, token_ids, last_token_logits)
        report["output_pt_path"] = out_pt
        report["output_report_path"] = out_report
        report["mode"] = "native_dist"
        report["timestamp_utc"] = timestamp

        out_dir = os.path.dirname(out_pt)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        torch.save(
            {
                "prompt": args.prompt,
                "token_ids": token_ids,
                "last_token_logits": last_token_logits,
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
