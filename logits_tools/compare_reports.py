#!/usr/bin/env python3
import argparse
import glob
import json
import os
from collections import defaultdict
from itertools import combinations
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect logits reports and print them side-by-side in a table."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="",
        help="Directory containing *.report.json files (default: <script_dir>/results)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.report.json",
        help="Glob pattern inside results dir",
    )
    parser.add_argument(
        "--full-keys",
        action="store_true",
        help="Show full hash values instead of shortened prefixes",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance used when comparing saved logits tensors.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance used when comparing saved logits tensors.",
    )
    return parser.parse_args()


def _short(value: str, full: bool = False, n: int = 12) -> str:
    if value is None:
        return ""
    if full:
        return value
    return value[:n]


def _fmt_float(v) -> str:
    try:
        return f"{float(v):.6f}"
    except Exception:
        return str(v)


def _fmt_top5_ids(v) -> str:
    if isinstance(v, list):
        return ",".join(str(int(x)) for x in v[:5])
    return ""


def _fmt_top5_logits(v) -> str:
    if isinstance(v, list):
        return ",".join(_fmt_float(x) for x in v[:5])
    return ""


def load_reports(results_dir: str, pattern: str) -> List[Dict]:
    paths = sorted(glob.glob(os.path.join(results_dir, pattern)))
    rows = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                report = json.load(f)
        except Exception as e:
            rows.append(
                {
                    "path": p,
                    "report_file": os.path.basename(p),
                    "mode": "ERROR",
                    "prompt_sha256": "",
                    "token_ids_sha256": "",
                    "logits_sha256": "",
                    "output_pt_path": "",
                    "top1_token_id": "",
                    "top1_logit": "",
                    "top5_token_ids": "",
                    "top5_logits": "",
                    "timestamp_utc": "",
                    "error": str(e),
                }
            )
            continue

        rows.append(
            {
                "path": p,
                "report_file": os.path.basename(p),
                "mode": report.get("mode", ""),
                "prompt_sha256": report.get("prompt_sha256", ""),
                "token_ids_sha256": report.get("token_ids_sha256", ""),
                "logits_sha256": report.get("logits_sha256", ""),
                "output_pt_path": report.get("output_pt_path", ""),
                "top1_token_id": report.get("top1_token_id", ""),
                "top1_logit": report.get("top1_logit", ""),
                "top5_token_ids": report.get("top5_token_ids", []),
                "top5_logits": report.get("top5_logits", []),
                "timestamp_utc": report.get("timestamp_utc", ""),
                "error": "",
            }
        )
    return rows


def render_table(rows: List[Dict], full_keys: bool):
    headers = [
        "report_file",
        "mode",
        "timestamp_utc",
        "top1",
        "top5_ids",
        "top5_logits",
        "logits_sha256",
    ]

    table_rows = []
    for r in rows:
        top1 = r["top1_token_id"]
        if top1 != "":
            top1 = f"{top1}:{_fmt_float(r['top1_logit'])}"
        table_rows.append(
            [
                r["report_file"],
                r["mode"],
                r["timestamp_utc"],
                top1,
                _fmt_top5_ids(r["top5_token_ids"]),
                _fmt_top5_logits(r["top5_logits"]),
                _short(r["logits_sha256"], full=full_keys),
            ]
        )

    widths = [len(h) for h in headers]
    for row in table_rows:
        for i, v in enumerate(row):
            widths[i] = max(widths[i], len(v))

    def fmt_row(values):
        return " | ".join(values[i].ljust(widths[i]) for i in range(len(values)))

    sep = "-+-".join("-" * w for w in widths)
    print(fmt_row(headers))
    print(sep)
    for row in table_rows:
        print(fmt_row(row))


def render_groups(rows: List[Dict], full_keys: bool):
    groups = defaultdict(list)
    for row in rows:
        key = row.get("logits_sha256", "")
        if key:
            groups[key].append(row["report_file"])

    print("")
    print("identical_groups:")
    for key, files in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        key_show = _short(key, full=full_keys)
        print(f"- key={key_show} count={len(files)} files={', '.join(files)}")


def _load_logits_tensor(path: str):
    if not path or not os.path.exists(path):
        return None, f"missing_pt:{path or '<empty>'}"
    try:
        import torch

        data = torch.load(path, map_location="cpu")
        logits = data.get("last_token_logits") if isinstance(data, dict) else None
        if logits is None:
            return None, "missing last_token_logits"
        return logits.float().cpu(), ""
    except Exception as exc:
        return None, str(exc)


def render_pairwise_checks(rows: List[Dict], atol: float, rtol: float):
    comparable = [
        row
        for row in rows
        if row.get("prompt_sha256") and row.get("token_ids_sha256") and row.get("mode") != "ERROR"
    ]
    groups = defaultdict(list)
    for row in comparable:
        groups[(row["prompt_sha256"], row["token_ids_sha256"])].append(row)

    print("")
    print("pairwise_checks:")
    any_pairs = False
    for _, group in sorted(groups.items(), key=lambda kv: (kv[0], len(kv[1]))):
        if len(group) < 2:
            continue
        for left, right in combinations(group, 2):
            any_pairs = True
            label = f"{left['report_file']} <-> {right['report_file']}"
            if left.get("logits_sha256") == right.get("logits_sha256"):
                print(f"- PASS exact {label}")
                continue

            left_logits, left_error = _load_logits_tensor(left.get("output_pt_path", ""))
            right_logits, right_error = _load_logits_tensor(right.get("output_pt_path", ""))
            if left_error or right_error:
                print(f"- CHECK {label} hashes differ; tensor compare unavailable ({left_error or right_error})")
                continue
            if tuple(left_logits.shape) != tuple(right_logits.shape):
                print(
                    f"- FAIL shape {label} left={tuple(left_logits.shape)} right={tuple(right_logits.shape)}"
                )
                continue

            import torch

            delta = (left_logits - right_logits).abs()
            max_abs = float(delta.max().item())
            mean_abs = float(delta.mean().item())
            passed = bool(torch.allclose(left_logits, right_logits, atol=atol, rtol=rtol))
            status = "PASS" if passed else "FAIL"
            print(
                f"- {status} close {label} max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} "
                f"atol={atol:g} rtol={rtol:g}"
            )
    if not any_pairs:
        print("- no reports share the same prompt/token fingerprint")


def main():
    args = parse_args()
    if args.results_dir:
        results_dir = args.results_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, "results")

    if not os.path.isdir(results_dir):
        raise SystemExit(f"results dir not found: {results_dir}")

    rows = load_reports(results_dir, args.pattern)
    if not rows:
        print(f"no report files found in {results_dir} matching {args.pattern}")
        return

    render_table(rows, full_keys=args.full_keys)
    render_groups(rows, full_keys=args.full_keys)
    render_pairwise_checks(rows, atol=args.atol, rtol=args.rtol)


if __name__ == "__main__":
    main()
