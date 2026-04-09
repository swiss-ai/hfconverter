#!/usr/bin/env python3
import argparse
import glob
import json
import os
from collections import defaultdict
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
                    "report_file": os.path.basename(p),
                    "mode": "ERROR",
                    "logits_sha256": "",
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
                "report_file": os.path.basename(p),
                "mode": report.get("mode", ""),
                "logits_sha256": report.get("logits_sha256", ""),
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
        "top5_ids",
        "top5_logits",
        "logits_sha256",
    ]

    table_rows = []
    for r in rows:
        table_rows.append(
            [
                r["report_file"],
                r["mode"],
                r["timestamp_utc"],
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
    print("")
    print("identical_groups:")
    for key, files in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        key_show = _short(key, full=full_keys)
        print(f"- key={key_show} count={len(files)} files={', '.join(files)}")


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


if __name__ == "__main__":
    main()
