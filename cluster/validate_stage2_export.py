#!/usr/bin/env python3
"""Check whether a Stage 2 Hugging Face export is safe for downstream gates.

The SLURM wrapper runs this dependency-free check before returning success.  The
submit-only orchestrator runs it again before reusing an existing export.
"""

from __future__ import print_function

import argparse
import json
from pathlib import Path
import sys


def validate(output_dir, expected_source, require_verify_load=True):
    output = Path(output_dir)
    source = Path(expected_source).resolve()
    info_path = output / "conversion_info.json"

    if (output / ".export_incomplete").exists():
        raise ValueError(".export_incomplete is still present")
    if not info_path.is_file():
        raise ValueError("conversion_info.json is missing")
    if not source.name.startswith("iter_"):
        raise ValueError("expected source is not an iter_XXXXXXX directory")
    if not (source / "common.pt").is_file():
        raise ValueError("expected source no longer contains common.pt")

    try:
        expected_iteration = int(source.name[len("iter_") :])
        info = json.loads(info_path.read_text())
        actual_source = Path(info["source_checkpoint"]).resolve()
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("conversion_info.json is malformed: {}".format(exc))

    settings = info.get("settings", {})
    problems = []
    if actual_source != source:
        problems.append("source_checkpoint does not match {}".format(source))
    if info.get("iteration") != expected_iteration:
        problems.append("iteration is not {}".format(expected_iteration))
    if info.get("verified") is not True:
        problems.append("verified is not true")
    if settings.get("verify") is not True:
        problems.append("settings.verify is not true")
    if require_verify_load and settings.get("verify_load") is not True:
        problems.append("settings.verify_load is not true")
    if problems:
        raise ValueError("; ".join(problems))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    parser.add_argument("expected_source")
    parser.add_argument(
        "--allow-no-verify-load",
        action="store_true",
        help="accept a generic export that skipped the full model load (the pipeline never uses this)",
    )
    args = parser.parse_args(argv)
    try:
        validate(
            args.output_dir,
            args.expected_source,
            require_verify_load=not args.allow_no_verify_load,
        )
    except ValueError as exc:
        print("Stage 2 export is not pipeline-ready: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
