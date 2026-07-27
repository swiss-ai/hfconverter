# Copyright 2026 the Swiss AI Initiative. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exclusive ownership of an exporter's destination directory."""

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

INCOMPLETE_MARKER = ".export_incomplete"


@dataclass(frozen=True)
class OutputClaim:
    """The marker contents that identify one export as the destination owner."""

    marker: Path
    contents: str


def claim_output_dir(output_dir: Path) -> OutputClaim:
    """Atomically claim an absent or empty directory before writing final-path files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    contents = (
        "export in progress; delete this directory and re-run\n"
        f"claim_token={token}\n"
    )
    marker = output_dir / INCOMPLETE_MARKER
    try:
        with marker.open("x", encoding="utf-8") as stream:
            stream.write(contents)
    except FileExistsError:
        raise ValueError(
            f"--output-dir is already claimed by another export: {output_dir}"
        ) from None

    claim = OutputClaim(marker=marker, contents=contents)
    unexpected = sorted(path.name for path in output_dir.iterdir() if path != marker)
    if unexpected:
        release_output_claim(claim)
        raise ValueError(
            f"--output-dir changed after validation and is no longer empty: {output_dir}; "
            f"found {unexpected}"
        )
    return claim


def release_output_claim(claim: OutputClaim) -> None:
    """Remove only the marker created by ``claim_output_dir``."""
    try:
        actual = claim.marker.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise RuntimeError(
            f"output ownership marker disappeared before completion: {claim.marker}"
        ) from None
    if actual != claim.contents:
        raise RuntimeError(
            f"output ownership marker changed before completion; refusing to remove it: "
            f"{claim.marker}"
        )
    claim.marker.unlink()
