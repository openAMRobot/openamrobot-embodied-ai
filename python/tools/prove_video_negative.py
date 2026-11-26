#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reproduce the old rounded-decimal video-boundary bug in an isolated copy."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = "python/tests/test_video_segments.py::test_repeating_30hz_boundary_has_no_leak_or_gap"


def main():
    def run(env=None):
        return subprocess.run(
            [sys.executable, "-m", "pytest", "-q", TARGET],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )

    current = run()
    print("CURRENT:", current.stdout, current.stderr)
    if current.returncode:
        return 1
    with tempfile.TemporaryDirectory(prefix="oa-video-boundary-revert-") as tmp:
        root = Path(tmp)
        shutil.copytree(ROOT / "python/oa_embodied_ai", root / "oa_embodied_ai")
        path = root / "oa_embodied_ai/dataset_qa/video.py"
        source = path.read_text()
        anchor = "return [i for i, p in enumerate(pts) if lo_tick <= p < hi_tick]"
        if source.count(anchor) != 1:
            raise RuntimeError("Expected one exact boundary-selection anchor")
        # Restore the original six-decimal ffprobe text comparison only in the
        # temporary copy. The target fixture retains both integer PTS and text.
        path.write_text(
            source.replace(
                anchor,
                "return [i for i, frame in enumerate(frames) if start <= float(frame['best_effort_timestamp_time']) < end]",
            )
        )
        mutated = run({**os.environ, "PYTHONPATH": str(root)})
        print("REVERTED:", mutated.stdout, mutated.stderr)
        if (
            mutated.returncode != 1
            or "1 failed" not in mutated.stdout
            or "AssertionError" not in mutated.stdout
            or "ERROR " in mutated.stdout
        ):
            return 1
    print(
        "PASS: restoring rounded-decimal selection causes exactly one assertion failure"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
