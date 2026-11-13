#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prove missing-lift regression catches deletion in an isolated source copy."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = "python/tests/test_conversion_validation.py::test_missing_lift_regression"


def main():
    baseline = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", TARGET],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    print("CURRENT:", baseline.stdout, baseline.stderr)
    if baseline.returncode != 0:
        return 1
    with tempfile.TemporaryDirectory(prefix="oa-missing-lift-mutation-") as tmp:
        root = Path(tmp)
        shutil.copytree(ROOT / "python/oa_embodied_ai", root / "oa_embodied_ai")
        path = root / "oa_embodied_ai/conversion/validation.py"
        text = path.read_text()
        target = "if component not in available or available[component] is None:"
        if text.count(target) != 1:
            raise RuntimeError("Mutation anchor must occur exactly once")
        path.write_text(
            text.replace(target, "if False:  # deliberate removed component guard")
        )
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", TARGET],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(root)},
            capture_output=True,
            text=True,
        )
        print("MUTATED:", result.stdout, result.stderr)
        if (
            result.returncode != 1
            or "1 failed" not in result.stdout
            or "DID NOT RAISE" not in result.stdout
            or "ERROR " in result.stdout
        ):
            return 1
    print(
        "PASS: removed missing-component validation causes exactly the intended assertion failure"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
