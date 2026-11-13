#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prove missing-lift regression catches deletion in an isolated source copy."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = "python/tests/test_conversion_validation.py::test_missing_lift_regression"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native",
        action="store_true",
        help="Use the actual CDR bag regression (requires conversion extra)",
    )
    args = parser.parse_args()
    target = (
        "python/native_tests/test_bag_conversion.py::test_bag_missing_declared_lift_regression"
        if args.native
        else TARGET
    )
    baseline = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", target],
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
        anchor = "if component not in available or available[component] is None:"
        if text.count(anchor) != 1:
            raise RuntimeError("Mutation anchor must occur exactly once")
        path.write_text(
            text.replace(anchor, "if False:  # deliberate removed component guard")
        )
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", target],
            cwd=ROOT,
            env={
                **os.environ,
                "PYTHONPATH": str(root),
                "OA_INTERFACES_ROOT": os.environ.get(
                    "OA_INTERFACES_ROOT", str(ROOT / "external/openamrobot-interfaces")
                ),
            },
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
