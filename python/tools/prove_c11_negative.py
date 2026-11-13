#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Execute focused tests against current code and deliberately weakened temp copies.

No checkout files change. Mutation failures must be assertion failures in exactly
one executed test, with no errors or skips; import/environment errors do not count.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASES = [
    (
        "canonical-readiness",
        "training/gate.py",
        "test_c11_gate.py::test_gate_itself_rejects_missing_diagnostics",
    ),
    (
        "physical-promotion",
        "training/cli.py",
        "test_c11_training.py::test_bounded_cpu_training_and_honest_checkpoint",
    ),
    (
        "zero-image-fallback",
        "training/model.py",
        "test_c11_training.py::test_every_camera_affects_prediction_and_receives_gradient",
    ),
]


def mutate(text, name):
    if name == "canonical-readiness":
        start = text.index("        decision = training_readiness(")
        end = text.index("        provenance =", start)
        return (
            text[:start]
            + '        decision = {"training_ready": True, "status": "PASS", "reasons": []}\n'
            + text[end:]
        )
    if name == "physical-promotion":
        return text.replace(
            '"acceptance_status": "NOT_VERIFIED_PHYSICAL"',
            '"acceptance_status": "VERIFIED_PHYSICAL"',
        )
    needle = "        B = state.shape[0]\n"
    start = text.index(needle, text.index("class ACTPolicy")) + len(needle)
    return (
        text[:start]
        + "        if images is None:\n            images = torch.zeros(B, self.num_cameras, 3, 32, 32, device=state.device)\n"
        + text[start:]
    )


def run(test, source, junit):
    env = dict(
        os.environ,
        PYTHONPATH=str(source),
        OA_INTERFACES_ROOT=str(ROOT / "external/openamrobot-interfaces"),
    )
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(ROOT / "python/tests" / test),
            "-q",
            f"--junitxml={junit}",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    print(process.stdout)
    print(process.stderr)
    suites = ET.parse(junit).getroot().findall("testsuite")
    counts = {
        key: sum(int(s.get(key, 0)) for s in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    return process.returncode, counts


def main():
    with tempfile.TemporaryDirectory(prefix="c11-mutation-proof-") as tmp:
        tmp = Path(tmp)
        for name, file, test in CASES:
            baseline = run(test, ROOT / "python", tmp / f"{name}-baseline.xml")
            if baseline != (0, {"tests": 1, "failures": 0, "errors": 0, "skipped": 0}):
                raise RuntimeError(f"Baseline failed: {name}: {baseline}")
            copy = tmp / name
            shutil.copytree(
                ROOT / "python/oa_embodied_ai",
                copy / "oa_embodied_ai",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            path = copy / "oa_embodied_ai" / file
            original = path.read_text()
            changed = mutate(original, name)
            if changed == original:
                raise RuntimeError(f"Mutation did not apply: {name}")
            path.write_text(changed)
            mutant = run(test, copy, tmp / f"{name}-mutant.xml")
            if mutant != (1, {"tests": 1, "failures": 1, "errors": 0, "skipped": 0}):
                raise RuntimeError(f"Expected one assertion failure: {name}: {mutant}")
            print(
                f"PASS: {name}: current test passed; deliberate weakening failed exactly once; checkout preserved"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
