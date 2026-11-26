#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prove a specific corruption test detects reverting affine arithmetic QA.

Only a temporary source copy is mutated. The production checkout and fixture
expectations are unchanged. Exit zero means the expected failure was observed.
"""

from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
TARGET = "test_c10_dataset.C10SemanticTests.test_reject_affine_arithmetic"
CHILD = """
import importlib.util
import importlib
from pathlib import Path
import sys
import unittest
root, source, expected = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(root / 'python/tools'))
sys.path.insert(0, str(root / 'python/tests'))
importlib.import_module('oa_embodied_ai.dataset_qa')
spec = importlib.util.spec_from_file_location(sys.argv[5], source)
module = importlib.util.module_from_spec(spec)
sys.modules[sys.argv[5]] = module
spec.loader.exec_module(module)
suite = unittest.defaultTestLoader.loadTestsFromName(sys.argv[4])
result = unittest.TextTestRunner(verbosity=2).run(suite)
valid = (result.testsRun == 1 and not result.errors and not result.skipped
         and len(result.failures) == int(expected))
raise SystemExit(0 if valid else 1)
"""


def prove(name, target, rule, replacement):
    source = ROOT / f"python/oa_embodied_ai/dataset_qa/{name}.py"
    original = source.read_text()
    if original.count(rule) != 1:
        raise SystemExit("FAIL: arithmetic rule changed; update the precise revert experiment")
    module = f"oa_embodied_ai.dataset_qa.{name}"
    subprocess.run([sys.executable,"-c",CHILD,str(ROOT),str(source),"0",target,module],check=True)
    with tempfile.TemporaryDirectory(prefix="c10-reverted-rule-") as directory:
        reverted = Path(directory) / source.name
        reverted.write_text(original.replace(rule,replacement))
        subprocess.run([sys.executable,"-c",CHILD,str(ROOT),str(reverted),"1",target,module],check=True)
    if source.read_text() != original:
        raise SystemExit("FAIL: source checkout changed during the experiment")
    print(f"PASS: {target} passed on current code and failed exactly once on the deliberate mutation; checkout preserved.")


def main():
    prove("semantics", TARGET, 'calculated == stamp.get("mapped_authoritative_ns")',
          "True  # deliberately reverted arithmetic comparison\n                                 ")
    prove("handoff", "test_c10_handoff.C10HandoffTests.test_synthetic_origin_cannot_be_relabelled_by_real_evidence_flag",
          "ready = not reasons", "ready = True  # deliberately promote blocked evidence")


if __name__ == "__main__":
    main()
