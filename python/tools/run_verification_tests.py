# SPDX-License-Identifier: MIT
"""Run a nonempty verification suite; skipped tests are failures."""

from pathlib import Path
import sys
import unittest

suite = unittest.defaultTestLoader.discover(str(Path(sys.argv[1])), pattern=sys.argv[2])
if suite.countTestCases() == 0:
    raise SystemExit("FAIL: no verification tests discovered")
result = unittest.TextTestRunner(verbosity=2).run(suite)
if result.skipped:
    raise SystemExit("FAIL: verification tests were skipped")
if not result.wasSuccessful():
    raise SystemExit(1)
