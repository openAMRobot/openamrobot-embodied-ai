<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# C10/C11 reproducible software verification

Date: 2 November 2026 (America/Chicago). Exact original heads: C10 PR #3
`79ac48e0fda292ddb52596337a745f0b77a35781`; C11 PR #4
`db199924543d888259fdf87c829481968e2112f3`. The integration worktree preserves
both commits and records the final execution HEAD, full source hashes, commands,
exit codes and environment in `.verification/c10-c11/final/final-execution.json`.
Run the complete recorder with:

```bash
.venv/bin/python python/tools/verify_c10_c11.py --output-dir .verification/c10-c11/final
```

Ignored local evidence remains available in this worktree; it is not a real
recording or a production artifact. Published CI artifacts will be separate
remote evidence only once those jobs actually execute.

## Environment and reproduction

macOS arm64, CPython 3.12.13, Torch 2.15.0, torchvision 0.29.2, PyArrow 26.0.0;
ffmpeg/ffprobe available. LeRobot is not installed or imported by this custom
model. Exact full installed package inventory is retained alongside logs.
These versions describe this execution; none is an approved release pin.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e 'python[test,native,training]' ruff build
# Check out canonical interfaces at 87ab8a936c97ab55ee74b8988ca3d7659160ea6a
# under external/openamrobot-interfaces (or set OA_INTERFACES_ROOT).
.venv/bin/python -m pytest python/tests -q
.venv/bin/python python/tools/run_verification_tests.py python/native_tests test_c10_native.py
.venv/bin/python python/tools/run_verification_tests.py python/native_tests test_c10_evidence.py
.venv/bin/python python/tools/prove_c10_negative.py
.venv/bin/python python/tools/prove_c11_negative.py
.venv/bin/python python/tools/run_c11_smoke.py --output-dir .verification/fresh-c11-smoke
.venv/bin/python python/tools/generate_c10_evidence.py --output-dir .verification/fresh-c10-evidence
```

Use a fresh output directory on each evidence run. `run_c11_smoke.py` makes two
explicitly synthetic episodes sharing native v3 Parquet/video artifacts, each
with two frames and four canonical cameras, plus synthetic-only test policy and
manifest. It performs two CPU epochs with at most two batches each, saves/hash
checks a strict checkpoint, and evaluates held-out images. The raw C10 report
has 324 PASS findings and two NOT_VERIFIED C10_REAL_EPISODE findings, readiness
false and exit 2. The software-test helper exits 0 only for the expected narrow
synthetic semantics. Physical and production acceptance remain NOT_VERIFIED.

Test logs below are relative to `.verification/c10-c11/final/`; the original
baseline log remains in `.verification/c10-c11/`. C11 QA/manifest hashes use
canonical JSON (`sort_keys=True`, compact separators); the original C10 CLI
report hash is over its exact serialized file bytes. Checkpoint hashes always
cover exact checkpoint bytes.

## Results

| Verification | Result | Evidence file |
| --- | --- | --- |
| Original combined PR #4 suite, before corrections | 114 passed + 32 subtests; no skips. This exposed inadequate assertions, not safe integration. | `baseline-pytest.log` |
| Integrated Python suite | 166 passed + 17 subtests; no skips | `integrated-pytest.log`, `pytest.xml` |
| C10 native suite | 34 passed, no skips; synthetic MP4 decode/corruption executed | `native-suite.log` |
| C10 evidence suite | 23 passed, no skips | `evidence-suite.log` |
| C10 mutation proof | Two baseline checks pass; each deliberate weakening produces its expected test failure; helper passes | `c10-negative.log` |
| C11 mutation proof | Three current tests pass; weakened canonical decision, physical promotion and zero-image fallback each fail exactly one assertion; no errors/skips; checkout unchanged | `c11-negative.log` |
| Bounded synthetic CPU training and offline evaluation | Software execution completed; C10/physical/production NOT_VERIFIED; zero executed rollouts, no success rate | `final-smoke/software-test-evidence.json` |
| Clean-HEAD C10 CLI evidence | Helper passes reporting-path invariant; raw synthetic gate exits 2, remains NOT_VERIFIED | `final-c10/evidence.json` |
| Scoped lint/format and diff whitespace | Passed | `lint.log`, `format.log`, `diff-check.log` |
| Wheel build/core package imports + pinned C9 schema loading | Passed | `wheel.log`, `wheel-import.log` |

Counts exclude the intentional failing mutants, which are negative proof evidence,
not regressions in the integration branch. In total the ordinary suites execute
223 distinct tests plus 17 pytest subtests. No test was skipped in the executed
suites. Unavailable native-inspection branches are tested with explicit fault
injection; installed dependencies were not removed from the environment.

The original upstream C11 tests were edited because several asserted acceptance
of fabricated incomplete reports, missing-camera zero fallbacks or invented
physical success. Replacement regressions retain model/config/metric coverage
and add actual image gradients, native shared offsets, per-episode checks,
startup-before-model assertions, source/input provenance, strict held-out splits
and JSON/Markdown agreement. See [audit](c11-integration-audit.md).

## Lint, packaging and evidence interpretation

Scoped Ruff invocation uses `--isolated --select F,I,B`; formatting uses
`ruff format --isolated`. There is no repository Ruff policy; this avoids
inheriting the execution host's unrelated parent-directory configuration. All
modified C11 Python/tests/helpers are included. C10's existing source formatting
is retained for the small policy-routing extension; Python compilation and the
C10 suites verify it. No ROS/safety-controller code is changed.

The new `C10 C11 CPU integration` workflow installs declared extras and CPU
Torch, checks the pinned C9 assets, runs all Python regressions and retains honest
synthetic software evidence. Existing C10 checks remain intact. No hosted result
is inferred from local results. No GPU training, external dataset download,
simulator/hardware motion or production deployment was attempted.

## Dependencies

Changed `python/pyproject.toml` experimental training extra:

- Removed unused LeRobot (`lerobot>=0.4.1`, Apache-2.0), SciPy (`scipy>=1.10.0`,
  BSD-3-Clause), statsmodels (`statsmodels>=0.14.0`, BSD-3-Clause). This custom
  architecture does not call LeRobot; Wilson intervals use the Python standard
  library. Original baseline tests installed SciPy/statsmodels only to reproduce
  the original suite before removing those runtime requirements.
- Added direct NumPy (BSD-3-Clause; [upstream](https://github.com/numpy/numpy)),
  PyArrow (`>=17,<27`, Apache-2.0; existing native-extra range reused;
  [upstream](https://github.com/apache/arrow)), and opencv-python-headless
  (Apache-2.0; [packaging source](https://github.com/opencv/opencv-python)).
- Retained Torch `>=2.2.0` and torchvision `>=0.17.0` (BSD-style licences;
  [Torch](https://github.com/pytorch/pytorch),
  [torchvision](https://github.com/pytorch/vision)). No arbitrary new release pin
  or validation approval was introduced. Runtime reproducibility is documented
  by the actual environment inventory and checkpoint/source/config hashes.

## Not verified and external acceptance

No authentic recorded dataset, signed/approved recording policy, release pin,
physical calibration, combined-model ROS execution, upstream LeRobot ACT/native
writer interoperability, simulator/hardware closed-loop rollout or production
acceptance is verified. Statistical helpers and synthetic media cannot establish
those facts. C11 owner must adopt/adapt the custom architecture and corrections;
recording/release/model owners must close the [remaining dependencies](c11-integration-audit.md#owner-dependencies).
