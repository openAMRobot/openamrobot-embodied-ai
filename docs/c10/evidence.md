<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Reproducible C10 evidence

After installing the optional native extra and external canonical assets, run
from a clean checkout at the revision being reviewed:

```bash
python python/tools/generate_c10_evidence.py \
  --output-dir .verification/c10-evidence/synthetic \
  --expected-checker-sha "$(git rev-parse HEAD)"
```

Output must be fresh. The helper records actual before/after Git SHA and clean
status, exact CLI argv/cwd, interpreter/platform/package versions, source/schema/
fixture hashes, native input hashes before/after, raw CLI output and report hash.
It retains `c10-report.json` verbatim. Exit 0 means the expected synthetic
NOT_VERIFIED/exit 2, false training readiness and explicit synthetic-origin finding
were verified. It does not convert that dataset result to PASS.

Without `--expected-checker-sha`, a dirty working tree is explicitly recorded;
that mode is development evidence only. Drift, unexpected PASS/REJECT, missing or
inconsistent reports and preexisting outputs fail the helper and retain available
partial evidence. Report bytes reproduce across independently generated directories.
No report-generation timestamp or absolute output paths enter the raw gate report.

## CI artifacts

`.github/workflows/c10-dataset-quality.yml` independently runs on Ubuntu 24.04,
Python 3.12, without ROS/Gazebo. It checks out the explicit PR head and the existing
canonical interfaces pin, installs PyArrow and system ffmpeg, runs nonempty
core/native/handoff/evidence suites and both mutation proofs, then generates the
exact-head synthetic bundle. Logs use `pipefail`; later evidence steps run unless
cancelled, including after earlier failures. Upload always runs, requires actual
files, includes hidden paths and retains evidence for 30 days.

Download `c10-evidence-<head-sha>-<attempt>` from that run's Actions page. Contents:
expected revision, suite logs, mutation proof output, generation log and
`synthetic/{c10-report.json,evidence.json,cli-stdout.log,cli-stderr.log,dataset/…}`.
Dataset payloads are temporary synthetic fixtures in CI artifacts, never Git.
The independent basic Python job still runs registry/core/handoff tests; the ROS
job and its runtime dependencies are unchanged. Uploaded failure artifacts are
diagnostic evidence, not a passing gate.

Hosted run and artifact links, when actually available, are recorded in the draft
PR. Local [verification](verification.md) is reported separately. No old hosted
interfaces run is represented as execution of this migration.
