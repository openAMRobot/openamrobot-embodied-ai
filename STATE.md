<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Repository state

24 October 2026 — C10 draft migration from interfaces PR #25. Registry QA remains
unchanged. Episode/native QA, semantic checks, optional media dependencies,
reproducible evidence and a fail-closed C11 handoff live under `python/`.
See [integration audit](docs/c10/integration-audit.md),
[verification](docs/c10/verification.md) and [acceptance checklist](docs/c10/acceptance-checklist.md).

Runtime location follows EMBODIED-AI-REPO. Implementation adoption and release,
profile/timing/calibration decisions remain pending. No real recording supplied;
REAL_EPISODE_NOT_VERIFIED, production acceptance NOT_VERIFIED. Synthetic tests
exercise reporting and rejection behavior only. C11/C14 and ROS runtime code are
outside this migration; the existing skeletons and safety boundaries remain intact.

2 November 2026 — C10/C11 integration corrections on a dedicated draft branch,
combining C10 `79ac48e` and C11 `db19992` without rewriting either contributor's
branch. Canonical C10 invocation, immutable input/policy bindings, native media
loading, profile checks and honest offline/reporting behavior are implemented.
The custom visual model is experimental; real rollout execution is unavailable.
See [audit](docs/c10/c11-integration-audit.md),
[mapping](docs/c10/robot-profile-compatibility-v1.md) and
[evidence](docs/c10/c11-integration-verification.md). C11-owner adoption and
actual recording/release/profile policies remain pending. REAL_EPISODE_NOT_VERIFIED;
physical and production acceptance NOT_VERIFIED. PR #3 remains draft.

12 November 2026 — PR #5 remains the proposed canonical integration path: exact
C10 and C11 contributor commits are ancestors, and #3/#4 stay open pending owner
coordination. Documented 16-channel interim versus historical 19-channel scope
and the current lift/base proposal dependencies. Existing-head local baseline:
166 tests + 17 subtests passed; hosted C10/CPU/Python/ROS pass at 59bb4bf. Five
Satya-authored commits still lack DCO sign-off; no shared history was rewritten.
Physical/production acceptance remains NOT_VERIFIED. Recorder/converter/benchmark
work is prepared separately for a dependent draft, not claimed as accepted here.
