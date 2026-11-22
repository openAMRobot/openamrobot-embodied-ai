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
