<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Prepared PR #3 integration status update

Target: [PR #3](https://github.com/openAMRobot/openamrobot-embodied-ai/pull/3),
`79ac48e0fda292ddb52596337a745f0b77a35781`. **Prepared, not posted.**

The independent C10/C11 review uses C11 PR #4 head
`db199924543d888259fdf87c829481968e2112f3`, which already includes this exact C10
commit. Corrections live on a separate integration branch; neither contributor's
branch was rewritten and neither PR's draft status changed.

C11 now reruns native `check_dataset` with an independently bound immutable
snapshot, reviewed policy and exact episode selection, then calls canonical
`training_readiness`. Stale/partial/contradictory or registry-only evidence blocks
with exit 2. Synthetic software testing retains the original C10 NOT_VERIFIED
result and permits no unrelated missing integrity evidence. A strict additive
per-episode policy map addresses the single-episode policy's inability to cover
a training/validation selection; the canonical readiness consumer and C9 schema
remain unchanged.

The [version 1 mapping](robot-profile-compatibility-v1.md) documents all 16 C9
and 19 historical C11 entries separately, including measured versus commanded
values, actual C9 degree units and unconfirmed physical bindings. The nineteen
entries contain six arm joints plus a gripper per side. They do not identify the
new lift-equipped combined model or authorize a lift command. Lift observations
need recording/model-owner agreement on storage and calibration provenance.

[Local verification](c11-integration-verification.md) includes core/native/evidence
suites, negative mutation proofs and a bounded synthetic four-camera CPU training
and offline evaluation run. The custom model is not verified upstream LeRobot
ACT. No actual rollout adapter or real recorded episode was validated.

Please retain the software migration approval with its existing boundaries,
review the per-episode policy extension and mapping with the C11/model owners,
and keep this PR draft. CODE COMPLETE for the scoped corrections is separate
from VALIDATED REAL RECORDING and PRODUCTION ACCEPTED, both NOT_VERIFIED.

AI disclosure: OpenAI Codex assisted source review, implementation, tests and
local evidence generation. No hardware, recording or production approval is claimed.
