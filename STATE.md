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

12 November 2026 — dependent capture/conversion/benchmark draft: shared component
validation, explicit-unit import, observational bag CLI, native LeRobot 0.6.1
writer/loader and inference-only timing/telemetry are implemented. Synthetic CDR
bag conversion loads all four cameras and C10 remains NOT_VERIFIED/exit 2.
Bounded synthetic visual C11 training/reload/offline evaluation executes; no rollout.
Actual capture profiles are unregistered pending owners; ROS/Gazebo and Jetson
execution remain BLOCKED/NOT_VERIFIED. See docs/execution and conversion/README.

12 November 2026 — continuation: atomic ordered multi-bag export now feeds the
actual native loader and a continuous explicitly synthetic C10/C11 smoke. Joint
and camera samples are checked before resampling. Shared video boundaries use
integer PTS and rational time bases; reverting to decimal-text comparison fails
the regression. Actual capture-profile acceptance and physical results remain
NOT_VERIFIED. Docker is running for isolated generated-model checks; no physical
commands or actual Gazebo recording are claimed.

19 November 2026: draft into satya/capture-conversion-benchmark (not main):
declared equal-timestamp rule `keep_last` per stream (default unchanged: equal or
decreasing stamps reject), dropped counts in the conversion manifest, tests on
messages recorded in Gazebo run 44448676604 that fail on 9643da1; an empty
header frame_id takes the declared configuration frame. Proposed
Baseline 1 capture profile under configs/capture_profiles/proposed/, listed in
PROPOSED_CAPTURE_PROFILES (never consulted for approval). Registration, origin
label and C9 units (interfaces #41) remain owner decisions. Simulation only.
On the recorded bag, validate_profile (development) and read_bag pass; export is
still refused (first frame selects samples before the episode origin; arm
command extraction, multi-source sidecar, lift units and ranges). No dataset.

19 November 2026 (continued): Proposed conversion rules on the same draft, each
declared in the profile and enabled by name (`--enable-proposed`): episode origin
on the first complete grid with pre-origin frames dropped and counted, command
semantics, controller-reference arm commands, multi-source vectors, lift stop
index to metres, range tolerance at model limits (kept, counted, never clamped),
held commands, lift_positioning from the lift commands and report-only lift-phase
evidence. Defaults unchanged. Tests on messages recorded in run 44448676604
(fixture baseline1_run44448676604_excerpts.json). The Baseline 1 profile now
declares these rules and its simulator provenance. None is approved; a dataset
converted with them is development evidence. Stowed-pose evidence is not
available: `park` is a placeholder and the recorded arms are not in it.

19 November 2026 (continued): Proposed `reference_integrity` on the same draft:
named integrity checks on the recorded messages (units and degrees, ranges with
tolerance, lift metres, gripper opening signs with the mirrored right finger,
read/kept/dropped counts against metadata.yaml, action[t] latest at or before t with
one-frame shift detection, camera presence and skew against the grid, camera_info
consistency, no template hash in the sidecar); recorded camera_info and model hashes
in the C9 sidecar; `conversion.json` binds bag, configuration and output hashes; and
`oa-verify-reference` recomputes the conversion from the bag. Tests on a 5 s slice of
run 44448676604 (python/tests/fixtures/baseline1_slice, recorded bytes) with one
positive and one negative per check; prove_reference_negative.py reverts each check.
Proposed profile digest now aad2e963 (integrity block added). Nothing is approved;
a dataset converted with these rules is development evidence, simulation only.

22 November 2026 — PR #8 follow-up at `cf7f283`: malformed timestamps and
invalid duplicate payloads fail before replacement. The Proposed recording
profile is bound to the released source bytes. Native LeRobot metadata stays
writer-owned. The version 2 reference manifest retains per-source selections
and timing. See [workstream review](docs/execution/baseline1-review.md) for
the proposed decisions, evidence and integration dependencies.
