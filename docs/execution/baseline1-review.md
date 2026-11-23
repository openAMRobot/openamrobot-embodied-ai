<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Baseline 1 and C9/C10/C11 review

Snapshot 22 November 2026. All recommendations below are proposals. The
recording is Gazebo Harmonic `firmware_sil` using the `sim-oracle` acceptance
profile and ground-truth planning and object poses. C10 remains
`NOT_VERIFIED`, exit 2, with training readiness false.
[Decision request](https://github.com/openAMRobot/openamrobot-embodied-ai/pull/8#issuecomment-7020185839).
[Integration review](https://github.com/openAMRobot/openamrobot-embodied-ai/pull/8#issuecomment-7057308793).
[Published recording and dataset](https://github.com/openAMRobot/openamr-upperbody-sw/releases/tag/dev-evidence-baseline1-c9-2026-11-19).

## Workstream matrix

| Task | Repository and state | Work and dependencies | Owner and autonomy | Acceptance |
|---|---|---|---|---|
| Reference converter | Embodied AI #8, draft `cf7f283`, based on #6 `9643da1`. Native conversion and software CI passed upstream. | Follow-up validates source timestamps and discarded payloads, checks published recording identity and keeps LeRobot metadata writer-owned. Profile policy depends on interfaces #39 and #41–45. | Satya owns conversion. Safe fixes implemented locally. Joint semantics, timing and thresholds require Satya and the software lead. | Verify published v1, reproduce v2 twice, load all four cameras, exercise negative regressions, retain raw C10 exit 2. |
| Component binding | Interfaces #39, draft `43ec677`. Source and generated model checks passed upstream. | Patch Proposed lift source and both fixtures using full-bag evidence. Later consumer repin changes the adjunct digest. | Satya owns C9. Source correction implemented locally. Software lead approves the mapping. | MCAP decode, exact ordering and units, three pinned generated URDF checks, ROS/DDS gate, identical accepted C9 v1 bytes. |
| Canonical QA and training | Embodied AI #3 `79ac48e` and #4 `db19992` are exact ancestors of draft #5 `2469ab4`. #6 and #7 depend on #5. | Bind the immutable input and prevent contradictory reports and unsupported profiles. Exact DCO omissions and CLA renewal remain open. | Satya and Marcus own C10 and C11 respectively. Local correctness fixes are autonomous. Contribution attestations and the PR disposition need their owners. | Snapshot and format regressions, four-camera tests, exact report comparison, synthetic training and offline evaluation with zero rollout claims. |
| Recorder and gripper | Upper body #91 draft `d58cb2a`. Published recording came from `a207463`. | Coordinate interfaces #45, record issue stamps per side and run information, check the profile generator against one canonical digest. Existing-head SIL failures need the owning team. | Upper-body recorder owner and software lead decide the new source and rerun. | Prove distinct per-side issuance, pre-first-goal status, exact config identity and capture completeness on a new run. |
| Benchmark and runtime sources | Upper body #35–39, platform #116 and interfaces #30/#38 remain open. | Reuse #6 missing-lift regression and inference harness. Confirm the benchmark target after the compute decision. Resolve platform action authority, CTL-017 producer and stowed-pose contract. | Platform, upper-body, manipulation and hardware owners. Software work and physical collection use their own gates. | Recorded command source and firmware telemetry; reference Jetson A/B/C results; reviewed stowed pose and controller fixture evidence. |

Default branches observed were Embodied AI `2c0ad9b`, interfaces `a029d2c`,
upper body `d351fe1`, platform `f45b9cc` and governance `70d3266`.
The contribution wording and CLA 2.0 activation landed across the repos.
Interfaces also landed `StopState` #35, platform landed command arbitration
#90, and upper body landed SIL and model updates. The published recording still
uses its own pins. Upper-body #91's current head has passing static, profile
and capture checks, with failed firmware-SIL drive/lift/cancel,
arms/grasp/single-writer and fault jobs. The discovery fix has a unit check.
A new capture at that head remains outstanding.
[Upper-body checks](https://github.com/openAMRobot/openamr-upperbody-sw/pull/91/checks).
[Platform arbitration](https://github.com/openAMRobot/openamr-platform-sw/pull/90).
[Governance activation](https://github.com/openAMRobot/.github/pull/97).

## Proposed decisions

| # | Recommendation | Technical basis and approval |
|---|---|---|
| 1 | **ADAPT** equal timestamps | Limit the policy to explicit opt-ins on the four affected streams. The current profile opts eight streams in. Stable `keep_last` over reader order reproduces the single-file recording. Keep strict defaults for cameras and other streams. Direct MCAP and rosbags audits agree on counts. A decreasing selected timestamp rejects before resampling. Headerless reset detection on general indexed or split bags needs a storage-order or clock-epoch check. **Satya**, with C9 semantics review under #44. |
| 2 | **ADOPT** reconciliation | Require source messages read = kept + equal-stamp drops = metadata count for each selected stream. Account for grid drops separately. Reject a malformed ROS stamp or discarded invalid payload before replacement and use strict metadata counts. The full bag hash allows replay of each choice. **Satya** approves the rule, and the converter hardening is ready for review. |
| 3 | **ADAPT** SI representation | Preserve rad for revolving joints and finger joints, rad/s for angular velocity, m for lift and m/s for base linear velocity in the Proposed profile. Degree conversions belong at an explicit adapter boundary. Founder approved SI direction. C9 v1 still supports its existing versioned degree example. Exact typed enforcement and compatible migration require **software lead** acceptance under #41/#42. |
| 4 | **ADAPT** action timing | Keep latest recorded value at or before grid time as an archival view. It mixes controller interpolated references, held goal publications, lift stop requests and pre-firmware base commands. C11 currently chunks from `action[t]`, then masks the episode tail. A fixed one-frame shift cannot recover original issue time or make nearest observations causal. Require a declared C11 observation/target rule with timing and tail tests before training. Mixed C9 v1 action labels need a decision by **Satya and software lead**, with **Marcus** for C11. |
| 5 | **ADAPT** lift stop decoding | `0 → 0.000`, `1 → 0.175`, `2 → 0.350 m` follows the recorded simulator's `LIFT_STOP_M` and register geometry `shoulder z = 1.000 + lift q`. The register names calibrated stops and joint state, while the integer encoding is specific to the recorder source. The bag exercises 1, 1, 0. A recorded metre target and its timestamp should govern later evidence when the producer contract provides it. **Software lead** approves the interim decoder and producer contract. |
| 6 | **ADAPT** gripper commands | Preserve signed finger radians with positive opening on the left and negative on the right. The configured `pinch_gripper` uses revolute coordinates. Current 20 Hz headers describe republishing a held value. Use independently recorded per-side issuance times and mark measured initialization explicitly. The legacy held topic stays development evidence. **Satya and software lead** with upper-body owner under #45. |
| 7 | **ADAPT** recording and configuration provenance | Bind the exact released bag, profile, schema and episode bytes and optionally supply a trusted manifest hash. Keep native `info.json` writer-owned. The run log supplies the model-file digest, and `/build1/run_info` is missing. Canonical configuration identity remains unknown. Retain the digest as model evidence and require the #43 artifact and identity for acceptance. The pre-release already preserves the bag. **Satya** owns manifest adoption and **software lead** owns durable storage and canonical identity. |
| 8 | **ADAPT** synchronization | Keep 100 ms at 5 Hz as a Proposed simulation parameter. The full run has 67 ms peak camera-to-grid and 68 ms peak span over all seven selected observation sources. C9 applies its tolerance to inter-observation span. Record both measurements and derive the robot threshold from reviewed evidence. **Satya and software lead**. |
| 9 | **ADAPT** calibration provenance | Use the recorded `CameraInfo` digest as simulation camera-model identity. Store capture configuration and calibration evidence as separate artifacts for acceptance. All four residuals remain unmeasured and the camera-info hash establishes no extrinsics. Head RGB is one rendered view in this recording. Make layout explicit in future profiles. **Software lead**. |
| 10 | **REJECT** command-derived intervals as stow proof | Retain the report with the interval source `command_to_arrival`. Of 393 moving lift samples, 385 fell within inferred intervals, 393 had zero odometry twist and zero matched the placeholder `park` within 0.01 rad. A reviewed manipulation pose and recorded phase are required. **Software lead and manipulation owner**. |
| 11 | **ADAPT** simulation QA | Establish a separate reviewed simulation development policy. Keep real-episode acceptance unavailable for known simulation, including C9 v1 sidecars labeled `actual`. Reject promotion of this dataset to training or production under the existing gate. **Satya and software lead** approve policy and thresholds. |

The released C9 v1 sidecar uses a single action timestamp and the aggregate
enums `requested_pre_clip` and `command_issued`. The preserved sidecar is a
compatibility artifact. Those fields cannot establish the semantics of the
mixed action sources. The v2 conversion manifest records the per-source
timestamps, receipts and stand-ins behind the composite and explicitly marks
mixed-source C9 semantics `NOT_VERIFIED`. The same manifest distinguishes
recorded configuration artifacts from physical acceptance.

Supporting contract threads include [units and migration #41](https://github.com/openAMRobot/openamrobot-interfaces/issues/41),
[v1 schema proposal #42](https://github.com/openAMRobot/openamrobot-interfaces/pull/42),
[configuration identity #43](https://github.com/openAMRobot/openamrobot-interfaces/issues/43),
[duplicate stamps #44](https://github.com/openAMRobot/openamrobot-interfaces/issues/44),
[gripper issuance #45](https://github.com/openAMRobot/openamrobot-interfaces/issues/45),
and [accepted C9](https://github.com/openAMRobot/openamrobot-interfaces/blob/a029d2c247705b68761abae12a4f207e0ac6c1d6/docs/c9-lerobot-dataset-contract.md).

## Additional converter recommendations

- **ADOPT** the declared first complete grid origin with pre-origin accounting
  and the native float32 grid check. The release has one pre-origin drop and
  543 frames from 7.6 to 116.0 s. Keep frame fallback limited to explicitly
  declared sources and require exact matching for nonempty headers.
- **ADAPT** the composed vector and held-command rules. Controller references
  are interpolated controller targets. The lift can hold its selected target
  under the Proposed source contract. The base can retain only a declared
  held zero with explicit accounting. The legacy lift stand-in precedes the
  first recorded target. Preserve that distinction and require owner acceptance
  of the composite policy before calling it conformant.
- **ADAPT** 1 mrad and 1 mm range monitoring for simulation only. Preserve
  values and count departures. Keep model bounds, data quality tolerances and
  physical limits separate. For this bag the base state is `/odom` and the
  base action is post-smoothing, post-collision-monitor `/cmd_vel` upstream of
  firmware. Production authority and applied values remain open.
  [Pinned base command map](https://github.com/openAMRobot/openamr-upperbody-sw/blob/a20746338701e7961c28ba4650e11996a1a76919/ros2/src/openamrobot_build1_scene/config/command_map.yaml#L72).
  [Platform #116](https://github.com/openAMRobot/openamr-platform-sw/issues/116).
- **ADAPT** profile ownership. Treat the converter copy as canonical for this
  reproduction. Replace the upper-body duplicate only after the recorder
  owner adopts one generator and an exact digest check. Verify the next
  capture with camera identity, the actual issue stamps, and configuration
  provenance. [Recorder review](https://github.com/openAMRobot/openamr-upperbody-sw/pull/91#issuecomment-7040693969).
- **ADAPT** the Jetson benchmark target. #38 predates the recorded decision for
  an 8 GB Orin NX and Seeed Robotics 104. Keep the harness and the supplied
  MoveIt workload. Collect hardware results against the confirmed target.
  [Benchmark workload](https://github.com/openAMRobot/openamr-upperbody-sw/issues/38#issuecomment-6944807709).
  [Decisions register](https://github.com/openAMRobot/.github/blob/70d3266b8354fe3c4894896efcfb58845d2a1e63/decisions.yaml).

## Recording measurements

| Stream | Metadata count | Kept | Dropped |
|---|---:|---:|---:|
| `/arm/joint_states` | 53,209 | 53,087 | 122 |
| `/left_arm_controller/controller_state` | 53,220 | 53,103 | 117 |
| `/right_arm_controller/controller_state` | 53,210 | 53,116 | 94 |
| `/cmd_vel` | 2,952 | 1,664 | 1,288 |

All other conversion input streams reconcile without losses. In the direct
bag audit 35 arm, 23 left controller and 2 right controller adjacent equal-time
pairs differ in serialized bytes. For base velocity, 237 pairs differ in
linear or angular command values. The full bag had zero decreases in
audited timestamps in record order and zero malformed header subsecond fields.
236 frames contain a post-grid observation, including 218 with post-grid
camera data. All observed action selections were at or before the grid.

Merged `/joint_states` contains lift in all 5,512 samples and arm-only
`/arm/joint_states` contains zero lift measurements. The existing converter
profile already selects the merged topic. The interfaces adjunct and two
fixtures need the local #39 correction. The current reference is still bound
to the published #39 bytes. Reissue the digest and regenerate after that
adjunct is reviewed. CTL-017 is the separately registered upper-body
controller. Its real telemetry and the manipulation stow authority need the
open owner contracts and bench evidence.
[Source review](https://github.com/openAMRobot/openamrobot-interfaces/pull/39#discussion_r4624823808).
[CTL-017 discussion](https://github.com/openAMRobot/openamrobot-interfaces/issues/30).

## Canonical integration

Retain #5 for canonical review and credit the exact Marcus commits
`d09a546` and `db19992`. #3 and #4 can be marked superseded after their owners
confirm. Carry #6 and #7 as separate children and #8 as the #6 child. #6's
`1aa975b` duplicates the patch in canonical `2469ab4`. The C9 arms-only adapter
and fixed-mast experiment remain separate, and C11 has no supported mobile-lift
profile for this dataset. Four required camera streams remain part of C10.
The local PR #5 worktree contains the remediation recipe, exact DCO SHAs and
the gate corrections.
[Canonical review thread](https://github.com/openAMRobot/openamrobot-embodied-ai/pull/5#pullrequestreview-5840396224).

## Evidence limits

The published manifest and source bytes were checked against release hashes.
The release reproduced 543 native frames in each camera, and local
reconversion kept the same ordered state and action values. The v2 draft
writer keeps identity in the sidecar and derives a new dataset revision from
the native bytes, episode profiles and selected source evidence. Local and hosted
MP4 bytes can differ across encoder environments. Verification covers
file digests, native row bindings and decoded frame counts, with the trusted
published digest anchoring exact release media. It does not independently
prove camera pixel equivalence for an untrusted re-authored manifest.
The raw C10 result on the published dataset has 26,090 PASS findings,
1,660 NOT_VERIFIED findings and zero REJECT findings. Physical testing,
production recording, policy adoption and benchmark hardware evidence
remain outstanding.

Local verification used Python 3.12.13, LeRobot 0.6.1 and rosbags 0.11.9
with C9 assets pinned to `87ab8a9`. The command `python -m pytest -q
python/tests python/native_tests` passed 359 tests and 25 subtests.
`python python/tools/prove_reference_negative.py --json <report>` reproduced
all 20 negative controls. Two independent runs of the full preparation and
native export path matched byte for byte. The verifier reproduced the
published and regenerated datasets, including all 543 frames in each of the
four videos. The #5 follow-up's C10 rerun on the regenerated revision observed
`simulation` and returned `NOT_VERIFIED`, exit 2, with zero REJECT findings.

AI disclosure. OpenAI Codex assisted the repository audit, source-bag review,
regressions, proposed converter and gate changes, and documentation. Contributor
review, DCO sign-off and acceptance decisions remain with the owners.
