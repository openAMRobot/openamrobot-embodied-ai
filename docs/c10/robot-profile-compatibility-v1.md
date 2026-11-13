<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# C10/C11 robot profile compatibility, review version 1

2 November 2026. **Proposed integration mapping for owner review. No new production
profile, bounds, recording policy or command channel is approved here.**

## Review clarification — 12 November 2026

PR #5 implements **the 16-channel interim arms/grippers adapter**, with a
separate historical fixed-mast 19-channel experiment. Neither is the proposed
lift-equipped/mobile full-body policy. Matching vector widths never establishes
profile equivalence. The November 10 review's full-body request is tracked in
[upperbody#35](https://github.com/openAMRobot/openamr-upperbody-sw/issues/35),
with the existing C9 v1 descriptors extended additively in interfaces.

Current model source at upperbody `2ca739202f5c51114de1520902e811817d9b2a62`
confirms `lift_joint`, prismatic, 0–0.350 m for `dold_v4_350`. Mock/Gazebo
profiles expose the position interface; the real profile does not. CTL-017
telemetry/command capture remains [interfaces#30](https://github.com/openAMRobot/openamrobot-interfaces/issues/30).
The model's `park` pose is a placeholder, not an approved stowed pose.

The proposed full-body mapping includes lift state and reserves separately
reviewed lift action semantics; it does not authorize a command because the
joint exists. [Platform#116](https://github.com/openAMRobot/openamr-platform-sw/issues/116)
requires separate requested policy/operator velocity, Nav2 goals/output,
accepted arbitration output and firmware-applied telemetry. `cmd_vel_policy`
and its age/timeout contract still need the platform contract process. Recorder
work observes these streams; it changes no control, timeout or interlock logic.
The following tables retain their November 2 provenance and profile versions.

## Exact sources and interpretation

- C9: `openAMRobot/openamrobot-interfaces` at
  `87ab8a936c97ab55ee74b8988ca3d7659160ea6a`,
  [`docs/c9-lerobot-dataset-contract.md`](https://github.com/openAMRobot/openamrobot-interfaces/blob/87ab8a936c97ab55ee74b8988ca3d7659160ea6a/docs/c9-lerobot-dataset-contract.md#L52)
  and `schemas/yaml/c9-lerobot-dataset-valid.yaml:40`. Schema SHA-256
  `692e04693bc738033af1da10f98e029f342497ddaeae4792945f829335f1938e`;
  example SHA-256 `e21218b4cd4ce09969ca589cc83e72d46bfa99eee9e197195a56ec1ad22a3e28`.
  C9 accepts additive semantics and profile-specific ordered dimensions; **16 is
  the established adapter/example profile, not a universal schema vector width**.
- Historical C11: PR #4 at `db199924543d888259fdf87c829481968e2112f3`,
  [`training/config.py:21`](https://github.com/openAMRobot/openamrobot-embodied-ai/blob/db199924543d888259fdf87c829481968e2112f3/python/oa_embodied_ai/training/config.py#L21),
  `configs/c11_act_default.yaml`, and `docs/c11/state_action_mapping.md:34`.
  Its `schema-v1.0-frozen` string is a historical configuration tag, not evidence
  of accepted physical limits or an approved whole-body training profile.
- Review direction: [PR #3 reviewer comment](https://github.com/openAMRobot/openamrobot-embodied-ai/pull/3)
  (review dated 27 October 2026):
  retain separate 16/19 profiles, agree ordered fields/units/sources, log lift
  state where relevant; URDF presence does not authorize lift actions.
- Current combined model, source inspection only:
  [`combined_robot.xacro:4`](https://github.com/openAMRobot/openamr-upperbody-sw/blob/17cf69374c5d2c3a9c058c89f54c6bf43b3741d7/ros2/src/openamrobot_upperbody_description/urdf/combined_robot.xacro#L4)
  at `17cf69374c5d2c3a9c058c89f54c6bf43b3741d7` and that checkout's
  `ros2/src/openamrobot_upperbody_description/config/upstream_pins.yaml`:
  platform `131e500fc4e5afc9f648cbd7dbf5b416280761fb`, arm description
  `14ff67b638ff1c738a1b9a6be8aaa5ce5ed2c831`, arm ROS
  `4e837e1d0dae692ff67b560b69d8d281d7a8d4ed`. The model distinguishes mock, Gazebo and real
  hardware profiles; its documentation claims simulation/mock verification,
  expressly no hardware result. This review did not run that model's ROS tests.
- Governance register, read at `openAMRobot/.github`
  `5ed600ef5595703ea1d13784cd7cb9227e64a04b`: entries `LIFT`, `ARM-MOUNT`,
  `CONFIG-IDS`, `PLATFORM-PIN`. The lift is approved in principle, and its height
  is joint state; calibrated stop IDs are not configuration IDs. The local
  governance checkout was older (`b1c8700`), so current remote register bytes were
  inspected separately. Neither register nor model source approves a C11 lift
  action profile.

## Profile A: established C9 adapter, 16 channels

Identifier: `lerobot-v0.6.1-bi-openarm-position`. Every row below specifies both
`observation.state[S]` and `action[A]`, using the same ordered dimension name.
For every row: state source is the adapter's measured position; action source
is the adapter requested/pre-clip command, timed at command issue in the supplied
C9 profile. The actual recorder/topic/device binding is **UNCONFIRMED** until
recording-owner evidence is supplied. Reference frame is joint space (null
Cartesian frame with the C9 reason); approved bounds are **UNCONFIRMED** for
both observations and commands. No conversion to ROS radians/metres is implicit.

| S / A index | Canonical dimension field | Observation meaning | Command meaning | Unit, both |
| --- | --- | --- | --- | --- |
| S0 / A0 | `left_joint_1.pos` | Left joint 1 measured position | Left joint 1 requested position | degree |
| S1 / A1 | `left_joint_2.pos` | Left joint 2 measured position | Left joint 2 requested position | degree |
| S2 / A2 | `left_joint_3.pos` | Left joint 3 measured position | Left joint 3 requested position | degree |
| S3 / A3 | `left_joint_4.pos` | Left joint 4 measured position | Left joint 4 requested position | degree |
| S4 / A4 | `left_joint_5.pos` | Left joint 5 measured position | Left joint 5 requested position | degree |
| S5 / A5 | `left_joint_6.pos` | Left joint 6 measured position | Left joint 6 requested position | degree |
| S6 / A6 | `left_joint_7.pos` | Left joint 7 measured position | Left joint 7 requested position | degree |
| S7 / A7 | `left_gripper.pos` | Left gripper measured position | Left gripper requested position | degree |
| S8 / A8 | `right_joint_1.pos` | Right joint 1 measured position | Right joint 1 requested position | degree |
| S9 / A9 | `right_joint_2.pos` | Right joint 2 measured position | Right joint 2 requested position | degree |
| S10 / A10 | `right_joint_3.pos` | Right joint 3 measured position | Right joint 3 requested position | degree |
| S11 / A11 | `right_joint_4.pos` | Right joint 4 measured position | Right joint 4 requested position | degree |
| S12 / A12 | `right_joint_5.pos` | Right joint 5 measured position | Right joint 5 requested position | degree |
| S13 / A13 | `right_joint_6.pos` | Right joint 6 measured position | Right joint 6 requested position | degree |
| S14 / A14 | `right_joint_7.pos` | Right joint 7 measured position | Right joint 7 requested position | degree |
| S15 / A15 | `right_gripper.pos` | Right gripper measured position | Right gripper requested position | degree |

Approval: accepted C9 v1 shape/adapter representation; actual robot mapping,
recording, thresholds and release pin remain NOT_VERIFIED. The `v0.6.1` in the
identifier records inspected upstream provenance, not a validated release.
Known incompatibilities: C11's original `openarm_*` names, radian joint claims,
metre gripper claims, historical base/mast channels, and newer lift state/actions.
No row may be silently renamed, scaled, padded, truncated or reinterpreted.

## Profile B: historical C11 fixed-mast configuration, 19 channels

Identifier: `openamrobot-fixed-mast-19dof`. Indices and field strings below are
verified from software; **all physical meanings, units, data sources, frames,
limits and approval status remain UNCONFIRMED for actual training**. The table
retains the historical C11 document's intended meanings/units as claims to
review, not approved values. `S` denotes observation, `A` denotes command.

For every row: measured state source is UNCONFIRMED; command endpoints below
are historical proposals, not implemented controllers in this package. Reference
frame is UNCONFIRMED (joint-space/null was not established by an approved
19-channel recording profile). Approved observation and command bounds are
UNCONFIRMED. In particular the old document's numerical ranges are not copied
as approvals. State channels must distinguish actual measurements from fixed
structural placeholders before a recording can be accepted.

| S / A index | Software dimension field | Historical observation claim | Historical command claim / proposed source | Claimed unit, UNCONFIRMED |
| --- | --- | --- | --- | --- |
| S0 / A0 | `base_linear_x` | Base forward velocity | Forward velocity, `/cmd_vel_nav` | m/s |
| S1 / A1 | `base_linear_y` | Base lateral velocity | Lateral velocity; differential-drive zero constraint proposed | m/s |
| S2 / A2 | `base_angular_z` | Base yaw velocity | Yaw rate, `/cmd_vel_nav` | rad/s |
| S3 / A3 | `torso_pitch` | Fixed mast pitch placeholder | Structural zero placeholder; no actuator endpoint | rad |
| S4 / A4 | `torso_yaw` | Fixed mast yaw placeholder | Structural zero placeholder; no actuator endpoint | rad |
| S5 / A5 | `left_shoulder_pan` | Left shoulder-pan position | Position, proposed left trajectory controller | rad |
| S6 / A6 | `left_shoulder_lift` | Left shoulder-lift position | Position, proposed left trajectory controller | rad |
| S7 / A7 | `left_elbow_flex` | Left elbow-flex position | Position, proposed left trajectory controller | rad |
| S8 / A8 | `left_wrist_flex` | Left wrist-flex position | Position, proposed left trajectory controller | rad |
| S9 / A9 | `left_wrist_roll` | Left wrist-roll position | Position, proposed left trajectory controller | rad |
| S10 / A10 | `left_wrist_yaw` | Left wrist-yaw position | Position, proposed left trajectory controller | rad |
| S11 / A11 | `left_gripper` | Left jaw position | Jaw position, proposed left gripper controller | m |
| S12 / A12 | `right_shoulder_pan` | Right shoulder-pan position | Position, proposed right trajectory controller | rad |
| S13 / A13 | `right_shoulder_lift` | Right shoulder-lift position | Position, proposed right trajectory controller | rad |
| S14 / A14 | `right_elbow_flex` | Right elbow-flex position | Position, proposed right trajectory controller | rad |
| S15 / A15 | `right_wrist_flex` | Right wrist-flex position | Position, proposed right trajectory controller | rad |
| S16 / A16 | `right_wrist_roll` | Right wrist-roll position | Position, proposed right trajectory controller | rad |
| S17 / A17 | `right_wrist_yaw` | Right wrist-yaw position | Position, proposed right trajectory controller | rad |
| S18 / A18 | `right_gripper` | Right jaw position | Jaw position, proposed right gripper controller | m |

The software has **six arm joints + one gripper per side**, after three base
and two structural channels: 3 + 2 + 12 + 2 = 19. The old description's claim
of two seven-joint arms plus grippers is arithmetically incompatible with this
list. Removing five base/mast entries leaves fourteen, not C9's sixteen. No
one-to-one joint mapping is asserted. C11 owner plus robot-model/recording owners
must resolve exact joint identities, measured/commanded sources, units, frames
and bounds against an immutable combined-model configuration before real use.

The historical zero-limit torso entries are unactuated placeholders, not lift
height or lift commands. Source code currently implements an offline model,
not runtime physical constraints on its outputs. Its outputs must not be sent
to hardware as an accepted controller.

## Lift-equipped combined model: separate, no C11 action profile

The inspected combined model declares `lift_joint` as a prismatic joint,
`lift_column_base_link` to `lift_carriage_link`, with travel in metres. Source
records 0..0.350 m travel and shoulder height 1.000 m + lift position. These are
model/register facts, **not accepted C11 action bounds**. Its real hardware
profile exposes no lift command interface; mock/Gazebo profiles are separate.

Lift observation: log measured `lift_joint` position with timestamp, origin,
unit, calibration evidence and configuration hash where relevant. The recorder
owner must approve its exact storage/binding; neither existing profile has a
lift observation index. **Do not append a dimension or invent a C9 v1 field.**
An agreed companion artifact/schema change is required if current provenance
artifacts cannot carry the needed binding. `L1000`, `L1175`, `L1350` identify
calibrated shoulder-height stops, not commanded channels or configuration IDs.

Lift commanded action: **none authorized by this review**. No inferred action
index, padding, lift/torso renaming, or acceptance of the full combined robot.
The C11, recording and combined-model owners must agree the observation mapping;
software lead reviews it separately from real-recording/production acceptance.

## Executable guards

`training/config.py:87` validates exact names/order, profile widths and the four
camera keys. `training/dataset.py:198` checks native names/shapes, C9 sidecar
profile and degree units for the adapter profile. Tests in
`python/tests/test_c11_training.py` compare the ordered names against the
hash-pinned C9 example, exercise 16/19 model shapes, reject dimension mixes and
verify no lift channel is added. These are software checks, not robot acceptance.
