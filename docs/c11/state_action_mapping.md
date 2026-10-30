# OpenAMRobot 2.0 — Deliverable C11: State & Action Mapping Specification

- **Project**: OpenAMRobot 2.0 | Workstream C (Embodied AI & Imitation Learning)
- **Deliverable**: C11 ACT Policy Baseline
- **Scope**: Explicit mapping of observation spaces, action command channels, kinematic profiles, and hardware invariants implemented in Deliverable C11 (`oa_embodied_ai/training/` and `oa_embodied_ai/evaluation/`).
- **Contributor**: Marcus Andrei Guinto (`@4u-79`, `mavguinto@gmail.com`), Embodied AI & Data (Division B)
- **References**: P-03 Addendum rev18.2, Deliverable C9 Contract, Deliverable C10 Gate Handoff, PR #4, Mohamed Sayed review directive (`openamrobot-interfaces#18`).

---

## 1. Profile Separation Invariant (Zero Silent Slicing or Zero-Padding)

To ensure strict safety and dataset integrity across disparate robotic fixtures, the Deliverable C11 codebase enforces complete profile separation via configuration schema (`configs/c11_act_default.yaml`) and strict tensor shape contracts:

1. **`openamrobot-fixed-mast-19dof`** (19-Channel Whole-Body Baseline):
   - Whole-body platform combining differential mobile base velocity commands, fixed structural mast joints, and bilateral 7-DoF manipulation arms with parallel jaw grippers.
   - State & action dimension: $D_{\text{state}} = 19$, $D_{\text{action}} = 19$.
2. **`lerobot-v0.6.1-bi-openarm-position`** (16-Channel Bimanual Arm Baseline):
   - Benchtop dual-arm fixture conforming to upstream C9 benchmark contracts.
   - State & action dimension: $D_{\text{state}} = 16$, $D_{\text{action}} = 16$.

### Strict Architectural Contract: No Runtime Slicing or Padding
- **Prohibition**: Runtime slicing (e.g., dynamically dropping base/mast channels from 19-DoF tensors to feed a 16-DoF model) and zero-padding (e.g., padding 16-DoF arm actions with zeros to publish to a 19-DoF robot) are **strictly forbidden**.
- **Enforcement**: Profile switching is explicitly governed by configuration schema and validated at dataloader initialization (`MultiCameraACTDataset`) and model construction (`ACTPolicy`). Any mismatch between dataset column counts and policy channel dimensions raises an unrecoverable shape error at initialization.

---

## 2. Channel Breakdown Tables

### Table 1: Profile B (`openamrobot-fixed-mast-19dof`, 19-Channel Whole-Body Baseline)

Governed by schema tag `schema-v1.0-frozen` (P-03 Addendum rev18.2):

| Ch # | Channel Name | Hardware Subsystem | Control Endpoint / Topic | Physical Limits | Unit | Safety & Hardware Invariant |
|:---:|---|---|---|:---:|:---:|---|
| **00** | `base_linear_x` | Diff-Drive Base | `/cmd_vel_nav` (`geometry_msgs/Twist`) | `[-0.20, +0.20]` | m/s | Arbitrated via `twist_mux`, velocity smoother ($\le 0.5\text{ m/s}^2$), collision monitor. |
| **01** | `base_linear_y` | Diff-Drive Base | `/cmd_vel_nav` (`geometry_msgs/Twist`) | `[0.00, 0.00]` | m/s | Non-holonomic constraint; must strictly remain `0.0`. |
| **02** | `base_angular_z` | Diff-Drive Base | `/cmd_vel_nav` (`geometry_msgs/Twist`) | `[-0.50, +0.50]` | rad/s | Angular yaw rate arbitrated through velocity smoother ($\le 1.0\text{ rad/s}^2$). |
| **03** | `torso_pitch` | Structural Mast | Fixed Structural Link | `[0.00, 0.00]` | rad | Fixed mast datum; unactuated structural link locked at `0.0` rad. |
| **04** | `torso_yaw` | Structural Mast | Fixed Structural Link | `[0.00, 0.00]` | rad | Fixed mast datum; unactuated structural link locked at `0.0` rad. |
| **05** | `left_shoulder_pan` | OpenArm 2.0 Left | `/left_arm_controller/follow_joint_trajectory` | `[-2.618, +2.618]` | rad | Joint 1 (Shoulder Pan). `FollowJointTrajectory` action interface. |
| **06** | `left_shoulder_lift`| OpenArm 2.0 Left | `/left_arm_controller/follow_joint_trajectory` | `[-1.745, +1.745]` | rad | Joint 2 (Shoulder Lift). |
| **07** | `left_elbow_flex` | OpenArm 2.0 Left | `/left_arm_controller/follow_joint_trajectory` | `[-2.618, +2.618]` | rad | Joint 3 (Elbow Flex). |
| **08** | `left_wrist_flex` | OpenArm 2.0 Left | `/left_arm_controller/follow_joint_trajectory` | `[-1.745, +1.745]` | rad | Joint 4 (Wrist Flex). |
| **09** | `left_wrist_roll` | OpenArm 2.0 Left | `/left_arm_controller/follow_joint_trajectory` | `[-2.618, +2.618]` | rad | Joint 5 (Wrist Roll). |
| **10** | `left_wrist_yaw` | OpenArm 2.0 Left | `/left_arm_controller/follow_joint_trajectory` | `[-1.745, +1.745]` | rad | Joint 6 (Wrist Yaw). |
| **11** | `left_gripper` | Left Gripper | `/left_gripper_controller/gripper_cmd` | `[0.000, +0.080]` | m | Parallel jaw gripper stroke / position command. |
| **12** | `right_shoulder_pan` | OpenArm 2.0 Right| `/right_arm_controller/follow_joint_trajectory` | `[-2.618, +2.618]` | rad | Joint 1 (Shoulder Pan). `FollowJointTrajectory` action interface. |
| **13** | `right_shoulder_lift`| OpenArm 2.0 Right| `/right_arm_controller/follow_joint_trajectory` | `[-1.745, +1.745]` | rad | Joint 2 (Shoulder Lift). |
| **14** | `right_elbow_flex` | OpenArm 2.0 Right| `/right_arm_controller/follow_joint_trajectory` | `[-2.618, +2.618]` | rad | Joint 3 (Elbow Flex). |
| **15** | `right_wrist_flex` | OpenArm 2.0 Right| `/right_arm_controller/follow_joint_trajectory` | `[-1.745, +1.745]` | rad | Joint 4 (Wrist Flex). |
| **16** | `right_wrist_roll` | OpenArm 2.0 Right| `/right_arm_controller/follow_joint_trajectory` | `[-2.618, +2.618]` | rad | Joint 5 (Wrist Roll). |
| **17** | `right_wrist_yaw` | OpenArm 2.0 Right| `/right_arm_controller/follow_joint_trajectory` | `[-1.745, +1.745]` | rad | Joint 6 (Wrist Yaw). |
| **18** | `right_gripper` | Right Gripper | `/right_gripper_controller/gripper_cmd` | `[0.000, +0.080]` | m | Parallel jaw gripper stroke / position command. |

---

### Table 2: Profile A (`lerobot-v0.6.1-bi-openarm-position`, 16-Channel Bimanual Arm Baseline)

Governed by C9 contract benchmark fixture:

| Ch # | Channel Name | Hardware Subsystem | Control Interface | Range / Limits | Unit |
|:---:|---|---|---|:---:|:---:|
| **00** | `openarm_left_joint1` | Left Arm J1 | `/left_arm_controller/follow_joint_trajectory` | `[-2.618, +2.618]` | rad |
| **01** | `openarm_left_joint2` | Left Arm J2 | `/left_arm_controller/follow_joint_trajectory` | `[-1.745, +1.745]` | rad |
| **02** | `openarm_left_joint3` | Left Arm J3 | `/left_arm_controller/follow_joint_trajectory` | `[-2.618, +2.618]` | rad |
| **03** | `openarm_left_joint4` | Left Arm J4 | `/left_arm_controller/follow_joint_trajectory` | `[-1.745, +1.745]` | rad |
| **04** | `openarm_left_joint5` | Left Arm J5 | `/left_arm_controller/follow_joint_trajectory` | `[-2.618, +2.618]` | rad |
| **05** | `openarm_left_joint6` | Left Arm J6 | `/left_arm_controller/follow_joint_trajectory` | `[-1.745, +1.745]` | rad |
| **06** | `openarm_left_joint7` | Left Arm J7 | `/left_arm_controller/follow_joint_trajectory` | `[-2.618, +2.618]` | rad |
| **07** | `openarm_left_gripper` | Left Gripper | `/left_gripper_controller/gripper_cmd` | `[0.000, +0.080]` | m |
| **08** | `openarm_right_joint1` | Right Arm J1 | `/right_arm_controller/follow_joint_trajectory` | `[-2.618, +2.618]` | rad |
| **09** | `openarm_right_joint2` | Right Arm J2 | `/right_arm_controller/follow_joint_trajectory` | `[-1.745, +1.745]` | rad |
| **10** | `openarm_right_joint3` | Right Arm J3 | `/right_arm_controller/follow_joint_trajectory` | `[-2.618, +2.618]` | rad |
| **11** | `openarm_right_joint4` | Right Arm J4 | `/right_arm_controller/follow_joint_trajectory` | `[-1.745, +1.745]` | rad |
| **12** | `openarm_right_joint5` | Right Arm J5 | `/right_arm_controller/follow_joint_trajectory` | `[-2.618, +2.618]` | rad |
| **13** | `openarm_right_joint6` | Right Arm J6 | `/right_arm_controller/follow_joint_trajectory` | `[-1.745, +1.745]` | rad |
| **14** | `openarm_right_joint7` | Right Arm J7 | `/right_arm_controller/follow_joint_trajectory` | `[-2.618, +2.618]` | rad |
| **15** | `openarm_right_gripper` | Right Gripper | `/right_gripper_controller/gripper_cmd` | `[0.000, +0.080]` | m |

---

## 3. DOLD Hexalift Invariant (Mohamed Sayed Review Directive)

In accordance with Division B Lead Mohamed Sayed's architectural directive on `openamrobot-interfaces#18`:

1. **Non-Actuated Action Space**:
   - The DOLD Hexalift V4 350 mm telescopic lift column (`lift_joint`) is **NOT** part of the policy's actuated action output space.
   - The ACT policy's predicted action chunk ($K=100$) never directly commands upper body carriage height or lift displacement.
2. **Observation Metadata Logging Only**:
   - Lift state and calibration height stops (`L1000` = 1.000 m, `L1175` = 1.175 m, `L1350` = 1.350 m datum) are captured strictly as contextual observation metadata in episode sidecars.
3. **Supervisory Transition Model**:
   - Carriage height changes occur only during pre-task setup or supervisory state machine transitions; autonomous whole-body imitation policies execute with the lift locked at a designated stop datum.

---

## 4. C9 Visual Conditioning Streams

Deliverable C11 models are conditioned on the 4 canonical camera streams designated in the C9 dataset contract:

| Canonical C9 Key | Sensor Hardware | Resolution ($H \times W$) | Role / Viewpoint | Frame Rate |
|---|---|:---:|---|:---:|
| `observation.images.head` | Stereo ZED-121210 | 720 × 1280 | Scene context & manipulation station view (mounted on upperbody lift carriage) | 30 Hz |
| `observation.images.base` | Orbbec Gemini 336L | 480 × 640 | Low-angle obstacle detection & forward navigation floor view | 30 Hz |
| `observation.images.camera_wrist_left` | RGB Micro Camera | 480 × 640 | Left end-effector close-up visual servoing & grasp alignment | 30 Hz |
| `observation.images.camera_wrist_right` | RGB Micro Camera | 480 × 640 | Right end-effector close-up visual servoing & grasp alignment | 30 Hz |

- **Preprocessing**: All camera streams are resized to $224 \times 224$ and normalized using standard ImageNet parameters (`mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`).
- **Feature Extraction**: Features are encoded through a shared ResNet-18 visual backbone and projected to $D_{\text{model}} = 256$ memory tokens.

---

## 5. Verification & Acceptance Status

Deliverable C11 strictly decouples trajectory imitation loss from physical closed-loop task execution:
- **Offline Trajectory Loss**: Evaluates open-loop multi-step prediction error (L1/MAE, MSE, chunk boundary discontinuity error) using a 15-line temporal ensembling buffer ($\lambda = 0.01$). Offline error reflects policy convergence, **not** real-world rollout competence.
- **Closed-Loop Rollout Metrics**: Evaluates physical or simulated execution tracking discrete phases across the canonical C13 taxonomy (phases 0..7: `IDLE`, `APPROACH`, `ALIGN`, `GRASP`, `TRANSPORT`, `PLACE`, `RETRACT`, `FAILURE_RECOVERY`), with Wilson 95% binomial confidence intervals on physical task success.
- **Mandatory Synthetic Marking**: Evaluated runs generated with synthetic, Gazebo, or test-fixture data are unconditionally stamped with:
  ```json
  "acceptance_status": "NOT_VERIFIED_PHYSICAL"
  ```
  Artifacts receive `"VERIFIED_PHYSICAL"` exclusively when real physical robot demonstration bags (`source_profile == "ROBOT"`) pass all C10 pre-flight quality verification gates without synthetic overrides.
