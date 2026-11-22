# C11 ACT Policy Training Engine

The `oa_embodied_ai.training` package implements Deliverable C11 (Action Chunking with Transformers Imitation Learning Baseline) for the OpenAMRobot 2.0 whole-body platform.

## CLI Entry Point: `oa-c11-train`

Execute training via console script or module:

```bash
# Rapid CPU smoke test (2 epochs, minimal batches)
oa-c11-train --dataset-path path/to/dataset --smoke-test --allow-synthetic-eval

# Full training run with custom config and output directory
oa-c11-train --dataset-path path/to/dataset --config configs/c11_act_default.yaml --output-dir outputs/c11_run --epochs 50 --device cuda
```

### CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--dataset-path` | `Path` | *Required* | Path to the `LeRobotDataset v3.0` directory tree. |
| `--config` | `Path` | `configs/c11_act_default.yaml` | Path to YAML configuration file specifying kinematic profile and hyperparameters. |
| `--qa-report` | `Path` | `None` (auto-detected) | Path to pre-flight C10 QA report JSON. If omitted, checks standard report paths or runs `check_dataset`. |
| `--output-dir` | `Path` | `outputs/c11_training` | Destination directory for checkpoints, hashes, and run summaries. |
| `--smoke-test` | `bool` | `False` | Run rapid 2-epoch CPU verification loop without training full epochs. |
| `--allow-synthetic-eval` | `bool` | `False` | Permit synthetic or Gazebo data override for CI dry-runs. Stamps output `NOT_VERIFIED_PHYSICAL`. |
| `--epochs` | `int` | Config default (`50`) | Total number of training epochs. |
| `--batch-size` | `int` | Config default (`8`) | Batch size for train and evaluation loaders. |
| `--device` | `str` | `"cpu"` / `"cuda"` | PyTorch device string. |
| `--seed` | `int` | Config default (`42`) | Deterministic random seed for episode partitioning and model weights. |

---

## Fail-Closed Training Gate Invariant

Per OpenAMRobot 2.0 Working Rules Section 10 and C10 handoff requirements, training **must fail closed** on unverified, synthetic, or contradictory data:

1. **Explicit Readiness**: `training_ready` must be explicitly `true`. If `false` or missing, the runner raises `TrainingBlockedError` and terminates with **exit code 2**.
2. **Quality Status**: `quality_status` / `status` must not be `BLOCKED`, `REJECT`, or `NOT_VERIFIED`.
3. **Physical Origin Requirement**: If `source_profile != "ROBOT"`, training aborts immediately with exit code 2 unless `--allow-synthetic-eval` is explicitly passed. When overridden, logs `WARNING: Synthetic data override active. Artifacts will be stamped NOT_VERIFIED_PHYSICAL.`
4. **Contradiction Invariant**: If the C10 QA report contains *any* finding with `severity == "REJECT"` or `status == "REJECT"`, the gate is blocked regardless of top-level boolean fields.

---

## Model Architecture

The vision-based ACT policy integrates:
- **Multi-Camera Visual Backbone**: Ingests all 4 canonical C9 camera streams (`observation.images.head`, `observation.images.base`, `observation.images.camera_wrist_left`, `observation.images.camera_wrist_right`). Applies 224×224 resize and ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`), then extracts feature tokens via a shared ResNet-18 backbone projected to dimension `d_model` (256).
- **CVAE Style Encoder**: Conditions on a learnable `[CLS]` token, current proprioceptive state, and target future action chunk ($K=100$). Uses a 4-layer transformer encoder to parameterize latent style distribution $\mathcal{N}(\mu, \sigma^2)$ where $D_{\text{latent}} = 32$. During inference, $z$ is clamped to zero.
- **Transformer Decoder**: Combines $K=100$ learned position queries with conditioning memory ($z$, state token, 4 camera tokens) through a 4-layer transformer decoder with 8 attention heads to predict continuous multi-step trajectories ($100 \times \text{action\_dim}$).
- **Loss Function**: $\mathcal{L}_{\text{total}} = \text{L1}(\hat{a}, a) \cdot \lambda_{\text{L1}} + D_{\text{KL}}(\mathcal{N}(\mu, \sigma^2) \parallel \mathcal{N}(0, I)) \cdot \lambda_{\text{KL}}$ ($\lambda_{\text{L1}} = 1.0, \lambda_{\text{KL}} = 10.0$).

---

## Supported Kinematic Profiles

| Profile Name | Channels | State/Action Dim | Subsystems Covered |
|---|---|---|---|
| `openamrobot-fixed-mast-19dof` | 19 | 19 | 3 base (`base_linear_x`, `base_linear_y`, `base_angular_z`) + 2 fixed mast (`torso_pitch`, `torso_yaw` locked `[0, 0]`) + 7 left arm/gripper + 7 right arm/gripper |
| `lerobot-v0.6.1-bi-openarm-position` | 16 | 16 | 8 left arm/gripper (`openarm_left_joint1..7` + gripper) + 8 right arm/gripper |

---

## Artifact Outputs

Training emits:
- `act_policy_best.pt`: PyTorch weights and configuration payload.
- `act_policy_best.pt.sha256`: Cryptographic SHA-256 hash of the checkpoint.
- `c11_training_summary.json`: Provenance metadata recording exact train/validation episode partitions, gate status, training/validation losses, git commit, and environment versions.
