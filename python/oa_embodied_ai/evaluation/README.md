# C11 Disambiguated Evaluation Engine

The `oa_embodied_ai.evaluation` package implements evaluation tooling for Deliverable C11 (and forward compatibility with C14), strictly resolving C10 handoff Finding #3: **Metric Disambiguation**.

## Overview: Disambiguation Contract

Offline imitation error and closed-loop rollout success represent fundamentally distinct evaluation domains:
- **Offline Trajectory Imitation**: Evaluates open-loop multi-step prediction error on held-out dataset trajectories. A low MAE/MSE indicates policy fitting, not physical task success.
- **Closed-Loop Rollout**: Evaluates interactive closed-loop execution against physical or simulated controllers, monitoring `/cmd_vel_nav` smoother constraints, joint limits, discrete phase attribution, and physical goal achievement.
- **Acceptance Rule**: Datasets generated with synthetic, Gazebo, or dry-run overrides are strictly stamped `acceptance_status: "NOT_VERIFIED_PHYSICAL"`. Only physical bags with `source_profile == "ROBOT"` that pass the pre-flight gate can receive `"VERIFIED_PHYSICAL"`.

---

## CLI Entry Point: `oa-c11-eval`

```bash
# Offline trajectory evaluation on validation episodes
oa-c11-eval --checkpoint checkpoints/act_policy_best.pt --dataset-path data/episodes --mode offline --output-report reports/c11_eval_report.json

# Closed-loop rollout evaluation in simulation
oa-c11-eval --checkpoint checkpoints/act_policy_best.pt --dataset-path data/episodes --mode rollout --allow-synthetic-eval

# Combined evaluation (both offline and closed-loop)
oa-c11-eval --checkpoint checkpoints/act_policy_best.pt --dataset-path data/episodes --mode both
```

### CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--checkpoint` | `Path` | *Required* | Path to `.pt` policy checkpoint. |
| `--dataset-path` | `Path` | *Required* | Path to the evaluation dataset directory. |
| `--qa-report` | `Path` | `None` (auto-detected) | Path to C10 QA report JSON. |
| `--mode` | `str` | `"offline"` | Evaluation mode: `"offline"`, `"rollout"`, or `"both"`. |
| `--output-report` | `Path` | `reports/c11_eval_report.json` | Destination report file adhering to the evaluation JSON schema. |
| `--allow-synthetic-eval` | `bool` | `False` | Permit synthetic evaluation. Stamps output `NOT_VERIFIED_PHYSICAL`. |
| `--device` | `str` | `"cpu"` | Evaluation compute device (`"cpu"` / `"cuda"`). |
| `--ensemble-lambda` | `float` | `0.01` | Temporal ensembling decay parameter $\lambda$. |

---

## Evaluation Metrics

### 1. Offline Trajectory Imitation
- **Per-Channel & Overall L1 / MAE & MSE**: Evaluates trajectory deviation across all active channels (radians for joints, m/s for base linear velocity, rad/s for base angular velocity).
- **Chunk Boundary Discontinuity Error**: Computes inter-chunk step boundary discontinuity between chunk $k$ step 99 and chunk $k+1$ step 0 ($|\hat{a}^{(k)}_{99} - \hat{a}^{(k+1)}_{0}|$).
- **Temporal Ensembling Buffer**: Implements exponential-decay temporal ensembling across overlapping ACT chunks with weighting $w_k = \exp(-\lambda k)$ ($\lambda = 0.01$).

### 2. Closed-Loop Rollout
- **Physical Success Rate**: Proportion of completed trials achieving target tolerance without exceeding timeout or triggering limits.
- **Wilson 95% Confidence Intervals**: Binomial score confidence intervals calculated over trial success counts ($k / n$).
- **Discrete Subtask Phase Attribution (Canonical C13 Taxonomy)**:
  - `0: IDLE` — Initial standby state.
  - `1: APPROACH` — Mobile base navigation towards manipulation station.
  - `2: ALIGN` — Upper body and dual-arm alignment.
  - `3: GRASP` — Gripper closure on target object.
  - `4: TRANSPORT` — Dual-arm trajectory transfer with grasped payload.
  - `5: PLACE` — Alignment and release of object at target location.
  - `6: RETRACT` — End-effector withdrawal to home pose.
  - `7: FAILURE_RECOVERY` — Safety deceleration or recovery routine.

---

## Strict Report Schema (`reports/c11_eval_report.json`)

```json
{
  "evaluation_type": "OFFLINE_TRAJECTORY",
  "acceptance_status": "NOT_VERIFIED_PHYSICAL",
  "provenance": {
    "checkpoint_sha256": "...",
    "dataset_sha256": "...",
    "git_commit": "...",
    "python_version": "3.12.x",
    "torch_version": "2.x.x",
    "lerobot_version": "0.4.1"
  },
  "offline_metrics": {
    "overall_mae": 0.042,
    "overall_mse": 0.0035,
    "chunk_boundary_discontinuity": 0.018,
    "per_channel": { ... }
  },
  "rollout_metrics": null
}
```
