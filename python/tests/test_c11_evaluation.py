# SPDX-License-Identifier: MIT
"""Unit tests for Deliverable C11 Disambiguated Evaluation Engine."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pytest

torch = pytest.importorskip("torch", reason="PyTorch required for C11 evaluation tests")
pyarrow = pytest.importorskip("pyarrow", reason="PyArrow required for parquet dataset tests")
scipy = pytest.importorskip("scipy", reason="SciPy required for evaluation tests")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from oa_embodied_ai.evaluation.offline_eval import (
    compute_chunk_discontinuity,
    compute_offline_metrics,
    temporal_ensemble_predict,
)
from oa_embodied_ai.evaluation.rollout_eval import (
    SUBTASK_PHASES,
    ClosedLoopRolloutEvaluator,
    compute_wilson_ci,
    run_rollout_eval,
)
from oa_embodied_ai.evaluation.cli import evaluate_c11
from oa_embodied_ai.training.config import get_profile_config
from oa_embodied_ai.training.model import ACTPolicy


class MockPolicy(torch.nn.Module):
    """Mock policy for evaluation unit tests."""

    def __init__(self, action_dim: int = 19, chunk_size: int = 10) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    def forward(
        self,
        state: torch.Tensor,
        images: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None, None]:
        B = state.shape[0]
        # Predict constant action repeated across chunk
        pred = state.unsqueeze(1).expand(B, self.chunk_size, self.action_dim)
        return pred, None, None


def create_eval_dataset(root: Path, num_episodes: int = 2, num_steps: int = 20) -> Path:
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "codebase_version": "v3.0",
        "robot_type": "openamrobot_2",
        "fps": 30,
        "profile": "openamrobot-fixed-mast-19dof",
        "schema_revision": "schema-v1.0-frozen",
    }
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")

    for ep_idx in range(num_episodes):
        states = np.zeros((num_steps, 19), dtype=np.float32)
        actions = np.zeros((num_steps, 19), dtype=np.float32)
        states[:, 5:19] = 0.05
        actions[:, 5:19] = 0.05

        table = pa.Table.from_arrays(
            [
                pa.array(states.tolist(), type=pa.list_(pa.float32(), 19)),
                pa.array(actions.tolist(), type=pa.list_(pa.float32(), 19)),
                pa.array(list(range(num_steps)), type=pa.int64()),
            ],
            names=["observation.state", "action", "frame_index"],
        )
        pq.write_table(table, data_dir / f"episode_{ep_idx:06d}.parquet")

    return root


class TestC11Evaluation(unittest.TestCase):
    def test_compute_offline_metrics(self):
        gt = np.ones((10, 19), dtype=np.float32)
        pred = np.zeros((10, 19), dtype=np.float32)
        channels = [f"ch_{i}" for i in range(19)]

        metrics = compute_offline_metrics(gt, pred, channel_names=channels, chunk_discontinuity=0.05)
        self.assertEqual(metrics["overall_mae"], 1.0)
        self.assertEqual(metrics["overall_mse"], 1.0)
        self.assertEqual(metrics["chunk_boundary_discontinuity"], 0.05)
        self.assertIn("ch_0", metrics["per_channel"])
        self.assertEqual(metrics["per_channel"]["ch_0"]["mae"], 1.0)
        self.assertEqual(metrics["per_channel"]["ch_0"]["mse"], 1.0)

    def test_temporal_ensemble_predict_buffer(self):
        model = MockPolicy(action_dim=19, chunk_size=10)
        states = np.random.randn(30, 19).astype(np.float32)

        ensembled = temporal_ensemble_predict(
            model=model,
            state_sequence=states,
            chunk_size=10,
            ensemble_lambda=0.01,
        )
        self.assertEqual(ensembled.shape, (30, 19))
        self.assertFalse(np.isnan(ensembled).any())

    def test_wilson_confidence_interval(self):
        # k=0, n=20
        low, high = compute_wilson_ci(0, 20, confidence=0.95)
        self.assertEqual(low, 0.0)
        self.assertTrue(0.15 < high < 0.20)

        # k=20, n=20
        low, high = compute_wilson_ci(20, 20, confidence=0.95)
        self.assertTrue(0.80 < low < 0.85)
        self.assertEqual(high, 1.0)

        # k=10, n=20 (50%)
        low, high = compute_wilson_ci(10, 20, confidence=0.95)
        self.assertTrue(0.25 < low < 0.35)
        self.assertTrue(0.65 < high < 0.75)
        self.assertAlmostEqual(0.5 - low, high - 0.5, places=2)

        # Edge case: n=0
        low_zero, high_zero = compute_wilson_ci(0, 0)
        self.assertEqual((low_zero, high_zero), (0.0, 0.0))

    def test_subtask_phases_completeness(self):
        expected_phases = {
            0: "IDLE",
            1: "APPROACH",
            2: "ALIGN",
            3: "GRASP",
            4: "TRANSPORT",
            5: "PLACE",
            6: "RETRACT",
            7: "FAILURE_RECOVERY",
        }
        self.assertEqual(SUBTASK_PHASES, expected_phases)

    def test_rollout_eval_phase_attribution(self):
        policy = MockPolicy()
        eval_result = run_rollout_eval(policy=policy, num_episodes=5)

        self.assertIn("success_rate", eval_result)
        self.assertIn("wilson_ci_95", eval_result)
        self.assertEqual(len(eval_result["wilson_ci_95"]), 2)
        self.assertEqual(eval_result["total_rollouts"], 5)
        self.assertIn("phase_attribution", eval_result)

        for phase_name in SUBTASK_PHASES.values():
            self.assertIn(phase_name, eval_result["phase_attribution"])

    def test_evaluation_cli_synthetic_acceptance_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_dir = create_eval_dataset(root / "dataset")
            qa_report_path = root / "qa_report.json"
            qa_report_path.write_text(
                json.dumps(
                    {
                        "report_version": 1,
                        "status": "PASS",
                        "quality_status": "PASS",
                        "exit_code": 0,
                        "training_ready": True,
                        "source_profile": "SYNTHETIC",
                        "findings": [{"check_id": "C10_SCHEMA", "status": "PASS", "severity": "INFO"}],
                        "episodes": [{"episode_index": 0, "status": "PASS", "data_origin": "synthetic"}],
                    }
                ),
                encoding="utf-8",
            )

            # Create mock checkpoint
            cfg = get_profile_config("openamrobot-fixed-mast-19dof")
            cfg.chunk_size = 10
            cfg.d_model = 64
            cfg.enc_layers = 1
            cfg.dec_layers = 1
            cfg.dim_feedforward = 128
            cfg.n_heads = 4

            policy = ACTPolicy(cfg)
            ckpt_path = root / "checkpoint.pt"
            torch.save(
                {
                    "model_state_dict": policy.state_dict(),
                    "config": cfg.to_dict(),
                    "val_indices": [0],
                },
                ckpt_path,
            )

            report_out = root / "eval_report.json"
            report = evaluate_c11(
                checkpoint_path=ckpt_path,
                dataset_path=dataset_dir,
                qa_report_path=qa_report_path,
                mode="offline",
                output_report=report_out,
                allow_synthetic_eval=True,
            )

            self.assertEqual(report["evaluation_type"], "OFFLINE_TRAJECTORY")
            self.assertEqual(report["acceptance_status"], "NOT_VERIFIED_PHYSICAL")
            self.assertIn("provenance", report)
            self.assertIn("checkpoint_sha256", report["provenance"])
            self.assertIn("dataset_sha256", report["provenance"])
            self.assertIn("offline_metrics", report)
            self.assertIsNone(report["rollout_metrics"])

    def test_evaluation_cli_robot_acceptance_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_dir = create_eval_dataset(root / "dataset")
            qa_report_path = root / "qa_report.json"
            qa_report_path.write_text(
                json.dumps(
                    {
                        "report_version": 1,
                        "status": "PASS",
                        "quality_status": "PASS",
                        "exit_code": 0,
                        "training_ready": True,
                        "source_profile": "ROBOT",
                        "provenance": {"source_profile": "ROBOT"},
                        "findings": [{"check_id": "C10_SCHEMA", "status": "PASS", "severity": "INFO"}],
                        "episodes": [{"episode_index": 0, "status": "PASS", "data_origin": "actual"}],
                    }
                ),
                encoding="utf-8",
            )

            cfg = get_profile_config("openamrobot-fixed-mast-19dof")
            cfg.chunk_size = 10
            cfg.d_model = 64
            cfg.enc_layers = 1
            cfg.dec_layers = 1
            cfg.dim_feedforward = 128
            cfg.n_heads = 4

            policy = ACTPolicy(cfg)
            ckpt_path = root / "checkpoint.pt"
            torch.save(
                {
                    "model_state_dict": policy.state_dict(),
                    "config": cfg.to_dict(),
                    "val_indices": [0],
                },
                ckpt_path,
            )

            report_out = root / "eval_report.json"
            report = evaluate_c11(
                checkpoint_path=ckpt_path,
                dataset_path=dataset_dir,
                qa_report_path=qa_report_path,
                mode="both",
                output_report=report_out,
                allow_synthetic_eval=False,
            )

            self.assertEqual(report["evaluation_type"], "COMBINED_OFFLINE_AND_ROLLOUT")
            self.assertEqual(report["acceptance_status"], "VERIFIED_PHYSICAL")
            self.assertIsNotNone(report["offline_metrics"])
            self.assertIsNotNone(report["rollout_metrics"])
            self.assertIn("wilson_ci_95", report["rollout_metrics"])


if __name__ == "__main__":
    unittest.main()

