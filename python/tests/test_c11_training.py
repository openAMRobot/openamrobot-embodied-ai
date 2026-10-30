# SPDX-License-Identifier: MIT
"""Unit tests for Deliverable C11 ACT Policy Training Engine."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pytest

torch = pytest.importorskip("torch", reason="PyTorch required for C11 training tests")
torchvision = pytest.importorskip("torchvision", reason="Torchvision required for C11 dataset tests")
pyarrow = pytest.importorskip("pyarrow", reason="PyArrow required for parquet dataset tests")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from oa_embodied_ai.training.config import (
    ACTConfig,
    CANONICAL_CAMERA_KEYS,
    OPENAMROBOT_19DOF_CHANNELS,
    LEROBOT_16DOF_CHANNELS,
    get_profile_config,
    load_config,
    save_config,
)
from oa_embodied_ai.training.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ImagePreprocessor,
    MultiCameraACTDataset,
    find_episode_parquet_files,
    partition_episodes,
)
from oa_embodied_ai.training.model import ACTPolicy
from oa_embodied_ai.training.cli import train_act


def create_dummy_dataset(root: Path, num_episodes: int = 3, num_steps: int = 20) -> Path:
    """Create a minimal mock LeRobotDataset v3.0 directory for tests."""
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
        # Small non-zero variations
        states[:, 5:19] = np.sin(np.linspace(0, 1, num_steps)[:, None])
        actions[:, 5:19] = np.cos(np.linspace(0, 1, num_steps)[:, None])

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


class TestC11Training(unittest.TestCase):
    def test_kinematic_profiles(self):
        cfg19 = get_profile_config("openamrobot-fixed-mast-19dof")
        self.assertEqual(cfg19.state_dim, 19)
        self.assertEqual(cfg19.action_dim, 19)
        self.assertEqual(len(cfg19.channel_names), 19)
        self.assertEqual(cfg19.channel_names, OPENAMROBOT_19DOF_CHANNELS)

        cfg16 = get_profile_config("lerobot-v0.6.1-bi-openarm-position")
        self.assertEqual(cfg16.state_dim, 16)
        self.assertEqual(cfg16.action_dim, 16)
        self.assertEqual(len(cfg16.channel_names), 16)
        self.assertEqual(cfg16.channel_names, LEROBOT_16DOF_CHANNELS)

        with self.assertRaises(ValueError):
            get_profile_config("unsupported-profile-42dof")

    def test_config_yaml_serialization(self):
        cfg = get_profile_config("openamrobot-fixed-mast-19dof")
        cfg.epochs = 15
        cfg.d_model = 128
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test_cfg.yaml"
            save_config(cfg, yaml_path)
            loaded = load_config(yaml_path)
            self.assertEqual(loaded.profile, cfg.profile)
            self.assertEqual(loaded.epochs, 15)
            self.assertEqual(loaded.d_model, 128)
            self.assertEqual(loaded.state_dim, 19)

    def test_image_preprocessor_normalization(self):
        preproc = ImagePreprocessor(image_size=(224, 224), normalize=True)
        ones_img = torch.ones((3, 224, 224), dtype=torch.float32)
        out = preproc(ones_img)
        self.assertEqual(out.shape, (3, 224, 224))

        # Check normalization matches ImageNet constants
        expected_ch0 = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        expected_ch1 = (1.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
        expected_ch2 = (1.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2]

        self.assertAlmostEqual(out[0, 0, 0].item(), expected_ch0, places=3)
        self.assertAlmostEqual(out[1, 0, 0].item(), expected_ch1, places=3)
        self.assertAlmostEqual(out[2, 0, 0].item(), expected_ch2, places=3)

    def test_mutation_reversion_image_normalization_contract(self):
        """Mutation test: Deleting or disabling normalization must fail the contract."""
        preproc_norm = ImagePreprocessor(image_size=(224, 224), normalize=True)
        preproc_unnorm = ImagePreprocessor(image_size=(224, 224), normalize=False)

        img = torch.ones((3, 224, 224), dtype=torch.float32)
        out_norm = preproc_norm(img)
        out_unnorm = preproc_unnorm(img)

        # Unnormalized image maintains value 1.0
        self.assertAlmostEqual(out_unnorm[0, 0, 0].item(), 1.0, places=4)
        # Normalized image must diverge from raw 1.0
        self.assertNotAlmostEqual(out_norm[0, 0, 0].item(), 1.0, places=2)

    def test_multi_camera_act_dataset_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = create_dummy_dataset(Path(tmpdir), num_episodes=2, num_steps=25)
            ep_files = find_episode_parquet_files(root)
            self.assertEqual(len(ep_files), 2)

            cfg = get_profile_config("openamrobot-fixed-mast-19dof")
            cfg.chunk_size = 100
            dataset = MultiCameraACTDataset(
                dataset_path=root,
                episode_paths=ep_files,
                config=cfg,
                load_images=True,
                normalize_images=True,
            )
            self.assertEqual(len(dataset), 50)  # 2 episodes * 25 steps

            sample = dataset[0]
            self.assertIn("state", sample)
            self.assertIn("images", sample)
            self.assertIn("actions", sample)
            self.assertIn("action_mask", sample)

            self.assertEqual(sample["state"].shape, (19,))
            self.assertEqual(sample["images"].shape, (4, 3, 224, 224))
            self.assertEqual(sample["actions"].shape, (100, 19))
            self.assertEqual(sample["action_mask"].shape, (100,))

    def test_partition_episodes_reproducibility(self):
        paths = [Path(f"ep_{i}.parquet") for i in range(10)]
        train1, val1, tr_idx1, val_idx1 = partition_episodes(paths, eval_split=0.2, seed=42)
        train2, val2, tr_idx2, val_idx2 = partition_episodes(paths, eval_split=0.2, seed=42)

        self.assertEqual(tr_idx1, tr_idx2)
        self.assertEqual(val_idx1, val_idx2)
        self.assertEqual(len(val_idx1), 2)
        self.assertEqual(len(tr_idx1), 8)
        self.assertEqual(len(set(tr_idx1).intersection(set(val_idx1))), 0)

        # Different seed yields different split
        _, _, tr_idx3, _ = partition_episodes(paths, eval_split=0.2, seed=999)
        self.assertNotEqual(tr_idx1, tr_idx3)

    def test_act_policy_forward_training_mode(self):
        cfg = get_profile_config("openamrobot-fixed-mast-19dof")
        cfg.chunk_size = 10
        cfg.d_model = 64
        cfg.enc_layers = 1
        cfg.dec_layers = 1
        cfg.dim_feedforward = 128
        cfg.n_heads = 4

        policy = ACTPolicy(cfg)
        policy.train()

        states = torch.randn(2, 19)
        images = torch.randn(2, 4, 3, 224, 224)
        actions = torch.randn(2, 10, 19)

        pred_actions, mu, log_var = policy(state=states, images=images, actions=actions)
        self.assertEqual(pred_actions.shape, (2, 10, 19))
        self.assertIsNotNone(mu)
        self.assertIsNotNone(log_var)
        self.assertEqual(mu.shape, (2, 32))
        self.assertEqual(log_var.shape, (2, 32))

        loss_dict = policy.compute_loss(pred_actions, actions, mu, log_var)
        self.assertIn("loss", loss_dict)
        self.assertIn("l1_loss", loss_dict)
        self.assertIn("kl_loss", loss_dict)
        self.assertTrue(loss_dict["loss"].item() > 0)

    def test_act_policy_forward_eval_mode(self):
        cfg = get_profile_config("openamrobot-fixed-mast-19dof")
        cfg.chunk_size = 10
        cfg.d_model = 64
        cfg.enc_layers = 1
        cfg.dec_layers = 1
        cfg.dim_feedforward = 128
        cfg.n_heads = 4

        policy = ACTPolicy(cfg)
        policy.eval()

        states = torch.randn(2, 19)
        images = torch.randn(2, 4, 3, 224, 224)

        with torch.no_grad():
            pred_actions, mu, log_var = policy(state=states, images=images)

        self.assertEqual(pred_actions.shape, (2, 10, 19))
        self.assertIsNone(mu)
        self.assertIsNone(log_var)

    def test_training_smoke_test_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_dir = create_dummy_dataset(root / "dataset", num_episodes=2, num_steps=10)
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

            cfg = get_profile_config("openamrobot-fixed-mast-19dof")
            cfg.chunk_size = 10
            cfg.d_model = 64
            cfg.enc_layers = 1
            cfg.dec_layers = 1
            cfg.dim_feedforward = 128
            cfg.n_heads = 4
            cfg.epochs = 2
            cfg.device = "cpu"

            out_dir = root / "output"
            summary = train_act(
                dataset_path=dataset_dir,
                config=cfg,
                qa_report_path=qa_report_path,
                output_dir=out_dir,
                smoke_test=True,
                allow_synthetic_eval=True,
            )

            self.assertEqual(summary["status"], "COMPLETED")
            self.assertEqual(summary["acceptance_status"], "NOT_VERIFIED_PHYSICAL")
            self.assertTrue((out_dir / "act_policy_best.pt").is_file())
            self.assertTrue((out_dir / "act_policy_best.pt.sha256").is_file())
            self.assertTrue((out_dir / "c11_training_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
