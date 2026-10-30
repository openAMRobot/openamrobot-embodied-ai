# SPDX-License-Identifier: MIT
"""CLI runner for Deliverable C11: ACT Policy Training Engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from .config import ACTConfig, load_config
from .dataset import MultiCameraACTDataset, find_episode_parquet_files, partition_episodes
from .gate import GateStatus, TrainingBlockedError, verify_training_gate
from .model import ACTPolicy

LOG = logging.getLogger("oa-c11-train")


def compute_file_sha256(filepath: Path) -> str:
    """Compute cryptographic SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit(repo_dir: Path) -> str:
    """Retrieve git HEAD commit SHA or return unknown."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def train_act(
    dataset_path: Path,
    config: ACTConfig,
    qa_report_path: Optional[Path] = None,
    output_dir: Path = Path("outputs/c11_training"),
    smoke_test: bool = False,
    allow_synthetic_eval: bool = False,
) -> Dict[str, Any]:
    """Execute ACT baseline training pipeline with mandatory pre-flight gate.

    Returns:
        Summary dict containing training metadata, checkpoint path, and SHA-256.

    Raises:
        TrainingBlockedError: If pre-flight gate fails.
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Mandatory Pre-Flight Gate Verification (Fail-Closed)
    LOG.info("Executing mandatory C10 pre-flight gate...")
    gate_status = verify_training_gate(
        dataset_path=dataset_path,
        qa_report_path=qa_report_path,
        allow_synthetic_override=allow_synthetic_eval,
    )
    LOG.info("Pre-flight gate PASSED (source_profile=%s)", gate_status.source_profile)

    # 2. Locate episodes and partition
    ep_files = find_episode_parquet_files(dataset_path)
    if not ep_files:
        raise FileNotFoundError(f"No episode parquet files found under {dataset_path}")

    train_eps, val_eps, train_indices, val_indices = partition_episodes(
        ep_files, eval_split=config.eval_split, seed=config.seed
    )
    LOG.info(
        "Partitioned %d episodes -> %d train, %d val",
        len(ep_files),
        len(train_eps),
        len(val_eps),
    )

    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    torch.manual_seed(config.seed)

    # 3. Build Datasets
    train_dataset = MultiCameraACTDataset(
        dataset_path=dataset_path,
        episode_paths=train_eps,
        config=config,
        load_images=True,
        normalize_images=True,
    )
    val_dataset = MultiCameraACTDataset(
        dataset_path=dataset_path,
        episode_paths=val_eps if val_eps else train_eps,
        config=config,
        load_images=True,
        normalize_images=True,
    )

    batch_size = 2 if smoke_test else config.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. Instantiate ACT Policy
    policy = ACTPolicy(config).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    epochs = 2 if smoke_test else config.epochs
    LOG.info("Starting training on %s (epochs=%d, smoke_test=%s)", device, epochs, smoke_test)

    train_loss_val = 0.0
    val_loss_val = 0.0

    for epoch in range(epochs):
        policy.train()
        epoch_loss = 0.0
        steps = 0

        for batch_idx, batch in enumerate(train_loader):
            states = batch["state"].to(device)
            images = batch["images"].to(device)
            actions = batch["actions"].to(device)
            masks = batch["action_mask"].to(device)

            optimizer.zero_grad()
            pred_actions, mu, log_var = policy(state=states, images=images, actions=actions)
            loss_dict = policy.compute_loss(
                pred_actions=pred_actions,
                target_actions=actions,
                mu=mu,
                log_var=log_var,
                action_mask=masks,
                l1_weight=config.l1_weight,
                kl_weight=config.kl_weight,
            )
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

            if smoke_test and steps >= 2:
                break

        train_loss_val = epoch_loss / max(1, steps)
        LOG.info("Epoch %d/%d - Train Loss: %.4f", epoch + 1, epochs, train_loss_val)

        # Validation step
        policy.eval()
        val_epoch_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                states = batch["state"].to(device)
                images = batch["images"].to(device)
                actions = batch["actions"].to(device)
                masks = batch["action_mask"].to(device)

                pred_actions, mu, log_var = policy(state=states, images=images, actions=actions)
                loss_dict = policy.compute_loss(
                    pred_actions=pred_actions,
                    target_actions=actions,
                    mu=mu,
                    log_var=log_var,
                    action_mask=masks,
                    l1_weight=config.l1_weight,
                    kl_weight=config.kl_weight,
                )
                val_epoch_loss += loss_dict["loss"].item()
                val_steps += 1
                if smoke_test and val_steps >= 2:
                    break

        val_loss_val = val_epoch_loss / max(1, val_steps)
        LOG.info("Epoch %d/%d - Val Loss: %.4f", epoch + 1, epochs, val_loss_val)

    # 5. Save Checkpoint and Hash
    ckpt_path = output_dir / "act_policy_best.pt"
    ckpt_payload = {
        "model_state_dict": policy.state_dict(),
        "config": config.to_dict(),
        "train_indices": train_indices,
        "val_indices": val_indices,
        "gate_status": {
            "source_profile": gate_status.source_profile,
            "quality_status": gate_status.quality_status,
            "is_synthetic_override": gate_status.is_synthetic_override,
        },
    }
    torch.save(ckpt_payload, ckpt_path)
    ckpt_sha256 = compute_file_sha256(ckpt_path)
    (output_dir / "act_policy_best.pt.sha256").write_text(ckpt_sha256, encoding="utf-8")
    LOG.info("Saved checkpoint to %s (SHA-256: %s)", ckpt_path, ckpt_sha256)

    # 6. Save Training Metadata Summary
    summary = {
        "status": "COMPLETED",
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha256,
        "dataset_path": str(dataset_path),
        "train_episodes": train_indices,
        "val_episodes": val_indices,
        "train_loss": train_loss_val,
        "val_loss": val_loss_val,
        "smoke_test": smoke_test,
        "gate_status": gate_status.quality_status,
        "source_profile": gate_status.source_profile,
        "is_synthetic_override": gate_status.is_synthetic_override,
        "acceptance_status": "NOT_VERIFIED_PHYSICAL" if gate_status.is_synthetic_override else "VERIFIED_PHYSICAL",
        "provenance": {
            "git_commit": get_git_commit(Path(__file__).resolve().parents[3]),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
        },
    }
    summary_path = output_dir / "c11_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("Saved training summary to %s", summary_path)

    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for oa-c11-train."""
    parser = argparse.ArgumentParser(description="Deliverable C11: ACT Baseline Trainer")
    parser.add_argument("--dataset-path", type=Path, required=True, help="Path to LeRobotDataset directory")
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config file")
    parser.add_argument("--qa-report", type=Path, default=None, help="Path to pre-flight QA report")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/c11_training"), help="Output directory")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid CPU smoke test")
    parser.add_argument("--allow-synthetic-eval", action="store_true", help="Allow synthetic/Gazebo data override")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Load configuration
    if args.config is not None and args.config.is_file():
        cfg = load_config(args.config)
    else:
        # Default config fallback
        repo_root = Path(__file__).resolve().parents[3]
        default_cfg_path = repo_root / "configs" / "c11_act_default.yaml"
        if default_cfg_path.is_file():
            cfg = load_config(default_cfg_path)
        else:
            from .config import get_profile_config

            cfg = get_profile_config("openamrobot-fixed-mast-19dof")

    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.device is not None:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed

    try:
        train_act(
            dataset_path=args.dataset_path,
            config=cfg,
            qa_report_path=args.qa_report,
            output_dir=args.output_dir,
            smoke_test=args.smoke_test,
            allow_synthetic_eval=args.allow_synthetic_eval,
        )
        return 0
    except TrainingBlockedError as exc:
        LOG.error("TRAINING GATE ERROR (Exit 2): %s", exc.message)
        sys.exit(exc.exit_code)
    except Exception as exc:
        LOG.error("Training failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())

