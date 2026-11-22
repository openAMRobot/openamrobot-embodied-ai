# SPDX-License-Identifier: MIT
"""CLI runner for Deliverable C11: Disambiguated Evaluation Engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence

import torch

from ..training.config import ACTConfig, load_config
from ..training.dataset import find_episode_parquet_files
from ..training.gate import verify_training_gate
from ..training.model import ACTPolicy
from .offline_eval import run_offline_eval
from .rollout_eval import run_rollout_eval

LOG = logging.getLogger("oa-c11-eval")


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_dataset_sha256(dataset_path: Path) -> str:
    """Compute aggregate SHA-256 over dataset parquet files and info.json."""
    dataset_path = Path(dataset_path)
    h = hashlib.sha256()

    info_json = dataset_path / "meta" / "info.json"
    if info_json.is_file():
        h.update(info_json.read_bytes())

    for ep_file in find_episode_parquet_files(dataset_path):
        h.update(ep_file.name.encode("utf-8"))
        h.update(ep_file.read_bytes()[:4096])

    return h.hexdigest()


def get_git_commit(repo_dir: Path) -> str:
    """Get HEAD git commit SHA."""
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


def evaluate_c11(
    checkpoint_path: Path,
    dataset_path: Path,
    qa_report_path: Optional[Path] = None,
    mode: str = "offline",
    output_report: Path = Path("reports/c11_eval_report.json"),
    allow_synthetic_eval: bool = False,
    device: str = "cpu",
    ensemble_lambda: float = 0.01,
) -> Dict[str, Any]:
    """Execute evaluation and emit strict schema report."""
    checkpoint_path = Path(checkpoint_path)
    dataset_path = Path(dataset_path)
    output_report = Path(output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Check pre-flight gate / dataset origin
    gate_status = verify_training_gate(
        dataset_path=dataset_path,
        qa_report_path=qa_report_path,
        allow_synthetic_override=allow_synthetic_eval,
    )

    # Determine acceptance status
    is_physical = (
        gate_status.source_profile == "ROBOT"
        and not gate_status.is_synthetic_override
        and not allow_synthetic_eval
    )
    acceptance_status = "VERIFIED_PHYSICAL" if is_physical else "NOT_VERIFIED_PHYSICAL"

    # Compute provenance hashes
    ckpt_sha256 = compute_file_sha256(checkpoint_path)
    dataset_sha256 = compute_dataset_sha256(dataset_path)

    # Load checkpoint
    ckpt_data = torch.load(checkpoint_path, map_location=device)
    cfg_data = ckpt_data.get("config", {})

    # Reconstruct ACTConfig
    from ..training.config import get_profile_config

    profile_name = cfg_data.get("profile", "openamrobot-fixed-mast-19dof")
    cfg = get_profile_config(profile_name)
    for k, v in cfg_data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.device = device

    # Instantiate model and load weights
    policy = ACTPolicy(cfg).to(device)
    if "model_state_dict" in ckpt_data:
        policy.load_state_dict(ckpt_data["model_state_dict"])
    policy.eval()

    # Find evaluation episodes
    ep_files = find_episode_parquet_files(dataset_path)
    val_indices = ckpt_data.get("val_indices")
    if val_indices and max(val_indices) < len(ep_files):
        eval_ep_files = [ep_files[i] for i in val_indices]
    else:
        eval_ep_files = ep_files

    offline_metrics: Optional[Dict[str, Any]] = None
    rollout_metrics: Optional[Dict[str, Any]] = None

    if mode in ("offline", "both"):
        LOG.info("Running offline trajectory imitation evaluation...")
        offline_metrics = run_offline_eval(
            model=policy,
            dataset_path=dataset_path,
            episode_paths=eval_ep_files,
            config=cfg,
            device=device,
            ensemble_lambda=ensemble_lambda,
        )

    if mode in ("rollout", "both"):
        LOG.info("Running closed-loop rollout evaluation...")
        rollout_metrics = run_rollout_eval(
            policy=policy,
            num_episodes=20,
            control_freq_hz=float(cfg.fps),
        )

    # Set evaluation_type
    if mode == "offline":
        eval_type = "OFFLINE_TRAJECTORY"
    elif mode == "rollout":
        eval_type = "CLOSED_LOOP_ROLLOUT"
    else:
        eval_type = "COMBINED_OFFLINE_AND_ROLLOUT"

    # Assemble report matching strict schema
    repo_dir = Path(__file__).resolve().parents[3]
    report: Dict[str, Any] = {
        "evaluation_type": eval_type,
        "acceptance_status": acceptance_status,
        "provenance": {
            "checkpoint_sha256": ckpt_sha256,
            "dataset_sha256": dataset_sha256,
            "git_commit": get_git_commit(repo_dir),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "lerobot_version": "0.4.1",
        },
        "offline_metrics": offline_metrics,
        "rollout_metrics": rollout_metrics,
    }

    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOG.info("Saved evaluation report to %s", output_report)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for oa-c11-eval."""
    parser = argparse.ArgumentParser(description="Deliverable C11: ACT Evaluation Runner")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--dataset-path", type=Path, required=True, help="Path to LeRobotDataset")
    parser.add_argument("--qa-report", type=Path, default=None, help="Path to pre-flight QA report")
    parser.add_argument("--mode", choices=["offline", "rollout", "both"], default="offline", help="Evaluation mode")
    parser.add_argument("--output-report", type=Path, default=Path("reports/c11_eval_report.json"), help="Output JSON")
    parser.add_argument("--allow-synthetic-eval", action="store_true", help="Permit synthetic evaluation")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--ensemble-lambda", type=float, default=0.01, help="Temporal ensembling lambda")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    try:
        evaluate_c11(
            checkpoint_path=args.checkpoint,
            dataset_path=args.dataset_path,
            qa_report_path=args.qa_report,
            mode=args.mode,
            output_report=args.output_report,
            allow_synthetic_eval=args.allow_synthetic_eval,
            device=args.device,
            ensemble_lambda=args.ensemble_lambda,
        )
        return 0
    except Exception as exc:
        LOG.error("Evaluation failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())

