# SPDX-License-Identifier: MIT
"""Gated offline visual evaluation; absent rollout evidence stays NOT_VERIFIED."""

import argparse
import logging
from pathlib import Path

import torch

from ..training.config import ACTConfig
from ..training.dataset import find_episode_parquet_files
from ..training.evidence import execution_provenance, write_report
from ..training.gate import TrainingBlockedError, verify_training_gate
from ..training.inputs import assert_unchanged, bound_snapshot, file_sha256, json_sha256
from ..training.model import ACTPolicy
from .offline_eval import run_offline_eval
from .rollout_eval import run_rollout_eval

LOG = logging.getLogger("oa-c11-eval")


def evaluate_c11(
    checkpoint_path,
    dataset_path,
    qa_report_path=None,
    mode="offline",
    output_report=Path("reports/c11_eval_report.json"),
    allow_synthetic_eval=False,
    device="cpu",
    ensemble_lambda=0.01,
    *,
    input_manifest_path=None,
    policy_path=None,
):
    if mode not in ("offline", "rollout", "both"):
        raise ValueError("Unsupported evaluation mode")
    output_report = Path(output_report).resolve()
    dataset_path = Path(dataset_path).resolve()
    if output_report.is_relative_to(dataset_path):
        raise TrainingBlockedError("Evaluation output must be outside the dataset")
    if output_report.exists() or output_report.with_suffix(".md").exists():
        raise FileExistsError("Evaluation output must be fresh")
    checkpoint_path = Path(checkpoint_path)
    with bound_snapshot(dataset_path, input_manifest_path, policy_path) as (
        snapshot,
        policy_copy,
    ):
        gate = verify_training_gate(
            snapshot,
            qa_report_path,
            allow_synthetic_eval,
            input_manifest_path=input_manifest_path,
            policy_path=policy_copy,
        )
        checkpoint_hash = file_sha256(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        cfg = ACTConfig(**checkpoint["config"])
        cfg.device = device
        cfg.validate()
        if cfg.profile != gate.input_manifest["profile"]:
            raise TrainingBlockedError("Checkpoint and evaluation profiles differ")
        source_manifest = checkpoint["gate_evidence"]["input_manifest"]
        train_ids, val_ids = checkpoint["train_indices"], checkpoint["val_indices"]
        if (
            not train_ids
            or not val_ids
            or any(type(i) is not int for i in train_ids + val_ids)
            or len(set(train_ids + val_ids)) != len(train_ids + val_ids)
        ):
            raise TrainingBlockedError(
                "Checkpoint has missing/overlapping episode splits"
            )
        same_dataset = (
            source_manifest["dataset_id"] == gate.input_manifest["dataset_id"]
        )
        if same_dataset:
            if json_sha256(source_manifest) != json_sha256(gate.input_manifest):
                raise TrainingBlockedError(
                    "Evaluation snapshot differs from checkpoint input"
                )
            selected = val_ids
        else:
            selected = gate.input_manifest["episode_indices"]
        available = {e.episode_index: e for e in find_episode_parquet_files(snapshot)}
        if not set(selected) <= set(gate.input_manifest["episode_indices"]) or not set(
            selected
        ) <= set(available):
            raise TrainingBlockedError(
                "Held-out evaluation episodes unavailable; no training fallback"
            )
        policy = ACTPolicy(cfg).to(device)
        policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
        policy.eval()
        offline = (
            run_offline_eval(
                policy,
                snapshot,
                [available[i] for i in selected],
                cfg,
                device=device,
                ensemble_lambda=ensemble_lambda,
            )
            if mode in ("offline", "both")
            else None
        )
        rollout = (
            run_rollout_eval(policy, control_freq_hz=cfg.fps)
            if mode in ("rollout", "both")
            else None
        )
        assert_unchanged(snapshot, gate.input_manifest, policy_copy)
        assert_unchanged(dataset_path, gate.input_manifest, policy_path)
        if file_sha256(checkpoint_path) != checkpoint_hash:
            raise TrainingBlockedError("Checkpoint bytes changed during evaluation")
        report = {
            "status": "COMPLETED" if mode == "offline" else "NOT_VERIFIED",
            "evaluation_type": {
                "offline": "OFFLINE_TRAJECTORY",
                "rollout": "CLOSED_LOOP_NOT_VERIFIED",
                "both": "OFFLINE_AND_ROLLOUT_NOT_VERIFIED",
            }[mode],
            "acceptance_status": "NOT_VERIFIED_PHYSICAL",
            "production_acceptance": "NOT_VERIFIED",
            "gate_status": gate.quality_status,
            "is_synthetic_override": gate.is_synthetic_override,
            "evaluation_episodes": selected,
            "offline_metrics": offline,
            "rollout_metrics": rollout,
            "provenance": {
                **execution_provenance(),
                "checkpoint_sha256": checkpoint_hash,
                "dataset_sha256": json_sha256(gate.input_manifest["files_sha256"]),
                "input_manifest": gate.input_manifest,
                "c10_report_sha256": gate.report_sha256,
                "training_provenance": checkpoint["provenance"],
            },
        }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    write_report(output_report, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="C11 offline visual evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--qa-report", type=Path)
    parser.add_argument(
        "--mode", choices=["offline", "rollout", "both"], default="offline"
    )
    parser.add_argument(
        "--output-report", type=Path, default=Path("reports/c11_eval_report.json")
    )
    parser.add_argument("--allow-synthetic-eval", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ensemble-lambda", type=float, default=0.01)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        report = evaluate_c11(
            args.checkpoint,
            args.dataset_path,
            args.qa_report,
            args.mode,
            args.output_report,
            args.allow_synthetic_eval,
            args.device,
            args.ensemble_lambda,
            input_manifest_path=args.input_manifest,
            policy_path=args.policy,
        )
        return 0 if report["status"] == "COMPLETED" else 2
    except TrainingBlockedError as error:
        LOG.error("Evaluation gate blocked: %s", error)
        return 2
    except Exception:
        LOG.exception("Evaluation failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
