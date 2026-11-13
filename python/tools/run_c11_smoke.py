#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded CPU software test using honest synthetic native C9/C10 artifacts.

The raw C10 decision and physical/production acceptance remain NOT_VERIFIED.
No external dataset, GPU, simulator or robot is used. Requires this source tree's
fixtures plus the test/native/training extras and ffmpeg/ffprobe.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from oa_embodied_ai.evaluation.cli import evaluate_c11
from oa_embodied_ai.training.cli import train_act
from oa_embodied_ai.training.config import get_profile_config, save_config
from oa_embodied_ai.training.evidence import execution_provenance
from oa_embodied_ai.training.inputs import file_sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from c11_fixture import synthetic_dataset

    torch.set_num_threads(1)
    root, policy, manifest = synthetic_dataset(output / "dataset")
    cfg = get_profile_config("lerobot-v0.6.1-bi-openarm-position")
    cfg.chunk_size, cfg.d_model, cfg.n_heads = 3, 32, 4
    cfg.enc_layers = cfg.dec_layers = 1
    cfg.dim_feedforward, cfg.image_size, cfg.dropout = 64, (32, 32), 0.0
    save_config(cfg, output / "synthetic-smoke-config.yaml")
    summary = train_act(
        root,
        cfg,
        output_dir=output / "training",
        smoke_test=True,
        allow_synthetic_eval=True,
        input_manifest_path=manifest,
        policy_path=policy,
    )
    evaluation = evaluate_c11(
        summary["checkpoint_path"],
        root,
        mode="both",
        output_report=output / "evaluation.json",
        allow_synthetic_eval=True,
        input_manifest_path=manifest,
        policy_path=policy,
    )
    if (
        summary["acceptance_status"] != "NOT_VERIFIED_PHYSICAL"
        or summary["gate_evidence"]["decision"]["training_ready"] is not False
        or summary["checkpoint_sha256"] != file_sha256(summary["checkpoint_path"])
        or evaluation["rollout_metrics"]["total_rollouts"] != 0
    ):
        raise RuntimeError("Synthetic software-test status invariant violated")
    evidence = {
        "purpose": "SYNTHETIC SOFTWARE TEST ONLY",
        "software_test_completed": True,
        "real_recording": "NOT_VERIFIED",
        "production_acceptance": "NOT_VERIFIED",
        "training": summary,
        "evaluation": evaluation,
        "environment": execution_provenance(),
    }
    (output / "software-test-evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        "CPU synthetic software test completed; C10/physical/production remain NOT_VERIFIED"
    )
    print(f"Evidence: {output / 'software-test-evidence.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
