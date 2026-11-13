#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Two SYNTHETIC CDR bags through native conversion, C10 and bounded visual C11.

QA bindings are explicitly synthetic fixtures; the raw decision remains blocked.
This helper never establishes recording, rollout or physical acceptance.
"""

import argparse
import json
import sys
from pathlib import Path

from oa_embodied_ai.conversion.batch import convert_collection
from oa_embodied_ai.dataset_qa.episode import check_dataset
from oa_embodied_ai.evaluation.cli import evaluate_c11
from oa_embodied_ai.training.cli import train_act
from oa_embodied_ai.training.config import get_profile_config
from oa_embodied_ai.training.evidence import execution_provenance
from oa_embodied_ai.training.inputs import file_sha256, inventory

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args(argv)
    before = execution_provenance()
    if before["git_commit"] != args.expected_head or not before["source_tree_clean"]:
        raise RuntimeError("Evidence requires the expected clean source HEAD")
    import torch

    sys.path.insert(0, str(ROOT / "python/tests"))
    from capture_fixture import (
        bind_converted_for_synthetic_training,
        collection_fixture,
    )

    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    collection = collection_fixture(root / "source-bags")
    conversion = convert_collection(
        collection, root / "native-dataset", development=True
    )
    unbound = check_dataset(root / "native-dataset")
    (root / "unbound-c10.json").write_text(json.dumps(unbound, indent=2) + "\n")
    native_hashes_before = inventory(root / "native-dataset")
    policy, manifest = bind_converted_for_synthetic_training(root / "native-dataset")
    native_hashes_after = inventory(root / "native-dataset")
    changes = {
        key: {"before": digest, "after": native_hashes_after.get(key)}
        for key, digest in native_hashes_before.items()
        if native_hashes_after.get(key) != digest
    }
    if any(not path.startswith("meta/openamrobot/episodes/") for path in changes):
        raise RuntimeError(
            "Synthetic QA fixture binding modified native writer payloads"
        )
    raw = check_dataset(root / "native-dataset", policy)
    (root / "raw-c10.json").write_text(json.dumps(raw, indent=2) + "\n")
    if (
        raw["status"] != "NOT_VERIFIED"
        or raw["exit_code"] != 2
        or raw["training_ready"]
        or any(f["status"] == "REJECT" for f in raw["findings"])
    ):
        raise RuntimeError(
            "Converted synthetic dataset did not retain the required C10 status"
        )
    cfg = get_profile_config("lerobot-v0.6.1-bi-openarm-position")
    cfg.chunk_size, cfg.d_model, cfg.n_heads = 3, 32, 4
    cfg.enc_layers = cfg.dec_layers = 1
    cfg.dim_feedforward, cfg.image_size, cfg.dropout = 64, (32, 32), 0.0
    torch.set_num_threads(1)
    training = train_act(
        root / "native-dataset",
        cfg,
        output_dir=root / "training",
        smoke_test=True,
        allow_synthetic_eval=True,
        input_manifest_path=manifest,
        policy_path=policy,
    )
    evaluation = evaluate_c11(
        training["checkpoint_path"],
        root / "native-dataset",
        mode="both",
        output_report=root / "evaluation.json",
        allow_synthetic_eval=True,
        input_manifest_path=manifest,
        policy_path=policy,
    )
    after = execution_provenance()
    if (
        training["acceptance_status"] != "NOT_VERIFIED_PHYSICAL"
        or evaluation["acceptance_status"] != "NOT_VERIFIED_PHYSICAL"
        or evaluation["rollout_metrics"]["total_rollouts"] != 0
        or training["checkpoint_sha256"] != file_sha256(training["checkpoint_path"])
        or set(training["train_episodes"]) & set(training["val_episodes"])
        or before["source_sha256"] != after["source_sha256"]
        or not after["source_tree_clean"]
    ):
        raise RuntimeError("Synthetic pipeline provenance/status invariant failed")
    report = {
        "purpose": "SYNTHETIC SOFTWARE PIPELINE ONLY; QA bindings are test fixtures",
        "software_pipeline_passed": True,
        "execution": before,
        "execution_after": after,
        "conversion": conversion,
        "training": training,
        "evaluation": evaluation,
        "synthetic_sidecar_changes": changes,
        "native_writer_payloads_unchanged": True,
        "raw_c10_status": raw["status"],
        "raw_c10_exit_code": raw["exit_code"],
        "physical_acceptance": "NOT_VERIFIED",
        "production_acceptance": "NOT_VERIFIED",
        "artifacts_sha256": inventory(root),
        "ci_archive_excludes": [str(p.relative_to(root)) for p in root.rglob("*.pt")]
        + [str(p.relative_to(root)) for p in root.rglob("*.pt.sha256")],
    }
    (root / "pipeline-manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        "PASS: synthetic CDR bags -> native LeRobot -> C10 NOT_VERIFIED -> explicit software-only C11 training -> offline evaluation"
    )
    print(
        "No real recording, executed rollout or physical acceptance; raw C10 exit 2 retained"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
