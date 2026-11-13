# SPDX-License-Identifier: MIT
"""Offline/rollout separation and evidence consistency regressions."""

import json
from unittest.mock import patch

import numpy as np
import pytest
import torch
from oa_embodied_ai.evaluation.cli import evaluate_c11, main
from oa_embodied_ai.evaluation.offline_eval import (
    compute_offline_metrics,
)
from oa_embodied_ai.evaluation.rollout_eval import (
    ClosedLoopRolloutEvaluator,
    compute_wilson_ci,
    run_rollout_eval,
)
from oa_embodied_ai.training.cli import train_act
from oa_embodied_ai.training.config import get_profile_config
from oa_embodied_ai.training.gate import TrainingBlockedError
from oa_embodied_ai.training.inputs import file_sha256


@pytest.fixture(scope="module")
def trained(synthetic_template, tmp_path_factory):
    root, policy, manifest = synthetic_template
    cfg = get_profile_config("lerobot-v0.6.1-bi-openarm-position")
    cfg.chunk_size, cfg.d_model, cfg.n_heads = 3, 32, 4
    cfg.enc_layers = cfg.dec_layers = 1
    cfg.dim_feedforward, cfg.image_size, cfg.dropout = 64, (32, 32), 0.0
    output = tmp_path_factory.mktemp("c11-trained") / "run"
    summary = train_act(
        root,
        cfg,
        output_dir=output,
        smoke_test=True,
        allow_synthetic_eval=True,
        input_manifest_path=manifest,
        policy_path=policy,
    )
    return root, policy, manifest, summary


def test_metrics_are_only_offline_errors():
    metrics = compute_offline_metrics(np.ones((4, 16)), np.zeros((4, 16)))
    assert metrics["overall_mae"] == metrics["overall_mse"] == 1
    assert "success_rate" not in metrics and "successful_rollouts" not in metrics
    with pytest.raises(ValueError):
        compute_offline_metrics(np.ones((4, 16)), np.zeros((4, 19)))


def test_missing_rollout_adapter_cannot_invent_success():
    class NeverCalled:
        def __call__(self, *args):
            raise AssertionError("No environment was supplied")

    result = run_rollout_eval(NeverCalled(), num_episodes=5)
    assert result["status"] == "NOT_VERIFIED"
    assert result["total_rollouts"] == 0
    assert (
        result["successful_rollouts"]
        is result["success_rate"]
        is result["wilson_ci_95"]
        is None
    )
    with pytest.raises(NotImplementedError):
        ClosedLoopRolloutEvaluator().evaluate_episode(None, None, None)


def test_wilson_interval_requires_executed_counts():
    assert compute_wilson_ci(0, 0) == (None, None)
    assert compute_wilson_ci(0, 20)[0] == 0
    assert compute_wilson_ci(20, 20)[1] == 1
    low, high = compute_wilson_ci(10, 20)
    assert low == pytest.approx(1 - high) and 0.25 < low < 0.35
    with pytest.raises(ValueError):
        compute_wilson_ci(21, 20)


@pytest.mark.parametrize("mode", ["offline", "both", "rollout"])
def test_executed_eval_keeps_offline_and_acceptance_separate(trained, tmp_path, mode):
    root, policy, manifest, summary = trained
    output = tmp_path / "evaluation.json"
    report = evaluate_c11(
        summary["checkpoint_path"],
        root,
        mode=mode,
        output_report=output,
        allow_synthetic_eval=True,
        input_manifest_path=manifest,
        policy_path=policy,
    )
    assert report["acceptance_status"] == "NOT_VERIFIED_PHYSICAL"
    assert report["production_acceptance"] == report["gate_status"] == "NOT_VERIFIED"
    assert report["evaluation_episodes"] == summary["val_episodes"]
    assert not set(report["evaluation_episodes"]) & set(summary["train_episodes"])
    assert report["provenance"]["checkpoint_sha256"] == file_sha256(
        summary["checkpoint_path"]
    )
    # Dataset interoperability tooling may coexist with the custom C11 model.
    # Installed-library provenance must report the environment, not assert absence.
    from importlib.metadata import PackageNotFoundError, version

    try:
        expected_lerobot = version("lerobot")
    except PackageNotFoundError:
        expected_lerobot = None
    assert report["provenance"]["dependencies"]["lerobot"] == expected_lerobot
    assert "custom ACT-like" in report["provenance"]["model_implementation"]
    assert "compatibility NOT_VERIFIED" in report["provenance"]["model_implementation"]
    assert json.loads(output.read_text()) == report
    markdown = output.with_suffix(".md").read_text()
    assert "NOT_VERIFIED_PHYSICAL" in markdown
    assert '"acceptance_status": "VERIFIED_PHYSICAL"' not in markdown
    if mode != "rollout":
        assert report["offline_metrics"]["overall_mae"] >= 0
        assert report["offline_metrics"]["chunk_boundary_discontinuity"] is None
    if mode != "offline":
        assert report["status"] == "NOT_VERIFIED"
        assert report["rollout_metrics"]["total_rollouts"] == 0


def test_offline_evaluation_actually_supplies_all_images(trained, tmp_path):
    root, policy, manifest, summary = trained
    from oa_embodied_ai.training.model import ACTPolicy

    original = ACTPolicy.forward
    calls = []

    def observe(self, state, images=None, **kwargs):
        assert images is not None and images.shape[1:3] == (4, 3)
        calls.append(images.detach().clone())
        return original(self, state, images, **kwargs)

    with patch.object(ACTPolicy, "forward", observe):
        evaluate_c11(
            summary["checkpoint_path"],
            root,
            output_report=tmp_path / "eval.json",
            allow_synthetic_eval=True,
            input_manifest_path=manifest,
            policy_path=policy,
        )
    assert len(calls) == 2
    assert not torch.equal(calls[0][:, 0], calls[0][:, 1])


@pytest.mark.parametrize(
    "fault",
    ["missing_val", "overlap", "out_of_range", "wrong_revision", "missing_weights"],
)
def test_checkpoint_selection_never_falls_back_to_training(trained, tmp_path, fault):
    root, policy, manifest, summary = trained
    ckpt = torch.load(summary["checkpoint_path"], weights_only=True)
    if fault == "missing_val":
        ckpt["val_indices"] = []
    elif fault == "overlap":
        ckpt["val_indices"] = ckpt["train_indices"]
    elif fault == "out_of_range":
        ckpt["val_indices"] = [999]
    elif fault == "wrong_revision":
        ckpt["gate_evidence"]["input_manifest"]["dataset_revision"] = "other"
    elif fault == "missing_weights":
        ckpt.pop("model_state_dict")
    path = tmp_path / "bad.pt"
    torch.save(ckpt, path)
    output = tmp_path / "eval.json"
    with pytest.raises((TrainingBlockedError, KeyError)):
        evaluate_c11(
            path,
            root,
            output_report=output,
            allow_synthetic_eval=True,
            input_manifest_path=manifest,
            policy_path=policy,
        )
    assert not output.exists()


def test_blocked_gate_prevents_eval_and_cli_returns_two(trained, tmp_path):
    root, policy, manifest, summary = trained
    output = tmp_path / "blocked.json"
    with patch("oa_embodied_ai.evaluation.cli.ACTPolicy") as model:
        result = main(
            [
                "--checkpoint",
                summary["checkpoint_path"],
                "--dataset-path",
                str(root),
                "--policy",
                str(policy),
                "--input-manifest",
                str(manifest),
                "--output-report",
                str(output),
            ]
        )
        assert result == 2
        model.assert_not_called()
    assert not output.exists()
