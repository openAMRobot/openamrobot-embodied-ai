# SPDX-License-Identifier: MIT
"""C11 startup fault injection. PASS envelopes are fabricated protocol tests."""

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
from oa_embodied_ai.dataset_qa import episode as c10
from oa_embodied_ai.training.cli import train_act
from oa_embodied_ai.training.config import get_profile_config
from oa_embodied_ai.training.gate import TrainingBlockedError, verify_training_gate
from oa_embodied_ai.training.inputs import file_sha256, inventory
from test_c10_handoff import protocol_envelope


@pytest.fixture
def protocol(tmp_path):
    """No real recording: mock checker output to exercise consumer/startup wiring."""
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "sidecar.json").write_text('{"purpose":"fabricated protocol envelope"}')
    policy = tmp_path / "policy.json"
    policy.write_text('{"purpose":"fabricated protocol test"}')
    manifest = {
        "manifest_version": 1,
        "dataset_id": "protocol-test",
        "dataset_revision": "v1",
        "episode_indices": [0],
        "data_origin": "actual",
        "profile": "lerobot-v0.6.1-bi-openarm-position",
        "evidence": "FABRICATED PROTOCOL TEST ONLY; no real acceptance",
        "policy_sha256": file_sha256(policy),
        "files_sha256": inventory(root),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    report = protocol_envelope()
    report["provenance"]["policy"]["sha256"] = file_sha256(policy)
    report["provenance"]["checker_files_sha256"] = {
        name: file_sha256(Path(c10.__file__).parent / name)
        for name in (
            "episode.py",
            "semantics.py",
            "strict_io.py",
            "contract.py",
            "handoff.py",
            "video.py",
        )
    }
    report["episodes"][0]["sidecar"] = {
        "path": "sidecar.json",
        "sha256": file_sha256(root / "sidecar.json"),
    }
    return root, policy, manifest_path, report


def test_complete_canonical_decision_is_consumed(protocol):
    root, policy, manifest, report = protocol
    from oa_embodied_ai.dataset_qa import training_readiness

    with (
        patch.object(c10, "check_dataset", return_value=report) as checker,
        patch(
            "oa_embodied_ai.training.gate.training_readiness", wraps=training_readiness
        ) as decision,
    ):
        status = verify_training_gate(
            root, input_manifest_path=manifest, policy_path=policy
        )
    assert status.training_ready and not status.is_synthetic_override
    checker.assert_called_once_with(root.resolve(), policy, [0])
    decision.assert_called_once_with(
        report, dataset_id="protocol-test", dataset_revision="v1", episode_indices=[0]
    )
    assert report["production_acceptance"] == "NOT_VERIFIED"


def remove_check(report, name):
    report["findings"] = [f for f in report["findings"] if f["check_id"] != name]


def mutate(report, fault):
    if fault == "missing_report":
        return None
    if fault == "missing_findings":
        report.pop("findings")
    elif fault == "empty_findings":
        report["findings"] = []
    elif fault == "blocked_overall_pass":
        report["quality_status"] = "BLOCKED"
    elif fault == "blocked_status_quality_pass":
        report.update(status="BLOCKED", quality_status="PASS")
    elif fault == "reject_ready_true":
        report["findings"][0]["status"] = "REJECT"
    elif fault == "not_verified":
        report["status"] = "NOT_VERIFIED"
    elif fault == "synthetic_labelled_robot":
        report["source_profile"] = "ROBOT"
        report["episodes"][0]["data_origin"] = "synthetic"
    elif fault in ("dataset_id", "dataset_revision"):
        report["episodes"][0]["binding"][fault] = "unrelated"
    elif fault == "selected_episode":
        report["episodes"][0]["episode_index"] = 7
    elif fault == "real_episode":
        remove_check(report, "C10_REAL_EPISODE")
    elif fault == "canonical_c9":
        report["provenance"]["c9_contract"] = None
    elif fault == "policy_missing":
        report["provenance"]["policy"] = None
    elif fault == "policy_unapproved":
        report["provenance"]["policy"]["approved"] = False
    elif fault == "diagnostic_coverage":
        remove_check(report, "C10_VIDEO")
    elif fault == "exit_missing":
        report.pop("exit_code")
    elif fault == "exit_invalid":
        report["exit_code"] = False
    elif fault == "checker_provenance":
        report["provenance"]["checker_files_sha256"]["episode.py"] = "0" * 64
    elif fault == "policy_provenance":
        report["provenance"]["policy"]["sha256"] = "0" * 64
    elif fault == "sidecar_provenance":
        report["episodes"][0]["sidecar"]["sha256"] = "0" * 64
    elif fault == "report_version":
        report["report_version"] = True
    elif fault == "malformed_findings":
        report["findings"] = [None]
    elif fault == "rejected_severity":
        report["findings"][0]["severity"] = "REJECT"
    elif fault == "registry_only":
        return {"dataset_id": "protocol-test", "quality_status": "passed"}
    return report


FAULTS = [
    "missing_report",
    "missing_findings",
    "empty_findings",
    "blocked_overall_pass",
    "blocked_status_quality_pass",
    "reject_ready_true",
    "not_verified",
    "synthetic_labelled_robot",
    "dataset_id",
    "dataset_revision",
    "selected_episode",
    "real_episode",
    "canonical_c9",
    "policy_missing",
    "policy_unapproved",
    "diagnostic_coverage",
    "exit_missing",
    "exit_invalid",
    "checker_provenance",
    "policy_provenance",
    "sidecar_provenance",
    "report_version",
    "malformed_findings",
    "rejected_severity",
    "registry_only",
]


@pytest.mark.parametrize("fault", FAULTS)
def test_rejected_evidence_stops_before_model_training(protocol, tmp_path, fault):
    root, policy, manifest, report = protocol
    output = tmp_path / "run"
    report = mutate(deepcopy(report), fault)
    with (
        patch.object(c10, "check_dataset", return_value=report),
        patch("oa_embodied_ai.training.cli.ACTPolicy") as model,
    ):
        with pytest.raises(TrainingBlockedError) as error:
            train_act(
                root,
                get_profile_config("lerobot-v0.6.1-bi-openarm-position"),
                output_dir=output,
                input_manifest_path=manifest,
                policy_path=policy,
                allow_synthetic_eval=True,
            )
        assert error.value.code == error.value.exit_code == 2
        model.assert_not_called()
    assert not list(tmp_path.rglob("*.pt"))
    assert not list(tmp_path.rglob("c11_training_summary.json"))


def test_missing_or_stale_supplied_report_blocks(protocol, tmp_path):
    root, policy, manifest, report = protocol
    old = tmp_path / "old.json"
    with (
        patch.object(c10, "check_dataset", return_value=report),
        patch("oa_embodied_ai.training.cli.ACTPolicy") as model,
    ):
        for payload in (None, {}, {"status": "PASS"}):
            if payload is not None:
                old.write_text(json.dumps(payload))
            with pytest.raises(TrainingBlockedError):
                train_act(
                    root,
                    get_profile_config("lerobot-v0.6.1-bi-openarm-position"),
                    qa_report_path=old,
                    output_dir=tmp_path / "run",
                    input_manifest_path=manifest,
                    policy_path=policy,
                )
        model.assert_not_called()


def test_dataset_modified_after_report_is_blocked(protocol, tmp_path):
    root, policy, manifest, report = protocol
    old = tmp_path / "old.json"
    old.write_text(json.dumps(report))
    (root / "sidecar.json").write_text("changed after QA")
    with (
        patch.object(c10, "check_dataset") as checker,
        patch("oa_embodied_ai.training.cli.ACTPolicy") as model,
    ):
        with pytest.raises(TrainingBlockedError):
            train_act(
                root,
                get_profile_config("lerobot-v0.6.1-bi-openarm-position"),
                qa_report_path=old,
                output_dir=tmp_path / "run",
                input_manifest_path=manifest,
                policy_path=policy,
            )
        checker.assert_not_called()
        model.assert_not_called()


def test_no_gate_clearance_or_trusted_inputs_blocks(protocol, tmp_path):
    root, _, _, _ = protocol
    with patch("oa_embodied_ai.training.cli.ACTPolicy") as model:
        with pytest.raises(TrainingBlockedError):
            train_act(
                root,
                get_profile_config("lerobot-v0.6.1-bi-openarm-position"),
                output_dir=tmp_path / "run",
            )
        model.assert_not_called()


def test_unrelated_working_directory_report_is_never_reused(
    protocol, tmp_path, monkeypatch
):
    root, policy, manifest, report = protocol
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/qa_report.json").write_text(json.dumps(report))
    monkeypatch.chdir(tmp_path)
    with patch.object(c10, "check_dataset", return_value=None) as checker:
        with pytest.raises(TrainingBlockedError):
            verify_training_gate(root, input_manifest_path=manifest, policy_path=policy)
        checker.assert_called_once()


def test_input_mutation_during_checker_blocks_startup(protocol, tmp_path):
    root, policy, manifest, report = protocol

    def mutate_while_checking(snapshot, policy_copy, indices):
        (snapshot / "sidecar.json").write_text("mutated while checker ran")
        return report

    with (
        patch.object(c10, "check_dataset", side_effect=mutate_while_checking),
        patch("oa_embodied_ai.training.cli.ACTPolicy") as model,
    ):
        with pytest.raises(TrainingBlockedError):
            train_act(
                root,
                get_profile_config("lerobot-v0.6.1-bi-openarm-position"),
                output_dir=tmp_path / "run",
                input_manifest_path=manifest,
                policy_path=policy,
            )
        model.assert_not_called()


def test_gate_itself_rejects_missing_diagnostics(protocol):
    root, policy, manifest, report = protocol
    remove_check(report, "C10_VIDEO")
    with (
        patch.object(c10, "check_dataset", return_value=report),
        pytest.raises(TrainingBlockedError),
    ):
        verify_training_gate(root, input_manifest_path=manifest, policy_path=policy)
