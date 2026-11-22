# SPDX-License-Identifier: MIT
"""Consumer fault injection. Fabricated PASS envelopes are protocol tests only."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from oa_embodied_ai.dataset_qa import validate_registry
from oa_embodied_ai.dataset_qa.contract import C9_SCHEMA_SHA256, load_c9_schema
from oa_embodied_ai.dataset_qa.handoff import REQUIRED_CHECKS, training_readiness
from oa_embodied_ai.dataset_qa.episode import check_dataset, main


def protocol_envelope():
    # This is deliberately fabricated to test the consumer protocol. It is never
    # emitted as dataset evidence, used for training or called a physical run.
    return {"report_version": 1, "status": "PASS", "exit_code": 0, "training_ready": True,
            "production_acceptance": "NOT_VERIFIED",
            "provenance": {"c9_contract": {"sha256": C9_SCHEMA_SHA256},
                           "policy": {"id": "protocol-test", "approved": True,
                                      "evidence": "fabricated protocol test only", "sha256": "1" * 64}},
            "findings": [{"check_id": name, "status": "PASS", "episode": 0,
                          "observed": "actual" if name == "C10_REAL_EPISODE" else None}
                         for name in sorted(REQUIRED_CHECKS)],
            "episodes": [{"episode_index": 0, "status": "PASS", "data_origin": "actual",
                          "binding": {"dataset_id": "protocol-test", "dataset_revision": "v1",
                                      "episode_id": "e0", "episode_index": 0}}]}


class C10HandoffTests(unittest.TestCase):
    def decision(self, report):
        return training_readiness(report, dataset_id="protocol-test", dataset_revision="v1", episode_indices=[0])

    def test_complete_protocol_envelope_is_consumable(self):
        report = protocol_envelope()
        before = deepcopy(report)
        self.assertTrue(self.decision(report)["training_ready"])
        self.assertEqual(report, before)
        self.assertEqual(report["production_acceptance"], "NOT_VERIFIED")

    def test_blocked_gate_cannot_be_promoted(self):
        for status, code in (("REJECT", 1), ("NOT_VERIFIED", 2), ("BLOCKED", 2)):
            with self.subTest(status=status):
                report = protocol_envelope()
                report.update(status=status, exit_code=code, training_ready=False)
                decision = self.decision(report)
                self.assertFalse(decision["training_ready"])
                self.assertNotEqual(decision["status"], "PASS")

    def test_top_level_pass_cannot_hide_blocked_findings(self):
        for status in ("REJECT", "NOT_VERIFIED", "BLOCKED", None):
            with self.subTest(status=status):
                report = protocol_envelope()
                report["findings"][0]["status"] = status
                self.assertFalse(self.decision(report)["training_ready"])

    def test_synthetic_origin_cannot_be_relabelled_by_real_evidence_flag(self):
        report = protocol_envelope()
        report["episodes"][0]["data_origin"] = "synthetic"
        self.assertFalse(self.decision(report)["training_ready"])
        report["episodes"][0]["data_origin"] = "actual"
        next(f for f in report["findings"] if f["check_id"] == "C10_REAL_EPISODE")["observed"] = "synthetic"
        self.assertFalse(self.decision(report)["training_ready"])

    def test_registry_pass_is_not_episode_evidence(self):
        registry = {"dataset_id": "protocol-test", "quality_status": "passed"}
        self.assertEqual(validate_registry(registry, {"type": "object"}), (1, []))
        self.assertFalse(self.decision(registry)["training_ready"])

    def test_empty_or_partial_diagnostics_are_blocked(self):
        report = protocol_envelope()
        report["findings"] = []
        self.assertFalse(self.decision(report)["training_ready"])
        report["findings"] = protocol_envelope()["findings"][1:]
        self.assertFalse(self.decision(report)["training_ready"])

    def test_missing_calibration_or_release_evidence_is_blocked(self):
        for name in ("C10_CALIBRATION", "C10_RELEASE", "C10_REAL_EPISODE"):
            report = protocol_envelope()
            report["findings"] = [f for f in report["findings"] if f["check_id"] != name]
            self.assertFalse(self.decision(report)["training_ready"])

    def test_wrong_snapshot_or_selection_is_blocked(self):
        for field, value in (("dataset_revision", "stale"), ("dataset_id", "other"), ("episode_index", 1)):
            report = protocol_envelope()
            report["episodes"][0]["binding"][field] = value
            self.assertFalse(self.decision(report)["training_ready"])
        report = protocol_envelope()
        report["episodes"].append(deepcopy(report["episodes"][0]))
        self.assertFalse(self.decision(report)["training_ready"])

    def test_no_expected_episodes_is_blocked(self):
        self.assertFalse(training_readiness(protocol_envelope(), dataset_id="protocol-test",
                                          dataset_revision="v1", episode_indices=[])["training_ready"])

    def test_unreviewed_policy_and_wrong_contract_are_blocked(self):
        for field in ("policy", "c9_contract"):
            report = protocol_envelope()
            report["provenance"][field] = None
            self.assertFalse(self.decision(report)["training_ready"])

    def test_exit_types_and_contradictions_are_blocked(self):
        for value in (False, "0", 2, None):
            report = protocol_envelope()
            report["exit_code"] = value
            self.assertFalse(self.decision(report)["training_ready"])

    def test_malformed_evidence_is_blocked(self):
        for report in (None, [], "PASS", {}, {"findings": 1}, {"findings": [None]},
                       {"episodes": [None], "findings": {}}, {"provenance": []}):
            with self.subTest(report=report):
                self.assertFalse(self.decision(report)["training_ready"])

    def test_missing_dataset_cannot_open_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = check_dataset(Path(temporary) / "missing")
            self.assertEqual(report["status"], "NOT_VERIFIED")
            self.assertFalse(self.decision(report)["training_ready"])

    def test_different_schema_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            schema = Path(temporary) / "schema.json"
            schema.write_text('{}')
            with self.assertRaisesRegex(ValueError, "differs from the pinned"):
                load_c9_schema(schema)

    def test_report_cannot_overwrite_dataset_or_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy = root / "policy.json"
            policy.write_text('{}')
            with self.assertRaises(SystemExit) as error:
                main(["--dataset-path", str(root), "--report", str(policy)])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(policy.read_text(), '{}')


if __name__ == "__main__":
    unittest.main()
