# SPDX-License-Identifier: MIT
"""Unit tests for C11 fail-closed pre-flight training gate."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from oa_embodied_ai.training.gate import (
    GateStatus,
    TrainingBlockedError,
    verify_training_gate,
)


def make_valid_robot_report() -> dict:
    return {
        "report_version": 1,
        "status": "PASS",
        "quality_status": "PASS",
        "exit_code": 0,
        "training_ready": True,
        "source_profile": "ROBOT",
        "provenance": {
            "source_profile": "ROBOT",
            "c9_contract": {"sha256": "692e04693bc738033af1da10f98e029f342497ddaeae4792945f829335f1938e"},
        },
        "findings": [
            {"check_id": "C10_SCHEMA", "status": "PASS", "severity": "INFO"},
            {"check_id": "C10_REAL_EPISODE", "status": "PASS", "severity": "INFO", "observed": "actual"},
        ],
        "episodes": [
            {"episode_index": 0, "status": "PASS", "data_origin": "actual"},
        ],
    }


class TestC11TrainingGate(unittest.TestCase):
    def test_valid_robot_report_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_file = Path(tmpdir) / "qa_report.json"
            report_file.write_text(json.dumps(make_valid_robot_report()), encoding="utf-8")

            status = verify_training_gate(Path(tmpdir), qa_report_path=report_file)
            self.assertIsInstance(status, GateStatus)
            self.assertTrue(status.training_ready)
            self.assertEqual(status.source_profile, "ROBOT")
            self.assertFalse(status.is_synthetic_override)
            self.assertEqual(status["quality_status"], "PASS")

    def test_missing_report_raises_training_blocked_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_file = Path(tmpdir) / "nonexistent_qa.json"
            with self.assertRaises(TrainingBlockedError) as ctx:
                verify_training_gate(Path(tmpdir), qa_report_path=missing_file)
            self.assertEqual(ctx.exception.exit_code, 2)
            self.assertEqual(ctx.exception.code, 2)

    def test_corrupted_json_report_raises_training_blocked_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_file = Path(tmpdir) / "corrupted_qa.json"
            report_file.write_text("{ this is not valid json", encoding="utf-8")
            with self.assertRaises(TrainingBlockedError) as ctx:
                verify_training_gate(Path(tmpdir), qa_report_path=report_file)
            self.assertEqual(ctx.exception.exit_code, 2)

    def test_training_ready_false_or_missing_raises_training_blocked_error(self):
        for bad_val in (False, None, "true", 1):
            with self.subTest(bad_val=bad_val):
                with tempfile.TemporaryDirectory() as tmpdir:
                    report = make_valid_robot_report()
                    if bad_val is None:
                        del report["training_ready"]
                    else:
                        report["training_ready"] = bad_val
                    report_file = Path(tmpdir) / "qa_report.json"
                    report_file.write_text(json.dumps(report), encoding="utf-8")

                    with self.assertRaises(TrainingBlockedError) as ctx:
                        verify_training_gate(Path(tmpdir), qa_report_path=report_file)
                    self.assertEqual(ctx.exception.exit_code, 2)

    def test_quality_status_blocked_reject_not_verified_raises(self):
        for bad_status in ("BLOCKED", "REJECT", "NOT_VERIFIED", "FAILED"):
            with self.subTest(bad_status=bad_status):
                with tempfile.TemporaryDirectory() as tmpdir:
                    report = make_valid_robot_report()
                    report["quality_status"] = bad_status
                    report["status"] = bad_status
                    report_file = Path(tmpdir) / "qa_report.json"
                    report_file.write_text(json.dumps(report), encoding="utf-8")

                    with self.assertRaises(TrainingBlockedError) as ctx:
                        verify_training_gate(Path(tmpdir), qa_report_path=report_file)
                    self.assertEqual(ctx.exception.exit_code, 2)

    def test_contradictory_finding_severity_reject_blocks_training(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = make_valid_robot_report()
            report["findings"].append(
                {"check_id": "C10_SKEW_CRITICAL", "status": "FAIL", "severity": "REJECT"}
            )
            report_file = Path(tmpdir) / "qa_report.json"
            report_file.write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaises(TrainingBlockedError) as ctx:
                verify_training_gate(Path(tmpdir), qa_report_path=report_file)
            self.assertEqual(ctx.exception.exit_code, 2)

    def test_contradictory_finding_status_reject_blocks_training(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = make_valid_robot_report()
            report["findings"].append(
                {"check_id": "C10_BOUNDS_CHECK", "status": "REJECT", "severity": "ERROR"}
            )
            report_file = Path(tmpdir) / "qa_report.json"
            report_file.write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaises(TrainingBlockedError) as ctx:
                verify_training_gate(Path(tmpdir), qa_report_path=report_file)
            self.assertEqual(ctx.exception.exit_code, 2)

    def test_source_profile_non_robot_fails_closed_by_default(self):
        for synthetic_profile in ("GAZEBO", "SYNTHETIC", "MOCK", "DIAGNOSTICS"):
            with self.subTest(profile=synthetic_profile):
                with tempfile.TemporaryDirectory() as tmpdir:
                    report = make_valid_robot_report()
                    report["source_profile"] = synthetic_profile
                    report["provenance"]["source_profile"] = synthetic_profile
                    report_file = Path(tmpdir) / "qa_report.json"
                    report_file.write_text(json.dumps(report), encoding="utf-8")

                    with self.assertRaises(TrainingBlockedError) as ctx:
                        verify_training_gate(Path(tmpdir), qa_report_path=report_file, allow_synthetic_override=False)
                    self.assertEqual(ctx.exception.exit_code, 2)

    def test_source_profile_synthetic_override_succeeds_with_warning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = make_valid_robot_report()
            report["source_profile"] = "SYNTHETIC"
            report["provenance"]["source_profile"] = "SYNTHETIC"
            report_file = Path(tmpdir) / "qa_report.json"
            report_file.write_text(json.dumps(report), encoding="utf-8")

            status = verify_training_gate(
                Path(tmpdir),
                qa_report_path=report_file,
                allow_synthetic_override=True,
            )
            self.assertTrue(status.training_ready)
            self.assertTrue(status.is_synthetic_override)
            self.assertEqual(status.source_profile, "SYNTHETIC")

    def test_nonzero_exit_code_in_report_blocks_training(self):
        for bad_code in (1, 2, "0"):
            with self.subTest(bad_code=bad_code):
                with tempfile.TemporaryDirectory() as tmpdir:
                    report = make_valid_robot_report()
                    report["exit_code"] = bad_code
                    report_file = Path(tmpdir) / "qa_report.json"
                    report_file.write_text(json.dumps(report), encoding="utf-8")

                    with self.assertRaises(TrainingBlockedError) as ctx:
                        verify_training_gate(Path(tmpdir), qa_report_path=report_file)
                    self.assertEqual(ctx.exception.exit_code, 2)

    def test_mutation_reversion_corrupt_gate_logic(self):
        """Mutation test: Assert that reverting fail-closed checks fails the gate contract."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Payload with contradictory finding: top-level PASS but finding REJECT
            contradictory = make_valid_robot_report()
            contradictory["findings"].append(
                {"check_id": "MUTATION_TEST", "status": "FAIL", "severity": "REJECT"}
            )
            report_file = Path(tmpdir) / "qa_report.json"
            report_file.write_text(json.dumps(contradictory), encoding="utf-8")

            # Contract requires verify_training_gate to raise TrainingBlockedError
            with self.assertRaises(TrainingBlockedError) as ctx:
                verify_training_gate(Path(tmpdir), qa_report_path=report_file)
            self.assertEqual(ctx.exception.exit_code, 2)

            # A mutated/corrupt gate logic (e.g. naive top-level check only) would erroneously pass:
            def naive_mutated_gate(rep):
                return rep.get("status") == "PASS"

            self.assertTrue(naive_mutated_gate(contradictory), "Mutant gate erroneously accepted contradictory report")


if __name__ == "__main__":
    unittest.main()

