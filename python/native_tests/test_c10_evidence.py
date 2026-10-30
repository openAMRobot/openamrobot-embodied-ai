# SPDX-License-Identifier: MIT
"""C10 evidence handoff integration; native Parquet dependency is explicit."""

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pyarrow  # Native evidence tests fail visibly if their dependency is absent.
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/tools"))
sys.path.insert(0, str(ROOT / "python/tests"))
import generate_c10_evidence as evidence
from c10_fixture import fixture, write_dataset


def file_hashes(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*")) if path.is_file()}


class C10EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.parent = Path(self.temporary.name)
        self.output = self.parent / "evidence"

    def invoke_cli(self, output=None, *extra):
        return subprocess.run([
            sys.executable, str(ROOT / "python/tools/generate_c10_evidence.py"),
            "--output-dir", str(output or self.output), *extra],
            capture_output=True, text=True, timeout=60)

    def manifest(self, output=None):
        return json.loads(((output or self.output) / "evidence.json").read_text())

    def test_actual_cli_keeps_not_verified_report_and_exact_provenance(self):
        process = self.invoke_cli()
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        report_path = self.output / "c10-report.json"
        raw_report = report_path.read_bytes()
        report = json.loads(raw_report)
        self.assertEqual(report["status"], "NOT_VERIFIED")
        self.assertEqual(report["exit_code"], 2)
        self.assertFalse(report["training_ready"])
        manifest = self.manifest()
        self.assertEqual(manifest["report_sha256"], hashlib.sha256(raw_report).hexdigest())
        self.assertEqual(manifest["expected_synthetic_result"], evidence.EXPECTED_SYNTHETIC_RESULT)
        self.assertTrue(manifest["reporting_path_success"])
        self.assertEqual(manifest["helper_exit_code"], 0)
        self.assertEqual(manifest["errors"], [])
        self.assertEqual(manifest["gate"]["process_exit_code"], 2)
        self.assertEqual(manifest["gate"]["report_exit_code"], 2)
        self.assertEqual(manifest["gate"]["status"], "NOT_VERIFIED")
        self.assertFalse(manifest["gate"]["training_ready"])
        self.assertEqual(manifest["gate"]["production_acceptance"], "NOT_VERIFIED")
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(manifest["checker"]["git_head"], head)
        self.assertIsNone(manifest["checker"]["expected_checker_sha"])
        self.assertFalse(manifest["checker"]["immutable_checkout_verified"])
        argv = manifest["invocation"]["argv"]
        self.assertEqual(Path(argv[0]).resolve(), Path(sys.executable).resolve())
        self.assertEqual(argv[1:3], ["-m", "oa_embodied_ai.dataset_qa.episode"])
        self.assertEqual(manifest["invocation"]["shell_command"], shlex.join(argv))
        self.assertEqual(Path(manifest["invocation"]["cwd"]).resolve(), ROOT)
        for option, path in (("--dataset-path", self.output / "dataset"),
                             ("--report", report_path),
                             ("--policy", self.output / "dataset/synthetic-policy.json")):
            self.assertEqual(Path(argv[argv.index(option) + 1]).resolve(), path.resolve())
        self.assertIn("C10 NOT_VERIFIED", (self.output / "cli-stdout.log").read_text())
        self.assertTrue((self.output / "cli-stderr.log").is_file())
        self.assertTrue(manifest["source_files_sha256"])
        self.assertEqual(manifest["source_files_sha256_after"], manifest["source_files_sha256"])
        for relative, digest in manifest["source_files_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest, relative)
        actual_hashes = file_hashes(self.output / "dataset")
        self.assertEqual(manifest["dataset_files_sha256_before"], actual_hashes)
        self.assertEqual(manifest["dataset_files_sha256_after"], actual_hashes)
        self.assertTrue(manifest["dataset_unchanged"])
        sidecar, native, policy = fixture()
        for key in ("dataset_id", "dataset_revision", "episode_id", "episode_index"):
            self.assertEqual(manifest["dataset"][key], sidecar["native_lerobot"][key])
        self.assertEqual(manifest["dataset"]["data_origin"], "synthetic")
        self.assertEqual(Path(manifest["dataset"]["path"]).resolve(), (self.output / "dataset").resolve())
        self.assertEqual(manifest["environment"]["python_executable"], sys.executable)
        self.assertEqual(manifest["environment"]["packages"]["pyarrow"], pyarrow.__version__)
        independent = self.parent / "independent-fixture"
        write_dataset(independent, sidecar, native, policy)
        self.assertEqual(file_hashes(independent), actual_hashes)

    def test_report_bytes_reproduce_across_independent_output_directories(self):
        first = self.invoke_cli(self.output)
        second_dir = self.parent / "another-output"
        second = self.invoke_cli(second_dir)
        self.assertEqual((first.returncode, second.returncode), (0, 0), first.stderr + second.stderr)
        self.assertEqual((self.output / "c10-report.json").read_bytes(),
                         (second_dir / "c10-report.json").read_bytes())
        self.assertEqual(self.manifest()["report_sha256"], self.manifest(second_dir)["report_sha256"])
        self.assertEqual(file_hashes(self.output / "dataset"), file_hashes(second_dir / "dataset"))

    def test_preexisting_output_is_refused_without_changing_any_bytes(self):
        self.output.mkdir()
        (self.output / "sentinel").write_bytes(b"existing evidence must remain intact\x00")
        nested = self.output / "nested"
        nested.mkdir()
        (nested / "raw-report.json").write_bytes(b'{"untouched":true}\n')
        before = file_hashes(self.output)
        process = self.invoke_cli()
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("exist", (process.stdout + process.stderr).lower())
        self.assertEqual(before, file_hashes(self.output))
        self.assertFalse((self.output / "evidence.json").exists())

    def test_existing_empty_output_is_also_refused(self):
        self.output.mkdir()
        process = self.invoke_cli()
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("exist", (process.stdout + process.stderr).lower())
        self.assertEqual(list(self.output.iterdir()), [])

    def simulate_checker(self, report_bytes, return_code):
        def run(argv, cwd, stdout_path, stderr_path):
            Path(stdout_path).write_bytes(b"raw synthetic gate stdout\n")
            Path(stderr_path).write_bytes(b"raw synthetic gate stderr\n")
            if report_bytes is not None:
                Path(argv[argv.index("--report") + 1]).write_bytes(report_bytes)
            return return_code
        return run

    def raw_gate_report(self, **overrides):
        report = {
            "status": "NOT_VERIFIED", "exit_code": 2, "training_ready": False,
            "production_acceptance": "NOT_VERIFIED",
            "findings": [{"check_id": "C10_REAL_EPISODE", "status": "NOT_VERIFIED", "observed": "synthetic"}],
        }
        report.update(overrides)
        return json.dumps(report, sort_keys=True, indent=1).encode() + b"\n"

    def check_failed_gate_artifacts(self, report_bytes, return_code, error_term):
        with patch.object(evidence, "run_checker", side_effect=self.simulate_checker(report_bytes, return_code)):
            helper_exit = evidence.generate_evidence(self.output)
        self.assertNotEqual(helper_exit, 0)
        manifest = self.manifest()
        self.assertFalse(manifest["reporting_path_success"])
        self.assertEqual(manifest["expected_synthetic_result"], evidence.EXPECTED_SYNTHETIC_RESULT)
        self.assertFalse(manifest["checker"]["immutable_checkout_verified"])
        self.assertEqual(manifest["gate"]["process_exit_code"], return_code)
        self.assertEqual(manifest["helper_exit_code"], helper_exit)
        self.assertIn(error_term, " ".join(manifest["errors"]).lower())
        self.assertEqual((self.output / "cli-stdout.log").read_bytes(), b"raw synthetic gate stdout\n")
        self.assertEqual((self.output / "cli-stderr.log").read_bytes(), b"raw synthetic gate stderr\n")
        report_path = self.output / "c10-report.json"
        if report_bytes is None:
            self.assertFalse(report_path.exists())
            self.assertIsNone(manifest["report_sha256"])
        else:
            self.assertEqual(report_path.read_bytes(), report_bytes)
            self.assertEqual(manifest["report_sha256"], hashlib.sha256(report_bytes).hexdigest())
        self.assertTrue(manifest["dataset_unchanged"])

    def test_rejects_unexpected_gate_pass(self):
        raw = self.raw_gate_report(status="PASS", exit_code=0, training_ready=True,
                                   production_acceptance="PASS")
        self.check_failed_gate_artifacts(raw, 0, "expected")

    def test_rejects_unexpected_gate_reject(self):
        raw = self.raw_gate_report(status="REJECT", exit_code=1, production_acceptance="REJECT")
        self.check_failed_gate_artifacts(raw, 1, "expected")

    def test_rejects_nonmatching_gate_process_exit(self):
        raw = self.raw_gate_report()
        self.check_failed_gate_artifacts(raw, 17, "expected")

    def test_rejects_gate_report_with_wrong_report_exit(self):
        raw = self.raw_gate_report(exit_code=0)
        self.check_failed_gate_artifacts(raw, 2, "expected")

    def test_rejects_noninteger_report_exit_even_when_numerically_two(self):
        self.check_failed_gate_artifacts(self.raw_gate_report(exit_code=2.0), 2, "expected")

    def test_rejects_report_without_explicit_false_readiness(self):
        report = json.loads(self.raw_gate_report())
        del report["training_ready"]
        self.check_failed_gate_artifacts(json.dumps(report).encode(), 2, "expected")

    def test_rejects_report_without_synthetic_origin_finding(self):
        self.check_failed_gate_artifacts(self.raw_gate_report(findings=[]), 2, "expected")

    def test_rejects_absent_report_and_preserves_logs(self):
        self.check_failed_gate_artifacts(None, 2, "report")

    def test_rejects_malformed_report_and_preserves_exact_bytes(self):
        self.check_failed_gate_artifacts(b'{"status": "NOT_VERIFIED", BROKEN\n', 2, "jsondecodeerror")

    def test_checker_sha_mismatch_never_claims_immutable_checkout(self):
        process = self.invoke_cli(None, "--expected-checker-sha", "0" * 40)
        self.assertNotEqual(process.returncode, 0)
        manifest = self.manifest()
        self.assertFalse(manifest["checker"]["immutable_checkout_verified"])
        self.assertEqual(manifest["checker"]["expected_checker_sha"], "0" * 40)
        self.assertFalse(manifest["reporting_path_success"])
        self.assertTrue(any("sha" in str(error).lower() for error in manifest["errors"]))

    def test_matching_sha_with_mocked_clean_identity_records_verification(self):
        mocked_head = "a" * 40
        with patch.object(evidence, "git_state", return_value=(mocked_head, True)):
            result = evidence.generate_evidence(self.output, mocked_head)
        self.assertEqual(result, 0)
        manifest = self.manifest()
        self.assertEqual(manifest["checker"]["git_head"], mocked_head)
        self.assertEqual(manifest["checker"]["git_head_after"], mocked_head)
        self.assertTrue(manifest["checker"]["immutable_checkout_verified"])
        self.assertTrue(manifest["reporting_path_success"])
        self.assertEqual(manifest["gate"]["process_exit_code"], 2)

    def test_matching_sha_with_mocked_dirty_tree_is_refused(self):
        mocked_head = "a" * 40
        with patch.object(evidence, "git_state", return_value=(mocked_head, False)):
            result = evidence.generate_evidence(self.output, mocked_head)
        self.assertEqual(result, 1)
        manifest = self.manifest()
        self.assertFalse(manifest["checker"]["immutable_checkout_verified"])
        self.assertFalse(manifest["reporting_path_success"])
        self.assertTrue(any("clean" in error.lower() for error in manifest["errors"]))
        self.assertIsNone(manifest["invocation"])
        self.assertFalse((self.output / "dataset").exists())

    def test_checker_mutating_generated_dataset_fails_and_retains_raw_report(self):
        real_run = evidence.run_checker
        def mutate_after_gate(argv, cwd, stdout_path, stderr_path):
            result = real_run(argv, cwd, stdout_path, stderr_path)
            dataset = Path(argv[argv.index("--dataset-path") + 1])
            (dataset / "unexpected-mutation.bin").write_bytes(b"deliberate read-only contract violation")
            return result
        with patch.object(evidence, "run_checker", side_effect=mutate_after_gate):
            result = evidence.generate_evidence(self.output)
        self.assertEqual(result, 1)
        manifest = self.manifest()
        self.assertFalse(manifest["dataset_unchanged"])
        self.assertFalse(manifest["reporting_path_success"])
        self.assertIn("unexpected-mutation.bin", manifest["dataset_files_sha256_after"])
        self.assertNotIn("unexpected-mutation.bin", manifest["dataset_files_sha256_before"])
        self.assertTrue(any("dataset" in error.lower() and ("changed" in error.lower() or "differs" in error.lower())
                            for error in manifest["errors"]))
        report = (self.output / "c10-report.json").read_bytes()
        self.assertEqual(json.loads(report)["status"], "NOT_VERIFIED")
        self.assertEqual(manifest["report_sha256"], hashlib.sha256(report).hexdigest())

    def test_head_change_during_generation_cannot_claim_immutable_checkout(self):
        before, after = "a" * 40, "b" * 40
        with patch.object(evidence, "git_state", side_effect=[(before, True), (after, True)]):
            result = evidence.generate_evidence(self.output, before)
        self.assertEqual(result, 1)
        manifest = self.manifest()
        self.assertFalse(manifest["checker"]["immutable_checkout_verified"])
        self.assertEqual(manifest["checker"]["git_head"], before)
        self.assertEqual(manifest["checker"]["git_head_after"], after)
        self.assertFalse(manifest["reporting_path_success"])
        self.assertTrue(any("source" in error.lower() and "changed" in error.lower() for error in manifest["errors"]))

    def test_source_bytes_drift_with_unchanged_head_invalidates_evidence(self):
        head = "a" * 40
        relative = evidence.SOURCE_FILES[0]
        source = (ROOT / relative).resolve()
        real_hash = evidence.sha256_file
        reads = 0
        def drifted_hash(path):
            nonlocal reads
            if Path(path).resolve() == source:
                reads += 1
                if reads > 1:
                    return "f" * 64
            return real_hash(path)
        with patch.object(evidence, "git_state", return_value=(head, True)), patch.object(
                evidence, "sha256_file", side_effect=drifted_hash):
            result = evidence.generate_evidence(self.output, head)
        self.assertEqual(result, 1)
        manifest = self.manifest()
        self.assertEqual(manifest["checker"]["git_head"], manifest["checker"]["git_head_after"])
        self.assertEqual(manifest["source_files_sha256"][relative], real_hash(source))
        self.assertEqual(manifest["source_files_sha256_after"][relative], "f" * 64)
        self.assertFalse(manifest["checker"]["immutable_checkout_verified"])
        self.assertFalse(manifest["reporting_path_success"])
        self.assertTrue(any("source" in error.lower() and "changed" in error.lower() for error in manifest["errors"]))
        self.assertEqual(json.loads((self.output / "c10-report.json").read_bytes())["status"], "NOT_VERIFIED")
        self.assertIn("C10 NOT_VERIFIED", (self.output / "cli-stdout.log").read_text())

    def test_clean_checkout_becoming_dirty_during_generation_is_refused(self):
        head = "a" * 40
        with patch.object(evidence, "git_state", side_effect=[(head, True), (head, False)]):
            result = evidence.generate_evidence(self.output, head)
        self.assertEqual(result, 1)
        manifest = self.manifest()
        self.assertTrue(manifest["checker"]["source_tree_clean"])
        self.assertFalse(manifest["checker"]["source_tree_clean_after"])
        self.assertFalse(manifest["checker"]["immutable_checkout_verified"])
        self.assertFalse(manifest["reporting_path_success"])
        self.assertTrue(any("dirty" in error.lower() for error in manifest["errors"]))


class C10EvidenceWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workflow = yaml.safe_load((ROOT / ".github/workflows/c10-dataset-quality.yml").read_text())
        cls.job = workflow["jobs"]["c10-reference"]
        cls.steps = cls.job["steps"]

    def test_upload_runs_after_failures_and_retains_hidden_evidence(self):
        upload = [step for step in self.steps if step.get("uses", "").startswith("actions/upload-artifact@")]
        self.assertEqual(len(upload), 1)
        upload = upload[0]
        self.assertEqual(upload["if"], "${{ always() }}")
        self.assertTrue(upload["with"]["include-hidden-files"])
        self.assertEqual(upload["with"]["if-no-files-found"], "error")
        self.assertEqual(upload["with"]["path"].rstrip("/"), self.job["env"]["C10_EVIDENCE_DIR"])
        self.assertGreaterEqual(upload["with"]["retention-days"], 1)
        names = [step.get("name") for step in self.steps]
        self.assertGreater(names.index(upload["name"]), names.index("Generate reproducible synthetic CLI evidence"))
        self.assertLess(names.index("Prepare evidence directory"), names.index("Install scoped C10 verification dependencies"))

    def test_checkout_and_generator_bind_the_same_explicit_checker_sha(self):
        checkout = next(step for step in self.steps if step.get("uses", "").startswith("actions/checkout@"))
        self.assertEqual(self.job["env"]["C10_CHECKER_SHA"], "${{ github.event.pull_request.head.sha || github.sha }}")
        self.assertEqual(checkout["with"]["ref"], self.job["env"]["C10_CHECKER_SHA"])
        generator = next(step for step in self.steps if step.get("name") == "Generate reproducible synthetic CLI evidence")
        self.assertIn('--expected-checker-sha "$C10_CHECKER_SHA"', generator["run"])
        self.assertEqual(generator["if"], "${{ !cancelled() }}")

    def test_each_captured_command_preserves_failure_status_through_tee(self):
        suites = {
            "Run nonempty C10 core suite": ("test_c10_dataset.py", "core-suite.log"),
            "Run nonempty C10 native suite": ("test_c10_native.py", "native-suite.log"),
            "Run nonempty C10 evidence suite": ("test_c10_evidence.py", "evidence-suite.log"),
            "Run nonempty C10 handoff suite": ("test_c10_handoff.py", "handoff-suite.log"),
            "Prove reverted arithmetic check is detected": ("python/tools/prove_c10_negative.py", "revert-negative.log"),
            "Generate reproducible synthetic CLI evidence": ("python/tools/generate_c10_evidence.py", "generation.log"),
        }
        self.assertEqual(self.job["defaults"]["run"]["shell"], "bash")
        for name, (pattern, log_name) in suites.items():
            with self.subTest(suite=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                binaries = root / "bin"
                binaries.mkdir()
                python = binaries / "python"
                python.write_text("#!/bin/sh\nprintf '%s\\n' 'injected suite failure output'\nexit 23\n")
                python.chmod(0o755)
                output = root / "captured"
                output.mkdir()
                step = next(step for step in self.steps if step.get("name") == name)
                self.assertEqual(step["if"], "${{ !cancelled() }}")
                self.assertIn(pattern, step["run"])
                self.assertIn("set -o pipefail", step["run"])
                environment = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"],
                                   C10_EVIDENCE_DIR=str(output))
                process = subprocess.run(["bash", "-e", "-c", step["run"]], cwd=root,
                                         env=environment, capture_output=True, text=True)
                self.assertEqual(process.returncode, 23, process.stdout + process.stderr)
                self.assertEqual((output / log_name).read_text(), "injected suite failure output\n")


if __name__ == "__main__":
    unittest.main()
