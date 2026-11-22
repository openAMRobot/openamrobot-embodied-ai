# SPDX-License-Identifier: MIT
"""Explicit native-Parquet C10 integration suite; requires pyarrow (no skips)."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

# This suite intentionally fails import when the optional native dependency is absent.
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/tools"))
sys.path.insert(0, str(ROOT / "python/tests"))
from c10_fixture import CAMERAS, fixture, write_dataset
import test_c10_dataset as semantics
from oa_embodied_ai.dataset_qa.episode import check_dataset

WRAPPER_CASES = {
    "orphan_native_episode_rows": ("C10_NATIVE_DATA", "REJECT"),
    "duplicate_json_key": ("C10_INPUT", "REJECT"),
    "invalid_c9_schema": ("C10_SCHEMA", "REJECT"),
    "duplicate_native_episode": ("C10_NATIVE_METADATA", "REJECT"),
    "native_row_range": ("C10_NATIVE_DATA", "REJECT"),
    "native_double_vectors": ("C10_PARQUET_DTYPE", "REJECT"),
    "decoded_video_frame_count": ("C10_VIDEO", "REJECT"),
    "artifact_bytes_changed": ("C10_ARTIFACT", "REJECT"),
    "calibration_identity": ("C10_CALIBRATION", "REJECT"),
    "recording_release_commit": ("C10_RELEASE", "REJECT"),
    "synthetic_is_not_real_evidence": ("C10_REAL_EPISODE", "NOT_VERIFIED"),
    "frozen_decoded_pixels": ("C10_VIDEO", "REJECT"),
    "configuration_declared_hash": ("C10_ARTIFACT", "REJECT"),
    "absent_configuration_artifact": ("C10_ARTIFACT", "NOT_VERIFIED"),
    "configuration_symlink_escape": ("C10_ARTIFACT", "REJECT"),
    "malformed_nested_policy": ("C10_INPUT", "REJECT"),
    "truncated_parquet": ("C10_NATIVE_DATA", "REJECT"),
    "unapproved_calibration": ("C10_CALIBRATION", "NOT_VERIFIED"),
    "unapproved_release": ("C10_RELEASE", "NOT_VERIFIED"),
    "source_profile_identity": ("C10_PROFILE", "REJECT"),
    "calibration_residual_exceeds": ("C10_CALIBRATION", "REJECT"),
}


class C10WrapperTests(unittest.TestCase):
    setUpClass = classmethod(semantics.C10SemanticTests.setUpClass.__func__)
    assert_c9_valid = semantics.C10SemanticTests.assert_c9_valid
    assert_finding = semantics.C10SemanticTests.assert_finding

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "dataset"
        self.sidecar, self.native, self.policy = fixture()

    def write(self):
        self.assert_c9_valid(self.sidecar)
        write_dataset(self.path, self.sidecar, self.native, self.policy)

    def run_gate(self):
        from oa_embodied_ai.dataset_qa.episode import check_dataset
        return check_dataset(self.path, self.path / "synthetic-policy.json")

    def test_synthetic_native_parquet_is_not_training_ready(self):
        self.write()
        report = self.run_gate()
        self.assertEqual(report["status"], "NOT_VERIFIED", report)
        self.assertEqual(report["exit_code"], 2)
        self.assertFalse(report["training_ready"])
        for check_id in ("C10_NATIVE_METADATA", "C10_NATIVE_DATA", "C10_PARQUET_DTYPE"):
            self.assert_finding(report["findings"], check_id, "PASS")
        self.assert_finding(report["findings"], "C10_VIDEO", "NOT_VERIFIED")
        self.assert_finding(report["findings"], "C10_REAL_EPISODE", "NOT_VERIFIED")

    def test_native_missing_parquet_is_not_verified(self):
        self.write()
        (self.path / "data/chunk-000/file-000.parquet").unlink()
        self.assert_finding(self.run_gate()["findings"], "C10_NATIVE_DATA", "NOT_VERIFIED")

    def test_cli_report_exit_and_read_only_dataset(self):
        self.write()
        before = {str(p.relative_to(self.path)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in self.path.rglob("*") if p.is_file()}
        report_path = Path(self.temporary.name) / "report.json"
        process = subprocess.run([
            sys.executable, "-m", "oa_embodied_ai.dataset_qa.episode",
            "--dataset-path", str(self.path), "--policy", str(self.path / "synthetic-policy.json"),
            "--report", str(report_path)], capture_output=True, text=True)
        self.assertEqual(process.returncode, 2, process.stdout + process.stderr)
        report = json.loads(report_path.read_text())
        self.assertEqual(report["exit_code"], process.returncode)
        self.assertEqual(report["status"], "NOT_VERIFIED")
        after = {str(p.relative_to(self.path)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in self.path.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        json.dumps(report, allow_nan=False)
        required = {"check_id", "severity", "status", "episode", "frame", "stream",
                    "observed", "expected", "units", "threshold", "policy", "fix"}
        self.assertTrue(report["findings"])
        for finding in report["findings"]:
            self.assertTrue(required <= set(finding), finding)

    def test_selection_rejects_an_unavailable_episode(self):
        from oa_embodied_ai.dataset_qa.episode import check_dataset
        self.write()
        report = check_dataset(self.path, self.path / "synthetic-policy.json", [999])
        self.assertEqual(report["status"], "REJECT")
        self.assert_finding(report["findings"], "C10_NATIVE_METADATA")

    def test_native_metadata_totals_cannot_contradict_episode_registry(self):
        self.native["info"]["total_frames"] = 99
        self.write()
        self.assert_finding(self.run_gate()["findings"],"C10_NATIVE_METADATA")

    def test_missing_dataset_is_environmental_not_verified(self):
        report = check_dataset(self.path)
        self.assertEqual(report["exit_code"],2)
        self.assert_finding(report["findings"],"C10_INPUT","NOT_VERIFIED")

    def test_nonfinite_parquet_values_produce_strict_json_rejection(self):
        self.native["rows"][0]["action"][0] = float("nan")
        self.write()
        report = self.run_gate()
        self.assert_finding(report["findings"], "C10_VECTOR_VALUES")
        self.assertEqual(report["exit_code"], 1)
        json.dumps(report, allow_nan=False)

    def test_actual_claim_without_external_review_stays_not_verified(self):
        self.sidecar["openamrobot"]["provenance"]["data_origin"] = "actual"
        self.write()
        self.assert_finding(self.run_gate()["findings"], "C10_REAL_EPISODE", "NOT_VERIFIED")

    def test_synthetic_origin_is_not_overridden_by_real_episode_policy(self):
        self.policy["real_episode"] = {key: self.sidecar["native_lerobot"][key] for key in
                                       ("dataset_id", "dataset_revision", "episode_id", "episode_index")}
        self.policy["real_episode"]["evidence"] = "Fabricated test assertion; never actual recording evidence"
        self.write()
        report = self.run_gate()
        self.assert_finding(report["findings"], "C10_REAL_EPISODE", "NOT_VERIFIED")
        self.assertFalse(report["training_ready"])

    def test_promoted_synthetic_report_still_blocks_consumer(self):
        from oa_embodied_ai.dataset_qa import training_readiness
        self.write()
        report = self.run_gate()
        report.update(status="PASS", exit_code=0, training_ready=True)
        binding = self.sidecar["native_lerobot"]
        decision = training_readiness(report, dataset_id=binding["dataset_id"],
                                      dataset_revision=binding["dataset_revision"], episode_indices=[0])
        self.assertFalse(decision["training_ready"])
        self.assertEqual(decision["status"], "NOT_VERIFIED")

    def test_report_binds_exact_policy_and_sidecar_bytes(self):
        self.write()
        report = self.run_gate()
        policy = self.path / "synthetic-policy.json"
        sidecar = self.path / "meta/openamrobot/episodes/episode-000000.json"
        self.assertEqual(report["provenance"]["policy"]["sha256"], hashlib.sha256(policy.read_bytes()).hexdigest())
        self.assertEqual(report["episodes"][0]["sidecar"]["sha256"], hashlib.sha256(sidecar.read_bytes()).hexdigest())
        self.assertEqual(report["episodes"][0]["data_origin"], "synthetic")
        self.assertEqual(report["episodes"][0]["binding"]["dataset_revision"], self.sidecar["native_lerobot"]["dataset_revision"])
        self.assertEqual(report, self.run_gate())

    def test_real_video_decode_or_explicit_unavailable(self):
        self.native["info"]["video_path"] = "videos/{video_key}/episode-{episode_index:06d}.mp4"
        self.policy["video_content"] = {name: {"max_identical_run": 1} for name in CAMERAS}
        self.write()
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            report = self.run_gate()
            missing = self.assert_finding(report["findings"], "C10_VIDEO", "NOT_VERIFIED")
            self.assertEqual({f["stream"] for f in missing}, set(CAMERAS))
            print("C10 real video decode branch unavailable: ffmpeg/ffprobe absent; explicit NOT_VERIFIED verified.")
            return
        seed = Path(self.temporary.name) / "two-frame-synthetic.mp4"
        process = subprocess.run([
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=30",
            "-frames:v", "2", "-pix_fmt", "yuv420p", "-c:v", "mpeg4", "-y", str(seed)],
            capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        videos = []
        for stream in CAMERAS:
            feature = self.sidecar["openamrobot"]["streams"][stream]["native_feature_key"]
            path = self.path / f"videos/{feature}/episode-000000.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(seed, path)
            videos.append(path)
        report = self.run_gate()
        successful = self.assert_finding(report["findings"], "C10_VIDEO", "PASS")
        self.assertEqual({f["stream"] for f in successful}, set(CAMERAS))
        videos[0].write_bytes(b"deliberately corrupt synthetic mp4")
        self.assert_finding(self.run_gate()["findings"], "C10_VIDEO")
        print("C10 synthetic MP4 creation/decode/corruption branch executed with ffmpeg/ffprobe; no robot recording.")

    def test_fractional_clock_lexemes_survive_json_and_yaml_loading(self):
        import yaml
        from oa_embodied_ai.dataset_qa.episode import preserve_clock_precision
        from oa_embodied_ai.dataset_qa.semantics import _mapped
        mapping = self.sidecar["openamrobot"]["streams"]["head"]["timestamps"]["device"]["mapping_to_authoritative"]
        mapping.update(source_origin=0, scale_ns_per_source_unit=0.5,
                       validity={"source_start": 0, "source_end": 100000000})
        first = self.sidecar["openamrobot"]["samples"][0]["streams"]["head"]
        first["device_timestamp"].update(raw="EXACT_RAW", mapped_authoritative_ns=1)
        first.update(selected_timestamp_ns=1, measured_skew_ns=1)
        second = self.sidecar["openamrobot"]["samples"][1]["streams"]["head"]
        second["device_timestamp"]["raw"] = 66666666
        for extension in ("json", "yaml"):
            with self.subTest(format=extension):
                path = Path(self.temporary.name) / f"lexical-tie.{extension}"
                text = json.dumps(self.sidecar) if extension == "json" else yaml.safe_dump(self.sidecar)
                text = text.replace('"EXACT_RAW"' if extension == "json" else "EXACT_RAW", "1.0000000000000000001")
                path.write_text(text)
                parsed = json.loads(text) if extension == "json" else yaml.safe_load(text)
                self.assert_c9_valid(parsed)
                precise = preserve_clock_precision(path, parsed)
                if precise is None:
                    precise = parsed
                raw = precise["openamrobot"]["samples"][0]["streams"]["head"]["device_timestamp"]["raw"]
                precise_mapping = precise["openamrobot"]["streams"]["head"]["timestamps"]["device"]["mapping_to_authoritative"]
                self.assertEqual(str(raw), "1.0000000000000000001")
                self.assertEqual(_mapped(raw, precise_mapping), 1)

    def run_named_fault(self, case):
        import pyarrow as pa
        import pyarrow.parquet as pq

        if case == "orphan_native_episode_rows":
            orphan = deepcopy(self.native["rows"][0])
            orphan["episode_index"] = 99
            self.native["rows"].append(orphan)
        elif case == "native_row_range":
            self.native["rows"][1]["index"] = 42
        elif case in ("recording_release_commit", "unapproved_release"):
            pin = {"release": "synthetic-test-release", "commit": "1" * 40,
                   "evidence": "Synthetic release-owner fixture; not a production pin."}
            self.sidecar["openamrobot"]["provenance"]["lerobot_compatibility"]["validated_release_pin"] = pin
            self.sidecar["openamrobot"]["provenance"]["software"].append({
                "name": "lerobot", "version": pin["release"], "commit": ("2" if case == "recording_release_commit" else "1") * 40,
                "repository": "https://github.com/huggingface/lerobot"})
            self.policy["release"] = {"pin": deepcopy(pin), "version": pin["release"], "commit": pin["commit"]}
        elif case in ("decoded_video_frame_count", "frozen_decoded_pixels"):
            self.native["info"]["video_path"] = "videos/{video_key}/episode-{episode_index:06d}.mp4"
            if case == "frozen_decoded_pixels":
                self.policy["video_content"] = {name: {"max_identical_run": 1} for name in CAMERAS}
        elif case == "unapproved_calibration":
            residual = self.sidecar["openamrobot"]["streams"]["base"]["calibration"]["residual"]
            self.policy["calibration"] = {"base": {
                key: residual[key] for key in ("name", "unit", "statistic", "method")}}
            self.policy["calibration"]["base"]["max_value"] = 0.3
        elif case == "calibration_residual_exceeds":
            residual = self.sidecar["openamrobot"]["streams"]["base"]["calibration"]["residual"]
            self.policy["calibration"] = {"base": {
                key: residual[key] for key in ("name", "unit", "statistic", "method")}}
            self.policy["calibration"]["base"]["max_value"] = 0.2
        self.write()
        if case == "duplicate_json_key":
            path = self.path / "meta/openamrobot/episodes/episode-000000.json"
            path.write_text('{"contract_version":1,"contract_version":1}')
        elif case == "invalid_c9_schema":
            del self.sidecar["contract_version"]
            (self.path / "meta/openamrobot/episodes/episode-000000.json").write_text(json.dumps(self.sidecar))
        elif case == "duplicate_native_episode":
            pq.write_table(pa.Table.from_pylist([self.native["episode"], self.native["episode"]]),
                           self.path / "meta/episodes/chunk-000/file-000.parquet")
        elif case == "native_double_vectors":
            pq.write_table(pa.Table.from_pylist(self.native["rows"]), self.path / "data/chunk-000/file-000.parquet")
        elif case == "artifact_bytes_changed":
            artifact = self.path / "synthetic-config.bin"
            artifact.write_bytes(b"mutated synthetic bytes")
            self.policy["artifacts"] = [{"kind": "robot_configuration", "path": artifact.name,
                                         "sha256": self.sidecar["openamrobot"]["provenance"]["robot_configuration_hash"]}]
        elif case in ("configuration_declared_hash", "configuration_symlink_escape", "absent_configuration_artifact"):
            artifact = self.path / "synthetic-config.bin"
            if case == "configuration_symlink_escape":
                external = Path(self.temporary.name) / "outside-config.bin"
                external.write_bytes(b"synthetic bytes outside dataset")
                artifact.symlink_to(external)
            elif case != "absent_configuration_artifact":
                artifact.write_bytes(b"synthetic bytes with a different declared hash")
            digest = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest() if artifact.exists() else "sha256:" + "1" * 64
            self.policy["artifacts"] = [{"kind": "robot_configuration", "path": artifact.name, "sha256": digest}]
        elif case == "malformed_nested_policy":
            self.policy["profiles"]["action"] = []
        elif case == "source_profile_identity":
            artifact = self.path / "synthetic-source-profile.json"
            artifact.write_text('{"synthetic_only":true}')
            self.policy["source_profile"] = {
                "id": "a-different-synthetic-source-profile", "artifact": {
                    "path": artifact.name, "sha256": "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()}}
        elif case == "truncated_parquet":
            path = self.path / "data/chunk-000/file-000.parquet"
            path.write_bytes(path.read_bytes()[:64])
        elif case == "calibration_identity":
            artifact = self.path / "synthetic-calibration.bin"
            artifact.write_bytes(b"synthetic calibration")
            self.policy["artifacts"] = [{"kind": "calibration", "stream": "base", "path": artifact.name,
                "sha256": "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "id": "a-different-synthetic-calibration", "version": "fixture-1"}]
        elif case in ("decoded_video_frame_count", "frozen_decoded_pixels"):
            for stream in CAMERAS:
                feature = self.sidecar["openamrobot"]["streams"][stream]["native_feature_key"]
                path = self.path / f"videos/{feature}/episode-000000.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic probe input")
            frames = [{"width": 64, "height": 48, "best_effort_timestamp_time": "0"}]
            if case == "frozen_decoded_pixels":
                frames.append({"width": 64, "height": 48, "best_effort_timestamp_time": "0.033333"})
            output = json.dumps({"frames": frames})
            def fake_process(command, **kwargs):
                value = output if command[0] == "ffprobe" else "# synthetic framemd5\n0,0,0,1,9216,samehash\n0,1,1,1,9216,samehash\n"
                return subprocess.CompletedProcess(command, 0, value, "")
            with patch("oa_embodied_ai.dataset_qa.episode.shutil.which", return_value="synthetic-ffprobe"), patch(
                    "oa_embodied_ai.dataset_qa.episode.subprocess.run", side_effect=fake_process):
                return self.run_gate()
        elif case in ("unapproved_calibration", "unapproved_release"):
            check_id = "C10_CALIBRATION" if case == "unapproved_calibration" else "C10_RELEASE"
            self.assert_finding(self.run_gate()["findings"], check_id, "PASS")
            self.policy["approved"] = False
        (self.path / "synthetic-policy.json").write_text(json.dumps(self.policy))
        return self.run_gate()



def _wrapper_test(case, check_id, status):
    def test(self):
        report = self.run_named_fault(case)
        self.assert_finding(report["findings"], check_id, status)
        self.assertEqual(report["exit_code"], 1 if status == "REJECT" else 2)
    test.__name__ = f"test_wrapper_{case}"
    return test


for _case, (_check_id, _status) in WRAPPER_CASES.items():
    setattr(C10WrapperTests, f"test_wrapper_{_case}", _wrapper_test(_case, _check_id, _status))


if __name__ == "__main__":
    unittest.main()
