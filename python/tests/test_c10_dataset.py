# SPDX-License-Identifier: MIT
"""C10 fault injection: every mutation is valid against the unchanged C9 schema."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

from jsonschema import Draft202012Validator
from oa_embodied_ai.dataset_qa.contract import c9_asset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/tests"))

from c10_fixture import fixture
from oa_embodied_ai.dataset_qa.semantics import CHECK_IDS, check_episode


# These names also form the machine-readable acceptance-matrix source.
NEGATIVE_CASES = {
    "camera_fps_descriptor": "C10_FEATURES",
    "success_with_abort_event": "C10_COMPLETENESS",
    "binding_revision": "C10_BINDING",
    "feature_shape": "C10_FEATURES",
    "task_text": "C10_TASK",
    "frame_coverage": "C10_COVERAGE",
    "timestamp_order": "C10_TIME_ORDER",
    "affine_arithmetic": "C10_MAPPING",
    "selection_source": "C10_SELECTION",
    "fresh_reused_sample": "C10_HEALTH",
    "sequence_reset": "C10_SEQUENCE",
    "reported_skew": "C10_SKEW",
    "inter_observation_skew": "C10_SYNC",
    "sequence_gap": "C10_STREAM_QUALITY",
    "vector_width": "C10_VECTOR_VALUES",
    "profile_units": "C10_PROFILE",
    "action_range": "C10_ACTION_BOUNDS",
    "command_latency": "C10_ACTION_LATENCY",
    "event_outside_recording": "C10_EVENTS",
    "interrupted_recording": "C10_COMPLETENESS",
    "mapping_validity_interval": "C10_MAPPING",
    "event_missing_frame": "C10_EVENTS",
    "vector_nonfinite": "C10_VECTOR_VALUES",
    "profile_dimension_order": "C10_PROFILE",
    "native_frame_index": "C10_COVERAGE",
    "format_version": "C10_BINDING",
    "stale_observation_age": "C10_STREAM_QUALITY",
    "measured_source_fps": "C10_STREAM_QUALITY",
    "action_value_semantics": "C10_ACTION_LATENCY",
    "action_timestamp_semantics": "C10_ACTION_LATENCY",
    "declared_policy_tolerance_mismatch": "C10_SYNC",
    "nonconsecutive_source_id_reuse": "C10_SEQUENCE",
    "duplicate_frame": "C10_COVERAGE",
    "episode_length": "C10_BINDING",
    "fresh_without_receipt": "C10_HEALTH",
    "descending_authoritative_time": "C10_TIME_ORDER",
    "selected_time_disagrees_with_mapping": "C10_SELECTION",
    "legacy_camera_aliases": "C10_FEATURES",
    "native_vector_shape": "C10_FEATURES",
    "native_vector_name_order": "C10_FEATURES",
    "vector_infinity": "C10_VECTOR_VALUES",
    "orphan_native_task": "C10_TASK",
    "orphan_native_episode": "C10_BINDING",
    "inverted_event": "C10_EVENTS",
    "duplicate_event_id": "C10_EVENTS",
    "external_native_identity_contradiction": "C10_BINDING",
}




def capture(sidecar, native, policy):
    findings = []

    def emit(check_id, status, observed=None, expected=None, units=None,
             threshold=None, policy=None, fix="", frame=None, stream=None):
        findings.append({
            "check_id": check_id, "status": status, "observed": observed,
            "expected": expected, "units": units, "threshold": threshold,
            "policy": policy, "fix": fix, "frame": frame, "stream": stream,
        })

    metrics = check_episode(sidecar, native, policy, emit)
    return findings, metrics


def set_time(sidecar, frame, stream, timestamp):
    """Keep mapping, selection and stored skew consistent for one mutation."""
    sample = sidecar["openamrobot"]["samples"][frame]
    item = sample["streams"][stream]
    source = item["selection_timestamp_source"]
    metadata = sidecar["openamrobot"]["streams"][stream]["timestamps"][source]
    mapping = metadata["mapping_to_authoritative"]
    stamp = item["device_timestamp" if source == "device" else "host_receipt"]
    stamp["raw"] = (mapping["source_origin"] +
                    (timestamp - mapping["authoritative_origin_ns"]) /
                    mapping["scale_ns_per_source_unit"])
    stamp["mapped_authoritative_ns"] = timestamp
    item["selected_timestamp_ns"] = timestamp
    item["measured_skew_ns"] = abs(timestamp - sample["authoritative_timestamp_ns"])


def mutate(name, sidecar, native, policy):
    manifest = sidecar["native_lerobot"]
    capture = sidecar["openamrobot"]
    first, second = capture["samples"]
    if name == "camera_fps_descriptor":
        native["info"]["features"]["observation.images.head"]["info"] = {"fps":15}
    elif name == "success_with_abort_event":
        capture["outcome"]["status"] = "success"
        capture["events"][0]["type"] = "abort"
    elif name == "binding_revision":
        native["info"]["dataset_revision"] = "different-synthetic-revision"
    elif name == "feature_shape":
        native["info"]["features"]["observation.images.head"]["shape"][0] += 1
    elif name == "task_text":
        native["tasks"][0] = "A different synthetic task."
    elif name == "frame_coverage":
        second["frame_index"] = 2
    elif name == "timestamp_order":
        second["authoritative_timestamp_ns"] = first["authoritative_timestamp_ns"]
    elif name == "affine_arithmetic":
        first["streams"]["head"]["device_timestamp"]["mapped_authoritative_ns"] = 1
    elif name == "selection_source":
        first["streams"]["head"]["selection_timestamp_source"] = "host_receipt"
    elif name == "fresh_reused_sample":
        second["streams"]["base"] = deepcopy(first["streams"]["base"])
        second["streams"]["base"]["measured_skew_ns"] = 33333333
    elif name == "sequence_reset":
        first["streams"]["base"]["sequence_id"] = 2
    elif name == "reported_skew":
        second["streams"]["base"]["measured_skew_ns"] = 1
    elif name == "inter_observation_skew":
        set_time(sidecar, 1, "base", second["authoritative_timestamp_ns"] - 101)
    elif name == "sequence_gap":
        second["streams"]["base"]["sequence_id"] = 2
    elif name == "vector_width":
        native["rows"][1]["action"] = [0.0] * 15
    elif name == "profile_units":
        manifest["features"]["action"]["dimensions"][0]["unit"] = "radian"
    elif name == "action_range":
        native["rows"][1]["action"][0] = 181.0
    elif name == "command_latency":
        set_time(sidecar, 1, "action", second["authoritative_timestamp_ns"] + 2001)
    elif name == "event_outside_recording":
        capture["events"][0]["end_timestamp_ns"] = capture["recording"]["end_timestamp_ns"] + 1
    elif name == "interrupted_recording":
        capture["recording"]["status"] = "interrupted"
    elif name == "mapping_validity_interval":
        capture["streams"]["head"]["timestamps"]["device"]["mapping_to_authoritative"][
            "validity"]["source_start"] = 1000001
    elif name == "event_missing_frame":
        capture["events"][0]["linked_frame_indices"] = [999]
    elif name == "vector_nonfinite":
        native["rows"][0]["observation.state"][0] = float("nan")
    elif name == "profile_dimension_order":
        dimensions = manifest["features"]["action"]["dimensions"]
        dimensions[0], dimensions[1] = dimensions[1], dimensions[0]
    elif name == "native_frame_index":
        native["rows"][1]["frame_index"] = 2
    elif name == "format_version":
        native["info"]["codebase_version"] = "v2.1"
    elif name == "producer_reports_stale":
        second["streams"]["base"]["health"] = "stale"
    elif name == "stale_observation_age":
        set_time(sidecar, 1, "base", second["authoritative_timestamp_ns"] - 101)
    elif name == "measured_source_fps":
        capture["recording"]["end_timestamp_ns"] = 50000000
        set_time(sidecar, 1, "base", 40000000)
    elif name == "action_value_semantics":
        capture["action_semantics"]["values"] = "sent_post_clip"
    elif name == "action_timestamp_semantics":
        capture["action_semantics"]["timestamp"] = "command_dispatched"
    elif name == "declared_policy_tolerance_mismatch":
        capture["synchronization"]["tolerance_ns"] = 101
    elif name == "nonconsecutive_source_id_reuse":
        third = deepcopy(second)
        third["frame_index"] = 2
        third["authoritative_timestamp_ns"] = 66666666
        capture["samples"].append(third)
        capture["recording"]["end_timestamp_ns"] = 70000000
        manifest["length"] = native["episode"]["length"] = 3
        native["episode"]["dataset_to_index"] = native["info"]["total_frames"] = 3
        row = deepcopy(native["rows"][1])
        row.update(frame_index=2, index=2, timestamp=2 / 30)
        native["rows"].append(row)
        for stream, item in third["streams"].items():
            item["sequence_id"] = 2
            item["source_frame_id"] = "synthetic-2"
            set_time(sidecar, 2, stream, 66666666 + (1000 if stream == "action" else 0))
        third["streams"]["base"]["source_frame_id"] = first["streams"]["base"]["source_frame_id"]
    elif name == "duplicate_frame":
        second["frame_index"] = 0
    elif name == "episode_length":
        native["episode"]["length"] = 3
    elif name == "fresh_without_receipt":
        second["streams"]["base"]["host_receipt"] = {
            "status": "unavailable", "reason": "Synthetic receipt evidence missing."}
    elif name == "descending_authoritative_time":
        first["authoritative_timestamp_ns"] = 33333333
        second["authoritative_timestamp_ns"] = 0
    elif name == "selected_time_disagrees_with_mapping":
        second["streams"]["base"]["selected_timestamp_ns"] += 1
        second["streams"]["base"]["measured_skew_ns"] = 1
    elif name == "legacy_camera_aliases":
        for key in ("head", "base"):
            features = native["info"]["features"]
            features[f"observation.images.{key}_camera"] = features.pop(f"observation.images.{key}")
    elif name == "native_vector_shape":
        native["info"]["features"]["observation.state"]["shape"] = [15]
    elif name == "native_vector_name_order":
        names = native["info"]["features"]["action"]["names"]
        names[0], names[1] = names[1], names[0]
    elif name == "vector_infinity":
        native["rows"][0]["action"][0] = float("inf")
    elif name == "orphan_native_task":
        native["rows"][1]["task_index"] = 999
    elif name == "orphan_native_episode":
        native["episode"]["episode_index"] = 999
    elif name == "inverted_event":
        capture["events"][0].update(start_timestamp_ns=2000, end_timestamp_ns=1000)
    elif name == "duplicate_event_id":
        capture["events"].append(deepcopy(capture["events"][0]))
    elif name == "external_native_identity_contradiction":
        policy["native_identity"] = {
            "dataset_id": manifest["dataset_id"], "dataset_revision": "a-contradictory-snapshot",
            "episode_ids": {"0": manifest["episode_id"]}, "evidence": "Synthetic snapshot evidence."}
    else:
        raise AssertionError(f"Unnamed C10 fault: {name}")


class C10SemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validator = Draft202012Validator(json.loads(
            c9_asset("schemas/json/c9-lerobot-dataset.schema.json").read_text()))

    def assert_c9_valid(self, sidecar):
        errors = sorted(self.validator.iter_errors(sidecar), key=lambda e: str(e.path))
        self.assertEqual([], [f"{list(e.path)}: {e.message}" for e in errors])

    def assert_finding(self, findings, check_id, status="REJECT"):
        matching = [f for f in findings if f["check_id"] == check_id and f["status"] == status]
        diagnostic = [{key: f.get(key) for key in ("check_id", "status", "frame", "stream")}
                      for f in findings if f["check_id"] == check_id]
        self.assertTrue(matching, f"Expected {check_id}/{status}; received {diagnostic}")
        return matching

    def test_positive_fixture_is_derived_and_c9_valid(self):
        sidecar, native, policy = fixture()
        self.assert_c9_valid(sidecar)
        findings, metrics = capture(sidecar, native, policy)
        unverified = [f for f in findings if f["status"] in ("REJECT", "ERROR", "NOT_VERIFIED")]
        self.assertFalse(unverified, unverified)
        self.assertIsInstance(metrics, dict)
        self.assertIn("SYNTHETIC", policy["id"])

    def test_every_core_diagnostic_has_named_corruption(self):
        self.assertEqual(set(CHECK_IDS), set(NEGATIVE_CASES.values()))

    def test_fixture_objects_are_independent(self):
        sidecar, native, policy = fixture()
        sidecar["native_lerobot"]["features"]["action"]["shape"][0] = 1
        native["rows"][0]["action"][0] = 42
        policy["action_bounds"].clear()
        clean_sidecar, clean_native, clean_policy = fixture()
        self.assertEqual(clean_sidecar["native_lerobot"]["features"]["action"]["shape"], [16])
        self.assertEqual(clean_native["rows"][0]["action"][0], 1)
        self.assertEqual(len(clean_policy["action_bounds"]), 16)

    def test_pending_tolerance_is_not_a_zero_skew_pass(self):
        sidecar, native, policy = fixture()
        sidecar["openamrobot"]["synchronization"] = {
            "status": "pending_evidence", "tolerance_ns": None,
            "basis": "Synthetic unresolved tolerance.",
        }
        self.assert_c9_valid(sidecar)
        findings, _ = capture(sidecar, native, policy)
        self.assert_finding(findings, "C10_SYNC", "NOT_VERIFIED")

    def test_partial_quality_policy_does_not_verify_stream(self):
        sidecar, native, policy = fixture()
        del policy["stream_quality"]["base"]["max_freshness_ns"]
        findings, _ = capture(sidecar, native, policy)
        matching = self.assert_finding(findings, "C10_STREAM_QUALITY", "NOT_VERIFIED")
        self.assertTrue(any(f["stream"] == "base" for f in matching))

    def test_unmapped_device_preserves_raw_and_does_not_pass_sync(self):
        sidecar, native, policy = fixture()
        sidecar["openamrobot"]["streams"]["head"]["timestamps"]["device"][
            "mapping_to_authoritative"] = {"status": "pending", "reason": "Synthetic clock evidence pending."}
        for sample in sidecar["openamrobot"]["samples"]:
            item = sample["streams"]["head"]
            raw = item["device_timestamp"]["raw"]
            item.update(health="unmapped", selected_timestamp_ns=None, measured_skew_ns=None)
            item["device_timestamp"] = {"status": "pending", "raw": raw, "reason": "Synthetic pending mapping."}
        self.assert_c9_valid(sidecar)
        findings, _ = capture(sidecar, native, policy)
        self.assert_finding(findings, "C10_SYNC", "NOT_VERIFIED")

    def test_affine_rounds_nearest_ties_to_even_and_validity_is_inclusive(self):
        sidecar, native, policy = fixture()
        mapping = sidecar["openamrobot"]["streams"]["head"]["timestamps"]["device"]["mapping_to_authoritative"]
        mapping.update(source_origin=0, authoritative_origin_ns=0,
                       scale_ns_per_source_unit=0.5,
                       validity={"source_start": 1, "source_end": 3})
        for frame, (raw, mapped) in enumerate(((1, 0), (3, 2))):
            item = sidecar["openamrobot"]["samples"][frame]["streams"]["head"]
            item["device_timestamp"].update(raw=raw, mapped_authoritative_ns=mapped)
            item["selected_timestamp_ns"] = mapped
            item["measured_skew_ns"] = abs(mapped - sidecar["openamrobot"]["samples"][frame]["authoritative_timestamp_ns"])
        self.assert_c9_valid(sidecar)
        findings, _ = capture(sidecar, native, policy)
        self.assertFalse([f for f in findings if f["check_id"] == "C10_MAPPING" and f["status"] == "REJECT"], findings)
        sidecar["openamrobot"]["samples"][1]["streams"]["head"]["device_timestamp"]["raw"] = 3.01
        findings, _ = capture(sidecar, native, policy)
        self.assert_finding(findings, "C10_MAPPING")

    def test_action_bounds_include_both_endpoints(self):
        sidecar, native, policy = fixture()
        native["rows"][0]["action"] = [-180.0] * 16
        native["rows"][1]["action"] = [180.0] * 16
        findings, _ = capture(sidecar, native, policy)
        self.assertFalse([f for f in findings if f["check_id"] == "C10_ACTION_BOUNDS" and f["status"] == "REJECT"], findings)

    def test_command_latency_includes_both_endpoints(self):
        sidecar, native, policy = fixture()
        set_time(sidecar, 0, "action", 0)
        set_time(sidecar, 1, "action", 33333333 + 2000)
        self.assert_c9_valid(sidecar)
        findings, _ = capture(sidecar, native, policy)
        self.assertFalse([f for f in findings if f["check_id"] == "C10_ACTION_LATENCY" and f["status"] == "REJECT"], findings)

    def test_sync_equal_to_threshold_is_accepted(self):
        sidecar, native, policy = fixture()
        set_time(sidecar, 1, "base", 33333333 - 100)
        findings, _ = capture(sidecar, native, policy)
        self.assertFalse([f for f in findings if f["check_id"] == "C10_SYNC" and f["status"] == "REJECT"], findings)

    def test_unapproved_numeric_policy_cannot_approve_ranges(self):
        sidecar, native, policy = fixture()
        policy["approved"] = False
        findings, _ = capture(sidecar, native, policy)
        self.assert_finding(findings, "C10_ACTION_BOUNDS", "NOT_VERIFIED")

    def test_native_identity_uses_external_snapshot_evidence(self):
        sidecar, native, policy = fixture()
        manifest = sidecar["native_lerobot"]
        for key in ("dataset_id", "dataset_revision"):
            del native["info"][key]
        del native["episode"]["episode_id"]
        policy["native_identity"] = {
            "dataset_id": manifest["dataset_id"], "dataset_revision": manifest["dataset_revision"],
            "episode_ids": {"0": manifest["episode_id"]},
            "evidence": "Synthetic immutable snapshot identity; not a production attestation.",
        }
        findings, _ = capture(sidecar, native, policy)
        self.assert_finding(findings, "C10_BINDING", "PASS")
        policy["native_identity"]["dataset_revision"] = "different-snapshot"
        findings, _ = capture(sidecar, native, policy)
        self.assert_finding(findings, "C10_BINDING")

    def test_episode_id_requires_native_or_external_snapshot_evidence(self):
        sidecar, native, policy = fixture()
        del native["episode"]["episode_id"]
        findings, _ = capture(sidecar, native, policy)
        self.assert_finding(findings, "C10_BINDING", "NOT_VERIFIED")

    def test_action_reference_offset_requires_explicit_policy(self):
        sidecar, native, policy = fixture()
        del policy["action_latency"]["reference_frame_offset"]
        findings, _ = capture(sidecar, native, policy)
        self.assert_finding(findings, "C10_ACTION_LATENCY", "NOT_VERIFIED")
        self.assertFalse([f for f in findings if f["check_id"] == "C10_ACTION_LATENCY" and f["status"] == "PASS"])

    def test_action_reference_offset_outside_episode_is_not_verified(self):
        sidecar, native, policy = fixture()
        policy["action_latency"]["reference_frame_offset"] = 999
        findings, _ = capture(sidecar, native, policy)
        self.assert_finding(findings, "C10_ACTION_LATENCY", "NOT_VERIFIED")
        self.assertFalse([f for f in findings if f["check_id"] == "C10_ACTION_LATENCY" and f["status"] == "PASS"])

    def test_honest_missing_and_dropped_slots_remain_not_verified(self):
        for health in ("missing", "dropped"):
            with self.subTest(health=health):
                sidecar, native, policy = fixture()
                sidecar["openamrobot"]["samples"][1]["streams"]["base"] = {
                    "sequence_id": None, "source_frame_id": None, "health": health,
                    "selection_timestamp_source": "none", "selected_timestamp_ns": None,
                    "host_receipt": {"status": "unavailable", "reason": "Synthetic absent slot."},
                    "device_timestamp": {"status": "unavailable", "reason": "Synthetic absent slot."},
                    "measured_skew_ns": None,
                }
                self.assert_c9_valid(sidecar)
                findings, metrics = capture(sidecar, native, policy)
                self.assert_finding(findings, "C10_SYNC", "NOT_VERIFIED")
                self.assertEqual(metrics["streams"]["base"][health + "_count"], 1)
                self.assertFalse([f for f in findings if f["check_id"] == "C10_SYNC" and
                                  f["frame"] == 1 and f["status"] == "PASS"])

    def test_stale_producer_label_requires_review_without_fabricated_limit(self):
        sidecar,native,policy = fixture()
        mutate("producer_reports_stale",sidecar,native,policy)
        self.assert_c9_valid(sidecar)
        findings,metrics = capture(sidecar,native,policy)
        self.assert_finding(findings,"C10_HEALTH","NOT_VERIFIED")
        self.assertFalse([f for f in findings if f["check_id"] == "C10_HEALTH" and f["status"] == "REJECT"])
        self.assertEqual(metrics["streams"]["base"]["max_stale_run"],1)




def _negative_test(case, check_id):
    def test(self):
        sidecar, native, policy = fixture()
        mutate(case, sidecar, native, policy)
        self.assert_c9_valid(sidecar)
        findings, _ = capture(sidecar, native, policy)
        self.assert_finding(findings, check_id)
    test.__name__ = f"test_reject_{case}"
    test.__doc__ = f"C9-valid {case} must produce {check_id}/REJECT."
    return test


for _case, _check_id in NEGATIVE_CASES.items():
    setattr(C10SemanticTests, f"test_reject_{_case}", _negative_test(_case, _check_id))




if __name__ == "__main__":
    unittest.main()
