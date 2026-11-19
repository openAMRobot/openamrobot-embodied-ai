#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Revert each reference_integrity check in an isolated source copy; its test must then fail.

For every (file, anchor, replacement, test): the test passes on the current source, the anchor
occurs exactly once, and with the check reverted the same test fails. Exit 0 only if all hold.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = "python/native_tests/test_reference_dataset.py::"
REF = "oa_embodied_ai/conversion/reference.py"

MUTATIONS = [
    ("degrees declared", REF, "if unit.lower() in DEGREE_UNITS:", "if False:",
     "test_a_channel_declared_in_degrees_is_refused"),
    ("source range", REF, "if limits is not None and not limits[0] - tolerance <= value <= limits[1] + tolerance:",
     "if False:", "test_degree_values_in_the_canonical_data_are_refused"),
    ("source range (lift command)", REF,
     "if limits is not None and not limits[0] - tolerance <= value <= limits[1] + tolerance:", "if False:",
     "test_a_lift_stop_index_read_as_metres_is_refused"),
    ("lift unit", REF, 'if c["component"] == "lift" and (', 'if False and (',
     "test_a_lift_declared_in_millimetres_is_refused"),
    ("action convention", REF, 'problem = compare_actions([f["action"] for f in frames], expected)', "problem = None",
     "test_a_one_frame_action_shift_is_detected"),
    ("camera missing", "oa_embodied_ai/conversion/batch.py",
     "named = reference.camera_missing(exc, profile) if checked else None", "named = None",
     "test_missing_camera_frames_are_refused"),
    ("camera sync", REF, "if skews[worst] > limit:", "if False:",
     "test_camera_skew_beyond_the_declared_limit_is_refused"),
    ("gripper value sign", REF, "if sign is not None and sign * value < -tolerance:", "if False:",
     "test_a_right_finger_with_the_left_finger_sign_is_refused"),
    ("gripper declared sign", REF,
     '            sign = gripper["channels"][c["name"]].get("opening_sign")\n            low, high = c["range"]\n',
     "            continue\n", "test_an_opening_sign_that_contradicts_the_range_is_refused"),
    ("timestamp reconcile", REF,
     'if recorded != entry["messages"] or kept + entry["equal_timestamp_dropped"] != entry["messages"]:', "if False:",
     "test_counts_that_do_not_reconcile_with_the_bag_metadata_are_refused"),
    ("calibration mismatch", REF,
     'if (content["height"], content["width"]) != (height, width) or content["frame_id"] != stream["frame_id"]:',
     "if False:", "test_camera_info_of_another_camera_is_refused"),
    ("template hash", REF, "if ZERO_HASH in text:", "if False:", "test_a_template_hash_never_remains_in_the_sidecar"),
    ("recorded sidecar hashes", "oa_embodied_ai/conversion/native.py", "    if reference:\n", "    if False:\n",
     "test_the_sidecar_carries_recorded_hashes_not_template_values"),
    ("keep_last only where declared", "oa_embodied_ai/recording/bag.py",
     'and stream.get("equal_timestamp_rule") != "keep_last"', "and False",
     "test_duplicate_stamps_without_keep_last_are_refused"),
    ("output binding", "oa_embodied_ai/conversion/native.py",
     'manifest["reference"]["output_sha256"] = reference.output_hashes(root)', "pass",
     "test_a_converted_slice_verifies_and_converts_identically"),
    ("verify output hashes", REF, "    if actual != expected:\n        changed", "    if False:\n        changed",
     "test_a_tampered_output_file_is_refused"),
    ("verify actions", REF, 'problem = compare_actions(rows["action"], [f["action"] for f in frames], float32=True)',
     "problem = None", "test_shifted_actions_in_a_rehashed_dataset_are_refused"),
    ("verify recomputed reports", REF, "if _roundtrip(value) != record[key]:", "if False:",
     "test_a_manifest_count_that_differs_from_the_bag_is_refused"),
    ("verify source bag", REF, 'if inventory(bag) != record["source_bag_sha256"]:', "if False:",
     "test_another_bag_is_refused"),
]


def run(test, pythonpath=None):
    env = dict(os.environ)
    env.setdefault("OA_INTERFACES_ROOT", str(ROOT / "external/openamrobot-interfaces"))
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", TESTS + test],
                          cwd=ROOT, env=env, capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    rows, ok = [], True
    for name, rel, anchor, replacement, test in MUTATIONS:
        current = run(test)
        with tempfile.TemporaryDirectory(prefix="oa-reference-revert-") as tmp:
            shutil.copytree(ROOT / "python/oa_embodied_ai", Path(tmp) / "oa_embodied_ai")
            path = Path(tmp) / rel
            text = path.read_text()
            count = text.count(anchor)
            if count == 1:
                path.write_text(text.replace(anchor, replacement))
                reverted = run(test, tmp)
            else:
                reverted = None
        row = {"check": name, "test": test, "current_exit": current.returncode,
               "anchor_count": count, "reverted_exit": None if reverted is None else reverted.returncode,
               "reverted_tail": None if reverted is None else (reverted.stdout.strip().splitlines() or [""])[-1]}
        row["proved"] = current.returncode == 0 and count == 1 and reverted is not None and reverted.returncode != 0
        ok &= row["proved"]
        rows.append(row)
        print(f"{'PROVED' if row['proved'] else 'NOT PROVED'}: {name}: {test}: current exit {current.returncode}, "
              f"reverted exit {row['reverted_exit']} ({row['reverted_tail']})", flush=True)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"{sum(r['proved'] for r in rows)} of {len(rows)} reverted checks make their test fail")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
