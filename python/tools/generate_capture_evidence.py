#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Retain a reproducible SYNTHETIC bag/native/QA bundle; never promote C10."""

import argparse
import json
import sys
from pathlib import Path

from oa_embodied_ai.conversion.native import export_native
from oa_embodied_ai.dataset_qa.episode import check_dataset
from oa_embodied_ai.recording.bag import align, read_bag
from oa_embodied_ai.recording.profile import validate_profile
from oa_embodied_ai.training.evidence import execution_provenance
from oa_embodied_ai.training.inputs import file_sha256

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args(argv)
    execution = execution_provenance()
    if (
        execution["git_commit"] != args.expected_head
        or not execution["source_tree_clean"]
    ):
        raise RuntimeError("Evidence requires the expected clean source HEAD")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT / "python/tests"))
    from c10_fixture import synthetic_policy
    from capture_fixture import profile_fixture, write_bag

    profile, template = profile_fixture()
    validate_profile(profile, development=True)
    bag = write_bag(output / "synthetic-bag", profile)
    frames = align(read_bag(bag, profile), profile)
    export = export_native(
        frames, profile, template, output / "native-dataset", source_bag=bag
    )
    sidecar = json.loads(
        (
            output / "native-dataset/meta/openamrobot/episodes/episode-000000.json"
        ).read_text()
    )
    policy = output / "synthetic-policy.json"
    policy.write_text(json.dumps(synthetic_policy(sidecar), indent=2) + "\n")
    report = check_dataset(output / "native-dataset", policy)
    (output / "raw-c10-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    after = execution_provenance()
    software_pass = (
        report["status"] == "NOT_VERIFIED"
        and report["exit_code"] == 2
        and report["training_ready"] is False
        and not any(f["status"] == "REJECT" for f in report["findings"])
        and execution["source_sha256"] == after["source_sha256"]
        and after["source_tree_clean"]
    )
    manifest = {
        "purpose": "SYNTHETIC SOFTWARE EVIDENCE ONLY",
        "software_path_passed": software_pass,
        "execution": execution,
        "execution_after": after,
        "native_loader": export["native_loader"],
        "raw_c10_status": report["status"],
        "raw_c10_exit_code": report["exit_code"],
        "physical_acceptance": "NOT_VERIFIED",
        "production_acceptance": "NOT_VERIFIED",
        "artifacts_sha256": {
            str(p.relative_to(output)): file_sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"Synthetic software path {'PASS' if software_pass else 'FAIL'}; raw C10 {report['status']}/exit {report['exit_code']}; physical NOT_VERIFIED"
    )
    print(f"Artifacts: {output}")
    return 0 if software_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
