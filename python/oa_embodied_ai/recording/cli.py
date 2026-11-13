# SPDX-License-Identifier: MIT
"""Bounded, observational ros2 bag capture with explicit episode identity."""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import subprocess
import time
import uuid
from pathlib import Path

from oa_embodied_ai.conversion.validation import ConversionError
from oa_embodied_ai.dataset_qa.strict_io import load_data
from oa_embodied_ai.recording.profile import validate_profile
from oa_embodied_ai.training.inputs import file_sha256


def check_graph(profile):
    import rclpy

    rclpy.init()
    node = rclpy.create_node("oa_bag_capture_preflight_" + uuid.uuid4().hex[:8])
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            actual = dict(node.get_topic_names_and_types())
            if all(s["type"] in actual.get(s["topic"], []) for s in profile["streams"]):
                return
        missing = [
            s["topic"] + ":" + s["type"]
            for s in profile["streams"]
            if s["type"] not in actual.get(s["topic"], [])
        ]
        raise ConversionError("Required topics unavailable: " + ", ".join(missing))
    finally:
        node.destroy_node()
        rclpy.shutdown()


def record(profile, output, seconds):
    if shutil.which("ros2") is None:
        raise ConversionError(
            "ROS 2 CLI unavailable; source the approved ROS environment"
        )
    check_graph(profile)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "recording_id": str(uuid.uuid4()),
        "episode_id": profile["episode_id"],
        "profile": profile,
        "state": "recording",
        "start_host_ns": time.time_ns(),
        "origin": profile["origin"],
        "physical_acceptance": "NOT_VERIFIED",
    }
    path = output / "recording.json"
    path.write_text(json.dumps(manifest, indent=2))
    command = [
        "ros2",
        "bag",
        "record",
        "--storage",
        "sqlite3",
        "--output",
        str(output / "bag"),
        *[s["topic"] for s in profile["streams"]],
    ]
    with (output / "record.log").open("wb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            process.wait(timeout=seconds)
            manifest["state"] = "interrupted"
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
                manifest["state"] = "interrupted"
            else:
                manifest["state"] = (
                    "captured"
                    if process.returncode in (0, -signal.SIGINT, 130)
                    else "interrupted"
                )
        finally:
            manifest.update(
                end_host_ns=time.time_ns(),
                command=command,
                returncode=process.returncode,
            )
            manifest["bag_sha256"] = {
                str(p.relative_to(output)): file_sha256(p)
                for p in (output / "bag").rglob("*")
                if p.is_file()
            }
            path.write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest["state"] != "captured":
        raise ConversionError(
            f"Recording interrupted; partial data/log retained at {output}"
        )
    from oa_embodied_ai.recording.bag import read_bag

    try:
        read_bag(output / "bag", profile)
    except (ConversionError, ImportError) as exc:
        manifest.update(state="capture_not_validated", validation_error=str(exc))
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise ConversionError(
            f"Captured bag failed topic/type/time validation: {exc}"
        ) from exc
    manifest["bag_validation"] = "PASS"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    # Capture completion never establishes dataset readiness or physical acceptance.
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--synthetic-development", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not 0 < args.seconds <= 86400:
            raise ConversionError("seconds must be positive and bounded to one day")
        profile = validate_profile(
            load_data(args.profile), development=args.synthetic_development
        )
        record(profile, args.output, args.seconds)
        return 0
    except (ConversionError, OSError, ImportError) as exc:
        print(f"REJECT: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
