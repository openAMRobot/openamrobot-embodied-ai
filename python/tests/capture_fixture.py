# SPDX-License-Identifier: MIT
"""Deterministic emitted-to-file CDR messages, explicitly synthetic, never Gazebo."""

import copy
import json

from oa_embodied_ai.dataset_qa.contract import c9_asset
from oa_embodied_ai.dataset_qa.strict_io import load_data


def profile_fixture():
    sidecar = load_data(c9_asset("schemas/yaml/c9-lerobot-dataset-valid.yaml"))
    channels = []
    for side in ("left", "right"):
        for i in range(1, 9):
            gripper = i == 8
            channels.append(
                {
                    "name": f"{side}_gripper.pos"
                    if gripper
                    else f"{side}_joint_{i}.pos",
                    "component": f"{side}_gripper" if gripper else f"{side}_arm",
                    "joint": f"openarm_{side}_finger_joint1"
                    if gripper
                    else f"openarm_{side}_joint{i}",
                    "unit": "degree",
                    "source_unit": "rad",
                    "field": "position",
                    "stream": "state",
                }
            )
    streams = [
        {
            "id": "state",
            "component": "left_arm",
            "topic": "/fixture/joint_states",
            "type": "sensor_msgs/msg/JointState",
            "frame_id": "base_link",
            "clock_domain": "fixture_ros",
            "timestamp_source": "header",
        },
        {
            "id": "command",
            "component": "right_arm",
            "topic": "/fixture/requested_joint_positions",
            "type": "sensor_msgs/msg/JointState",
            "frame_id": "base_link",
            "clock_domain": "fixture_ros",
            "timestamp_source": "header",
        },
    ]
    for name in ("head", "base", "camera_wrist_left", "camera_wrist_right"):
        streams.append(
            {
                "id": name,
                "component": "camera",
                "topic": "/fixture/" + name,
                "type": "sensor_msgs/msg/Image",
                "frame_id": name,
                "clock_domain": "fixture_ros",
                "timestamp_source": "header",
                "camera_key": "observation.images." + name,
                "calibration_id": "synthetic-" + name,
            }
        )
    action = copy.deepcopy(channels)
    for c in action:
        c["stream"] = "command"
    return {
        "capture_version": 1,
        "profile_id": "synthetic-c9-capture-v1",
        "native_profile_id": "lerobot-v0.6.1-bi-openarm-position",
        "status": "proposed",
        "origin": "synthetic",
        "episode_id": "synthetic-episode-000",
        "dataset_id": "openamrobot/synthetic-capture",
        "dataset_revision": "synthetic-capture-revision-1",
        "task": "Synthetic software test only",
        "robot_configuration_id": "synthetic-config",
        "fps": 30,
        "max_skew_ns": 100,
        "streams": streams,
        "components": ["left_arm", "left_gripper", "right_arm", "right_gripper"],
        "state_channels": channels,
        "action_channels": action,
        "state_names": [c["name"] for c in channels],
        "action_names": [c["name"] for c in action],
        "source_commits": {
            "openAMRobot/openamrobot-interfaces": "87ab8a936c97ab55ee74b8988ca3d7659160ea6a"
        },
    }, sidecar


def write_bag(
    path, profile, *, omit=None, corrupt_clock=False, value_offset=0.0, image_offset=0
):
    import numpy as np
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore

    store = get_typestore(Stores.ROS2_JAZZY)
    types = store.types
    with Writer(path, version=9) as writer:
        connections = {
            s["id"]: writer.add_connection(s["topic"], s["type"], typestore=store)
            for s in profile["streams"]
            if s["id"] != omit
        }
        for i in range(4):
            stamp = 1_000_000_000 + round(i * 1_000_000_000 / 30)
            for stream in profile["streams"]:
                if stream["id"] == omit:
                    continue
                acquisition = stamp if not corrupt_clock or i != 2 else 1_000_000_000
                header = types["std_msgs/msg/Header"](
                    types["builtin_interfaces/msg/Time"](
                        acquisition // 1_000_000_000, acquisition % 1_000_000_000
                    ),
                    stream["frame_id"],
                )
                if stream.get("camera_key"):
                    msg = types[stream["type"]](
                        header,
                        32,
                        32,
                        "rgb8",
                        0,
                        96,
                        np.full(32 * 32 * 3, 20 + image_offset + i * 30, np.uint8),
                    )
                else:
                    msg = types[stream["type"]](
                        header,
                        [c["joint"] for c in profile["state_channels"]],
                        np.full(16, value_offset + i / 100, np.float64),
                        np.zeros(16),
                        np.zeros(16),
                    )
                writer.write(
                    connections[stream["id"]],
                    stamp,
                    store.serialize_cdr(msg, stream["type"]),
                )
    return path


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    profile, template = profile_fixture()
    (args.output / "profile.json").write_text(json.dumps(profile, indent=2))
    (args.output / "sidecar-template.json").write_text(json.dumps(template, indent=2))
    write_bag(args.output / "bag", profile)


def collection_fixture(root, *, different_tasks=True):
    """Two independent synthetic bags; no dataset rows/episodes fabricated by a merger."""
    from pathlib import Path

    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    entries = []
    for i in range(2):
        profile, template = profile_fixture()
        profile["episode_id"] = f"synthetic-bag-episode-{i}"
        if different_tasks:
            profile["task"] = f"Synthetic task {i}; no robot execution"
        write_bag(root / f"bag-{i}", profile, value_offset=i / 20, image_offset=i * 10)
        (root / f"profile-{i}.json").write_text(json.dumps(profile))
        (root / f"template-{i}.json").write_text(json.dumps(template))
        entries.append(
            {
                "bag": f"bag-{i}",
                "profile": f"profile-{i}.json",
                "sidecar_template": f"template-{i}.json",
            }
        )
    manifest = root / "collection.json"
    manifest.write_text(
        json.dumps({"collection_version": 1, "episodes": entries}, indent=2)
    )
    return manifest


def bind_converted_for_synthetic_training(root):
    """Attach explicit TEST-ONLY QA fixtures, then bind an immutable snapshot.

    This never makes a real dataset ready. Native writer output is not modified;
    synthetic configuration/calibration/release policy fixtures supplement the
    intentionally incomplete converter output so the C11 software override can
    exercise training. Raw C10 always remains NOT_VERIFIED with false readiness.
    """
    from pathlib import Path

    from c10_fixture import synthetic_policy
    from oa_embodied_ai.training.inputs import file_sha256, inventory

    root = Path(root)
    sidecars = sorted((root / "meta/openamrobot/episodes").glob("episode-*.json"))
    native_before = {
        str(p.relative_to(root)): file_sha256(p)
        for p in root.rglob("*")
        if p.is_file() and "openamrobot" not in p.relative_to(root).parts
    }
    policy = None

    def artifact(name):
        path = root / "synthetic-software-qa" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "SYNTHETIC SOFTWARE FIXTURE ONLY; no measured or owner-approved evidence: "
            + name
        )
        return {
            "path": str(path.relative_to(root)),
            "sha256": "sha256:" + file_sha256(path),
        }

    for path in sidecars:
        sidecar = json.loads(path.read_text())
        capture = sidecar["openamrobot"]
        assert capture["provenance"]["data_origin"] == "synthetic"
        artifacts, calibration = [], {}
        robot = artifact("robot.txt")
        capture["provenance"]["robot_configuration_hash"] = robot["sha256"]
        artifacts.append({"kind": "robot_configuration", **robot})
        for name, stream in capture["streams"].items():
            config = artifact(name + "-config.txt")
            stream["configuration_hash"] = config["sha256"]
            artifacts.append({"kind": "stream_configuration", "stream": name, **config})
            if "calibration" in stream:
                residual = {
                    "status": "measured",
                    "name": "synthetic_reprojection_error",
                    "value": 0.25,
                    "unit": "pixel",
                    "statistic": "rms",
                    "method": "synthetic fixture value; no physical measurement",
                }
                stream["calibration"]["residual"] = residual
                cal = stream["calibration"]
                calibration[name] = {
                    k: residual[k] for k in ("name", "unit", "statistic", "method")
                }
                calibration[name]["max_value"] = 0.3
                artifacts.append(
                    {
                        "kind": "calibration",
                        "stream": name,
                        "id": cal["id"],
                        "version": cal["version"],
                        **artifact(name + "-calibration.txt"),
                    }
                )
        # Deliberately synthetic release fixture; never a project-approved pin.
        pin = {
            "release": "synthetic-test-only",
            "commit": "1" * 40,
            "evidence": "SYNTHETIC SOFTWARE TEST ONLY; no release-owner acceptance",
        }
        capture["provenance"]["lerobot_compatibility"]["validated_release_pin"] = pin
        capture["provenance"]["software"].append(
            {
                "name": "lerobot",
                "version": pin["release"],
                "commit": pin["commit"],
                "repository": "https://github.com/huggingface/lerobot",
            }
        )
        path.write_text(json.dumps(sidecar, indent=2))
        if policy is None:
            policy = synthetic_policy(sidecar)
            policy["video_content"] = {
                name: {"max_identical_run": 1}
                for name in ("head", "base", "camera_wrist_left", "camera_wrist_right")
            }
            policy.update(
                artifacts=artifacts,
                calibration=calibration,
                source_profile={
                    "id": capture["provenance"]["source_profile"],
                    "artifact": artifact("source.txt"),
                },
                release={
                    "pin": pin,
                    "version": pin["release"],
                    "commit": pin["commit"],
                },
                native_identity={
                    "evidence": "SYNTHETIC caller selection only",
                    "episode_ids": {},
                },
            )
        policy["native_identity"]["episode_ids"][
            str(sidecar["native_lerobot"]["episode_index"])
        ] = sidecar["native_lerobot"]["episode_id"]
    native_after = {
        str(p.relative_to(root)): file_sha256(p)
        for p in root.rglob("*")
        if p.is_file()
        and "openamrobot" not in p.relative_to(root).parts
        and "synthetic-software-qa" not in p.relative_to(root).parts
    }
    assert native_before == native_after
    policy_path = root.parent / "converted-synthetic-policy.json"
    policy_path.write_text(json.dumps(policy, indent=2))
    first = json.loads(sidecars[0].read_text())["native_lerobot"]
    manifest = {
        "manifest_version": 1,
        "dataset_id": first["dataset_id"],
        "dataset_revision": first["dataset_revision"],
        "episode_indices": list(range(len(sidecars))),
        "profile": first["features"]["action"]["profile"],
        "data_origin": "synthetic",
        "evidence": "Synthetic converted bag pipeline; software execution only",
        "policy_sha256": file_sha256(policy_path),
        "files_sha256": inventory(root),
    }
    manifest_path = root.parent / "converted-synthetic-inputs.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return policy_path, manifest_path
