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


def write_bag(path, profile, *, omit=None, corrupt_clock=False):
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
                        np.full(32 * 32 * 3, 20 + i * 30, np.uint8),
                    )
                else:
                    msg = types[stream["type"]](
                        header,
                        [c["joint"] for c in profile["state_channels"]],
                        np.full(16, i / 100, np.float64),
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
