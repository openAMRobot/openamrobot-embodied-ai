# SPDX-License-Identifier: MIT
"""Equal timestamps under simulation time, on samples recorded in a Gazebo run.

The fixture holds consecutive messages copied from the bag of openamr-upperbody-sw
run 44448676604 (Baseline 1, opt-in AI capture, simulation time): windows of
/arm/joint_states with two messages carrying one header stamp, and windows of
/cmd_vel (no header) with two messages received at one /clock value. The tests
write those messages, with their recorded receipt times, to a bag and read it
with read_bag. Simulation evidence only.
"""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from oa_embodied_ai.conversion.validation import ConversionError
from oa_embodied_ai.recording.bag import read_bag
from oa_embodied_ai.recording.profile import validate_profile

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "baseline1_run44448676604_stamps.json"
)


def recorded():
    return json.loads(FIXTURE.read_text())


def profile(rule=None):
    joints = recorded()["/arm/joint_states"]["windows"][0]["rows"][0]["name"]
    streams = [
        {
            "id": "arm_joint_states",
            "component": "left_arm",
            "topic": "/arm/joint_states",
            "type": "sensor_msgs/msg/JointState",
            "frame_id": recorded()["/arm/joint_states"]["windows"][0]["rows"][0][
                "frame_id"
            ],
            "clock_domain": "ros.sim_time",
            "timestamp_source": "header",
        },
        {
            "id": "cmd_vel",
            "component": "base",
            "topic": "/cmd_vel",
            "type": "geometry_msgs/msg/Twist",
            "frame_id": "base_link",
            "frame_id_source": "configuration",
            "clock_domain": "ros.sim_time",
            "timestamp_source": "bag",
        },
    ]
    if rule:
        for stream in streams:
            stream["equal_timestamp_rule"] = rule
    left = next(j for j in joints if j.startswith("openarm_left_joint"))
    return {
        "profile_id": "recorded-stamp-excerpt",
        "components": ["left_arm"],
        "streams": streams,
        "state_channels": [
            {
                "name": left,
                "component": "left_arm",
                "stream": "arm_joint_states",
                "joint": left,
                "field": "position",
                "unit": "rad",
                "source_unit": "rad",
            }
        ],
    }


def write(path, data):
    """Write the recorded windows in their recorded receipt order, one bag."""
    import numpy as np
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore

    store = get_typestore(Stores.ROS2_JAZZY)
    t = store.types
    rows = []
    for row in (r for w in data["/arm/joint_states"]["windows"] for r in w["rows"]):
        sec, nsec = divmod(row["stamp_ns"], 1_000_000_000)
        msg = t["sensor_msgs/msg/JointState"](
            t["std_msgs/msg/Header"](
                t["builtin_interfaces/msg/Time"](sec, nsec), row["frame_id"]
            ),
            row["name"],
            *[
                np.array(row[k], dtype=np.float64)
                for k in ("position", "velocity", "effort")
            ],
        )
        rows.append(
            (row["receipt_ns"], "/arm/joint_states", "sensor_msgs/msg/JointState", msg)
        )
    for row in (r for w in data["/cmd_vel"]["windows"] for r in w["rows"]):
        msg = t["geometry_msgs/msg/Twist"](
            t["geometry_msgs/msg/Vector3"](*row["linear"]),
            t["geometry_msgs/msg/Vector3"](*row["angular"]),
        )
        rows.append((row["receipt_ns"], "/cmd_vel", "geometry_msgs/msg/Twist", msg))
    rows.sort(key=lambda r: r[0])  # stable: equal receipt times keep recorded order
    with Writer(path, version=9) as writer:
        conns = {
            topic: writer.add_connection(topic, kind, typestore=store)
            for topic, kind in {(r[1], r[2]) for r in rows}
        }
        for receipt, topic, kind, msg in rows:
            writer.write(conns[topic], receipt, store.serialize_cdr(msg, kind))
    return path


def equal_pairs(rows, key):
    return sum(1 for a, b in zip(rows, rows[1:], strict=False) if a[key] == b[key])


def test_recorded_equal_stamps_are_refused_without_a_declared_rule(tmp_path):
    bag = write(tmp_path / "bag", recorded())
    with pytest.raises(ConversionError, match="Nonmonotonic or duplicate timestamps"):
        read_bag(bag, profile())


def test_recorded_equal_stamps_are_read_with_keep_last(tmp_path):
    bag = write(tmp_path / "bag", recorded())
    result = read_bag(bag, profile("keep_last"))
    for stream in ("arm_joint_states", "cmd_vel"):
        times = [s.time_ns for s in result[stream]]
        assert all(b > a for a, b in zip(times, times[1:], strict=False))


@pytest.mark.parametrize(
    ("topic", "stream", "key"),
    [
        ("/arm/joint_states", "arm_joint_states", "stamp_ns"),
        ("/cmd_vel", "cmd_vel", "receipt_ns"),
    ],
)
def test_keep_last_keeps_the_last_received_sample_of_each_stamp(
    tmp_path, topic, stream, key
):
    data = recorded()
    bag = write(tmp_path / "bag", data)
    report = {}
    result = read_bag(bag, profile("keep_last"), report=report)
    times = [s.time_ns for s in result[stream]]
    assert times == sorted(set(times)), "one sample per stamp, strictly increasing"
    rows = [r for w in data[topic]["windows"] for r in w["rows"]]
    last = {}
    for row in rows:
        last[row[key]] = row  # recorded order is receipt order
    assert times == sorted(last)
    for sample in result[stream]:
        row = last[sample.time_ns]
        if key == "stamp_ns":
            assert list(sample.message.position) == row["position"]
            assert sample.receipt_ns == row["receipt_ns"]
        else:
            assert sample.message.linear.x == row["linear"][0]
            assert sample.message.angular.z == row["angular"][2]
    assert report[stream]["messages"] == len(rows)
    assert report[stream]["equal_timestamp_dropped"] == equal_pairs(rows, key) > 0
    assert report[stream]["equal_timestamp_rule"] == "keep_last"


def test_a_decreasing_stamp_is_refused_even_with_keep_last(tmp_path):
    data = recorded()
    rows = data["/arm/joint_states"]["windows"][0]["rows"]
    rows[-1] = copy.deepcopy(rows[-1])
    rows[-1]["stamp_ns"] = rows[0]["stamp_ns"] - 1
    bag = write(tmp_path / "bag", data)
    with pytest.raises(ConversionError, match="Nonmonotonic or duplicate timestamps"):
        read_bag(bag, profile("keep_last"))


def test_only_keep_last_is_a_valid_rule():
    from capture_fixture import profile_fixture

    capture, _ = profile_fixture()
    capture["streams"][0]["equal_timestamp_rule"] = "average"
    with pytest.raises(ConversionError, match="equal_timestamp_rule"):
        validate_profile(capture, development=True)
    capture["streams"][0]["equal_timestamp_rule"] = "keep_last"
    assert validate_profile(capture, development=True) is capture


def test_the_proposed_baseline1_profile_is_bound_and_not_registered():
    from oa_embodied_ai.recording.profile import (
        APPROVED_CAPTURE_PROFILES,
        PROPOSED_CAPTURE_PROFILES,
        capture_contract_digest,
    )

    root = Path(__file__).resolve().parents[2]  # repository root
    for profile_id, entry in PROPOSED_CAPTURE_PROFILES.items():
        capture = json.loads((root / entry["path"]).read_text())
        assert capture["profile_id"] == profile_id
        assert capture["status"] == entry["status"] == "proposed"
        assert capture_contract_digest(capture) == entry["sha256"]
        assert profile_id not in APPROVED_CAPTURE_PROFILES
        with pytest.raises(ConversionError, match="enable it explicitly"):
            validate_profile(capture)
        # its Proposed rules enabled, it is still not an approved profile
        with pytest.raises(ConversionError, match="Unapproved profile"):
            validate_profile(capture, enabled=capture.get("proposed_rules", []))


def test_an_empty_header_frame_takes_the_declared_configuration_frame(tmp_path):
    """JointState from /build1/capture/gripper_command and the controller states
    carry an empty frame_id in run 44448676604. Here the recorded /arm/joint_states
    rows are written with their frame emptied, to exercise that case."""
    data = recorded()
    for window in data["/arm/joint_states"]["windows"]:
        for row in window["rows"]:
            row["frame_id"] = ""
    bag = write(tmp_path / "bag", data)
    capture = profile("keep_last")
    capture["streams"][0]["frame_id"] = "base_link"
    with pytest.raises(ConversionError, match="frame ID mismatch"):
        read_bag(bag, capture)
    capture["streams"][0]["frame_id_source"] = "configuration"
    assert read_bag(bag, capture)["arm_joint_states"]


def test_a_non_empty_header_frame_must_still_match_the_configuration(tmp_path):
    bag = write(tmp_path / "bag", recorded())
    capture = profile("keep_last")
    capture["streams"][0]["frame_id"] = "some_other_frame"
    capture["streams"][0]["frame_id_source"] = "configuration"
    with pytest.raises(ConversionError, match="frame ID mismatch"):
        read_bag(bag, capture)
