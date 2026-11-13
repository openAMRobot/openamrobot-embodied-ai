# SPDX-License-Identifier: MIT
"""ROS 2 bag inspection and deterministic bounded-skew alignment (read only)."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path

from oa_embodied_ai.conversion.validation import (
    ConversionError,
    normalize_vector,
    require_components,
)


@dataclass
class Sample:
    time_ns: int
    receipt_ns: int
    message: object


def read_bag(path, profile):
    from rosbags.highlevel import AnyReader

    path = Path(path)
    if not (path / "metadata.yaml").is_file():
        raise ConversionError(f"ROS bag metadata missing: {path}/metadata.yaml")
    result = {s["id"]: [] for s in profile["streams"]}
    by_topic = {s["topic"]: s for s in profile["streams"]}
    try:
        with AnyReader([path]) as reader:
            actual = {c.topic: c.msgtype for c in reader.connections}
            for stream in profile["streams"]:
                if actual.get(stream["topic"]) != stream["type"]:
                    raise ConversionError(
                        f"Component '{stream['component']}' topic '{stream['topic']} missing or wrong message type; expected {stream['type']}"
                    )
            for conn, receipt, raw in reader.messages(
                connections=[c for c in reader.connections if c.topic in by_topic]
            ):
                stream = by_topic[conn.topic]
                msg = reader.deserialize(raw, conn.msgtype)
                header = getattr(msg, "header", None)
                if header is None and stream.get("frame_id_source") != "configuration":
                    raise ConversionError(
                        f"Stream '{stream['id']}' missing header/frame ID; explicit configuration source required"
                    )
                if header is not None and header.frame_id != stream["frame_id"]:
                    raise ConversionError(
                        f"Component '{stream['component']}' frame ID mismatch on '{conn.topic}'"
                    )
                stamp = receipt
                if stream["timestamp_source"] == "header":
                    if header is None:
                        raise ConversionError(
                            f"Stream '{stream['id']}' missing acquisition timestamp"
                        )
                    stamp = int(header.stamp.sec) * 1_000_000_000 + int(
                        header.stamp.nanosec
                    )
                samples = result[stream["id"]]
                if samples and stamp <= samples[-1].time_ns:
                    raise ConversionError(
                        f"Nonmonotonic or duplicate timestamps / clock discontinuity: '{stream['id']}'"
                    )
                samples.append(Sample(stamp, receipt, msg))
    except ConversionError:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        raise ConversionError(f"Cannot read ROS bag {path}: {exc}") from exc
    available = {}
    for c in profile["state_channels"]:
        samples = result.get(c["stream"], [])
        if samples and (
            not c.get("joint") or c["joint"] in getattr(samples[0].message, "name", [])
        ):
            available[c["component"]] = True
    require_components(profile["components"], available, profile["profile_id"])
    for stream in profile["streams"]:
        if not result[stream["id"]]:
            raise ConversionError(
                f"Required stream '{stream['id']}' contains no messages"
            )
    return result


def extract(sample, channel):
    value = sample.message
    if channel.get("joint"):
        names = list(getattr(value, "name", []))
        if len(names) != len(set(names)) or channel["joint"] not in names:
            raise ConversionError(
                f"Component '{channel['component']}' missing or duplicate joint '{channel['joint']}'"
            )
        vector = getattr(value, channel["field"], None)
        idx = names.index(channel["joint"])
        if vector is None or len(vector) != len(names):
            raise ConversionError(
                f"Component '{channel['component']}' missing field '{channel['field']}' or wrong dimensions"
            )
        return float(vector[idx])
    for field in channel["field"].split("."):
        value = getattr(value, field, None)
        if value is None:
            raise ConversionError(
                f"Component '{channel['component']}' missing field '{channel['field']}'"
            )
    return float(value)


def decode_image(msg):
    import cv2
    import numpy as np

    if hasattr(msg, "format"):
        image = cv2.imdecode(np.asarray(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ConversionError("Corrupted compressed camera frame")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if msg.encoding not in ("rgb8", "bgr8", "mono8"):
        raise ConversionError(f"Unsupported camera encoding {msg.encoding}")
    channels = 1 if msg.encoding == "mono8" else 3
    if msg.step < msg.width * channels or len(msg.data) != msg.height * msg.step:
        raise ConversionError("Malformed camera frame dimensions/stride")
    image = np.asarray(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[
        :, : msg.width * channels
    ]
    image = image.reshape(msg.height, msg.width, channels)
    if channels == 1:
        image = np.repeat(image, 3, axis=2)
    return np.ascontiguousarray(image[:, :, ::-1] if msg.encoding == "bgr8" else image)


def align(data, profile):
    """Nearest observations, causal commands; reject reuse, gaps and excess skew.

    The output is a sampling grid; acquisition/arrival times are retained in
    the sidecar, never replaced by that grid. No interpolation or padding.
    """
    start = max(v[0].time_ns for v in data.values())
    end = min(v[-1].time_ns for v in data.values())
    if end <= start:
        raise ConversionError("No overlapping nonempty episode interval")
    times = {k: [s.time_ns for s in v] for k, v in data.items()}
    action_streams = {c["stream"] for c in profile["action_channels"]}
    used = {s["id"]: set() for s in profile["streams"] if s.get("camera_key")}
    frames = []
    n = 0
    while (stamp := start + round(n * 1_000_000_000 / profile["fps"])) <= end:
        selected = {}
        for key, ts in times.items():
            if key in action_streams:
                index = bisect.bisect_right(ts, stamp) - 1
            else:
                pos = bisect.bisect_left(ts, stamp)
                options = [i for i in (pos - 1, pos) if 0 <= i < len(ts)]
                index = min(options, key=lambda i: (abs(ts[i] - stamp), ts[i]))
            if index < 0 or abs(ts[index] - stamp) > profile["max_skew_ns"]:
                raise ConversionError(
                    f"Missing/stale frame for stream '{key}' at {stamp}"
                )
            if key in used:
                if index in used[key]:
                    raise ConversionError(
                        f"Duplicated/resampled camera frame in stream '{key}'"
                    )
                used[key].add(index)
            selected[key] = data[key][index]
        observations = [
            s.time_ns for k, s in selected.items() if k not in action_streams
        ]
        if max(observations) - min(observations) > profile["max_skew_ns"]:
            raise ConversionError(
                "Inter-observation timestamp skew exceeds explicit tolerance"
            )
        frame = {"time_ns": stamp, "selected": selected}
        for kind in ("state", "action"):
            channels = profile[kind + "_channels"]
            frame[kind] = normalize_vector(
                [extract(selected[c["stream"]], c) for c in channels],
                {
                    "representation": "qpos",
                    "names": profile[kind + "_names"],
                    "units": [c["source_unit"] for c in channels],
                },
                channels,
                profile=profile["profile_id"],
                components=profile["components"],
            )
        frame["images"] = {
            s["camera_key"]: decode_image(selected[s["id"]].message)
            for s in profile["streams"]
            if s.get("camera_key")
        }
        frames.append(frame)
        n += 1
    if len(frames) < 2:
        raise ConversionError("Episode requires at least two aligned frames")
    return frames


def validate_recording_phases(data, profile):
    """Check measured lift movement against timestamped phase and pose evidence.

    Original source samples are inspected, not just the lower-rate export grid.
    No approved stowed pose or velocity tolerance is supplied by this module.
    """
    from oa_embodied_ai.conversion.validation import validate_lift_phase

    lift = [c for c in profile["state_channels"] if c["component"] == "lift"]
    if not lift:
        return
    if len(lift) != 1 or lift[0].get("unit") != "m":
        raise ConversionError(
            "Lift phase needs one explicitly mapped lift position in metres"
        )
    samples = data[lift[0]["stream"]]
    moving = [
        s
        for previous, s in zip(samples, samples[1:], strict=False)
        if extract(previous, lift[0]) != extract(s, lift[0])
    ]
    evidence = profile.get("phase_evidence", {})
    if not moving:
        return
    if not evidence.get("stowed_pose_review"):
        raise ConversionError(
            "Lift phase requires reviewed stowed-pose evidence; park is a placeholder"
        )
    base = data.get(evidence.get("base_stream"), [])
    if not base:
        raise ConversionError(
            "Lift phase requires measured base odometry; stopped-base evidence missing"
        )
    joints = evidence.get("stowed_joints_rad", {})
    arm_joints = {
        c.get("joint")
        for c in profile["state_channels"]
        if c["component"] in ("left_arm", "right_arm")
    }
    if (
        not joints
        or set(joints) != arm_joints
        or not isinstance(evidence.get("pose_tolerance_rad"), (int, float))
    ):
        raise ConversionError(
            "Lift phase requires a complete reviewed stowed-arm pose and explicit tolerance"
        )
    phases = profile.get("phases", [])
    for sample in moving:
        phase = next(
            (
                p
                for p in phases
                if p.get("start_ns", -1) <= sample.time_ns <= p.get("end_ns", -1)
            ),
            {},
        )
        nearest = min(base, key=lambda s: abs(s.time_ns - sample.time_ns))
        if abs(nearest.time_ns - sample.time_ns) > profile["max_skew_ns"]:
            raise ConversionError("Lift phase base evidence is stale")
        twist = getattr(getattr(nearest.message, "twist", None), "twist", None)
        if twist is None:
            raise ConversionError(
                "Lift phase requires nav_msgs/Odometry measured twist"
            )
        stopped = all(
            float(getattr(twist.linear, axis)) == 0
            and float(getattr(twist.angular, axis)) == 0
            for axis in ("x", "y", "z")
        )
        names = list(getattr(sample.message, "name", []))
        positions = getattr(sample.message, "position", [])
        stowed = all(
            name in names
            and abs(float(positions[names.index(name)]) - target)
            <= evidence["pose_tolerance_rad"]
            for name, target in joints.items()
        )
        validate_lift_phase(
            [
                {
                    "lift_moving": True,
                    "phase": phase.get("name"),
                    "base_stopped": stopped,
                    "arms_stowed": stowed,
                }
            ],
            stowed_pose_review=evidence["stowed_pose_review"],
        )
