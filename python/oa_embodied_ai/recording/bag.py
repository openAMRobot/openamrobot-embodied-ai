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


def read_bag(path, profile, report=None):
    """Read the declared streams in bag (receipt) order.

    A stream's time is its header stamp (`timestamp_source: header`) or the bag
    receipt time (`bag`). A decreasing time is a clock discontinuity and always
    rejects. An equal time rejects too, unless the stream declares
    `equal_timestamp_rule: keep_last`: then, of a group of samples with one
    time, the last one in bag order (receipt order) is kept and the earlier ones are counted in
    `report`, never silently. Under simulation time two messages can carry one
    /clock value; the rule names which of them a frame uses, it does not
    reorder, interpolate or invent a time.
    """
    from rosbags.highlevel import AnyReader

    from oa_embodied_ai.conversion import reference

    counts = {
        s["id"]: {"messages": 0, "equal_timestamp_dropped": 0}
        for s in profile["streams"]
    }

    path = Path(path)
    if not (path / "metadata.yaml").is_file():
        raise ConversionError(f"ROS bag metadata missing: {path}/metadata.yaml")
    result = {s["id"]: [] for s in profile["streams"]}
    by_topic = {s["topic"]: s for s in profile["streams"]}
    vectors = _vector_sources(profile)
    reference_checks = reference.enabled(profile)
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
                # An empty header frame (JointState, controller states) takes the
                # declared configuration frame; a non-empty one must still match.
                configured = (
                    header is not None
                    and header.frame_id == ""
                    and stream.get("frame_id_source") == "configuration"
                )
                if (
                    header is not None
                    and not configured
                    and header.frame_id != stream["frame_id"]
                ):
                    raise ConversionError(
                        f"Component '{stream['component']}' frame ID mismatch on '{conn.topic}'"
                    )
                stamp = receipt
                if stream["timestamp_source"] == "header":
                    if header is None:
                        raise ConversionError(
                            f"Stream '{stream['id']}' missing acquisition timestamp"
                        )
                    if not 0 <= header.stamp.nanosec < 1_000_000_000:
                        raise ConversionError(
                            f"Invalid ROS header timestamp on '{stream['id']}' "
                            f"at receipt {receipt}: nanosec={header.stamp.nanosec}"
                        )
                    stamp = int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)
                samples = result[stream["id"]]
                counts[stream["id"]]["messages"] += 1
                if samples and (
                    stamp < samples[-1].time_ns
                    or (
                        stamp == samples[-1].time_ns
                        and stream.get("equal_timestamp_rule") != "keep_last"
                    )
                ):
                    raise ConversionError(
                        f"Nonmonotonic or duplicate timestamps / clock discontinuity: '{stream['id']}'"
                    )
                if samples and stamp == samples[-1].time_ns:
                    # Validate bytes that resampling will no longer see.
                    if stream.get("camera_key"):
                        decode_image(samples[-1].message)
                    if reference_checks:
                        reference.check_source(result, profile, discarded=(stream["id"], samples[-1]))
                    else:
                        for kind, stream_id, channels in vectors:
                            if stream_id == stream["id"]:
                                _validate_vector_sample(
                                    samples[-1], channels, profile, kind, stream_id
                                )
                    samples[-1] = Sample(stamp, receipt, msg)
                    counts[stream["id"]]["equal_timestamp_dropped"] += 1
                    continue
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
    if report is not None:
        for stream in profile["streams"]:
            entry = counts[stream["id"]]
            entry["timestamp_source"] = stream["timestamp_source"]
            entry["equal_timestamp_rule"] = stream.get("equal_timestamp_rule")
            report[stream["id"]] = entry
    return result


def _field(value, path):
    for field in path.split("."):
        value = getattr(value, field, None)
        if value is None:
            return None
    return value


def extract(sample, channel):
    """One channel value of a recorded message.

    A joint channel reads the joint's entry of `field`, indexed by `name`
    (JointState). Proposed (controller_reference): a channel declaring
    `names_field: joint_names` indexes a nested field such as
    `reference.positions` by `joint_names`, as in
    control_msgs/JointTrajectoryControllerState; every vector must then name
    each joint once and have one entry per joint.
    """
    value = sample.message
    if channel.get("joint"):
        names = list(getattr(value, channel.get("names_field", "name"), []))
        if len(names) != len(set(names)) or channel["joint"] not in names:
            raise ConversionError(
                f"Component '{channel['component']}' missing or duplicate joint '{channel['joint']}'"
            )
        vector = (
            _field(value, channel["field"])
            if "names_field" in channel
            else getattr(value, channel["field"], None)
        )
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


def _vector_sources(profile):
    sources = []
    for kind in ("state", "action"):
        by_stream = {}
        for channel in profile.get(kind + "_channels", []):
            by_stream.setdefault(channel["stream"], []).append(channel)
        sources.extend((kind, stream, channels) for stream, channels in by_stream.items())
    return sources


def _validate_vector_sample(sample, channels, profile, kind, stream, flags=None):
    try:
        normalize_vector(
            [extract(sample, c) for c in channels],
            {
                "representation": "qpos",
                "names": [c["name"] for c in channels],
                "units": [c["source_unit"] for c in channels],
            },
            channels,
            profile=profile["profile_id"],
            components=list(dict.fromkeys(c["component"] for c in channels)),
            flags=flags,
        )
    except (TypeError, AttributeError, IndexError) as exc:
        raise ConversionError(
            f"Malformed {kind} source sample in stream '{stream}' at {sample.time_ns}"
        ) from exc


def validate_source_samples(data, profile, flags=None):
    """Reject invalid kept messages even between grid frames and count range flags."""
    for kind, stream, channels in _vector_sources(profile):
        for sample in data[stream]:
            _validate_vector_sample(sample, channels, profile, kind, stream, flags)
    for stream in profile["streams"]:
        if stream.get("camera_key"):
            for sample in data[stream["id"]]:
                decode_image(sample.message)


def _single(value, channel, profile):
    """Normalize one value of one channel (units, declared stop targets, range)."""
    return normalize_vector(
        [value],
        {
            "representation": "qpos",
            "names": [channel["name"]],
            "units": [channel["source_unit"]],
        },
        [channel],
        profile=profile["profile_id"],
        components=[channel["component"]],
    )[0]


def _stand_ins(data, profile):
    """Proposed held_command: the declared value a command stream holds before its
    first message, checked against the first measured sample of a state channel."""
    from types import SimpleNamespace

    result = {}
    states = {c["name"]: c for c in profile["state_channels"]}
    for stream in profile["streams"]:
        first = (stream.get("hold") or {}).get("before_first")
        if not first:
            continue
        channels = [c for c in profile["action_channels"] if c["stream"] == stream["id"]]
        fields = {c["field"] for c in channels}
        if len(fields) != 1 or "." in next(iter(fields)):
            raise ConversionError(
                f"hold.before_first on '{stream['id']}' needs one top-level command field"
            )
        message = SimpleNamespace(**{next(iter(fields)): first["field_value"]})
        state = states[first["check_state_channel"]]
        measured = _single(extract(data[state["stream"]][0], state), state, profile)
        for channel in channels:
            target = _single(first["field_value"], channel, profile)
            if abs(target - measured) > first["tolerance"]:
                raise ConversionError(
                    f"Held command stand-in of '{stream['id']}' ({target} {channel['unit']}) differs from the "
                    f"first measured '{state['name']}' ({measured}) by more than {first['tolerance']}"
                )
        result[stream["id"]] = Sample(None, None, message)
    return result


def align(data, profile, report=None):
    """Nearest observations, causal commands; reject reuse, gaps and excess skew.

    The output is a sampling grid; acquisition/arrival times are retained in
    the sidecar, never replaced by that grid. No interpolation or padding.

    Proposed, only where a validated profile declares them (19 Nov 2026):
    episode_origin first_complete_grid: grid times are integer multiples of
    1/fps in the declared clock domain; the origin is the first grid time at or
    after the latest first sample of the required streams, so every required
    stream has a sample at or before it. Leading frames that select a sample
    (time or receipt) before the origin are dropped and counted; such a
    selection in a later frame refuses. held_command: a command stream with
    `hold` keeps its last message beyond max_skew_ns (until_next), or only
    while every commanded value of that message is zero (while_zero); a
    declared before_first value stands in before the first message. Held
    streams do not bound the episode. Counts go to `report`.
    """
    source_flags = {}
    validate_source_samples(data, profile, source_flags)
    holds = {s["id"]: s["hold"] for s in profile["streams"] if s.get("hold")}
    stand_ins = _stand_ins(data, profile)
    latest_first = max(v[0].time_ns for k, v in data.items() if k not in stand_ins)
    end = min(v[-1].time_ns for k, v in data.items() if k not in holds)
    fps = profile["fps"]
    origin_rule = profile.get("episode_origin")
    if origin_rule == "first_complete_grid":
        first_index = -(-latest_first * fps // 1_000_000_000)
        grid = lambda n: round((first_index + n) * 1_000_000_000 / fps)
    else:
        grid = lambda n: latest_first + round(n * 1_000_000_000 / fps)
    origin = grid(0)
    if end <= origin:
        raise ConversionError("No overlapping nonempty episode interval")
    times = {k: [s.time_ns for s in v] for k, v in data.items()}
    action_streams = {c["stream"] for c in profile["action_channels"]}
    action_channels = {}
    for c in profile["action_channels"]:
        action_channels.setdefault(c["stream"], []).append(c)
    used = {s["id"]: set() for s in profile["streams"] if s.get("camera_key")}
    held = {k: {"rule": h["rule"], "held_frames": 0, "stand_in_frames": 0} for k, h in holds.items()}
    frame_flags = {}
    frames = []
    dropped = 0
    dropped_offsets = {}
    n = 0
    while (stamp := grid(n)) <= end:
        n += 1
        selected = {}
        for key, ts in times.items():
            if key in action_streams:
                index = bisect.bisect_right(ts, stamp) - 1
                if index < 0 and key in stand_ins:
                    selected[key] = stand_ins[key]
                    held[key]["stand_in_frames"] += 1
                    continue
                if index >= 0 and stamp - ts[index] > profile["max_skew_ns"] and key in holds:
                    if holds[key]["rule"] == "while_zero" and any(
                        extract(data[key][index], c) != 0 for c in action_channels[key]
                    ):
                        raise ConversionError(
                            f"Held command of '{key}' is not zero beyond max_skew_ns at {stamp}"
                        )
                    held[key]["held_frames"] += 1
                    selected[key] = data[key][index]
                    continue
            else:
                pos = bisect.bisect_left(ts, stamp)
                options = [i for i in (pos - 1, pos) if 0 <= i < len(ts)]
                index = min(options, key=lambda i: (abs(ts[i] - stamp), ts[i]))
            if index < 0 or abs(ts[index] - stamp) > profile["max_skew_ns"]:
                raise ConversionError(
                    f"Missing/stale frame for stream '{key}' at {stamp}"
                )
            selected[key] = data[key][index]
        if origin_rule == "first_complete_grid":
            early = {
                k: min(s.time_ns, s.receipt_ns) - origin
                for k, s in selected.items()
                if s.time_ns is not None and min(s.time_ns, s.receipt_ns) < origin
            }
            if early:
                if frames:
                    raise ConversionError(
                        f"Frame at {stamp} selects samples before the episode origin {origin}: {sorted(early)}"
                    )
                dropped += 1
                if not dropped_offsets:
                    dropped_offsets = early
                continue
        for key, sample in selected.items():
            if key in used:
                index = times[key].index(sample.time_ns)
                if index in used[key]:
                    raise ConversionError(
                        f"Duplicated/resampled camera frame in stream '{key}'"
                    )
                used[key].add(index)
        observations = [
            s.time_ns for k, s in selected.items() if k not in action_streams
        ]
        if max(observations) - min(observations) > profile["max_skew_ns"]:
            raise ConversionError(
                "Inter-observation timestamp skew exceeds explicit tolerance"
            )
        frame = {"time_ns": stamp, "selected": selected}
        if origin_rule == "first_complete_grid":
            frame["episode_origin_ns"] = origin
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
                flags=frame_flags,
            )
        frame["images"] = {
            s["camera_key"]: decode_image(selected[s["id"]].message)
            for s in profile["streams"]
            if s.get("camera_key")
        }
        frames.append(frame)
    if len(frames) < 2:
        raise ConversionError("Episode requires at least two aligned frames")
    if report is not None:
        if origin_rule:
            report["episode_origin"] = {
                "rule": origin_rule,
                "origin_ns": origin,
                "latest_first_sample_ns": latest_first,
                "pre_origin_frames_dropped": dropped,
                "first_dropped_frame_offsets_ns": dropped_offsets,
            }
        if holds:
            report["held_commands"] = held
        if any(c.get("range_tolerance") is not None for k in ("state", "action") for c in profile[k + "_channels"]):
            report["range_tolerance_flags"] = {
                "tolerance": {
                    c["name"]: [c["range_tolerance"], c["unit"]]
                    for k in ("state", "action")
                    for c in profile[k + "_channels"]
                    if c.get("range_tolerance") is not None
                },
                "source_samples": source_flags,
                "exported_frames": frame_flags,
                "values": "kept unchanged, never clamped",
            }
        report["frames"] = len(frames)
    return frames


def lift_positioning_intervals(data, profile):
    """Proposed lift_positioning_from_command: one lift_positioning interval per
    recorded lift command, from the command to the first measured lift sample
    within `arrival_tolerance` of its target, or to the next command, or to the
    end of the recording, whichever comes first."""
    rule = profile["phase_evidence"]["lift_positioning"]
    command = next(c for c in profile["action_channels"] if c["name"] == rule["command_channel"])
    state = next(c for c in profile["state_channels"] if c["name"] == rule["state_channel"])
    measured = [(s.time_ns, _single(extract(s, state), state, profile)) for s in data[state["stream"]]]
    intervals = []
    commands = data[command["stream"]]
    for i, sample in enumerate(commands):
        target = _single(extract(sample, command), command, profile)
        # a later command supersedes this one; without arrival the interval
        # ends at the next command or at the end of the recording
        until = commands[i + 1].time_ns if i + 1 < len(commands) else measured[-1][0]
        arrival = next(
            (
                t
                for t, v in measured
                if sample.time_ns <= t <= until and abs(v - target) <= rule["arrival_tolerance"]
            ),
            until,
        )
        intervals.append(
            {"name": "lift_positioning", "start_ns": sample.time_ns, "end_ns": arrival, "target": target}
        )
    return intervals


def _reviewed(review):
    """A stowed-pose review object counts only with status reviewed; a string is
    the earlier (synthetic) form and is unchanged."""
    if isinstance(review, dict):
        return review.get("status") == "reviewed" and bool(review.get("approval_ref"))
    return bool(review)


def validate_recording_phases(data, profile, report=None):
    """Check measured lift movement against timestamped phase and pose evidence.

    Original source samples are inspected, not just the lower-rate export grid.
    No approved stowed pose or velocity tolerance is supplied by this module.

    Proposed (19 Nov 2026): `phase_evidence.lift_positioning` derives the
    lift_positioning intervals from the recorded lift commands
    (lift_positioning_from_command); `phase_evidence.on_missing_evidence:
    report` (lift_phase_report) writes the evidence of every moving sample to
    `report` with status NOT_VERIFIED instead of refusing. A stowed-pose review
    object needs status reviewed and an approval_ref; a placeholder never
    passes.
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
    reporting = evidence.get("on_missing_evidence") == "report"
    reviewed = _reviewed(evidence.get("stowed_pose_review"))
    phases = list(profile.get("phases", []))
    if "lift_positioning" in evidence:
        phases += lift_positioning_intervals(data, profile)
    joints = evidence.get("stowed_joints_rad", {})
    arm_joints = {
        c.get("joint")
        for c in profile["state_channels"]
        if c["component"] in ("left_arm", "right_arm")
    }
    pose_complete = bool(joints) and set(joints) == arm_joints and isinstance(
        evidence.get("pose_tolerance_rad"), (int, float)
    )
    base = data.get(evidence.get("base_stream"), [])
    if not reporting:
        if not reviewed:
            raise ConversionError(
                "Lift phase requires reviewed stowed-pose evidence; park is a placeholder"
            )
        if not base:
            raise ConversionError(
                "Lift phase requires measured base odometry; stopped-base evidence missing"
            )
        if not pose_complete:
            raise ConversionError(
                "Lift phase requires a complete reviewed stowed-arm pose and explicit tolerance"
            )
    counts = {"moving_samples": len(moving), "in_lift_positioning": 0, "base_stopped": 0, "arms_within_pose_tolerance": 0}
    outside = []
    for sample in moving:
        phase = next(
            (
                p
                for p in phases
                if p.get("start_ns", -1) <= sample.time_ns <= p.get("end_ns", -1)
            ),
            {},
        )
        stopped = False
        if base:
            nearest = min(base, key=lambda s: abs(s.time_ns - sample.time_ns))
            if abs(nearest.time_ns - sample.time_ns) > profile["max_skew_ns"]:
                if not reporting:
                    raise ConversionError("Lift phase base evidence is stale")
            else:
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
        stowed = pose_complete and all(
            name in names
            and abs(float(positions[names.index(name)]) - target)
            <= evidence["pose_tolerance_rad"]
            for name, target in joints.items()
        )
        in_phase = phase.get("name") == "lift_positioning"
        counts["in_lift_positioning"] += in_phase
        counts["base_stopped"] += stopped
        counts["arms_within_pose_tolerance"] += stowed
        if not in_phase:
            if outside and sample.time_ns - outside[-1]["end_ns"] <= profile["max_skew_ns"]:
                outside[-1]["end_ns"] = sample.time_ns
                outside[-1]["samples"] += 1
            else:
                outside.append({"start_ns": sample.time_ns, "end_ns": sample.time_ns, "samples": 1})
        if reporting:
            continue
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
    if report is not None:
        review = evidence.get("stowed_pose_review")
        verified = reviewed and pose_complete and all(
            counts[k] == len(moving) for k in ("in_lift_positioning", "base_stopped", "arms_within_pose_tolerance")
        )
        report.update(
            status="PASS" if verified else "NOT_VERIFIED",
            mode="report" if reporting else "enforce",
            stowed_pose_review=review,
            stowed_pose_reviewed=reviewed,
            pose_tolerance_rad=evidence.get("pose_tolerance_rad"),
            lift_positioning_intervals=[p for p in phases if p.get("name") == "lift_positioning"],
            moving_outside_lift_positioning=outside,
            **counts,
        )
