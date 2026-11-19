# SPDX-License-Identifier: MIT
"""Proposed reference_integrity (19 Nov 2026): integrity checks, source binding and verify.

Only where a validated profile lists `reference_integrity` and declares an
`integrity` object. The checks run inside prepare_episode on the recorded
messages and refuse with a named code (IntegrityError):

UNITS_DEGREES     a channel declares degrees; canonical data is radians (rad/s)
UNITS             an arm or gripper channel is not rad, or has no declared range
UNITS_RANGE       a source value lies outside its range plus declared tolerance
                  (the message says when the value fits degrees or millimetres)
LIFT_UNIT         the lift is not in metres
GRIPPER_SIGN      a gripper range or value contradicts its declared opening sign
TIMESTAMP_RECONCILE  read, kept and dropped counts do not add up to the bag's
                  metadata.yaml message counts, or the grid has a gap
ACTION_CONVENTION an action[t] is not the latest command at or before t
                  (a one-frame shift is named)
CAMERA_MISSING    a camera has no frame within max_skew_ns of a grid time
CAMERA_SYNC       a camera frame is further from its grid time than declared
CALIBRATION_MISMATCH  camera_info differs from the image stream or changes
TEMPLATE_HASH     a template (all-zero) hash would remain in the C9 sidecar

`verify` recomputes everything from the bag and the dataset: file hashes,
bag hashes, the conversion and its reports, the sidecar, the stored vectors
and timestamps, and (with --run-log) the robot model hash. Development
evidence only: nothing here approves a profile, a rule or a dataset.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import json
import math
import sys
from pathlib import Path

from oa_embodied_ai.conversion.validation import ConversionError, normalize_vector

ZERO_HASH = "sha256:" + "0" * 64
DEGREE_UNITS = {"deg", "degree", "degrees", "deg/s", "degree/s", "degrees/s"}
ANGULAR_COMPONENTS = ("left_arm", "right_arm", "left_gripper", "right_gripper")
GRIPPER_COMPONENTS = ("left_gripper", "right_gripper")
OUTPUT_MANIFEST = "meta/openamrobot/conversion.json"


class IntegrityError(ConversionError):
    """A reference-integrity refusal with a stable code."""

    def __init__(self, code, message):
        super().__init__(f"{code}: {message}")
        self.code = code


def enabled(profile):
    return "reference_integrity" in profile.get("proposed_rules", [])


def _json_sha256(value):
    from oa_embodied_ai.training.inputs import json_sha256

    return json_sha256(value)


# Declarations -------------------------------------------------------------------------------------------------------


def check_declaration(profile):
    """Validate the `integrity` object and the channel units it governs."""
    integrity = profile.get("integrity")
    if not isinstance(integrity, dict):
        raise ConversionError("reference_integrity needs an integrity object")
    if integrity.get("canonical_angular_unit") != "rad" or integrity.get(
        "canonical_angular_velocity_unit"
    ) != "rad/s":
        raise IntegrityError(
            "UNITS_DEGREES",
            "canonical_angular_unit must be rad and canonical_angular_velocity_unit rad/s; "
            "degrees only through an explicit LeRobot adapter outside the canonical dataset",
        )
    if integrity.get("action_convention") != "latest_at_or_before":
        raise ConversionError("integrity.action_convention: only latest_at_or_before is defined")
    limit = integrity.get("camera_sync_max_skew_ns")
    if type(limit) is not int or not 0 <= limit <= profile["max_skew_ns"]:
        raise ConversionError(
            "integrity.camera_sync_max_skew_ns: an integer between 0 and max_skew_ns"
        )
    sources = integrity.get("calibration_sources")
    cameras = {s["id"] for s in profile["streams"] if s.get("camera_key")}
    if not isinstance(sources, dict) or set(sources) != cameras:
        raise ConversionError(
            "integrity.calibration_sources: one camera_info topic (or null with a reason) per camera stream"
        )
    model = integrity.get("robot_model")
    if not isinstance(model, dict) or not (
        (isinstance(model.get("sha256"), str) and len(model["sha256"]) == 64 and model.get("source"))
        or (model.get("sha256") is None and model.get("reason"))
    ):
        raise ConversionError("integrity.robot_model: sha256 with source, or null with a reason")
    for kind in ("state", "action"):
        for c in profile[kind + "_channels"]:
            for unit in (c["unit"], c["source_unit"]):
                if unit.lower() in DEGREE_UNITS:
                    raise IntegrityError(
                        "UNITS_DEGREES",
                        f"{kind} channel '{c['name']}' declares {unit}; canonical joint data is radians "
                        "(rad/s for velocities), degrees only through an explicit LeRobot adapter",
                    )
            if c["component"] in ANGULAR_COMPONENTS:
                if c["unit"] != "rad" or c["source_unit"] != "rad" or not c.get("range"):
                    raise IntegrityError(
                        "UNITS",
                        f"{kind} channel '{c['name']}' must be rad from a rad source with a declared range",
                    )
            if c["component"] == "lift" and (
                c["unit"] != "m" or c["source_unit"] not in ("m", "stop_index")
            ):
                raise IntegrityError(
                    "LIFT_UNIT",
                    f"{kind} channel '{c['name']}' is {c['unit']} from {c['source_unit']}; the lift is metres "
                    "(a stop index maps to its declared target in metres)",
                )
            if c["component"] == "base" and c["unit"] not in ("m/s", "rad/s"):
                raise IntegrityError("UNITS", f"base channel '{c['name']}' must be m/s or rad/s")
    gripper = integrity.get("gripper")
    names = {
        c["name"]
        for k in ("state", "action")
        for c in profile[k + "_channels"]
        if c["component"] in GRIPPER_COMPONENTS
    }
    if (
        not isinstance(gripper, dict)
        or gripper.get("unit") != "rad"
        or not isinstance(gripper.get("channels"), dict)
        or set(gripper["channels"]) != names
        or not gripper.get("source")
    ):
        raise IntegrityError(
            "GRIPPER_SIGN",
            "integrity.gripper: unit rad, an opening_sign for every gripper channel and its source",
        )
    for kind in ("state", "action"):
        for c in profile[kind + "_channels"]:
            if c["name"] not in gripper["channels"]:
                continue
            sign = gripper["channels"][c["name"]].get("opening_sign")
            low, high = c["range"]
            if sign not in (1, -1) or (sign == 1 and not (low == 0 and high > 0)) or (
                sign == -1 and not (high == 0 and low < 0)
            ):
                raise IntegrityError(
                    "GRIPPER_SIGN",
                    f"{kind} channel '{c['name']}' range {c['range']} contradicts opening_sign {sign} "
                    "(+1: [0, open]; -1, mirrored: [-open, 0])",
                )


# Source samples -----------------------------------------------------------------------------------------------------


def _unbounded(channel):
    copy_ = dict(channel)
    copy_.pop("range", None)
    copy_.pop("range_tolerance", None)
    return copy_


def _convert(value, channel, profile):
    """The channel's normalized value, before its range check."""
    return normalize_vector(
        [value],
        {"representation": "qpos", "names": [channel["name"]], "units": [channel["source_unit"]]},
        [_unbounded(channel)],
        profile=profile["profile_id"],
        components=[channel["component"]],
    )[0]


def _hint(value, channel):
    tolerance = channel.get("range_tolerance", 0)
    low, high = channel["range"][0] - tolerance, channel["range"][1] + tolerance
    if channel["unit"] == "rad" and low <= value * math.pi / 180 <= high:
        return " (the value fits the range in degrees: degree data in the canonical dataset)"
    if channel["unit"] == "m" and channel["source_unit"] != "stop_index" and value == int(value) and value > high:
        return " (an integer: a stop index read as metres)"
    if channel["unit"] == "m" and low <= value / 1000 <= high:
        return " (the value fits the range in millimetres: wrong lift unit)"
    return ""


def check_source(data, profile):
    """Gripper sign and range (plus tolerance) of every recorded source sample."""
    from oa_embodied_ai.recording.bag import extract

    gripper = profile["integrity"]["gripper"]["channels"]
    counts = {}
    for kind in ("state", "action"):
        for channel in profile[kind + "_channels"]:
            limits = channel.get("range")
            tolerance = channel.get("range_tolerance", 0)
            sign = gripper.get(channel["name"], {}).get("opening_sign")
            n = 0
            for sample in data[channel["stream"]]:
                value = _convert(extract(sample, channel), channel, profile)
                n += 1
                if sign is not None and sign * value < -tolerance:
                    raise IntegrityError(
                        "GRIPPER_SIGN",
                        f"{kind} '{channel['name']}' = {value} rad at {sample.time_ns} has the sign of the other "
                        f"finger; declared opening_sign {sign}",
                    )
                if limits is not None and not limits[0] - tolerance <= value <= limits[1] + tolerance:
                    raise IntegrityError(
                        "UNITS_RANGE",
                        f"{kind} '{channel['name']}' = {value} {channel['unit']} at {sample.time_ns} is outside "
                        f"{limits} plus tolerance {tolerance}{_hint(value, channel)}",
                    )
            counts[f"{kind}.{channel['name']}"] = n
    return {"source_samples_checked": counts, "status": "PASS"}


# Timestamps ---------------------------------------------------------------------------------------------------------


def bag_message_counts(bag):
    """Per-topic message counts from the bag's metadata.yaml."""
    import yaml

    meta = yaml.safe_load((Path(bag) / "metadata.yaml").read_text())
    info = meta["rosbag2_bagfile_information"]
    return {
        t["topic_metadata"]["name"]: int(t["message_count"])
        for t in info["topics_with_message_count"]
    }


def check_timestamps(data, profile, timestamps, alignment, frames, metadata):
    streams = {}
    for stream in profile["streams"]:
        entry = timestamps[stream["id"]]
        kept = len(data[stream["id"]])
        recorded = metadata.get(stream["topic"])
        row = {
            "topic": stream["topic"],
            "bag_metadata_messages": recorded,
            "read": entry["messages"],
            "kept": kept,
            "equal_timestamp_dropped": entry["equal_timestamp_dropped"],
        }
        if recorded != entry["messages"] or kept + entry["equal_timestamp_dropped"] != entry["messages"]:
            raise IntegrityError(
                "TIMESTAMP_RECONCILE",
                f"stream '{stream['id']}': bag metadata {recorded}, read {entry['messages']}, kept {kept} "
                f"+ dropped {entry['equal_timestamp_dropped']} do not reconcile",
            )
        streams[stream["id"]] = row
    fps = profile["fps"]
    origin = alignment.get("episode_origin", {})
    dropped = origin.get("pre_origin_frames_dropped", 0)
    if origin:
        first_index = round(origin["origin_ns"] * fps / 1_000_000_000)
        expected = [round((first_index + dropped + i) * 1_000_000_000 / fps) for i in range(len(frames))]
    else:
        expected = [frames[0]["time_ns"] + round(i * 1_000_000_000 / fps) for i in range(len(frames))]
    actual = [f["time_ns"] for f in frames]
    if actual != expected:
        index = next(i for i, (a, b) in enumerate(zip(actual, expected, strict=True)) if a != b)
        raise IntegrityError(
            "TIMESTAMP_RECONCILE", f"exported frame {index} at {actual[index]} is not grid time {expected[index]}"
        )
    return {
        "streams": streams,
        "grid": {
            "fps": fps,
            "exported_frames": len(frames),
            "pre_origin_frames_dropped": dropped,
            "first_grid_ns": actual[0],
            "last_grid_ns": actual[-1],
            "consecutive": True,
        },
        "status": "PASS",
    }


# Actions ------------------------------------------------------------------------------------------------------------


def expected_actions(data, profile, grid_times):
    """action[t] by the declared convention: per action stream, the latest recorded command
    at or before t, or the declared stand-in before the first one."""
    from oa_embodied_ai.recording.bag import _stand_ins, extract

    stand_ins = _stand_ins(data, profile)
    times = {k: [s.time_ns for s in v] for k, v in data.items()}
    channels = profile["action_channels"]
    result = []
    for t in grid_times:
        selected = {}
        for stream in dict.fromkeys(c["stream"] for c in channels):
            index = bisect.bisect_right(times[stream], t) - 1
            if index < 0:
                if stream not in stand_ins:
                    raise IntegrityError(
                        "ACTION_CONVENTION", f"no command of '{stream}' at or before grid time {t}"
                    )
                selected[stream] = stand_ins[stream]
            else:
                selected[stream] = data[stream][index]
        result.append(
            normalize_vector(
                [extract(selected[c["stream"]], c) for c in channels],
                {
                    "representation": "qpos",
                    "names": profile["action_names"],
                    "units": [c["source_unit"] for c in channels],
                },
                channels,
                profile=profile["profile_id"],
                components=profile["components"],
            )
        )
    return result


def _same(a, b, float32):
    if float32:
        import numpy as np

        return bool(np.array_equal(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)))
    return list(a) == list(b)


def compare_actions(actual, expected, *, float32=False):
    """None when equal; otherwise a message naming the first differing frame and a shift."""
    n = len(expected)
    if len(actual) != n:
        return f"{len(actual)} action rows for {n} frames"
    bad = [i for i in range(n) if not _same(actual[i], expected[i], float32)]
    if not bad:
        return None
    for shift in (1, -1):
        rows = [i for i in range(n) if 0 <= i - shift < n]
        if all(_same(actual[i], expected[i - shift], float32) for i in rows):
            return (
                f"action is shifted by {shift:+d} frame (action[t] holds the command of frame t{-shift:+d}); "
                f"first differing frame {bad[0]}; convention latest_at_or_before"
            )
    return f"action[{bad[0]}] differs from the latest command at or before its grid time"


def check_actions(frames, data, profile):
    expected = expected_actions(data, profile, [f["time_ns"] for f in frames])
    problem = compare_actions([f["action"] for f in frames], expected)
    if problem:
        raise IntegrityError("ACTION_CONVENTION", problem)
    changes = sum(1 for a, b in zip(expected, expected[1:]) if a != b)
    return {
        "convention": "latest_at_or_before",
        "statement": "action[t] is the latest recorded command at or before grid time t (declared stand-in "
        "before the first); observation[t] is the measured sample nearest t within max_skew_ns",
        "frames": len(expected),
        "action_changes": changes,
        "shift_detectable": changes > 0,
        "status": "PASS",
    }


# Cameras ------------------------------------------------------------------------------------------------------------


def check_cameras(frames, profile):
    limit = profile["integrity"]["camera_sync_max_skew_ns"]
    cameras = [s["id"] for s in profile["streams"] if s.get("camera_key")]
    per_frame = []
    for i, frame in enumerate(frames):
        skews = {k: abs(frame["selected"][k].time_ns - frame["time_ns"]) for k in cameras}
        worst = max(skews, key=skews.get)
        if skews[worst] > limit:
            raise IntegrityError(
                "CAMERA_SYNC",
                f"frame {i}: camera '{worst}' is {skews[worst]} ns from grid time {frame['time_ns']}; "
                f"declared limit {limit} ns",
            )
        per_frame.append(skews[worst])
    return {
        "limit_ns": limit,
        "reference": "grid time",
        "max_skew_ns": max(per_frame),
        "frame_of_max": per_frame.index(max(per_frame)),
        "per_frame_max_skew_ns": per_frame,
        "status": "PASS",
    }


def camera_missing(exc, profile):
    """Name an align refusal for a camera stream as CAMERA_MISSING."""
    text = str(exc)
    for stream in profile["streams"]:
        if stream.get("camera_key") and f"stream '{stream['id']}'" in text and "Missing/stale" in text:
            return IntegrityError("CAMERA_MISSING", text)
    return None


# Calibration --------------------------------------------------------------------------------------------------------


def _camera_info(msg):
    roi = msg.roi
    return {
        "frame_id": msg.header.frame_id,
        "height": int(msg.height),
        "width": int(msg.width),
        "distortion_model": msg.distortion_model,
        "d": [float(x) for x in msg.d],
        "k": [float(x) for x in msg.k],
        "r": [float(x) for x in msg.r],
        "p": [float(x) for x in msg.p],
        "binning_x": int(msg.binning_x),
        "binning_y": int(msg.binning_y),
        "roi": {
            "x_offset": int(roi.x_offset),
            "y_offset": int(roi.y_offset),
            "height": int(roi.height),
            "width": int(roi.width),
            "do_rectify": bool(roi.do_rectify),
        },
    }


def calibration(bag, profile, frames):
    """sha256 of the recorded camera_info content (stamps excluded) per camera stream."""
    from rosbags.highlevel import AnyReader

    sources = profile["integrity"]["calibration_sources"]
    streams = {s["id"]: s for s in profile["streams"]}
    result = {}
    with AnyReader([Path(bag)]) as reader:
        topics = {c.topic: c for c in reader.connections}
        contents = {}
        wanted = [topics[t] for t in sources.values() if isinstance(t, str) and t in topics]
        for conn, _, raw in reader.messages(connections=wanted):
            content = _camera_info(reader.deserialize(raw, conn.msgtype))
            contents.setdefault(conn.topic, {})
            key = _json_sha256(content)
            contents[conn.topic].setdefault(key, [content, 0])[1] += 1
    for stream_id, topic in sorted(sources.items()):
        stream = streams[stream_id]
        if not isinstance(topic, str) or topic not in contents:
            result[stream_id] = {
                "status": "NOT_AVAILABLE",
                "topic": topic,
                "reason": "camera_info topic not declared or not recorded in the bag",
            }
            continue
        distinct = contents[topic]
        if len(distinct) != 1:
            raise IntegrityError(
                "CALIBRATION_MISMATCH", f"{topic}: {len(distinct)} distinct camera_info contents in one episode"
            )
        (digest, (content, count)), = distinct.items()
        height, width, _ = frames[0]["images"][stream["camera_key"]].shape
        if (content["height"], content["width"]) != (height, width) or content["frame_id"] != stream["frame_id"]:
            raise IntegrityError(
                "CALIBRATION_MISMATCH",
                f"{topic}: {content['width']}x{content['height']} {content['frame_id']} differs from "
                f"'{stream_id}' {width}x{height} {stream['frame_id']}",
            )
        result[stream_id] = {
            "status": "recorded",
            "topic": topic,
            "sha256": digest,
            "hashed": "camera_info content without header stamp (frame_id, size, distortion model, D, K, R, P, binning, ROI)",
            "messages": count,
            "fx": content["k"][0],
            "residual": "not measured: simulated camera_info",
        }
    return result


# Episode and sidecar ------------------------------------------------------------------------------------------------


def not_available(profile, calibration_report):
    items = []
    if profile["robot_configuration_id"].startswith("unknown"):
        items.append({"item": "robot_configuration_id", "reason": profile["robot_configuration_id"]})
    run_info = profile.get("provenance", {}).get("run_info")
    if run_info:
        items.append({"item": "run_info", "reason": run_info})
    model = profile["integrity"]["robot_model"]
    if model.get("sha256") is None:
        items.append({"item": "robot_model_sha256", "reason": model["reason"]})
    for stream, entry in sorted(calibration_report.items()):
        if entry["status"] == "NOT_AVAILABLE":
            items.append({"item": f"calibration {stream}", "reason": entry["reason"]})
        items.append({"item": f"calibration residual {stream}", "reason": "not measured: simulated camera_info"})
    review = (profile.get("phase_evidence") or {}).get("stowed_pose_review")
    if isinstance(review, dict) and review.get("status") != "reviewed":
        items.append({"item": "stowed-pose evidence", "reason": f"stowed_pose_review status {review.get('status')}"})
    return items


def episode_report(bag, profile, data, frames, timestamps, alignment):
    from oa_embodied_ai.training.inputs import inventory

    calib = calibration(bag, profile, frames)
    return {
        "rule": "reference_integrity (Proposed, not approved)",
        "source": {
            "bag_files_sha256": inventory(bag),
            "bag_metadata_message_counts": bag_message_counts(bag),
            "recording": {
                k: profile.get("provenance", {}).get(k)
                for k in ("run_id", "run_head_sha", "artifact", "launch_profile", "acceptance_profile")
            },
            "source_commits": profile["source_commits"],
        },
        "robot_model": profile["integrity"]["robot_model"],
        "calibration": calib,
        "not_available": not_available(profile, calib),
        "checks": {
            "units": check_source(data, profile),
            "timestamps": check_timestamps(
                data, profile, timestamps, alignment, frames, bag_message_counts(bag)
            ),
            "actions": check_actions(frames, data, profile),
            "cameras": check_cameras(frames, profile),
            "gripper": {
                "unit": "rad",
                "channels": profile["integrity"]["gripper"]["channels"],
                "source": profile["integrity"]["gripper"]["source"],
                "status": "PASS",
            },
        },
    }


def fill_sidecar(sidecar, profile, report):
    """Real configuration and calibration hashes in place of template values."""
    capture = sidecar["openamrobot"]
    streams = {s["camera_key"].removeprefix("observation.images."): s for s in profile["streams"] if s.get("camera_key")}
    for name, stream in streams.items():
        entry = report["calibration"][stream["id"]]
        target = capture["streams"][name]
        if entry["status"] == "recorded":
            target["configuration_hash"] = "sha256:" + entry["sha256"]
            target["calibration"]["version"] = f"camera_info sha256 {entry['sha256']}"
        else:
            target["configuration_hash"] = "sha256:" + _json_sha256(stream)
            target["calibration"]["version"] = f"NOT_AVAILABLE: {entry['reason']}"
        target["calibration"]["residual"] = {
            "status": "not_measured",
            "reason": "Simulated camera_info (Gazebo); no calibration residual measured",
        }
    for name, kind in (("robot_state", "state"), ("action", "action")):
        capture["streams"][name]["configuration_hash"] = "sha256:" + _json_sha256(profile[kind + "_channels"])
    # independent of the source_commits key order, so a regeneration from the
    # (sorted-key) manifest gives the same sidecar
    capture["provenance"]["software"].sort(key=lambda s: s["name"])
    model = profile["integrity"]["robot_model"]
    if model.get("sha256"):
        capture["provenance"]["robot_configuration_hash"] = "sha256:" + model["sha256"]
    text = json.dumps(sidecar)
    if ZERO_HASH in text:
        raise IntegrityError("TEMPLATE_HASH", "a template all-zero hash remains in the C9 sidecar")
    return sidecar


def conversion_reference(episodes, template_hashes):
    profile = episodes[0].profile
    from oa_embodied_ai.dataset_qa.contract import load_c9_schema
    from oa_embodied_ai.recording.profile import capture_contract_digest

    _, schema = load_c9_schema()
    return {
        "manifest_version": 1,
        "status": "development evidence: Proposed rules, unregistered profile; not validated",
        "converter": {"repository": "openAMRobot/openamrobot-embodied-ai", "see": "execution"},
        "configuration": {
            "profile_sha256": _json_sha256(profile),
            "capture_contract_digest": capture_contract_digest(profile),
            "proposed_rules_enabled": list(episodes[0].enabled),
            "sidecar_templates": [e.template for e in episodes],
            "sidecar_template_sha256": template_hashes,
            "c9_schema": schema,
            "component_contract": profile.get("component_contract"),
        },
        "verify": "oa-verify-reference DATASET --bag BAG [--run-log LOG] recomputes this manifest",
    }


def output_hashes(root):
    from oa_embodied_ai.training.inputs import file_sha256

    root = Path(root)
    return {
        p.relative_to(root).as_posix(): file_sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.relative_to(root).as_posix() != OUTPUT_MANIFEST
    }


# Verify -------------------------------------------------------------------------------------------------------------


def _roundtrip(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _read_vectors(dataset):
    import pyarrow.parquet as pq

    files = sorted((Path(dataset) / "data").rglob("*.parquet"))
    if not files:
        raise IntegrityError("TAMPERED_OUTPUT", "no parquet data files")
    tables = [pq.read_table(f) for f in files]
    rows = {}
    for column in ("observation.state", "action", "timestamp", "frame_index", "episode_index"):
        rows[column] = [x for t in tables for x in t.column(column).to_pylist()]
    return rows


def _video_frames(dataset):
    try:
        import av
    except ImportError:
        return "NOT_VERIFIED: PyAV not installed"
    counts = {}
    for path in sorted((Path(dataset) / "videos").rglob("*.mp4")):
        with av.open(str(path)) as container:
            counts[path.relative_to(dataset).as_posix()] = sum(1 for _ in container.decode(video=0))
    return counts


def verify(dataset, bag, *, run_log=None, require_clean_converter=False):
    """Recompute the conversion from the bag and compare it with the dataset; raise on the first difference."""
    import numpy as np

    from oa_embodied_ai.conversion.batch import prepare_episode
    from oa_embodied_ai.conversion.native import native_timestamp_matches, sidecar_for
    from oa_embodied_ai.training.evidence import execution_provenance
    from oa_embodied_ai.training.inputs import inventory

    dataset, bag = Path(dataset), Path(bag)
    manifest = json.loads((dataset / OUTPUT_MANIFEST).read_text())
    reference = manifest.get("reference")
    if not reference or len(manifest["episodes"]) != 1:
        raise IntegrityError("MANIFEST", "no reference binding for a single-episode dataset")
    result = {"dataset": str(dataset), "checks": {}}
    actual = output_hashes(dataset)
    expected = reference["output_sha256"]
    if actual != expected:
        changed = sorted(k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k))
        raise IntegrityError("TAMPERED_OUTPUT", f"dataset files differ from the manifest: {changed}")
    result["checks"]["output_sha256"] = {"files": len(actual), "status": "PASS"}
    record = manifest["episodes"][0]
    if inventory(bag) != record["source_bag_sha256"]:
        raise IntegrityError("SOURCE_MISMATCH", f"bag files differ from the manifest: {bag}")
    result["checks"]["source_bag_sha256"] = {"files": len(record["source_bag_sha256"]), "status": "PASS"}
    current = execution_provenance()
    converter = manifest["execution"]
    if require_clean_converter and not converter.get("source_tree_clean"):
        raise IntegrityError("CONVERTER", "the dataset was written from a converter tree with local changes")
    result["checks"]["converter"] = {
        "manifest_commit": converter.get("git_commit"),
        "verifier_commit": current.get("git_commit"),
        "same_source": converter.get("source_sha256") == current.get("source_sha256"),
        "manifest_tree_clean": converter.get("source_tree_clean"),
    }
    template = reference["configuration"]["sidecar_templates"][0]
    episode = prepare_episode(
        bag, copy.deepcopy(record["profile"]), template,
        development=record["profile"].get("origin") == "synthetic",
        enabled=manifest["proposed_rules_enabled"],
    )
    for key, value in (
        ("stream_timestamps", episode.timestamp_report),
        ("alignment", episode.alignment_report),
        ("lift_phase", episode.lift_phase_report),
        ("reference", episode.reference_report),
    ):
        if _roundtrip(value) != record[key]:
            raise IntegrityError("MANIFEST_MISMATCH", f"episode {key} recomputed from the bag differs from the manifest")
    result["checks"]["recomputed_reports"] = {"status": "PASS"}
    sidecar = sidecar_for(episode.profile, episode.frames, template, episode_index=0, task_index=0,
                          reference=episode.reference_report)
    stored = json.loads((dataset / "meta/openamrobot/episodes/episode-000000.json").read_text())
    if _roundtrip(sidecar) != stored:
        raise IntegrityError("SIDECAR_MISMATCH", "C9 sidecar recomputed from the bag differs from the stored sidecar")
    result["checks"]["c9_sidecar"] = {"status": "PASS"}
    rows = _read_vectors(dataset)
    frames = episode.frames
    if len(rows["action"]) != len(frames):
        raise IntegrityError("FRAME_COUNT", f"{len(rows['action'])} rows for {len(frames)} recomputed frames")
    if not np.array_equal(np.asarray(rows["observation.state"], dtype=np.float32),
                          np.asarray([f["state"] for f in frames], dtype=np.float32)):
        raise IntegrityError("STATE_MISMATCH", "observation.state differs from the recomputed measured state")
    problem = compare_actions(rows["action"], [f["action"] for f in frames], float32=True)
    if problem:
        raise IntegrityError("ACTION_SHIFTED" if "shifted" in problem else "ACTION_MISMATCH", problem)
    fps = record["profile"]["fps"]
    for i, (stamp, index) in enumerate(zip(rows["timestamp"], rows["frame_index"], strict=True)):
        if index != i or not native_timestamp_matches(stamp, i, fps):
            raise IntegrityError("TIMESTAMP_MISMATCH", f"row {i}: frame_index {index}, timestamp {stamp}")
    result["checks"]["vectors_and_timestamps"] = {"frames": len(frames), "status": "PASS"}
    videos = _video_frames(dataset)
    if isinstance(videos, dict):
        wrong = {k: v for k, v in videos.items() if v != len(frames)}
        if wrong or len(videos) != 4:
            raise IntegrityError("VIDEO_FRAMES", f"decoded frame counts {videos} for {len(frames)} frames")
        result["checks"]["video_frames"] = {"files": videos, "status": "PASS"}
    else:
        result["checks"]["video_frames"] = {"status": videos}
    model = record["reference"]["robot_model"]
    if run_log is None or not model.get("sha256"):
        result["checks"]["robot_model"] = {"status": "NOT_RECOMPUTED", "reason": "no --run-log" if run_log is None else model.get("reason")}
    else:
        text = Path(run_log).read_text(errors="replace")
        if f'"model_sha256": "{model["sha256"]}"' not in text:
            raise IntegrityError("ROBOT_MODEL", f"model_sha256 {model['sha256']} is not in the run log {run_log}")
        result["checks"]["robot_model"] = {"status": "PASS", "run_log": str(run_log)}
    result["dataset_sha256"] = _json_sha256(actual)
    result["manifest_sha256"] = _json_sha256(manifest)
    result["status"] = "PASS"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Recompute a reference dataset from its bag and compare.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--run-log", type=Path)
    parser.add_argument("--require-clean-converter", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify(args.dataset, args.bag, run_log=args.run_log,
                        require_clean_converter=args.require_clean_converter)
    except (ConversionError, OSError, ValueError, KeyError) as exc:
        print(f"REJECT: {exc}")
        return 2
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.json:
        args.json.write_text(text + "\n")
    print(text)
    print("PASS: recomputed from the bag; development evidence, training/physical acceptance NOT_VERIFIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
