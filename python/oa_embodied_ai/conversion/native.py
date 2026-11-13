# SPDX-License-Identifier: MIT
"""Native LeRobot writer adapter; C9 remains an additive, canonical sidecar."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import tempfile
from pathlib import Path

from jsonschema import Draft202012Validator

from oa_embodied_ai.conversion.validation import ConversionError
from oa_embodied_ai.dataset_qa.contract import load_c9_schema
from oa_embodied_ai.training.evidence import execution_provenance
from oa_embodied_ai.training.inputs import file_sha256

LEROBOT_VERSION = "0.6.1"  # inspected development pin; not project release acceptance


def _dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sidecar_for(profile, frames, template):
    schema, _ = load_c9_schema()
    Draft202012Validator(schema).validate(template)
    sidecar = copy.deepcopy(template)
    native, capture = sidecar["native_lerobot"], sidecar["openamrobot"]
    native.update(
        dataset_id=profile["dataset_id"],
        dataset_revision=profile["dataset_revision"],
        episode_id=profile["episode_id"],
        episode_index=0,
        length=len(frames),
        fps=profile["fps"],
        task={"task_index": 0, "instruction": profile["task"]},
    )
    for kind, key in [("state", "observation.state"), ("action", "action")]:
        channels = profile[kind + "_channels"]
        native["features"][key] = {
            "dtype": "float32",
            "shape": [len(channels)],
            "profile": profile.get("native_profile_id", profile["profile_id"]),
            "dimensions": [
                {
                    "name": c["name"],
                    "unit": c["unit"],
                    "meaning": "measured state"
                    if kind == "state"
                    else "requested command",
                }
                for c in channels
            ],
        }
    start, end = frames[0]["time_ns"], frames[-1]["time_ns"]
    stream_map = {}
    for stream in profile["streams"]:
        if stream.get("camera_key"):
            key = stream["camera_key"]
            name = key.removeprefix("observation.images.")
            height, width, _ = frames[0]["images"][key].shape
            native["features"][key].update(
                shape=[height, width, 3], fps=profile["fps"], encoding="rgb8"
            )
            capture["streams"][name]["frame"].update(
                width=width,
                height=height,
                fps=profile["fps"],
                encoding="rgb8",
                stereo_view="mono",
            )
            capture["streams"][name]["physical_id"] = stream["id"]
            capture["streams"][name]["calibration"]["id"] = stream["calibration_id"]
            stream_map[name] = stream
    for kind, name in [("state", "robot_state"), ("action", "action")]:
        source_ids = {c["stream"] for c in profile[kind + "_channels"]}
        if len(source_ids) != 1:
            raise ConversionError(
                f"C9 v1 {kind} stream needs one acquisition source; multi-source aggregation requires a reviewed adapter"
            )
        stream_map[name] = next(s for s in profile["streams"] if s["id"] in source_ids)
    capture["samples"] = []
    for name, stream in stream_map.items():
        first = frames[0]["selected"][stream["id"]]
        last = frames[-1]["selected"][stream["id"]]
        mapping = lambda lo, hi: {
            "status": "valid",
            "method": "explicit same-domain ROS nanoseconds minus episode origin",
            "source_origin": start,
            "authoritative_origin_ns": 0,
            "scale_ns_per_source_unit": 1,
            "validity": {"source_start": lo, "source_end": hi},
        }
        timestamps = {
            "capture_source": "device"
            if stream["timestamp_source"] == "header"
            else "host_receipt",
            "host_receipt": {
                "clock_domain": stream["clock_domain"],
                "raw_unit": "ns",
                "mapping_to_authoritative": mapping(first.receipt_ns, last.receipt_ns),
            },
            "device": {
                "availability": "unavailable",
                "reason": "No acquisition header in source",
            },
        }
        if stream["timestamp_source"] == "header":
            timestamps["device"] = {
                "availability": "available",
                "clock_domain": stream["clock_domain"],
                "raw_unit": "ns",
                "mapping_to_authoritative": mapping(first.time_ns, last.time_ns),
            }
        capture["streams"][name]["timestamps"] = timestamps
    for i, frame in enumerate(frames):
        row = {
            "episode_index": 0,
            "frame_index": i,
            "task_index": 0,
            "authoritative_timestamp_ns": frame["time_ns"] - start,
            "streams": {},
        }
        for name, stream in stream_map.items():
            sample = frame["selected"][stream["id"]]
            selected = sample.time_ns - start
            if min(selected, sample.receipt_ns - start) < 0:
                raise ConversionError(
                    "Selected frame/receipt precedes episode origin; explicit timebase adapter required"
                )
            row["streams"][name] = {
                "sequence_id": i,
                "source_frame_id": f"{stream['id']}:{sample.time_ns}",
                "health": "fresh",
                "selection_timestamp_source": "device"
                if stream["timestamp_source"] == "header"
                else "host_receipt",
                "selected_timestamp_ns": selected,
                "host_receipt": {
                    "status": "valid",
                    "raw": sample.receipt_ns,
                    "mapped_authoritative_ns": sample.receipt_ns - start,
                },
                "device_timestamp": {
                    "status": "valid",
                    "raw": sample.time_ns,
                    "mapped_authoritative_ns": selected,
                }
                if stream["timestamp_source"] == "header"
                else {
                    "status": "unavailable",
                    "reason": "No acquisition header in source",
                },
                "measured_skew_ns": abs(sample.time_ns - frame["time_ns"]),
            }
        capture["samples"].append(row)
    capture["synchronization"] = {
        "status": "declared",
        "tolerance_ns": profile["max_skew_ns"],
        "basis": profile["profile_id"] + "; owner acceptance not inferred",
    }
    capture["timebase"]["clock_domain"] = next(iter(stream_map.values()))[
        "clock_domain"
    ]
    capture["events"] = []
    capture["outcome"] = {
        "status": "unknown",
        "reason": "No task outcome evidence supplied",
    }
    capture["recording"] = {
        "start_timestamp_ns": 0,
        "end_timestamp_ns": end - start,
        "status": "complete",
    }
    capture["provenance"]["data_origin"] = (
        "synthetic" if profile["origin"] == "synthetic" else "actual"
    )
    capture["provenance"]["source_profile"] = profile["profile_id"]
    # Preserve supplied configuration/calibration artifacts; never invent approval.
    capture["provenance"]["software"] = [
        {"name": key, "version": "source-commit", "commit": value, "repository": key}
        for key, value in profile["source_commits"].items()
    ]
    capture["provenance"]["lerobot_compatibility"]["validated_release_pin"] = None
    Draft202012Validator(schema).validate(sidecar)
    return sidecar


def export_native(
    frames, profile, template, output, *, source_bag, expected_source_sha256=None
):
    import numpy as np
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not frames:
        raise ConversionError("Cannot export an empty episode")
    if importlib.metadata.version("lerobot") != LEROBOT_VERSION:
        raise ConversionError(
            f"Native adapter requires inspected LeRobot {LEROBOT_VERSION}"
        )
    output = Path(output).resolve()
    if output.exists():
        raise ConversionError(f"Output exists; refusing overwrite: {output}")
    bag = Path(source_bag)
    before = {
        str(p.relative_to(bag)): file_sha256(p)
        for p in sorted(bag.rglob("*"))
        if p.is_file()
    }
    if not before:
        raise ConversionError("Source bag contains no artifacts")
    if expected_source_sha256 is not None and before != expected_source_sha256:
        raise ConversionError("Source bag artifacts changed after capture validation")
    sidecar = sidecar_for(profile, frames, template)
    features = {
        key: {
            "dtype": "video",
            "shape": tuple(image.shape),
            "names": ["height", "width", "channels"],
        }
        for key, image in frames[0]["images"].items()
    }
    for kind, key in [("state", "observation.state"), ("action", "action")]:
        features[key] = {
            "dtype": "float32",
            "shape": (len(profile[kind + "_channels"]),),
            "names": profile[kind + "_names"],
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Work is staged beside destination, then atomically renamed after loader success.
    with tempfile.TemporaryDirectory(
        prefix="oa-convert-", dir=output.parent
    ) as temporary:
        root = Path(temporary) / "dataset"
        writer = LeRobotDataset.create(
            profile["dataset_id"],
            profile["fps"],
            features,
            root=root,
            robot_type=profile["profile_id"],
            video_backend="pyav",
            rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="ultrafast", crf=18),
            encoder_threads=1,
        )
        try:
            for frame in frames:
                writer.add_frame(
                    {
                        "observation.state": np.array(frame["state"], dtype=np.float32),
                        "action": np.array(frame["action"], dtype=np.float32),
                        "task": profile["task"],
                        **frame["images"],
                    }
                )
            writer.save_episode(parallel_encoding=False)
        finally:
            writer.finalize()
        # Dataset identity is additive metadata; primary storage is upstream writer output.
        info_path = root / "meta/info.json"
        info = json.loads(info_path.read_text())
        info.update(
            dataset_id=profile["dataset_id"],
            dataset_revision=profile["dataset_revision"],
        )
        _dump(info_path, info)
        _dump(root / "meta/openamrobot/episodes/episode-000000.json", sidecar)
        bag = Path(source_bag)
        manifest = {
            "origin": profile["origin"],
            "profile": profile,
            "execution": execution_provenance(),
            "source_bag_sha256": before,
            "training_readiness": "NOT_VERIFIED",
            "physical_acceptance": "NOT_VERIFIED",
        }
        loaded = LeRobotDataset(profile["dataset_id"], root=root, video_backend="pyav")
        if len(loaded) != len(frames):
            raise ConversionError("Native LeRobot frame count mismatch")
        for index in range(len(frames)):
            row = loaded[index]
            if row["action"].shape != (len(profile["action_channels"]),) or row[
                "observation.state"
            ].shape != (len(profile["state_channels"]),):
                raise ConversionError("Native LeRobot vector dimensions mismatch")
            if not np.allclose(
                row["action"].numpy(), frames[index]["action"], atol=1e-5
            ) or not np.allclose(
                row["observation.state"].numpy(), frames[index]["state"], atol=1e-5
            ):
                raise ConversionError(
                    "Native LeRobot vector values differ from normalized source"
                )
            if (
                int(row["episode_index"]) != 0
                or int(row["frame_index"]) != index
                or int(row["task_index"]) != 0
                or abs(float(row["timestamp"]) - index / profile["fps"]) > 1e-6
            ):
                raise ConversionError(
                    "Native LeRobot episode/frame/task/timestamp binding mismatch"
                )
            for key in frames[index]["images"]:
                if row[key].shape != (3, *frames[index]["images"][key].shape[:2]):
                    raise ConversionError(f"Native LeRobot image load failed: {key}")
        manifest["native_loader"] = {
            "version": LEROBOT_VERSION,
            "frames_loaded": len(loaded),
            "status": "PASS",
        }
        _dump(root / "meta/openamrobot/conversion.json", manifest)
        after = {
            str(p.relative_to(bag)): file_sha256(p)
            for p in sorted(bag.rglob("*"))
            if p.is_file()
        }
        if before != after:
            raise ConversionError("Source bag artifacts changed during conversion")
        root.rename(output)
    return manifest
