# SPDX-License-Identifier: MIT
"""Deterministic, explicitly synthetic C10 inputs derived from the C9 example.

No value here constitutes an approved recording threshold, calibration, device
specification, configuration identity, or production release pin.
"""

from copy import deepcopy
import json
from pathlib import Path

import yaml
from oa_embodied_ai.dataset_qa.contract import c9_asset


ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_HASH = "sha256:" + "1" * 64
CAMERAS = ("camera_wrist_left", "camera_wrist_right", "head", "base")


def fixture():
    """Return independent sidecar/native/policy objects for a two-frame episode."""
    sidecar = deepcopy(yaml.safe_load(
        c9_asset("schemas/yaml/c9-lerobot-dataset-valid.yaml").read_text()
    ))
    capture = sidecar["openamrobot"]
    capture["synchronization"] = {
        "status": "declared",
        "tolerance_ns": 100,
        "basis": "SYNTHETIC TEST ONLY: no physical evidence or production approval.",
    }
    for name, stream in capture["streams"].items():
        stream["configuration_hash"] = SYNTHETIC_HASH
        if name in CAMERAS:
            stream["calibration"]["residual"] = {
                "status": "measured", "name": "synthetic_reprojection_error",
                "value": 0.25, "unit": "pixel", "statistic": "rms",
                "method": "synthetic deterministic test fixture; no measurement",
            }
    capture["provenance"]["robot_configuration_hash"] = SYNTHETIC_HASH
    # The C9 example is structural only: its marker starts after linked frame 0.
    # Include that observation time to make this a positive C10 linkage fixture.
    capture["events"][0]["start_timestamp_ns"] = 0
    manifest = sidecar["native_lerobot"]
    features = {
        key: {"dtype": descriptor["dtype"], "shape": deepcopy(descriptor["shape"])}
        for key, descriptor in manifest["features"].items()
    }
    # Native LeRobot vector names carry the same ordering as C9 dimensions.
    for key in ("observation.state", "action"):
        features[key]["names"] = [d["name"] for d in manifest["features"][key]["dimensions"]]
    for key, descriptor in features.items():
        if descriptor["dtype"] == "video":
            descriptor["names"] = ["height", "width", "channels"]
    native = {
        "info": {
            "codebase_version": manifest["format_version"],
            "dataset_id": manifest["dataset_id"],
            "dataset_revision": manifest["dataset_revision"],
            "fps": manifest["fps"], "features": features,
            "total_episodes": 1, "total_frames": 2, "total_tasks": 1,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        },
        "episode": {
            "episode_index": 0, "episode_id": manifest["episode_id"],
            "length": 2, "tasks": [manifest["task"]["instruction"]],
            "data/chunk_index": 0, "data/file_index": 0,
            "dataset_from_index": 0, "dataset_to_index": 2,
        },
        "tasks": {0: manifest["task"]["instruction"]},
        "rows": [
            {"episode_index": 0, "frame_index": i, "task_index": 0,
             "timestamp": i / 30, "index": i,
             "observation.state": [float(i)] * 16,
             "action": [float(i + 1)] * 16}
            for i in range(2)
        ],
    }
    return sidecar, native, synthetic_policy(sidecar)


def synthetic_policy(sidecar):
    """Explicit test thresholds; ``approved`` is only the checker input flag."""
    features = sidecar["native_lerobot"]["features"]
    return {
        "id": "SYNTHETIC-TEST-ONLY-never-production-approval",
        "evidence": "Deterministic synthetic unit tests; no physical validation.",
        "approved": True,
        "profiles": {
            key: {"profile": features[key]["profile"],
                  "dimensions": deepcopy(features[key]["dimensions"]),
                  "reference_frame": None}
            for key in ("observation.state", "action")
        },
        "action_bounds": {
            d["name"]: {"min": -180, "max": 180, "unit": d["unit"]}
            for d in features["action"]["dimensions"]
        },
        "synchronization": {"max_skew_ns": 100},
        "stream_quality": {
            name: {"min_fps": 29, "max_freshness_ns": 100,
                   "max_sequence_gap": 0, "max_reuse": 0}
            for name in sidecar["openamrobot"]["streams"]
        },
        "action_latency": {
            "min_ns": 0, "max_ns": 2000,
            "reference_frame_offset": 0,
            "values": sidecar["openamrobot"]["action_semantics"]["values"],
            "timestamp": sidecar["openamrobot"]["action_semantics"]["timestamp"],
            "observation_relationship": sidecar["openamrobot"]["action_semantics"][
                "observation_relationship"],
        },
    }


def write_dataset(path, sidecar, native, policy=None):
    """Write actual Parquet metadata/data, keeping the dependency explicit."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    (path / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)
    (path / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    (path / "meta/openamrobot/episodes").mkdir(parents=True, exist_ok=True)
    (path / "meta/info.json").write_text(json.dumps(native["info"]))
    tasks = [{"task_index": i, "task": task} for i, task in native["tasks"].items()]
    pq.write_table(pa.Table.from_pylist(tasks), path / "meta/tasks.parquet")
    pq.write_table(pa.Table.from_pylist([native["episode"]]),
                   path / "meta/episodes/chunk-000/file-000.parquet")
    table = pa.Table.from_pylist(native["rows"])
    for key in ("observation.state", "action"):
        column_index = table.column_names.index(key)
        values = [row[key] for row in native["rows"]]
        table = table.set_column(column_index, key, pa.array(values, type=pa.list_(pa.float32())))
    pq.write_table(table, path / "data/chunk-000/file-000.parquet")
    (path / "meta/openamrobot/episodes/episode-000000.json").write_text(json.dumps(sidecar))
    if policy is not None:
        (path / "synthetic-policy.json").write_text(json.dumps(policy))
    return path
