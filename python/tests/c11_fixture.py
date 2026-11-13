# SPDX-License-Identifier: MIT
"""Complete native synthetic QA fixture; every policy value is TEST ONLY.

No generated identifier, approval flag, release or threshold is recording evidence.
"""

import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from c10_fixture import CAMERAS, fixture, write_dataset
from oa_embodied_ai.training.inputs import file_sha256, inventory


def synthetic_dataset(root):
    root = Path(root)
    sidecar, native, policy = fixture()
    info = native["info"]
    info["video_path"] = (
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    info["total_episodes"], info["total_frames"] = 2, 4
    policy["video_content"] = {name: {"max_identical_run": 1} for name in CAMERAS}
    pin = {
        "release": "synthetic-test-only",
        "commit": "1" * 40,
        "evidence": "SYNTHETIC FIXTURE ONLY; no approved LeRobot release",
    }
    sidecar["openamrobot"]["provenance"]["lerobot_compatibility"][
        "validated_release_pin"
    ] = pin
    sidecar["openamrobot"]["provenance"]["software"].append(
        {
            "name": "lerobot",
            "version": pin["release"],
            "commit": pin["commit"],
            "repository": "https://github.com/huggingface/lerobot",
        }
    )
    policy["release"] = {"pin": pin, "version": pin["release"], "commit": pin["commit"]}
    policy["calibration"] = {}
    artifacts = []

    def artifact(name):
        path = root / "synthetic-artifacts" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("SYNTHETIC TEST ONLY: " + name)
        return {
            "path": path.relative_to(root).as_posix(),
            "sha256": "sha256:" + file_sha256(path),
        }

    robot = artifact("robot.txt")
    sidecar["openamrobot"]["provenance"]["robot_configuration_hash"] = robot["sha256"]
    artifacts.append({"kind": "robot_configuration", **robot})
    source = artifact("source.txt")
    policy["source_profile"] = {
        "id": sidecar["openamrobot"]["provenance"]["source_profile"],
        "artifact": source,
    }
    for name, stream in sidecar["openamrobot"]["streams"].items():
        config = artifact(name + "-config.txt")
        stream["configuration_hash"] = config["sha256"]
        artifacts.append({"kind": "stream_configuration", "stream": name, **config})
        if name in CAMERAS:
            cal = stream["calibration"]
            artifacts.append(
                {
                    "kind": "calibration",
                    "stream": name,
                    "id": cal["id"],
                    "version": cal["version"],
                    **artifact(name + "-calibration.txt"),
                }
            )
            policy["calibration"][name] = {
                k: cal["residual"][k] for k in ("name", "unit", "statistic", "method")
            }
            policy["calibration"][name]["max_value"] = 0.3
    policy["artifacts"] = artifacts
    for key in ("calibration",):
        if key not in policy:
            policy[key] = {}
    write_dataset(root, sidecar, native)
    episodes, rows = [], []
    for index in range(2):
        episode = deepcopy(native["episode"])
        episode.update(
            episode_index=index,
            episode_id=f"synthetic-e{index}",
            dataset_from_index=2 * index,
            dataset_to_index=2 * index + 2,
        )
        for key in (k for k in info["features"] if k.startswith("observation.images.")):
            episode.update(
                {
                    f"videos/{key}/chunk_index": 0,
                    f"videos/{key}/file_index": 0,
                    f"videos/{key}/from_timestamp": 2 * index / 30,
                    f"videos/{key}/to_timestamp": (2 * index + 2) / 30,
                }
            )
        episodes.append(episode)
        sc = deepcopy(sidecar)
        sc["native_lerobot"].update(
            episode_index=index, episode_id=episode["episode_id"]
        )
        for sample in sc["openamrobot"]["samples"]:
            sample["episode_index"] = index
        (root / f"meta/openamrobot/episodes/episode-{index:06d}.json").write_text(
            json.dumps(sc)
        )
        for row in native["rows"]:
            row = deepcopy(row)
            row.update(episode_index=index, index=2 * index + row["frame_index"])
            rows.append(row)
    pq.write_table(
        pa.Table.from_pylist(episodes),
        root / "meta/episodes/chunk-000/file-000.parquet",
    )
    table = pa.Table.from_pylist(rows)
    for key in ("observation.state", "action"):
        table = table.set_column(
            table.column_names.index(key),
            key,
            pa.array([r[key] for r in rows], type=pa.list_(pa.float32())),
        )
    pq.write_table(table, root / "data/chunk-000/file-000.parquet")
    for camera_index, name in enumerate(CAMERAS):
        key = sidecar["openamrobot"]["streams"][name]["native_feature_key"]
        path = root / f"videos/{key}/chunk-000/file-000.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=64x48:rate=30",
                "-vf",
                f"hue=h={camera_index * 60}",
                "-frames:v",
                "4",
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "mpeg4",
                str(path),
            ],
            check=True,
            capture_output=True,
        )
    policy_path = root.parent / "synthetic-policy.json"
    policy_path.write_text(json.dumps(policy))
    manifest = {
        "manifest_version": 1,
        "dataset_id": info["dataset_id"],
        "dataset_revision": info["dataset_revision"],
        "episode_indices": [0, 1],
        "profile": "lerobot-v0.6.1-bi-openarm-position",
        "data_origin": "synthetic",
        "evidence": "SYNTHETIC SOFTWARE TEST ONLY; no recording or production approval",
        "policy_sha256": file_sha256(policy_path),
        "files_sha256": inventory(root),
    }
    manifest_path = root.parent / "synthetic-input-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return root, policy_path, manifest_path
