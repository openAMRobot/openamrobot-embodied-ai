# SPDX-License-Identifier: MIT
"""Vision-based multi-camera dataset and preprocessor for Deliverable C11.

Ingests LeRobotDataset v3.0 format:
  - Parquet state and action tables
  - 4 canonical C9 camera video streams:
    * observation.images.head (720x1280)
    * observation.images.base (480x640)
    * observation.images.camera_wrist_left (480x640)
    * observation.images.camera_wrist_right (480x640)
Applies ImageNet normalization and temporal chunking (K=100).
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from ..dataset_qa.episode import inside, native_metadata, read_rows
from ..dataset_qa.strict_io import load_data

LOG = logging.getLogger("oa-c11-dataset")

# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImagePreprocessor:
    """Preprocesses camera images with resize and standard ImageNet normalization."""

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
    ) -> None:
        self.image_size = image_size
        self.normalize = normalize
        transform_list: List[Any] = [
            T.Resize(image_size, antialias=True),
        ]
        if normalize:
            transform_list.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        self.transform = T.Compose(transform_list)

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Process tensor.

        Args:
            img_tensor: Tensor of shape (C, H, W) or (B, C, H, W) with values in [0, 1].

        Returns:
            Processed tensor of shape (..., C, *image_size).
        """
        if img_tensor.dtype == torch.uint8:
            img_tensor = img_tensor.float() / 255.0
        elif img_tensor.max() > 1.0:
            img_tensor = img_tensor / 255.0

        return self.transform(img_tensor)


@dataclass(frozen=True)
class NativeEpisode:
    """An episode identity, even when several episodes share a Parquet file."""

    episode_index: int
    path: Path
    metadata: dict


def find_episode_parquet_files(dataset_path: Path) -> List[NativeEpisode]:
    """Resolve native data_path and episode metadata, without filename guessing."""
    root = Path(dataset_path).resolve()
    info, _, episodes = native_metadata(root)
    result = []
    for index, episode in sorted(episodes.items()):
        if type(index) is not int or index < 0:
            raise ValueError("Invalid native episode identity")
        template = info["data_path"]
        path = inside(
            root,
            template.format(
                episode_index=index,
                episode_chunk=index // info.get("chunks_size", 1000),
                chunk_index=episode.get("data/chunk_index", 0),
                file_index=episode.get("data/file_index", 0),
            ),
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing native data artifact: {path}")
        result.append(NativeEpisode(index, path, episode))
    return result


def partition_episodes(episode_paths, eval_split=0.2, seed=42):
    """Split by actual episode identity; never fall back to training as validation."""
    n = len(episode_paths)
    if n < 2 or not 0 < eval_split < 1:
        raise ValueError("Training requires at least two episodes and a held-out split")
    indices = np.random.RandomState(seed).permutation(n).tolist()
    num_eval = min(n - 1, max(1, round(n * eval_split)))
    val = [episode_paths[i] for i in sorted(indices[:num_eval])]
    train = [episode_paths[i] for i in sorted(indices[num_eval:])]
    return train, val, [e.episode_index for e in train], [e.episode_index for e in val]


def extract_video_frame(video_path, frame_idx):
    """Decode the exact native segment frame ordinal or fail visibly."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if (
            not ok
            or frame is None
            or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != frame_idx + 1
        ):
            raise ValueError(f"Cannot decode camera frame {frame_idx}: {video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def _video_binding(root, info, episode, key):
    template = info["video_path"]
    index = episode["episode_index"]
    shared = "{file_index" in template
    prefix = f"videos/{key}/"
    if shared and any(
        prefix + name not in episode
        for name in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
    ):
        raise ValueError(f"Missing shared-video binding: {key}")
    path = inside(
        root,
        template.format(
            video_key=key,
            episode_index=index,
            episode_chunk=index // info.get("chunks_size", 1000),
            chunk_index=episode.get(prefix + "chunk_index"),
            file_index=episode.get(prefix + "file_index"),
        ),
    )
    process = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    if process.stderr.strip():
        raise ValueError(process.stderr)
    frames = json.loads(process.stdout)["frames"]
    # Same half-open native video segment convention as C10. The kth native
    # episode frame maps to the kth decoded frame within that segment.
    positions = list(range(len(frames)))
    if shared:
        start, end = (
            episode[prefix + "from_timestamp"],
            episode[prefix + "to_timestamp"],
        )
        positions = [
            i
            for i, f in enumerate(frames)
            if start <= float(f["best_effort_timestamp_time"]) < end
        ]
    if len(positions) != episode["length"]:
        raise ValueError(f"Camera/native frame count mismatch: {key}")
    return path, positions


class MultiCameraACTDataset(Dataset):
    """Native per-episode vectors plus four required video segments.

    No alias, guessed video path, unreadable-episode skip or zero-image fallback.
    """

    def __init__(
        self,
        dataset_path,
        episode_paths,
        config,
        load_images=True,
        normalize_images=True,
    ):
        config.validate()
        import cv2  # noqa: F401 - require the actual training decoder before model startup

        if not load_images:
            raise ValueError("Visual ACT requires image loading")
        self.dataset_path = Path(dataset_path).resolve()
        self.episode_paths = list(episode_paths)
        self.config = config
        self.chunk_size = config.chunk_size
        self.camera_keys = list(config.camera_keys)
        self.preprocessor = ImagePreprocessor(config.image_size, normalize_images)
        self.samples = []
        self._load_dataset_index()
        if not self.samples:
            raise ValueError("No samples in selected native episodes")

    def _load_dataset_index(self):
        info, _, all_episodes = native_metadata(self.dataset_path)
        for key, dim in (
            ("observation.state", self.config.state_dim),
            ("action", self.config.action_dim),
        ):
            feature = info["features"][key]
            if (
                feature["shape"] != [dim]
                or feature.get("names") != self.config.channel_names
            ):
                raise ValueError(
                    f"Native {key} shape/names differ from selected profile"
                )
        if info["fps"] != self.config.fps:
            raise ValueError("Native FPS differs from checkpoint configuration")
        for key in self.camera_keys:
            if info["features"].get(key, {}).get("dtype") != "video":
                raise ValueError(f"Missing canonical video feature: {key}")
        for episode_ref in self.episode_paths:
            index = episode_ref.episode_index
            episode = all_episodes[index]
            sidecars = [
                path
                for path in (self.dataset_path / "meta/openamrobot/episodes").glob(
                    "episode-*"
                )
                if path.suffix in (".json", ".yaml", ".yml")
            ]
            matching = [
                load_data(path)
                for path in sidecars
                if load_data(path)["native_lerobot"]["episode_index"] == index
            ]
            if len(matching) != 1:
                raise ValueError("Expected one C9 sidecar per selected episode")
            for key in ("observation.state", "action"):
                descriptor = matching[0]["native_lerobot"]["features"][key]
                if descriptor["profile"] != self.config.profile:
                    raise ValueError("Sidecar profile differs from configured profile")
                if self.config.state_dim == 16 and any(
                    d["unit"] != "degree" for d in descriptor["dimensions"]
                ):
                    raise ValueError(
                        "C9 adapter profile requires degree-valued channels"
                    )
            findings = []
            rows = read_rows(
                self.dataset_path,
                info,
                episode,
                lambda check, status, findings=findings, **kw: findings.append(status),
                set(all_episodes),
            )
            if rows is None or any(status != "PASS" for status in findings):
                raise ValueError(f"Native data inspection failed for episode {index}")
            if [r["frame_index"] for r in rows] != list(range(episode["length"])):
                raise ValueError(
                    "Native frame order/range differs from episode metadata"
                )
            states = np.asarray(
                [r["observation.state"] for r in rows], dtype=np.float32
            )
            actions = np.asarray([r["action"] for r in rows], dtype=np.float32)
            if (
                states.shape != (len(rows), self.config.state_dim)
                or actions.shape != (len(rows), self.config.action_dim)
                or not np.isfinite(states).all()
                or not np.isfinite(actions).all()
            ):
                raise ValueError("Invalid state/action vectors")
            videos = {
                key: _video_binding(self.dataset_path, info, episode, key)
                for key in self.camera_keys
            }
            for t in range(len(rows)):
                self.samples.append(
                    dict(
                        episode_index=index,
                        time_step=t,
                        state=states[t],
                        act_raw=actions,
                        video_map=videos,
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        t, actions = sample["time_step"], sample["act_raw"]
        count = min(self.chunk_size, len(actions) - t)
        chunk = np.zeros((self.chunk_size, self.config.action_dim), dtype=np.float32)
        chunk[:count] = actions[t : t + count]
        mask = torch.arange(self.chunk_size) < count
        images = []
        for key in self.camera_keys:
            path, positions = sample["video_map"][key]
            frame = extract_video_frame(path, positions[t])
            images.append(
                self.preprocessor(torch.from_numpy(frame.transpose(2, 0, 1).copy()))
            )
        return {
            "state": torch.from_numpy(sample["state"]),
            "images": torch.stack(images),
            "actions": torch.from_numpy(chunk),
            "action_mask": mask,
        }
