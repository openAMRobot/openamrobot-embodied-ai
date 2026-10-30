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

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset
import torchvision.transforms as T

from .config import ACTConfig, CANONICAL_CAMERA_KEYS

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


def find_episode_parquet_files(dataset_path: Path) -> List[Path]:
    """Locate all episode parquet files within the dataset directory."""
    dataset_path = Path(dataset_path)
    parquet_files: List[Path] = []

    # Check data/chunk-* or direct data/
    for pattern in ("data/**/*.parquet", "data/*.parquet", "*.parquet"):
        found = list(dataset_path.glob(pattern))
        if found:
            parquet_files.extend(found)

    # Filter for episode_*.parquet
    ep_files = [p for p in parquet_files if "episode_" in p.name]
    ep_files = sorted(list(set(ep_files)), key=lambda p: p.name)
    return ep_files


def partition_episodes(
    episode_paths: Sequence[Path],
    eval_split: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[int], List[int]]:
    """Partition episodes into train and held-out validation sets.

    Returns:
        (train_paths, val_paths, train_indices, val_indices)
    """
    n = len(episode_paths)
    if n == 0:
        return [], [], [], []

    indices = list(range(n))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    num_eval = max(1, int(round(n * eval_split))) if n > 1 else 0
    val_idx = sorted(indices[:num_eval])
    train_idx = sorted(indices[num_eval:])

    train_paths = [episode_paths[i] for i in train_idx]
    val_paths = [episode_paths[i] for i in val_idx]

    return train_paths, val_paths, train_idx, val_idx


def extract_video_frame(video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
    """Extract a single frame from an MP4 file as RGB numpy array (H, W, 3)."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception as exc:
        LOG.debug("Error decoding frame %d from %s: %s", frame_idx, video_path, exc)
    return None


class MultiCameraACTDataset(Dataset):
    """Dataset extracting (state, multi-view images, action_chunk) samples."""

    def __init__(
        self,
        dataset_path: Path,
        episode_paths: Sequence[Path],
        config: ACTConfig,
        load_images: bool = True,
        normalize_images: bool = True,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.episode_paths = list(episode_paths)
        self.config = config
        self.chunk_size = config.chunk_size
        self.camera_keys = config.camera_keys
        self.load_images = load_images
        self.preprocessor = ImagePreprocessor(
            image_size=config.image_size,
            normalize=normalize_images,
        )

        self.samples: List[Dict[str, Any]] = []
        self._load_dataset_index()

    def _locate_video(self, ep_path: Path, camera_key: str) -> Optional[Path]:
        """Locate video stream file for an episode and camera head."""
        ep_stem = ep_path.stem  # e.g. "episode_000000"
        chunk_name = ep_path.parent.name  # e.g. "chunk-000"

        candidates = [
            self.dataset_path / "videos" / chunk_name / camera_key / f"{ep_stem}.mp4",
            self.dataset_path / "videos" / camera_key / f"{ep_stem}.mp4",
            self.dataset_path / "videos" / chunk_name / f"{camera_key}_{ep_stem}.mp4",
            self.dataset_path / "videos" / f"{camera_key}_{ep_stem}.mp4",
        ]
        for c in candidates:
            if c.is_file():
                return c
        return None

    def _load_dataset_index(self) -> None:
        """Scan episodes and build sample indexing."""
        for ep_path in self.episode_paths:
            try:
                table = pq.read_table(ep_path)
            except Exception as exc:
                LOG.warning("Failed to read parquet file %s: %s", ep_path, exc)
                continue

            state_col = table.column("observation.state").to_pylist()
            act_col = table.column("action").to_pylist()

            state_raw = np.array(state_col, dtype=np.float32)
            act_raw = np.array(act_col, dtype=np.float32)
            num_rows = len(state_raw)
            if num_rows < 1:
                continue

            # Check video paths
            video_map: Dict[str, Optional[Path]] = {}
            if self.load_images:
                for cam in self.camera_keys:
                    video_map[cam] = self._locate_video(ep_path, cam)

            for t in range(num_rows):
                self.samples.append(
                    {
                        "ep_path": ep_path,
                        "time_step": t,
                        "total_steps": num_rows,
                        "state": state_raw[t],
                        "act_raw": act_raw,
                        "video_map": video_map,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        t = sample["time_step"]
        total_steps = sample["total_steps"]
        act_raw = sample["act_raw"]

        # Extract state vector
        state = torch.from_numpy(sample["state"]).float()

        # Extract target action chunk of length chunk_size
        K = self.chunk_size
        act_dim = self.config.action_dim
        action_chunk = np.zeros((K, act_dim), dtype=np.float32)
        action_mask = np.zeros(K, dtype=np.float32)

        available_len = min(K, total_steps - t)
        if available_len > 0:
            action_chunk[:available_len] = act_raw[t : t + available_len]
            action_mask[:available_len] = 1.0

        # Pad remaining steps with last valid action
        if available_len < K:
            last_act = act_raw[min(t + available_len - 1, total_steps - 1)]
            action_chunk[available_len:] = last_act
            # Mask remains 1.0 for action replication or 0.0 for pure padding

        actions_tensor = torch.from_numpy(action_chunk).float()
        mask_tensor = torch.from_numpy(action_mask).float()

        # Load multi-view images: (num_cameras, 3, H, W)
        num_cams = len(self.camera_keys)
        H, W = self.config.image_size
        cam_tensors: List[torch.Tensor] = []

        if self.load_images and sample["video_map"]:
            for cam in self.camera_keys:
                v_path = sample["video_map"].get(cam)
                frame = None
                if v_path is not None and v_path.is_file():
                    frame = extract_video_frame(v_path, t)

                if frame is not None:
                    # Convert (H, W, 3) uint8 to (3, H, W) float
                    t_frame = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
                    processed = self.preprocessor(t_frame)
                else:
                    # Synthetic / placeholder frame
                    processed = torch.zeros((3, H, W), dtype=torch.float32)
                cam_tensors.append(processed)
        else:
            for _ in range(num_cams):
                cam_tensors.append(torch.zeros((3, H, W), dtype=torch.float32))

        images_tensor = torch.stack(cam_tensors, dim=0)  # (num_cams, 3, H, W)

        return {
            "state": state,
            "images": images_tensor,
            "actions": actions_tensor,
            "action_mask": mask_tensor,
        }
