# SPDX-License-Identifier: MIT
"""Kinematic profile configurations for Deliverable C11 ACT Baseline.

Supports:
  - openamrobot-fixed-mast-19dof (19 channels: 3 base + 2 fixed mast [0,0] + 14 arm + 2 grippers)
  - lerobot-v0.6.1-bi-openarm-position (16 channels: 14 arm + 2 grippers)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

OPENAMROBOT_19DOF_CHANNELS: List[str] = [
    "base_linear_x",
    "base_linear_y",
    "base_angular_z",
    "torso_pitch",
    "torso_yaw",
    "left_shoulder_pan",
    "left_shoulder_lift",
    "left_elbow_flex",
    "left_wrist_flex",
    "left_wrist_roll",
    "left_wrist_yaw",
    "left_gripper",
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow_flex",
    "right_wrist_flex",
    "right_wrist_roll",
    "right_wrist_yaw",
    "right_gripper",
]

LEROBOT_16DOF_CHANNELS: List[str] = [
    "openarm_left_joint1",
    "openarm_left_joint2",
    "openarm_left_joint3",
    "openarm_left_joint4",
    "openarm_left_joint5",
    "openarm_left_joint6",
    "openarm_left_joint7",
    "openarm_left_gripper",
    "openarm_right_joint1",
    "openarm_right_joint2",
    "openarm_right_joint3",
    "openarm_right_joint4",
    "openarm_right_joint5",
    "openarm_right_joint6",
    "openarm_right_joint7",
    "openarm_right_gripper",
]

CANONICAL_CAMERA_KEYS: List[str] = [
    "observation.images.head",
    "observation.images.base",
    "observation.images.camera_wrist_left",
    "observation.images.camera_wrist_right",
]


@dataclass
class ACTConfig:
    """Configuration for ACT policy training and evaluation."""

    profile: str = "openamrobot-fixed-mast-19dof"
    schema_revision: str = "schema-v1.0-frozen"
    state_dim: int = 19
    action_dim: int = 19
    chunk_size: int = 100
    fps: int = 30
    channel_names: List[str] = field(default_factory=lambda: list(OPENAMROBOT_19DOF_CHANNELS))
    camera_keys: List[str] = field(default_factory=lambda: list(CANONICAL_CAMERA_KEYS))
    vision_backbone: str = "resnet18"
    image_size: Tuple[int, int] = (224, 224)
    d_model: int = 256
    n_heads: int = 8
    enc_layers: int = 4
    dec_layers: int = 4
    dim_feedforward: int = 512
    latent_dim: int = 32
    dropout: float = 0.1
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    kl_weight: float = 10.0
    l1_weight: float = 1.0
    epochs: int = 50
    eval_split: float = 0.2
    seed: int = 42
    device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_profile_config(profile_name: str) -> ACTConfig:
    """Return default ACTConfig for a given kinematic profile name."""
    if profile_name == "openamrobot-fixed-mast-19dof":
        return ACTConfig(
            profile="openamrobot-fixed-mast-19dof",
            schema_revision="schema-v1.0-frozen",
            state_dim=19,
            action_dim=19,
            channel_names=list(OPENAMROBOT_19DOF_CHANNELS),
        )
    elif profile_name in {"lerobot-v0.6.1-bi-openarm-position", "bi-openarm-position"}:
        return ACTConfig(
            profile="lerobot-v0.6.1-bi-openarm-position",
            schema_revision="c9-contract-fixture",
            state_dim=16,
            action_dim=16,
            channel_names=list(LEROBOT_16DOF_CHANNELS),
        )
    else:
        raise ValueError(f"Unknown kinematic profile: '{profile_name}'. Supported: 'openamrobot-fixed-mast-19dof', 'lerobot-v0.6.1-bi-openarm-position'")


def load_config(path: Union[str, Path]) -> ACTConfig:
    """Load ACT configuration from a YAML file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config at {path}: expected mapping")

    # If nested under 'model', 'training', etc.
    if "model" in data:
        m = data["model"]
        t = data.get("training", {})
        l = data.get("loss", {})
        profile = m.get("profile", "openamrobot-fixed-mast-19dof")
        cfg = get_profile_config(profile)
        cfg.schema_revision = m.get("schema_revision", cfg.schema_revision)
        cfg.state_dim = m.get("state_dim", cfg.state_dim)
        cfg.action_dim = m.get("action_dim", cfg.action_dim)
        cfg.chunk_size = m.get("chunk_size", cfg.chunk_size)
        cfg.fps = m.get("fps", cfg.fps)
        cfg.d_model = m.get("d_model", cfg.d_model)
        cfg.n_heads = m.get("n_heads", cfg.n_heads)
        cfg.enc_layers = m.get("enc_layers", cfg.enc_layers)
        cfg.dec_layers = m.get("dec_layers", cfg.dec_layers)
        cfg.dim_feedforward = m.get("dim_feedforward", cfg.dim_feedforward)
        cfg.latent_dim = m.get("latent_dim", cfg.latent_dim)
        cfg.dropout = m.get("dropout", cfg.dropout)
        cfg.vision_backbone = m.get("vision_backbone", cfg.vision_backbone)
        if "camera_keys" in m:
            cfg.camera_keys = list(m["camera_keys"])
        if "channel_names" in m:
            cfg.channel_names = list(m["channel_names"])
        cfg.batch_size = t.get("batch_size", cfg.batch_size)
        cfg.lr = t.get("lr", cfg.lr)
        cfg.weight_decay = t.get("weight_decay", cfg.weight_decay)
        cfg.kl_weight = t.get("kl_weight", l.get("kl_weight", cfg.kl_weight))
        cfg.l1_weight = l.get("l1_weight", cfg.l1_weight)
        cfg.epochs = t.get("epochs", cfg.epochs)
        cfg.eval_split = t.get("eval_split", cfg.eval_split)
        cfg.seed = t.get("seed", cfg.seed)
        cfg.device = t.get("device", cfg.device)
        return cfg

    # Flat dict format
    profile = data.get("profile", "openamrobot-fixed-mast-19dof")
    cfg = get_profile_config(profile)
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def save_config(config: ACTConfig, path: Union[str, Path]) -> None:
    """Save ACT configuration to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {
        "model": {
            "name": "act",
            "profile": config.profile,
            "schema_revision": config.schema_revision,
            "state_dim": config.state_dim,
            "action_dim": config.action_dim,
            "chunk_size": config.chunk_size,
            "fps": config.fps,
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "enc_layers": config.enc_layers,
            "dec_layers": config.dec_layers,
            "dim_feedforward": config.dim_feedforward,
            "latent_dim": config.latent_dim,
            "dropout": config.dropout,
            "vision_backbone": config.vision_backbone,
            "camera_keys": config.camera_keys,
            "channel_names": config.channel_names,
        },
        "training": {
            "batch_size": config.batch_size,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "kl_weight": config.kl_weight,
            "epochs": config.epochs,
            "eval_split": config.eval_split,
            "seed": config.seed,
            "device": config.device,
        },
        "loss": {
            "type": "l1_plus_kl",
            "l1_weight": config.l1_weight,
            "kl_weight": config.kl_weight,
        },
    }
    path.write_text(yaml.dump(d, sort_keys=False), encoding="utf-8")

