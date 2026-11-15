# SPDX-License-Identifier: MIT
"""Offline trajectory imitation evaluation with temporal ensembling.

Deliverable C11: ACT Imitation Learning Baseline.
Resolves C10->C11 handoff findings:
3. Metric Disambiguation: Explicitly separate offline imitation loss
   from closed-loop rollout success.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from ..training.config import ACTConfig
from ..training.dataset import MultiCameraACTDataset

LOG = logging.getLogger("oa-c11-offline-eval")


def temporal_ensemble_predict(
    model: torch.nn.Module,
    state_sequence: np.ndarray,
    images_sequence: Optional[torch.Tensor] = None,
    chunk_size: int = 100,
    device: str = "cpu",
    ensemble_lambda: float = 0.01,
) -> np.ndarray:
    """Exponential-decay temporal ensembling across overlapping ACT chunk predictions.

    Evaluates output of a 15-line Temporal Ensembling buffer (lambda = 0.01).
    At each step t the model predicts a chunk of chunk_size future actions.
    Overlapping chunk predictions are accumulated with exp(-lambda * k) weights.
    """
    T = len(state_sequence)
    if (
        T == 0
        or chunk_size < 1
        or not np.isfinite(ensemble_lambda)
        or ensemble_lambda < 0
    ):
        raise ValueError("Invalid temporal ensemble inputs")
    action_dim = model.action_dim
    accum = np.zeros((T + chunk_size, action_dim), dtype=np.float64)
    weight_sum = np.zeros(T + chunk_size, dtype=np.float64)

    weights = np.exp(-ensemble_lambda * np.arange(chunk_size, dtype=np.float64))
    model.eval()
    with torch.no_grad():
        for t in range(T):
            s_t = torch.from_numpy(state_sequence[t : t + 1]).float().to(device)
            img_t = (
                images_sequence[t : t + 1].to(device)
                if images_sequence is not None
                else None
            )
            pred_chunk, _, _ = model(state=s_t, images=img_t)
            if (
                pred_chunk.shape != (1, chunk_size, action_dim)
                or not torch.isfinite(pred_chunk).all()
            ):
                raise ValueError("Invalid predicted action chunk")
            pred_np = pred_chunk[0].cpu().numpy().astype(np.float64)
            accum[t : t + chunk_size] += weights[:, None] * pred_np
            weight_sum[t : t + chunk_size] += weights

    safe_w = np.where(weight_sum[:T] > 0.0, weight_sum[:T], 1.0)
    return (accum[:T] / safe_w[:, None]).astype(np.float32)


def compute_chunk_discontinuity(
    model: torch.nn.Module,
    state_sequence: np.ndarray,
    images_sequence: Optional[torch.Tensor] = None,
    chunk_size: int = 100,
    device: str = "cpu",
) -> float:
    """Compute chunk boundary discontinuity error between chunk k step K-1 and chunk k+1 step 0."""
    T = len(state_sequence)
    if T < chunk_size * 2:
        return None

    model.eval()
    discontinuities: List[float] = []

    with torch.no_grad():
        for start_t in range(0, T - chunk_size * 2 + 1, chunk_size):
            s_k = (
                torch.from_numpy(state_sequence[start_t : start_t + 1])
                .float()
                .to(device)
            )
            img_k = (
                images_sequence[start_t : start_t + 1].to(device)
                if images_sequence is not None
                else None
            )
            chunk_k, _, _ = model(state=s_k, images=img_k)

            next_t = start_t + chunk_size
            s_k1 = (
                torch.from_numpy(state_sequence[next_t : next_t + 1]).float().to(device)
            )
            img_k1 = (
                images_sequence[next_t : next_t + 1].to(device)
                if images_sequence is not None
                else None
            )
            chunk_k1, _, _ = model(state=s_k1, images=img_k1)

            # Delta between last step of chunk k (index chunk_size - 1) and first step of chunk k+1 (index 0)
            last_k = chunk_k[0, chunk_size - 1].cpu().numpy()
            first_k1 = chunk_k1[0, 0].cpu().numpy()
            delta = float(np.mean(np.abs(last_k - first_k1)))
            discontinuities.append(delta)

    return float(np.mean(discontinuities)) if discontinuities else None


def compute_offline_metrics(
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    channel_names: Optional[Sequence[str]] = None,
    chunk_discontinuity: float = 0.0,
) -> Dict[str, Any]:
    """Compute per-channel and overall L1 / MAE and MSE metrics.

    Args:
        gt_actions: Ground truth action array (N, action_dim).
        pred_actions: Predicted action array (N, action_dim).
        channel_names: Optional channel names.
        chunk_discontinuity: Computed chunk boundary discontinuity error.

    Returns:
        Dictionary containing overall_mae, overall_mse, per_channel dict, and chunk discontinuity.
    """
    if (
        gt_actions.ndim != 2
        or gt_actions.shape != pred_actions.shape
        or not len(gt_actions)
        or not np.isfinite(gt_actions).all()
        or not np.isfinite(pred_actions).all()
    ):
        raise ValueError(
            "Offline metrics require finite matching nonempty action matrices"
        )
    error = pred_actions - gt_actions
    overall_mae = float(np.mean(np.abs(error)))
    overall_mse = float(np.mean(error**2))

    num_channels = gt_actions.shape[-1]
    per_channel: Dict[str, Dict[str, float]] = {}

    for ch in range(num_channels):
        ch_name = (
            channel_names[ch]
            if channel_names and ch < len(channel_names)
            else f"channel_{ch}"
        )
        ch_err = error[:, ch]
        per_channel[ch_name] = {
            "mae": float(np.mean(np.abs(ch_err))),
            "mse": float(np.mean(ch_err**2)),
        }

    return {
        "overall_mae": overall_mae,
        "overall_mse": overall_mse,
        "chunk_boundary_discontinuity": chunk_discontinuity,
        "per_channel": per_channel,
    }


def run_offline_eval(
    model: torch.nn.Module,
    dataset_path: Path,
    episode_paths: Sequence[Path],
    config: ACTConfig,
    device: str = "cpu",
    ensemble_lambda: float = 0.01,
) -> Dict[str, Any]:
    """Run offline evaluation over specified episode parquet files."""
    abs_err_sum = np.zeros(config.action_dim, dtype=np.float64)
    sq_err_sum = np.zeros(config.action_dim, dtype=np.float64)
    total_steps = 0
    discontinuity_accum: List[float] = []

    for episode in episode_paths:
        dataset = MultiCameraACTDataset(dataset_path, [episode], config)
        state_raw = np.stack([sample["state"] for sample in dataset.samples])
        act_raw = dataset.samples[0]["act_raw"]

        # Decode only the requested frames, keeping image memory bounded.
        class EpisodeImages:
            def __getitem__(self, selection, dataset=dataset):
                return torch.stack(
                    [
                        dataset[i]["images"]
                        for i in range(*selection.indices(len(dataset)))
                    ]
                )

        images = EpisodeImages()
        pred_ensembled = temporal_ensemble_predict(
            model=model,
            state_sequence=state_raw,
            images_sequence=images,
            chunk_size=config.chunk_size,
            device=device,
            ensemble_lambda=ensemble_lambda,
        )
        disc = compute_chunk_discontinuity(
            model=model,
            state_sequence=state_raw,
            images_sequence=images,
            chunk_size=config.chunk_size,
            device=device,
        )
        if disc is not None:
            discontinuity_accum.append(disc)

        if (
            act_raw.ndim != 2
            or act_raw.shape != pred_ensembled.shape
            or not len(act_raw)
            or not np.isfinite(act_raw).all()
            or not np.isfinite(pred_ensembled).all()
        ):
            raise ValueError(
                "Offline metrics require finite matching nonempty action matrices"
            )

        err = pred_ensembled.astype(np.float64) - act_raw.astype(np.float64)
        abs_err_sum += np.sum(np.abs(err), axis=0)
        sq_err_sum += np.sum(err**2, axis=0)
        total_steps += len(act_raw)

    if total_steps == 0:
        raise ValueError(
            f"No valid episodes found for offline evaluation under {dataset_path}"
        )

    mean_disc = float(np.mean(discontinuity_accum)) if discontinuity_accum else None
    overall_mae = float(np.sum(abs_err_sum) / (total_steps * config.action_dim))
    overall_mse = float(np.sum(sq_err_sum) / (total_steps * config.action_dim))

    channel_names = config.channel_names
    per_channel: Dict[str, Dict[str, float]] = {}
    for ch in range(config.action_dim):
        ch_name = (
            channel_names[ch]
            if channel_names and ch < len(channel_names)
            else f"channel_{ch}"
        )
        per_channel[ch_name] = {
            "mae": float(abs_err_sum[ch] / total_steps),
            "mse": float(sq_err_sum[ch] / total_steps),
        }

    return {
        "overall_mae": overall_mae,
        "overall_mse": overall_mse,
        "chunk_boundary_discontinuity": mean_disc,
        "per_channel": per_channel,
    }
