# SPDX-License-Identifier: MIT
"""Closed-loop simulator/hardware rollout evaluation runner.

Deliverable C11: ACT Imitation Learning Baseline.
Resolves C10->C11 handoff findings:
3. Metric Disambiguation: Explicitly separate offline imitation loss
   from closed-loop rollout success, with mandatory NOT_VERIFIED
   marking on synthetic data.
"""

from __future__ import annotations

import logging
import math
from typing import Dict

LOG = logging.getLogger("oa-c11-rollout-eval")

# Discrete subtask phase attribution across phases 0..7 (Canonical C13 taxonomy)
SUBTASK_PHASES: Dict[int, str] = {
    0: "IDLE",
    1: "APPROACH",
    2: "ALIGN",
    3: "GRASP",
    4: "TRANSPORT",
    5: "PLACE",
    6: "RETRACT",
    7: "FAILURE_RECOVERY",
}


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95):
    """Wilson interval for independently obtained rollout counts only."""
    from statistics import NormalDist

    if (
        type(k) is not int
        or type(n) is not int
        or not 0 <= k <= n
        or not 0 < confidence < 1
    ):
        raise ValueError("Invalid binomial counts/confidence")
    if n == 0:
        return None, None
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    return (
        0.0 if k == 0 else max(0.0, center - margin),
        1.0 if k == n else min(1.0, center + margin),
    )


class ClosedLoopRolloutEvaluator:
    """Reserved integration point; no simulator or hardware adapter supplied."""

    def evaluate_episode(self, *args, **kwargs):
        raise NotImplementedError("No executed policy/environment rollout adapter")


def run_rollout_eval(policy, num_episodes=20, control_freq_hz=30.0):
    """Return missing-evidence status; never invent trajectories or outcomes."""
    return {
        "status": "NOT_VERIFIED",
        "reason": "No executed policy/environment rollout adapter",
        "requested_rollouts": num_episodes,
        "total_rollouts": 0,
        "successful_rollouts": None,
        "success_rate": None,
        "wilson_ci_95": None,
        "phase_attribution": None,
        "acceptance_status": "NOT_VERIFIED_PHYSICAL",
    }
