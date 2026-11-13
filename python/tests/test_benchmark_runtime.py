# SPDX-License-Identifier: MIT
"""Timing/deadline math and fail-closed scenario/workload checks."""

from unittest.mock import patch

import numpy as np
import pytest
from oa_embodied_ai.benchmarking.runtime import (
    BenchmarkError,
    benchmark,
    scenario_nodes,
    summarize,
)


def test_deadlines_include_scheduling_delay():
    r = summarize([0.01, 0.01, 0.01], [0.01, 0.21, 0.22], elapsed=0.3, target_hz=10)
    assert r["missed_deadlines"] == 1
    assert r["latency_over_period"] == 0
    assert r["achieved_hz"] == 10
    assert r["p99_ms"] == r["max_ms"] == 10


def test_empty_metrics_fail():
    with pytest.raises(BenchmarkError, match="empty"):
        summarize([], [], elapsed=1, target_hz=30)


@pytest.mark.parametrize(
    "scenario,ros,moveit,message",
    [("B", [], [], "explicit expected ROS"), ("C", ["/state"], [], "expected MoveIt")],
)
def test_scenario_requires_evidence(scenario, ros, moveit, message):
    with pytest.raises(BenchmarkError, match=message):
        scenario_nodes(scenario, ros, moveit)


def test_scenario_c_requires_live_nodes():
    with patch("oa_embodied_ai.benchmarking.runtime.command", return_value="/state"):
        with pytest.raises(
            BenchmarkError, match="Missing required scenario nodes.*move_group"
        ):
            scenario_nodes("C", ["/state"], ["/move_group"])


def test_a_rejects_active_ros():
    with patch("oa_embodied_ai.benchmarking.runtime.command", return_value="/state"):
        with pytest.raises(BenchmarkError, match="Policy-alone"):
            scenario_nodes("A", [], [])


def test_local_benchmark_keeps_physical_unverified(tmp_path, tiny_config):
    import torch
    from oa_embodied_ai.training.model import ACTPolicy

    torch.set_num_threads(1)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "config": tiny_config.to_dict(),
            "model_state_dict": ACTPolicy(tiny_config).state_dict(),
        },
        checkpoint,
    )
    observations = tmp_path / "observations.npz"
    rng = np.random.default_rng(17)
    np.savez(
        observations,
        state=rng.normal(size=(2, 16)).astype("float32"),
        images=rng.uniform(size=(2, 4, 3, 32, 32)).astype("float32"),
    )
    with patch("oa_embodied_ai.benchmarking.runtime.command", return_value=None):
        r = benchmark(
            checkpoint,
            observations,
            output=tmp_path / "run",
            scenario="A",
            seconds=0.12,
            target_hz=10,
            device="cpu",
            warmup=1,
        )
    assert r["metrics"]["samples"] > 0
    assert r["status"] == "DEVELOPMENT_ONLY"
    assert r["physical_acceptance"] == "NOT_VERIFIED"
    assert "NOT_VERIFIED" in r["fit_decision"]
    assert r["workload"]["hashes"]["checkpoint"]
    assert "thermal_throttling" in r["missing_telemetry"]


def test_wrong_camera_shape_rejected(tmp_path, tiny_config):
    import torch
    from oa_embodied_ai.training.model import ACTPolicy

    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "config": tiny_config.to_dict(),
            "model_state_dict": ACTPolicy(tiny_config).state_dict(),
        },
        checkpoint,
    )
    obs = tmp_path / "obs.npz"
    np.savez(
        obs,
        state=np.zeros((1, 16), np.float32),
        images=np.zeros((1, 3, 3, 32, 32), np.float32),
    )
    with patch("oa_embodied_ai.benchmarking.runtime.command", return_value=None):
        with pytest.raises(BenchmarkError, match="Observation shapes"):
            benchmark(
                checkpoint,
                obs,
                output=tmp_path / "bad",
                scenario="A",
                seconds=1,
                target_hz=10,
                device="cpu",
            )
