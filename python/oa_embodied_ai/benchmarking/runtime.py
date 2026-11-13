# SPDX-License-Identifier: MIT
"""Checkpoint-bound ACT-like inference benchmark, with honest device evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from oa_embodied_ai.training.evidence import execution_provenance
from oa_embodied_ai.training.inputs import file_sha256


class BenchmarkError(ValueError):
    """Invalid benchmark input or missing scenario evidence."""


def read_text(path):
    try:
        return Path(path).read_text().strip().strip("\0")
    except OSError:
        return None


def command(argv):
    if shutil.which(argv[0]) is None:
        return None
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=5, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def environment():
    import torch

    model = read_text("/proc/device-tree/model")
    return {
        "host": platform.platform(),
        "machine": platform.machine(),
        "device_model": model,
        "jetson_orin_nx_detected": bool(model and "Orin NX" in model),
        "jetpack_l4t": read_text("/etc/nv_tegra_release"),
        "power_mode": command(["nvpmodel", "-q"]),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tensorrt": _version("tensorrt"),
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "cuda_device": torch.cuda.get_device_name()
        if torch.cuda.is_available()
        else None,
    }


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


_PROCESS = None


def telemetry(device):
    import psutil
    import torch

    global _PROCESS
    if _PROCESS is None:
        _PROCESS = psutil.Process()
    memory = psutil.virtual_memory()
    temperatures = {}
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        value = read_text(zone / "temp")
        if value:
            temperatures[read_text(zone / "type") or zone.name] = float(value) / 1000
    gpu_loads = {
        str(p): read_text(p) for p in Path("/sys/devices").glob("platform/*gpu*/load")
    }
    return {
        "monotonic_s": time.monotonic(),
        "system_cpu_percent": psutil.cpu_percent(),
        "process_cpu_percent": _PROCESS.cpu_percent(),
        "ram_used_bytes": memory.used,
        "ram_available_bytes": memory.available,
        "ram_total_bytes": memory.total,
        "process_rss_bytes": _PROCESS.memory_info().rss,
        "cuda_allocated_bytes": torch.cuda.memory_allocated()
        if device == "cuda"
        else None,
        "cuda_reserved_bytes": torch.cuda.memory_reserved()
        if device == "cuda"
        else None,
        "temperatures_c": temperatures,
        "gpu_load_raw_per_mille": gpu_loads,
        "thermal_throttling": None,
    }


def scenario_nodes(scenario, ros_nodes, moveit_nodes):
    if scenario not in ("A", "B", "C"):
        raise BenchmarkError("Scenario must be A, B or C")
    if scenario in ("B", "C") and not ros_nodes:
        raise BenchmarkError("ROS scenario requires explicit expected ROS nodes")
    if scenario == "C" and not moveit_nodes:
        raise BenchmarkError("MoveIt scenario requires explicit expected MoveIt nodes")
    listing = command(["ros2", "node", "list"])
    actual = set(listing.splitlines()) if listing is not None else set()
    if scenario == "A" and actual:
        raise BenchmarkError(
            "Policy-alone scenario found ROS nodes; isolate ROS_DOMAIN_ID"
        )
    expected = set(ros_nodes) | (set(moveit_nodes) if scenario == "C" else set())
    if not expected <= actual:
        raise BenchmarkError(
            "Missing required scenario nodes: " + ", ".join(sorted(expected - actual))
        )
    if scenario == "B" and (
        set(moveit_nodes) & actual or any("move_group" in n for n in actual)
    ):
        raise BenchmarkError("Scenario B unexpectedly contains MoveIt")
    return {
        "required": sorted(expected),
        "observed": sorted(actual),
        "graph_query_available": listing is not None,
    }


def summarize(durations, completions, *, elapsed, target_hz):
    if not durations or elapsed <= 0 or target_hz <= 0:
        raise BenchmarkError("Cannot summarize empty/invalid measurements")
    latency = np.asarray(durations) * 1000
    missed = sum(t > (i + 1) / target_hz for i, t in enumerate(completions))
    return {
        "samples": len(durations),
        "elapsed_s": elapsed,
        "target_hz": target_hz,
        "achieved_hz": len(durations) / elapsed,
        "missed_deadlines": missed,
        "p50_ms": float(np.percentile(latency, 50)),
        "p95_ms": float(np.percentile(latency, 95)),
        "p99_ms": float(np.percentile(latency, 99)),
        "max_ms": float(latency.max()),
        "latency_over_period": int((latency > 1000 / target_hz).sum()),
    }


def benchmark(
    checkpoint,
    observations,
    *,
    output,
    scenario,
    seconds,
    target_hz,
    device,
    warmup=10,
    ros_nodes=(),
    moveit_nodes=(),
    ros_config=None,
    moveit_config=None,
    threads=1,
):
    import torch

    from oa_embodied_ai.training.config import ACTConfig
    from oa_embodied_ai.training.model import ACTPolicy

    if (
        not math.isfinite(seconds)
        or not math.isfinite(target_hz)
        or seconds <= 0
        or target_hz <= 0
        or warmup < 1
    ):
        raise BenchmarkError(
            "Positive finite duration/rate and nonzero warmup required"
        )
    if device == "cuda" and not torch.cuda.is_available():
        raise BenchmarkError("CUDA requested but unavailable; no silent CPU fallback")
    if threads < 1:
        raise BenchmarkError("Positive CPU thread count required")
    torch.set_num_threads(threads)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint, observations = Path(checkpoint), Path(observations)
    hashes = {
        "checkpoint": file_sha256(checkpoint),
        "observations": file_sha256(observations),
    }
    env = environment()
    graph = scenario_nodes(scenario, ros_nodes, moveit_nodes)
    if (
        scenario in ("B", "C")
        and not ros_config
        or scenario == "C"
        and not moveit_config
    ):
        raise BenchmarkError("Scenario stack configuration artifacts required")
    stack_hashes = {
        k: file_sha256(Path(v))
        for k, v in [("ros", ros_config), ("moveit", moveit_config)]
        if v
    }
    load_start = time.perf_counter()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = ACTConfig(**payload["config"]).validate()
    model = ACTPolicy(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    arrays = np.load(observations, allow_pickle=False)
    if set(arrays.files) != {"state", "images"}:
        raise BenchmarkError("Observation NPZ must contain exactly state and images")
    states, images = arrays["state"], arrays["images"]
    if (
        states.ndim != 2
        or states.shape[1:] != (cfg.state_dim,)
        or images.shape != (len(states), 4, 3, *cfg.image_size)
        or len(states) < 1
    ):
        raise BenchmarkError(
            "Observation shapes differ from checkpoint state/cameras/resolution"
        )
    if (
        states.dtype != np.float32
        or images.dtype != np.float32
        or not np.isfinite(states).all()
        or not np.isfinite(images).all()
    ):
        raise BenchmarkError("Observation tensors must be finite float32")
    states = torch.from_numpy(states).to(device)
    images = torch.from_numpy(images).to(device)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    sync()
    startup = time.perf_counter() - load_start
    durations, completions, samples = [], [], []
    errors = []
    tegra = None
    tegra_log = (output / "tegrastats.log").open("w")
    if env["jetson_orin_nx_detected"] and shutil.which("tegrastats"):
        tegra = subprocess.Popen(
            ["tegrastats", "--interval", "1000"],
            stdout=tegra_log,
            stderr=subprocess.STDOUT,
        )
    try:
        with torch.inference_mode():
            for i in range(warmup):
                model(
                    states[i % len(states) : i % len(states) + 1],
                    images[i % len(states) : i % len(states) + 1],
                )
            sync()
            start = time.perf_counter()
            last_telemetry = start - 1
            while time.perf_counter() - start < seconds:
                i = len(durations)
                release = start + i / target_hz
                if release > time.perf_counter():
                    time.sleep(release - time.perf_counter())
                before = time.perf_counter()
                prediction = model(
                    states[i % len(states) : i % len(states) + 1],
                    images[i % len(states) : i % len(states) + 1],
                )[0]
                sync()
                after = time.perf_counter()
                if not torch.isfinite(prediction).all():
                    raise BenchmarkError("Nonfinite model output")
                durations.append(after - before)
                completions.append(after - start)
                if after - last_telemetry >= 1:
                    samples.append(telemetry(device))
                    last_telemetry = after
                    if scenario in ("B", "C"):
                        scenario_nodes(scenario, ros_nodes, moveit_nodes)
            elapsed = time.perf_counter() - start
    except (
        BenchmarkError,
        RuntimeError,
        OSError,
        ValueError,
        KeyboardInterrupt,
    ) as exc:
        errors.append(str(exc))
        elapsed = time.perf_counter() - start if "start" in locals() else 0
    finally:
        if tegra:
            tegra.terminate()
            try:
                tegra.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tegra.kill()
                tegra.wait()
        tegra_log.close()
    metrics = (
        summarize(durations, completions, elapsed=elapsed, target_hz=target_hz)
        if durations
        else None
    )
    if hashes != {
        "checkpoint": file_sha256(checkpoint),
        "observations": file_sha256(observations),
    }:
        errors.append("Checkpoint/input bytes changed during benchmark")
    if errors:
        status = "FAIL"
    elif env["jetson_orin_nx_detected"] and seconds >= 600 and elapsed >= 600:
        status = "MEASURED_ON_JETSON_REVIEW_PENDING"
    else:
        status = "DEVELOPMENT_ONLY"
    workload = {
        "hashes": hashes,
        "config": cfg.to_dict(),
        "precision": "float32",
        "batch_size": 1,
        "observation_sequence_length": 1,
        "device": device,
        "cpu_threads": threads,
    }
    raw = output / "inference-samples.jsonl"
    raw.write_text(
        "".join(
            json.dumps({"index": i, "latency_s": latency, "completion_s": completion})
            + "\n"
            for i, (latency, completion) in enumerate(
                zip(durations, completions, strict=True)
            )
        )
    )
    report = {
        "status": status,
        "scenario": scenario,
        "environment": env,
        "workload": workload,
        "workload_sha256": hashlib.sha256(
            json.dumps(workload, sort_keys=True).encode()
        ).hexdigest(),
        "inference_samples_sha256": file_sha256(raw),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "warmup_iterations": warmup,
        "startup_loading_s": startup,
        "metrics": metrics,
        "telemetry": samples,
        "scenario_nodes": graph,
        "stack_config_sha256": stack_hashes,
        "errors": errors,
        "missing_telemetry": [
            name
            for name, available in [
                ("jetpack", env["jetpack_l4t"]),
                ("power_mode", env["power_mode"]),
                ("gpu_utilization", any(s["gpu_load_raw_per_mille"] for s in samples)),
                ("temperatures", any(s["temperatures_c"] for s in samples)),
                ("thermal_throttling", False),
            ]
            if not available
        ],
        "execution": execution_provenance(),
        "physical_acceptance": "NOT_VERIFIED",
        "fit_decision": "NOT_VERIFIED: reviewer decision requires matched sustained A/B/C device runs",
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (output / "report.md").write_text(
        "# ACT-like inference benchmark\n\n"
        + status
        + "\n\n"
        + report["fit_decision"]
        + "\n\n```json\n"
        + json.dumps({k: v for k, v in report.items() if k != "telemetry"}, indent=2)
        + "\n```\n"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=["A", "B", "C"], required=True)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--target-hz", type=float, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--ros-nodes", nargs="*", default=[])
    parser.add_argument("--moveit-nodes", nargs="*", default=[])
    parser.add_argument("--ros-config", type=Path)
    parser.add_argument("--moveit-config", type=Path)
    args = parser.parse_args(argv)
    try:
        result = benchmark(**vars(args))
        print(result["status"], args.output / "metrics.json")
        return 2 if result["errors"] else 0
    except (
        BenchmarkError,
        OSError,
        ValueError,
        RuntimeError,
        ImportError,
        KeyError,
    ) as exc:
        print(f"REJECT: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
