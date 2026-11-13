#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run bounded software verification and retain exact commands/results/HEAD.

Requires installed test/native/training/benchmark extras, build, Ruff and uv, plus ffmpeg.
This runner downloads no datasets and invokes no GPU, simulator or hardware job.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from oa_embodied_ai.training.evidence import execution_provenance
from oa_embodied_ai.training.inputs import file_sha256

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    evidence = {
        "purpose": "SOFTWARE ONLY; real recording and production NOT_VERIFIED",
        "execution": execution_provenance(),
        "platform": platform.platform(),
        "packages": dict(
            sorted(
                (d.metadata["Name"], d.version)
                for d in importlib.metadata.distributions()
                if d.metadata.get("Name")
            )
        ),
        "commands": [],
    }
    (output / "environment-requirements.txt").write_text(
        "\n".join(f"{k}=={v}" for k, v in evidence["packages"].items()) + "\n"
    )

    def run(name, argv, env=None):
        with (output / f"{name}.log").open("wb") as log:
            result = subprocess.run(
                argv,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        evidence["commands"].append(
            {
                "name": name,
                "argv": list(map(str, argv)),
                "cwd": str(ROOT),
                "exit_code": result.returncode,
                "log": f"{name}.log",
            }
        )
        print(f"{name}: exit {result.returncode}", flush=True)
        return result.returncode

    py = sys.executable
    run(
        "integrated-pytest",
        [
            py,
            "-m",
            "pytest",
            "python/tests",
            "-q",
            f"--junitxml={output / 'pytest.xml'}",
        ],
    )
    for pattern, name in [
        ("test_c10_native.py", "native-suite"),
        ("test_c10_evidence.py", "evidence-suite"),
    ]:
        run(
            name,
            [
                py,
                "python/tools/run_verification_tests.py",
                "python/native_tests",
                pattern,
            ],
        )
    run("c10-negative", [py, "python/tools/prove_c10_negative.py"])
    run("c11-negative", [py, "python/tools/prove_c11_negative.py"])
    run("converter-negative", [py, "python/tools/prove_converter_negative.py"])
    run("video-negative", [py, "python/tools/prove_video_negative.py"])
    run(
        "smoke",
        [
            py,
            "python/tools/run_c11_smoke.py",
            "--output-dir",
            str(output / "final-smoke"),
        ],
    )
    run(
        "c10-cli",
        [
            py,
            "python/tools/generate_c10_evidence.py",
            "--output-dir",
            str(output / "final-c10"),
            "--expected-checker-sha",
            evidence["execution"]["git_commit"],
        ],
    )
    scoped = [
        "python/oa_embodied_ai/recording",
        "python/oa_embodied_ai/conversion",
        "python/oa_embodied_ai/benchmarking",
        "python/tests/test_conversion_validation.py",
        "python/tests/test_benchmark_runtime.py",
        "python/tests/test_recording_lifecycle.py",
        "python/tests/capture_fixture.py",
        "python/native_tests/test_bag_conversion.py",
        "python/tools/prove_converter_negative.py",
        "python/tools/generate_capture_evidence.py",
        "python/tools/run_bag_training_smoke.py",
        "python/tools/prove_video_negative.py",
        "python/tests/test_video_segments.py",
        "python/oa_embodied_ai/dataset_qa/video.py",
        "python/oa_embodied_ai/training",
        "python/oa_embodied_ai/evaluation",
        "python/tests/test_c11_gate.py",
        "python/tests/test_c11_training.py",
        "python/tests/test_c11_evaluation.py",
        "python/tests/test_c10_multi_episode_policy.py",
        "python/tests/c11_fixture.py",
        "python/tests/conftest.py",
        "python/tools/prove_c11_negative.py",
        "python/tools/run_c11_smoke.py",
        "python/tools/verify_c10_c11.py",
    ]
    run("lint", [py, "-m", "ruff", "check", "--isolated", "--select", "F,I,B", *scoped])
    run("format", [py, "-m", "ruff", "format", "--isolated", "--check", *scoped])
    run("diff-check", ["git", "diff", "--check"])
    if (
        run(
            "wheel",
            [py, "-m", "build", "--wheel", "python", "--outdir", str(output / "wheel")],
        )
        == 0
    ):
        wheels = list((output / "wheel").glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError("Expected exactly one wheel")
        env_python = output / "wheel-env/bin/python"
        run("wheel-env", ["uv", "venv", "--python", py, str(env_python.parent.parent)])
        run(
            "wheel-install",
            ["uv", "pip", "install", "--python", str(env_python), str(wheels[0])],
        )
        env = dict(
            os.environ, OA_INTERFACES_ROOT=str(ROOT / "external/openamrobot-interfaces")
        )
        env.pop("PYTHONPATH", None)
        script = "from oa_embodied_ai.dataset_qa import training_readiness; from oa_embodied_ai.training import verify_training_gate; from oa_embodied_ai.dataset_qa.contract import load_c9_schema; from importlib.metadata import version; load_c9_schema(); print('Installed wheel core imports/C9 PASS', version('oa_embodied_ai'))"
        run("wheel-import", [str(env_python), "-c", script], env=env)
        evidence["wheel_sha256"] = file_sha256(wheels[0])
    evidence["execution_after"] = execution_provenance()
    evidence["source_unchanged"] = (
        evidence["execution"]["source_sha256"]
        == evidence["execution_after"]["source_sha256"]
    )
    evidence["software_verification_passed"] = (
        all(c["exit_code"] == 0 for c in evidence["commands"])
        and evidence["source_unchanged"]
    )
    evidence["logs_sha256"] = {p.name: file_sha256(p) for p in output.glob("*.log")}
    (output / "final-execution.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(f"Evidence: {output / 'final-execution.json'}", flush=True)
    return 0 if evidence["software_verification_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
