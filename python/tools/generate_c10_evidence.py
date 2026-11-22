#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate a fresh synthetic CLI evidence bundle without approving any data.

Exit zero verifies the reporting path's expected synthetic NOT_VERIFIED result;
the checker still exits two and its original report is retained unchanged.
Without --expected-checker-sha, dirty source is permitted and explicitly
recorded. With that option, both the exact Git HEAD and a clean tree are
required. The ignored .verification/c10-evidence/ directory is recommended.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = ('python/oa_embodied_ai/dataset_qa/episode.py', 'python/oa_embodied_ai/dataset_qa/semantics.py', 'python/oa_embodied_ai/dataset_qa/strict_io.py', 'python/oa_embodied_ai/dataset_qa/contract.py', 'python/oa_embodied_ai/dataset_qa/handoff.py', 'python/tests/c10_fixture.py', 'python/tools/generate_c10_evidence.py', 'external/openamrobot-interfaces/schemas/json/c9-lerobot-dataset.schema.json', 'external/openamrobot-interfaces/schemas/yaml/c9-lerobot-dataset-valid.yaml')
EXPECTED_SYNTHETIC_RESULT = {
    "process_exit_code": 2,
    "status": "NOT_VERIFIED",
    "report_exit_code": 2,
    "training_ready": False,
    "production_acceptance": "NOT_VERIFIED",
}


def sha256_file(path):
    """Hash file bytes without loading a dataset artifact into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_inventory(directory):
    """Inventory only this helper's newly generated synthetic directory."""
    directory = Path(directory)
    if not directory.exists():
        return {}
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Unexpected symbolic link in generated dataset: {path}")
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = sha256_file(path)
    return result


def git_state():
    """Report actual checkout identity; never substitute an expected SHA."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                            cwd=ROOT, check=True, capture_output=True, text=True).stdout
    return head, not bool(status)


def run_checker(argv, cwd, stdout_path, stderr_path):
    """Run the actual CLI and preserve its stdout and stderr bytes verbatim."""
    with Path(stdout_path).open("wb") as stdout, Path(stderr_path).open("wb") as stderr:
        return subprocess.run(argv, cwd=cwd, stdout=stdout, stderr=stderr, check=False).returncode


def _fixture_module():
    # Import the existing fixture by path to avoid collisions with installed
    # packages named tests. No alternative fixture or policy lives here.
    spec = importlib.util.spec_from_file_location("c10_evidence_fixture", ROOT / "python/tests/c10_fixture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _environment():
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name] = distribution.version
    return {"python_executable": sys.executable, "python_version": sys.version,
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(), "packages": dict(sorted(packages.items()))}


def _read_json(path):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    return json.loads(Path(path).read_text(), object_pairs_hook=unique_pairs,
                      parse_constant=invalid_constant)


def _synthetic_identity(dataset):
    sidecars = sorted((dataset / "meta/openamrobot/episodes").glob("episode-*.json"))
    if len(sidecars) != 1:
        raise ValueError("Expected exactly one generated synthetic sidecar")
    sidecar = _read_json(sidecars[0])
    binding = sidecar["native_lerobot"]
    identity = {key: binding[key] for key in
                ("dataset_id", "dataset_revision", "episode_id", "episode_index")}
    identity["data_origin"] = sidecar["openamrobot"]["provenance"]["data_origin"]
    if identity["data_origin"] != "synthetic":
        raise ValueError("Generated fixture does not declare synthetic data_origin")
    return identity


def expected_gate_result(report, process_exit):
    """Independently verify the expected synthetic result, without changing it."""
    if not isinstance(report, dict):
        return False
    findings = report.get("findings", [])
    # Exact types matter: False equals 0, but it is not an integer exit code.
    return (type(process_exit) is int and process_exit == 2
            and type(report.get("exit_code")) is int and report["exit_code"] == 2
            and report.get("status") == "NOT_VERIFIED"
            and report.get("training_ready") is False
            and report.get("production_acceptance") == "NOT_VERIFIED"
            and isinstance(findings, list) and any(
                isinstance(item, dict) and item.get("check_id") == "C10_REAL_EPISODE"
                and item.get("status") == "NOT_VERIFIED" and item.get("observed") == "synthetic"
                for item in findings))


def generate_evidence(output_dir, expected_checker_sha=None):
    """Create an evidence bundle, returning 0 only for the expected raw result.

    An existing output path is never modified, even if it is empty. Once a
    fresh directory exists, failures retain the generated files, logs and a
    partial evidence.json whenever the filesystem permits writing it.
    """
    output_dir = Path(output_dir).expanduser().absolute()
    try:
        # mkdir(exist_ok=False) refuses existing files, directories and symlinks.
        output_dir.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        print(f"C10 evidence output must be a fresh directory: {error}", file=sys.stderr)
        return 1

    output_dir = output_dir.resolve()
    dataset = output_dir / "dataset"
    report_path = output_dir / "c10-report.json"
    stdout_path, stderr_path = output_dir / "cli-stdout.log", output_dir / "cli-stderr.log"
    manifest = {
        "evidence_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Synthetic CLI reporting-path evidence only; owner adoption and production acceptance remain pending.",
        "checker": {"git_head": None, "expected_checker_sha": expected_checker_sha,
                    "source_tree_clean": None, "immutable_checkout_verified": False},
        "invocation": None,
        "dataset": {"path": str(dataset)},
        "artifacts": {"report": "c10-report.json", "stdout": "cli-stdout.log",
                      "stderr": "cli-stderr.log", "manifest": "evidence.json"},
        "environment": None,
        "source_files_sha256": {},
        "source_files_sha256_after": {},
        "dataset_files_sha256_before": {},
        "dataset_files_sha256_after": {},
        "dataset_unchanged": None,
        "report_sha256": None,
        "gate": {key: None for key in EXPECTED_SYNTHETIC_RESULT},
        "expected_synthetic_result": dict(EXPECTED_SYNTHETIC_RESULT),
        "reporting_path_success": False,
        "helper_exit_code": 1,
        "errors": [],
    }
    try:
        stdout_path.touch(exist_ok=False)
        stderr_path.touch(exist_ok=False)
        head, clean = git_state()
        manifest["checker"].update(git_head=head, source_tree_clean=clean)
        manifest["environment"] = _environment()
        manifest["source_files_sha256"] = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
        if expected_checker_sha is not None:
            if not re.fullmatch(r"[0-9a-f]{40}", expected_checker_sha):
                raise ValueError("--expected-checker-sha must be a full lowercase 40-character Git SHA")
            if head != expected_checker_sha:
                raise ValueError(f"Checker HEAD {head} differs from expected SHA {expected_checker_sha}")
            if not clean:
                raise ValueError("--expected-checker-sha requires a clean tracked and untracked source tree")

        fixture_module = _fixture_module()
        sidecar, native, policy = fixture_module.fixture()
        if sidecar["openamrobot"]["provenance"]["data_origin"] != "synthetic":
            raise ValueError("Existing fixture generator did not return synthetic data")
        fixture_module.write_dataset(dataset, sidecar, native, policy)
        identity = _synthetic_identity(dataset)
        manifest["dataset"].update(identity)
        manifest["dataset_files_sha256_before"] = file_inventory(dataset)
        argv = [sys.executable, "-m", "oa_embodied_ai.dataset_qa.episode",
                "--dataset-path", str(dataset), "--policy", str(dataset / "synthetic-policy.json"),
                "--report", str(report_path)]
        manifest["invocation"] = {"argv": argv, "cwd": str(ROOT), "shell_command": shlex.join(argv)}
        manifest["gate"]["process_exit_code"] = run_checker(argv, ROOT, stdout_path, stderr_path)
        manifest["dataset_files_sha256_after"] = file_inventory(dataset)
        manifest["dataset_unchanged"] = (manifest["dataset_files_sha256_before"] == manifest["dataset_files_sha256_after"])
        if not report_path.is_file() or report_path.is_symlink():
            raise ValueError("Checker did not create an ordinary report file")
        manifest["report_sha256"] = sha256_file(report_path)
        try:
            report = _read_json(report_path)
        except (OSError, ValueError) as error:
            raise ValueError(f"Checker report could not be read ({type(error).__name__}): {error}") from error
        if not isinstance(report, dict):
            raise ValueError("Checker report must be a JSON object")
        manifest["gate"].update(status=report.get("status"), report_exit_code=report.get("exit_code"),
                                training_ready=report.get("training_ready"),
                                production_acceptance=report.get("production_acceptance"))
        if not manifest["dataset_unchanged"]:
            raise ValueError("Generated dataset changed while the read-only checker ran")
        if _synthetic_identity(dataset) != identity:
            raise ValueError("Generated synthetic identity changed while the checker ran")
        if not expected_gate_result(report, manifest["gate"]["process_exit_code"]):
            raise ValueError("Checker outcome differs from the expected synthetic NOT_VERIFIED result or lacks synthetic-origin evidence")
        manifest["reporting_path_success"] = True
        manifest["helper_exit_code"] = 0
    except Exception as error:
        manifest["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        # Preserve useful partial evidence even if generation or invocation
        # failed. The original report and CLI logs are never rewritten here.
        for field, operation in (
            ("dataset_files_sha256_after", lambda: file_inventory(dataset)),
            ("source_files_sha256_after", lambda: {name: sha256_file(ROOT / name) for name in SOURCE_FILES}),
        ):
            try:
                manifest[field] = operation()
            except Exception as error:
                manifest["errors"].append(f"Could not record {field}: {error}")
                manifest["reporting_path_success"] = False
                manifest["helper_exit_code"] = 1
        if manifest["dataset_files_sha256_before"]:
            manifest["dataset_unchanged"] = manifest["dataset_files_sha256_before"] == manifest["dataset_files_sha256_after"]
        try:
            after_head, after_clean = git_state()
            manifest["checker"].update(git_head_after=after_head, source_tree_clean_after=after_clean)
            if (manifest["checker"]["git_head"] != after_head
                    or manifest["source_files_sha256"] != manifest["source_files_sha256_after"]):
                raise ValueError("Checker source changed during evidence generation")
            if expected_checker_sha is not None and not after_clean:
                raise ValueError("Source tree is dirty after the exact-SHA evidence run")
            if manifest["dataset_unchanged"] is False:
                raise ValueError("Generated dataset differs from its pre-checker inventory")
            manifest["checker"]["immutable_checkout_verified"] = bool(
                expected_checker_sha == after_head and manifest["checker"]["source_tree_clean"] and after_clean)
        except Exception as error:
            manifest["errors"].append(f"Final identity verification failed: {error}")
            manifest["reporting_path_success"] = False
            manifest["helper_exit_code"] = 1
        if report_path.is_file() and not report_path.is_symlink():
            try:
                manifest["report_sha256"] = sha256_file(report_path)
            except OSError as error:
                manifest["errors"].append(f"Could not hash report: {error}")
                manifest["reporting_path_success"] = False
                manifest["helper_exit_code"] = 1
        try:
            (output_dir / "evidence.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        except (OSError, ValueError) as error:
            print(f"C10 partial evidence could not be saved: {error}", file=sys.stderr)
            return 1

    print(f"Checker HEAD: {manifest['checker']['git_head']}; source_tree_clean={manifest['checker']['source_tree_clean']}")
    if "dataset_revision" in manifest["dataset"]:
        print(f"Synthetic dataset: {manifest['dataset']['dataset_id']} at revision {manifest['dataset']['dataset_revision']}")
    if manifest["invocation"]:
        print(f"Checker command (cwd={ROOT}): {manifest['invocation']['shell_command']}")
    if manifest["reporting_path_success"]:
        print(f"C10 synthetic reporting path verified; raw gate remains NOT_VERIFIED (exit 2): {output_dir}")
    else:
        print(f"C10 evidence generation failed; retained available artifacts at {output_dir}", file=sys.stderr)
        for error in manifest["errors"]:
            print(f"  {error}", file=sys.stderr)
    return manifest["helper_exit_code"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New evidence directory; existing paths are never modified")
    parser.add_argument("--expected-checker-sha",
                        help="Require this full Git HEAD SHA and a clean source tree")
    args = parser.parse_args(argv)
    return generate_evidence(args.output_dir, args.expected_checker_sha)


if __name__ == "__main__":
    raise SystemExit(main())
