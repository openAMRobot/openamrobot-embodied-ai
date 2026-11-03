# SPDX-License-Identifier: MIT
"""Execution evidence from actual bytes/environment; no acceptance promotion."""

import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from .inputs import file_sha256, json_sha256


def execution_provenance():
    root = Path(__file__).resolve().parents[3]

    def git(*args):
        process = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        )
        return process.stdout.strip() if process.returncode == 0 else "unknown"

    packages = {}
    for name in (
        "torch",
        "torchvision",
        "pyarrow",
        "opencv-python-headless",
        "lerobot",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    sources = {
        str(p.relative_to(root)): file_sha256(p)
        for p in sorted((root / "python/oa_embodied_ai").rglob("*.py"))
    }
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "source_tree_clean": not bool(
            git("status", "--porcelain", "--untracked-files=all")
        ),
        "source_files_sha256": sources,
        "source_sha256": json_sha256(sources),
        "python_version": sys.version.split()[0],
        "dependencies": packages,
        "model_implementation": "custom ACT-like CVAE/ResNet18; upstream LeRobot compatibility NOT_VERIFIED",
    }


def write_report(path, report):
    """JSON and Markdown share one status source."""
    path = Path(path)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    path.with_suffix(".md").write_text(
        "# C11 software execution evidence\n\n"
        f"Acceptance: **{report['acceptance_status']}**\n\n"
        f"Production acceptance: **{report['production_acceptance']}**\n\n"
        "Offline metrics are prediction errors. Physical operation is NOT_VERIFIED.\n\n"
        "```json\n"
        + json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
        + "\n```\n"
    )
