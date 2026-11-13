# SPDX-License-Identifier: MIT
"""Fresh native C10 QA followed by the canonical fail-closed decision API."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..dataset_qa import episode as c10
from ..dataset_qa import training_readiness
from ..dataset_qa.contract import C9_SCHEMA_SHA256
from ..dataset_qa.handoff import REQUIRED_CHECKS
from ..dataset_qa.strict_io import load_data
from .config import get_profile_config
from .inputs import assert_unchanged, file_sha256, json_sha256, load_input_manifest

LOG = logging.getLogger("oa-c11-gate")


class TrainingBlockedError(Exception):
    """No training authorization; CLI exit code 2."""

    def __init__(
        self,
        message="Training gate is CLOSED: dataset failed quality verification.",
        exit_code=2,
    ):
        super().__init__(message)
        self.message = message
        self.exit_code = self.code = exit_code


@dataclass
class GateStatus:
    training_ready: bool
    quality_status: str
    source_profile: str
    qa_report: dict
    reasons: list = field(default_factory=list)
    is_synthetic_override: bool = False
    input_manifest: dict = field(default_factory=dict)
    decision: dict = field(default_factory=dict)
    report_sha256: str = ""

    def __getitem__(self, key):
        return getattr(self, key)


def _synthetic_testing_ready(report, manifest):
    """Permit ONLY the real-episode exception; never manufacture a C10 PASS.

    This is software-test execution permission, separate from canonical training
    readiness. Every other diagnostic must be present and PASS.
    """
    findings, episodes = report.get("findings"), report.get("episodes")
    if (
        report.get("report_version") != 1
        or type(report.get("report_version")) is not int
        or report.get("status") != "NOT_VERIFIED"
        or type(report.get("exit_code")) is not int
        or report["exit_code"] != 2
        or report.get("training_ready") is not False
        or report.get("production_acceptance") != "NOT_VERIFIED"
        or not isinstance(findings, list)
        or not findings
        or not isinstance(episodes, list)
        or not episodes
    ):
        return False
    provenance = report.get("provenance", {})
    policy = provenance.get("policy", {})
    if (
        provenance.get("c9_contract", {}).get("sha256") != C9_SCHEMA_SHA256
        or policy.get("approved") is not True
        or not all(
            isinstance(policy.get(k), str) and policy[k].strip()
            for k in ("id", "evidence", "sha256")
        )
    ):
        return False
    for finding in findings:
        if (
            not isinstance(finding, dict)
            or not isinstance(finding.get("check_id"), str)
            or (
                finding.get("episode") is not None
                and type(finding["episode"]) is not int
            )
        ):
            return False
        if finding["check_id"] == "C10_REAL_EPISODE":
            if (
                finding.get("status") != "NOT_VERIFIED"
                or finding.get("observed") != "synthetic"
            ):
                return False
        elif finding.get("status") != "PASS":
            return False
    actual = [e.get("episode_index") if isinstance(e, dict) else None for e in episodes]
    if any(type(i) is not int for i in actual) or sorted(actual) != sorted(
        manifest["episode_indices"]
    ):
        return False
    for episode in episodes:
        binding = episode.get("binding", {})
        if (
            episode.get("status") != "NOT_VERIFIED"
            or episode.get("data_origin") != "synthetic"
            or binding.get("dataset_id") != manifest["dataset_id"]
            or binding.get("dataset_revision") != manifest["dataset_revision"]
            or binding.get("episode_index") != episode["episode_index"]
        ):
            return False
        applicable = [
            f for f in findings if f.get("episode") in (None, episode["episode_index"])
        ]
        if REQUIRED_CHECKS - {f["check_id"] for f in applicable}:
            return False
        if not any(
            f["check_id"] == "C10_REAL_EPISODE"
            and f.get("episode") == episode["episode_index"]
            for f in applicable
        ):
            return False
    return True


def verify_training_gate(
    dataset_path,
    qa_report_path=None,
    allow_synthetic_override=False,
    *,
    input_manifest_path=None,
    policy_path=None,
):
    """Rerun C10 on the independently bound snapshot and exact selection.

    An optional old report is only a strict comparison artifact, never a gate
    substitute. No automatic discovery in the dataset or working directory.
    """
    try:
        if input_manifest_path is None or policy_path is None:
            raise ValueError("Explicit input manifest and reviewed policy are required")
        root = Path(dataset_path).resolve(strict=True)
        manifest = load_input_manifest(input_manifest_path)
        get_profile_config(manifest["profile"]).validate()
        assert_unchanged(root, manifest, policy_path)
        report = c10.check_dataset(root, Path(policy_path), manifest["episode_indices"])
        assert_unchanged(root, manifest, policy_path)
        decision = training_readiness(
            report,
            dataset_id=manifest["dataset_id"],
            dataset_revision=manifest["dataset_revision"],
            episode_indices=manifest["episode_indices"],
        )
        provenance = report["provenance"]
        if provenance["policy"]["sha256"] != manifest["policy_sha256"]:
            raise ValueError("C10 policy provenance mismatch")
        expected_checker = {
            name: file_sha256(Path(c10.__file__).parent / name)
            for name in (
                "episode.py",
                "semantics.py",
                "strict_io.py",
                "contract.py",
                "handoff.py",
                "video.py",
            )
        }
        if provenance.get("checker_files_sha256") != expected_checker:
            raise ValueError("C10 checker provenance mismatch")
        for episode in report["episodes"]:
            sidecar = episode["sidecar"]
            if sidecar["sha256"] != manifest["files_sha256"].get(sidecar["path"]):
                raise ValueError("C10 sidecar provenance mismatch")
            if episode["data_origin"] != manifest["data_origin"]:
                raise ValueError("C10 origin contradicts trusted input origin")
        if report.get("quality_status", report["status"]) != report["status"]:
            raise ValueError("Contradictory report status")
        if any(
            f.get("severity") in ("REJECT", "BLOCKED", "NOT_VERIFIED")
            and f.get("status") == "PASS"
            for f in report["findings"]
        ):
            raise ValueError("Contradictory finding severity")
        if qa_report_path is not None and load_data(Path(qa_report_path)) != report:
            raise ValueError("Supplied QA report differs from fresh native QA")
        synthetic = manifest["data_origin"] == "synthetic"
        if synthetic and report.get("source_profile") == "ROBOT":
            raise ValueError("ROBOT label contradicts synthetic origin")
        if synthetic:
            if not allow_synthetic_override or not _synthetic_testing_ready(
                report, manifest
            ):
                raise ValueError(
                    "Synthetic testing requires complete QA except real-episode evidence"
                )
            LOG.warning(
                "Synthetic software test only; canonical C10 remains NOT_VERIFIED"
            )
        elif not decision["training_ready"]:
            raise ValueError("; ".join(decision["reasons"]))
        return GateStatus(
            training_ready=decision["training_ready"],
            quality_status=decision["status"],
            source_profile="SYNTHETIC" if synthetic else "ROBOT",
            qa_report=report,
            reasons=decision["reasons"],
            is_synthetic_override=synthetic,
            input_manifest=manifest,
            decision=decision,
            report_sha256=json_sha256(report),
        )
    except TrainingBlockedError:
        raise
    except Exception as error:
        raise TrainingBlockedError(f"Training gate is CLOSED: {error}") from error
