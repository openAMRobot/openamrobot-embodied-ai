# SPDX-License-Identifier: MIT
"""Fail-closed C10 pre-flight training gate consumer.

Deliverable C11: ACT Imitation Learning Baseline.
Resolves C10->C11 handoff findings:
1. Gating Rule (Fail-Closed): Training must fail closed on unverified,
   synthetic, or contradictory C10 reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG = logging.getLogger("oa-c11-gate")


class TrainingBlockedError(Exception):
    """Raised when the pre-flight quality verification gate fails."""

    def __init__(
        self,
        message: str = "Training gate is CLOSED: dataset failed quality verification.",
        exit_code: int = 2,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code
        self.code = exit_code


@dataclass
class GateStatus:
    """Status record returned when dataset passes the pre-flight gate."""

    training_ready: bool
    quality_status: str
    source_profile: str
    qa_report: Dict[str, Any]
    reasons: List[str] = field(default_factory=list)
    is_synthetic_override: bool = False

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def verify_training_gate(
    dataset_path: Path,
    qa_report_path: Optional[Path] = None,
    allow_synthetic_override: bool = False,
) -> GateStatus:
    """Verify that the dataset satisfies all C10 quality criteria before training.

    Fail-Closed Invariants:
      a) training_ready MUST be explicitly True. If false or missing, raise
         TrainingBlockedError and exit code 2.
      b) quality_status MUST NOT be BLOCKED, REJECT, or NOT_VERIFIED.
      c) If source_profile != "ROBOT", it MUST fail closed unless
         allow_synthetic_override is explicitly True, in which case it logs:
         "WARNING: Synthetic data override active. Artifacts will be stamped NOT_VERIFIED_PHYSICAL."
      d) Contradictory check: If the report contains ANY finding with
         severity == "REJECT" (or status == "REJECT"), training is blocked
         regardless of top-level boolean fields.

    Args:
        dataset_path: Directory path to LeRobotDataset.
        qa_report_path: Optional direct path to QA report JSON.
        allow_synthetic_override: If True, permits synthetic data for CI/dry-runs.

    Returns:
        GateStatus containing gate verification metadata.

    Raises:
        TrainingBlockedError: If any gate invariant fails.
    """
    dataset_path = Path(dataset_path)
    report_dict: Optional[Dict[str, Any]] = None

    # 1. Resolve and load QA report
    if qa_report_path is not None:
        qa_path = Path(qa_report_path)
        if not qa_path.is_file():
            LOG.error("QA report file not found at %s", qa_path)
            raise TrainingBlockedError(
                "Training gate is CLOSED: dataset failed quality verification.", exit_code=2
            )
        try:
            report_dict = json.loads(qa_path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOG.error("Failed to parse QA report JSON at %s: %s", qa_path, exc)
            raise TrainingBlockedError(
                "Training gate is CLOSED: dataset failed quality verification.", exit_code=2
            ) from exc
    else:
        # Check standard report locations
        candidate_paths = [
            dataset_path / "reports" / "qa_report.json",
            dataset_path / "qa_report.json",
            Path("reports") / "qa_report.json",
        ]
        for candidate in candidate_paths:
            if candidate.is_file():
                try:
                    report_dict = json.loads(candidate.read_text(encoding="utf-8"))
                    LOG.info("Loaded pre-flight QA report from %s", candidate)
                    break
                except Exception as exc:
                    LOG.warning("Candidate report at %s failed to parse: %s", candidate, exc)

        # If still not found, execute programmatic check_dataset
        if report_dict is None:
            try:
                from oa_embodied_ai.dataset_qa.episode import check_dataset

                LOG.info("Running programmatic dataset QA check on %s", dataset_path)
                report_dict = check_dataset(dataset_path)
            except Exception as exc:
                LOG.error("Programmatic check_dataset execution failed: %s", exc)
                raise TrainingBlockedError(
                    "Training gate is CLOSED: dataset failed quality verification.", exit_code=2
                ) from exc

    if not isinstance(report_dict, dict):
        LOG.error("Invalid QA report: expected dictionary, got %s", type(report_dict))
        raise TrainingBlockedError(
            "Training gate is CLOSED: dataset failed quality verification.", exit_code=2
        )

    # 2. Check exit code if present
    exit_code = report_dict.get("exit_code")
    if exit_code is not None and (exit_code != 0 or type(exit_code) is not int):
        LOG.error("QA report reports nonzero exit_code: %s", exit_code)
        raise TrainingBlockedError(
            "Training gate is CLOSED: dataset failed quality verification.", exit_code=2
        )

    # 3. Invariant (a): training_ready MUST be explicitly True
    if report_dict.get("training_ready") is not True:
        LOG.error("training_ready is not True (value=%s)", report_dict.get("training_ready"))
        raise TrainingBlockedError(
            "Training gate is CLOSED: dataset failed quality verification.", exit_code=2
        )

    # 4. Invariant (b): quality_status MUST NOT be BLOCKED, REJECT, or NOT_VERIFIED
    blocked_statuses = {"BLOCKED", "REJECT", "NOT_VERIFIED"}
    quality_status = (
        report_dict.get("quality_status")
        or report_dict.get("status")
        or "NOT_VERIFIED"
    )
    if not isinstance(quality_status, str):
        LOG.error("Invalid quality_status type: %s", type(quality_status))
        raise TrainingBlockedError(
            "Training gate is CLOSED: dataset failed quality verification.", exit_code=2
        )

    normalized_status = quality_status.upper()
    if normalized_status in blocked_statuses:
        LOG.error("QA report indicates blocked status: %s", quality_status)
        raise TrainingBlockedError(
            "Training gate is CLOSED: dataset failed quality verification.", exit_code=2
        )

    if normalized_status not in {"PASS", "PASSED"}:
        LOG.error("QA report quality_status '%s' is not PASS/PASSED", quality_status)
        raise TrainingBlockedError(
            "Training gate is CLOSED: dataset failed quality verification.", exit_code=2
        )

    # 5. Invariant (d): Contradictory check: If ANY finding has severity == "REJECT" or status == "REJECT"
    findings = report_dict.get("findings")
    if isinstance(findings, list):
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            f_sev = str(finding.get("severity", "")).upper()
            f_stat = str(finding.get("status", "")).upper()
            if f_sev in {"REJECT", "BLOCKED", "NOT_VERIFIED"} or f_stat in {"REJECT", "BLOCKED", "NOT_VERIFIED"}:
                LOG.error(
                    "Contradictory finding detected: check_id=%s, severity=%s, status=%s",
                    finding.get("check_id"),
                    finding.get("severity"),
                    finding.get("status"),
                )
                raise TrainingBlockedError(
                    "Training gate is CLOSED: dataset failed quality verification.", exit_code=2
                )

    # 6. Invariant (c): source_profile check
    source_profile = (
        report_dict.get("source_profile")
        or report_dict.get("provenance", {}).get("source_profile")
        or report_dict.get("meta", {}).get("source_profile")
    )
    if source_profile is None:
        # Check episode-level data origin
        episodes = report_dict.get("episodes")
        if isinstance(episodes, list) and episodes:
            all_actual = all(
                isinstance(e, dict) and e.get("data_origin") == "actual" for e in episodes
            )
            any_synthetic = any(
                isinstance(e, dict) and e.get("data_origin") == "synthetic" for e in episodes
            )
            if any_synthetic:
                source_profile = "SYNTHETIC"
            elif all_actual:
                source_profile = "ROBOT"
            else:
                source_profile = "UNKNOWN"
        else:
            source_profile = "UNKNOWN"

    source_profile_norm = str(source_profile).upper()
    is_synthetic_override = False

    if source_profile_norm != "ROBOT":
        if not allow_synthetic_override:
            LOG.error(
                "Source profile '%s' is not ROBOT and allow_synthetic_override is False.",
                source_profile,
            )
            raise TrainingBlockedError(
                "Training gate is CLOSED: dataset failed quality verification.", exit_code=2
            )
        LOG.warning("WARNING: Synthetic data override active. Artifacts will be stamped NOT_VERIFIED_PHYSICAL.")
        is_synthetic_override = True

    return GateStatus(
        training_ready=True,
        quality_status=quality_status,
        source_profile=str(source_profile),
        qa_report=report_dict,
        is_synthetic_override=is_synthetic_override,
    )

