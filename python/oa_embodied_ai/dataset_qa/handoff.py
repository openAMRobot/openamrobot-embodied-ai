# SPDX-License-Identifier: MIT
"""Fail-closed C10 report consumption; caller retains responsibility for authenticity.

Use a freshly executed check_dataset result on the immutable training snapshot.
This consistency check cannot authenticate an arbitrary JSON document or approve policy.
"""

from .contract import C9_SCHEMA_SHA256
from .semantics import CHECK_IDS

REQUIRED_CHECKS = frozenset(CHECK_IDS) | {
    "C10_SCHEMA", "C10_NATIVE_METADATA", "C10_NATIVE_DATA", "C10_PARQUET_DTYPE",
    "C10_VIDEO", "C10_ARTIFACT", "C10_CALIBRATION", "C10_RELEASE", "C10_REAL_EPISODE",
}


def training_readiness(report, *, dataset_id, dataset_revision, episode_indices):
    """Return an explicit decision without promoting incomplete/blocked reports.

    Expected identity is supplied independently by C11, never inferred from a registry
    quality label. Selection must match exactly. Production acceptance is separate.
    """
    reasons = []
    if not isinstance(report, dict):
        return {"training_ready": False, "status": "NOT_VERIFIED", "reasons": ["missing C10 report"]}
    findings, episodes = report.get("findings"), report.get("episodes")
    if report.get("report_version") != 1 or type(report.get("report_version")) is not int:
        reasons.append("unsupported report version")
    if (report.get("status") != "PASS" or type(report.get("exit_code")) is not int
            or report.get("exit_code") != 0 or report.get("training_ready") is not True):
        reasons.append("C10 gate is not a consistent PASS")
    if not isinstance(findings, list) or not findings or any(
            not isinstance(f, dict) or f.get("status") != "PASS" for f in findings):
        reasons.append("missing, blocked or rejected findings")
    findings = findings if isinstance(findings, list) else []
    if any(not isinstance(f, dict) or not isinstance(f.get("check_id"), str)
           or (f.get("episode") is not None and type(f["episode"]) is not int) for f in findings):
        reasons.append("malformed finding identity")
        findings = []
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    contract = provenance.get("c9_contract")
    if not isinstance(contract, dict) or contract.get("sha256") != C9_SCHEMA_SHA256:
        reasons.append("canonical contract provenance missing")
    policy = provenance.get("policy")
    if (not isinstance(policy, dict) or policy.get("approved") is not True
            or not all(isinstance(policy.get(k), str) and policy[k].strip()
                       for k in ("id", "evidence", "sha256"))):
        reasons.append("reviewed policy provenance missing")
    expected = list(episode_indices)
    if (not expected or any(type(i) is not int or i < 0 for i in expected)
            or len(set(expected)) != len(expected)):
        reasons.append("expected episode selection is empty or invalid")
    if not isinstance(episodes, list) or not episodes:
        reasons.append("no checked episodes")
        episodes = []
    actual = [e.get("episode_index") if isinstance(e, dict) else None for e in episodes]
    if (any(type(i) is not int for i in actual) or sorted(actual, key=str) != sorted(expected, key=str)):
        reasons.append("episode selection differs from training input")
    for episode in episodes:
        if not isinstance(episode, dict):
            reasons.append("malformed episode")
            continue
        binding = episode.get("binding")
        if (episode.get("status") != "PASS" or episode.get("data_origin") != "actual"
                or not isinstance(binding, dict) or binding.get("dataset_id") != dataset_id
                or binding.get("dataset_revision") != dataset_revision
                or binding.get("episode_index") != episode.get("episode_index")):
            reasons.append("episode identity/origin/verdict does not establish actual recording evidence")
        applicable = [f for f in findings if isinstance(f, dict)
                      and f.get("episode") in (None, episode.get("episode_index"))]
        if REQUIRED_CHECKS - {f.get("check_id") for f in applicable}:
            reasons.append("required diagnostic coverage missing")
        if not any(f.get("check_id") == "C10_REAL_EPISODE" and f.get("observed") == "actual"
                   and f.get("status") == "PASS" and f.get("episode") == episode.get("episode_index")
                   for f in applicable):
            reasons.append("real episode evidence missing")
    rejected = report.get("status") == "REJECT" or any(
        isinstance(f, dict) and f.get("status") == "REJECT" for f in findings)
    ready = not reasons
    return {"training_ready": ready, "status": "PASS" if ready else "REJECT" if rejected else "NOT_VERIFIED",
            "reasons": sorted(set(reasons))}
