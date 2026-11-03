# SPDX-License-Identifier: MIT
"""Per-episode policy routing using fabricated metadata, never recording evidence."""

import json

import pytest
from c10_fixture import fixture
from oa_embodied_ai.dataset_qa.episode import check_dataset, evidence


def real_finding(sidecar, policy, root):
    findings = []
    evidence(
        root,
        sidecar,
        policy,
        lambda name, status, **kw: findings.append((name, status)),
    )
    return next(status for name, status in findings if name == "C10_REAL_EPISODE")


def test_explicit_policy_bindings_cover_each_episode_without_fallback(tmp_path):
    sidecar, _, policy = fixture()
    # Fabricated unit-level metadata tests decision routing only. No native
    # artifacts or PASS report are generated, and no actual recording is claimed.
    sidecar["openamrobot"]["provenance"]["data_origin"] = "actual"
    binding = sidecar["native_lerobot"]
    policy["real_episodes"] = {}
    for index in (0, 7):
        item = {
            k: binding[k]
            for k in ("dataset_id", "dataset_revision", "episode_id", "episode_index")
        }
        item.update(
            episode_index=index,
            episode_id=f"fabricated-protocol-{index}",
            evidence="FABRICATED UNIT TEST ONLY; no physical approval",
        )
        policy["real_episodes"][str(index)] = item
    for index in (0, 7):
        sidecar["native_lerobot"].update(
            episode_index=index, episode_id=f"fabricated-protocol-{index}"
        )
        assert real_finding(sidecar, policy, tmp_path) == "PASS"
    sidecar["native_lerobot"]["episode_index"] = 8
    assert real_finding(sidecar, policy, tmp_path) == "NOT_VERIFIED"
    sidecar["native_lerobot"].update(
        episode_index=0, episode_id="fabricated-protocol-0"
    )
    sidecar["openamrobot"]["provenance"]["data_origin"] = "synthetic"
    assert real_finding(sidecar, policy, tmp_path) == "NOT_VERIFIED"
    sidecar["openamrobot"]["provenance"]["data_origin"] = "actual"
    policy["approved"] = False
    assert real_finding(sidecar, policy, tmp_path) == "NOT_VERIFIED"


@pytest.mark.parametrize(
    "bad", [[], {"01": {}}, {"-1": {}}, {"0": []}, {"0": {"episode_index": 1}}]
)
def test_malformed_multi_episode_policy_is_rejected_before_qa(tmp_path, bad):
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"real_episodes": bad}))
    report = check_dataset(tmp_path, policy, [0, 1])
    assert report["status"] == "REJECT" and not report["training_ready"]
    assert any(
        f["check_id"] == "C10_INPUT" and f["status"] == "REJECT"
        for f in report["findings"]
    )


def test_ambiguous_single_and_multi_episode_approvals_are_rejected(tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"real_episode": {}, "real_episodes": {}}))
    assert check_dataset(tmp_path, policy, [0, 1])["status"] == "REJECT"
