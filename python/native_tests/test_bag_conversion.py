# SPDX-License-Identifier: MIT
"""Actual CDR bag -> actual LeRobot writer/loader; synthetic, never Gazebo."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from capture_fixture import profile_fixture, write_bag
from oa_embodied_ai.conversion.native import export_native
from oa_embodied_ai.conversion.validation import ConversionError
from oa_embodied_ai.recording.bag import align, read_bag
from oa_embodied_ai.recording.profile import validate_profile


def test_native_bag_to_lerobot_loader(tmp_path):
    profile, template = profile_fixture()
    validate_profile(profile, development=True)
    bag = write_bag(tmp_path / "bag", profile)
    frames = align(read_bag(bag, profile), profile)
    result = export_native(
        frames, profile, template, tmp_path / "dataset", source_bag=bag
    )
    assert result["native_loader"] == {
        "version": "0.6.1",
        "frames_loaded": 4,
        "status": "PASS",
    }
    assert result["origin"] == "synthetic"
    import json

    from c10_fixture import synthetic_policy
    from oa_embodied_ai.dataset_qa.episode import check_dataset

    dataset = tmp_path / "dataset"
    sidecar = json.loads(
        (dataset / "meta/openamrobot/episodes/episode-000000.json").read_text()
    )
    policy = tmp_path / "synthetic-policy.json"
    policy.write_text(json.dumps(synthetic_policy(sidecar)))
    decision = check_dataset(dataset, policy)
    assert not [f for f in decision["findings"] if f["status"] == "REJECT"]
    assert decision["status"] == "NOT_VERIFIED"
    assert decision["exit_code"] == 2
    assert decision["training_ready"] is False
    assert (
        result["training_readiness"] == result["physical_acceptance"] == "NOT_VERIFIED"
    )


def test_missing_camera_bag(tmp_path):
    profile, _ = profile_fixture()
    bag = write_bag(tmp_path / "bag", profile, omit="head")
    with pytest.raises(
        ConversionError, match="camera.*head.*missing or wrong message type"
    ):
        read_bag(bag, profile)


def test_clock_reset_bag(tmp_path):
    profile, _ = profile_fixture()
    bag = write_bag(tmp_path / "bag", profile, corrupt_clock=True)
    with pytest.raises(ConversionError, match="Nonmonotonic or duplicate timestamps"):
        read_bag(bag, profile)
