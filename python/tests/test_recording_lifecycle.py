# SPDX-License-Identifier: MIT
"""Capture subprocess lifecycle only; these mocks are not ROS execution."""

import json
import subprocess
from unittest.mock import patch

import pytest
from capture_fixture import profile_fixture
from oa_embodied_ai.conversion.validation import ConversionError
from oa_embodied_ai.recording.cli import record


class CaptureProcess:
    def __init__(self, command, **kwargs):
        self.command = command
        self.returncode = None
        self.waits = 0
        self.signals = []

    def wait(self, timeout):
        self.waits += 1
        if self.waits == 1:
            raise subprocess.TimeoutExpired(self.command, timeout)
        self.returncode = 0
        return 0

    def send_signal(self, value):
        self.signals.append(value)


def test_capture_stops_explicitly_and_requires_bag_validation(tmp_path):
    profile, _ = profile_fixture()
    with (
        patch(
            "oa_embodied_ai.recording.cli.shutil.which", return_value="/usr/bin/ros2"
        ),
        patch("oa_embodied_ai.recording.cli.check_graph"),
        patch("oa_embodied_ai.recording.cli.subprocess.Popen", CaptureProcess),
        patch("oa_embodied_ai.recording.bag.read_bag") as validator,
    ):
        result = record(profile, tmp_path / "episode", 0.1)
    validator.assert_called_once()
    assert result["state"] == "captured"
    assert result["bag_validation"] == "PASS"
    assert result["physical_acceptance"] == "NOT_VERIFIED"
    assert result["command"][:3] == ["ros2", "bag", "record"]
    assert result["episode_id"] == profile["episode_id"]


def test_capture_invalid_bag_retains_failure_manifest(tmp_path):
    profile, _ = profile_fixture()
    with (
        patch(
            "oa_embodied_ai.recording.cli.shutil.which", return_value="/usr/bin/ros2"
        ),
        patch("oa_embodied_ai.recording.cli.check_graph"),
        patch("oa_embodied_ai.recording.cli.subprocess.Popen", CaptureProcess),
        patch(
            "oa_embodied_ai.recording.bag.read_bag",
            side_effect=ConversionError("missing lift topic"),
        ),
    ):
        with pytest.raises(ConversionError, match="missing lift topic"):
            record(profile, tmp_path / "episode", 0.1)
    result = json.loads((tmp_path / "episode/recording.json").read_text())
    assert result["state"] == "capture_not_validated"
    assert "bag_validation" not in result


def test_missing_required_topic_stops_before_creating_recording(tmp_path):
    profile, _ = profile_fixture()
    with (
        patch(
            "oa_embodied_ai.recording.cli.shutil.which", return_value="/usr/bin/ros2"
        ),
        patch(
            "oa_embodied_ai.recording.cli.check_graph",
            side_effect=ConversionError("Required topics unavailable: head"),
        ),
    ):
        with pytest.raises(ConversionError, match="Required topics unavailable: head"):
            record(profile, tmp_path / "episode", 0.1)
    assert not (tmp_path / "episode").exists()
