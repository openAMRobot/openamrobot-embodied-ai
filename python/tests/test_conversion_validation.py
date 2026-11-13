# SPDX-License-Identifier: MIT
"""Negative pre-export checks; fixtures provide no robot/simulation evidence."""

import math

import pytest
from capture_fixture import profile_fixture
from oa_embodied_ai.conversion.validation import (
    ConversionError,
    normalize_vector,
    require_components,
    validate_lift_phase,
)
from oa_embodied_ai.recording.profile import validate_profile


def channels(n):
    items = [
        {
            "name": f"joint{i}",
            "component": "left_arm"
            if i < 7
            else "left_gripper"
            if i == 7
            else "right_arm"
            if i < 15
            else "right_gripper",
            "unit": "rad",
            "source_unit": "rad",
        }
        for i in range(16)
    ]
    if n == 17:
        items.append(
            {
                "name": "lift_joint",
                "component": "lift",
                "unit": "m",
                "source_unit": "mm",
                "range": [0, 0.35],
            }
        )
    return items


def normalize(values, cs, **kw):
    meta = {
        "representation": "qpos",
        "names": [c["name"] for c in cs],
        "units": [c["source_unit"] for c in cs],
    }
    meta.update(kw.pop("metadata", {}))
    return normalize_vector(
        values,
        meta,
        cs,
        profile="explicit-test",
        components=kw.pop(
            "components", list(dict.fromkeys(c["component"] for c in cs))
        ),
        **kw,
    )


@pytest.mark.parametrize("n", [16, 17])
def test_explicit_qpos_profiles(n):
    values = [0.0] * n
    if n == 17:
        values[-1] = 350
    result = normalize(values, channels(n))
    assert len(result) == n
    assert result[-1] == (0.35 if n == 17 else 0)


def test_missing_lift_regression():
    cs = channels(16)
    with pytest.raises(
        ConversionError, match="Missing required component 'lift'.*explicit-test"
    ):
        normalize(
            [0.0] * 16,
            cs,
            components=[
                "left_arm",
                "left_gripper",
                "right_arm",
                "right_gripper",
                "lift",
            ],
        )


@pytest.mark.parametrize(
    "component", ["left_arm", "right_arm", "left_gripper", "right_gripper", "lift"]
)
def test_missing_component_names_requirement(component):
    with pytest.raises(
        ConversionError, match=f"Missing required component '{component}'"
    ):
        require_components([component], {}, "profile-test")


@pytest.mark.parametrize("bad", [15, 18, 19])
def test_invalid_dimensions(bad):
    with pytest.raises(ConversionError, match="Wrong vector dimensions"):
        normalize([0.0] * bad, channels(16))


@pytest.mark.parametrize(
    "metadata, message",
    [
        ({"representation": "pose"}, "explicit qpos"),
        ({"names": ["wrong"] * 16}, "channel ordering"),
        ({"units": ["deg"] * 16}, "Unit mismatch"),
        ({"units": ["m"] * 16}, "Unit mismatch"),
        ({"units": []}, "one explicit unit"),
    ],
)
def test_bad_source_metadata(metadata, message):
    with pytest.raises(ConversionError, match=message):
        normalize([0.0] * 16, channels(16), metadata=metadata)


def test_degrees_radians_explicit():
    cs = channels(16)
    cs[0]["source_unit"] = "deg"
    assert normalize([180.0] + [0.0] * 15, cs)[0] == math.pi


@pytest.mark.parametrize("value", [-1, 351, 350000])
def test_lift_limits_after_mm_conversion(value):
    with pytest.raises(ConversionError, match="Component 'lift'.*outside"):
        normalize([0.0] * 16 + [value], channels(17))


def test_duplicate_components():
    with pytest.raises(ConversionError, match="Duplicate component"):
        require_components(["lift", "lift"], {"lift": True}, "test")


def test_unsupported_component():
    with pytest.raises(ConversionError, match="Unsupported component"):
        require_components(["torso"], {}, "test")


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("capture_version", 2, "Incompatible capture schema"),
        ("episode_id", "", "Missing episode_id"),
        ("origin", "simulation", "Unapproved profile"),
        ("state_names", ["wrong"], "state channel ordering"),
    ],
)
def test_invalid_capture_profile(field, value, message):
    p, _ = profile_fixture()
    p[field] = value
    with pytest.raises(ConversionError, match=message):
        validate_profile(p, development=True)


def test_missing_camera_and_calibration():
    p, _ = profile_fixture()
    p["streams"].pop()
    with pytest.raises(ConversionError, match="Missing required camera"):
        validate_profile(p, development=True)
    p, _ = profile_fixture()
    p["streams"][-1]["calibration_id"] = None
    with pytest.raises(ConversionError, match="Missing calibration ID"):
        validate_profile(p, development=True)


def test_invalid_clock_mapping():
    p, _ = profile_fixture()
    p["streams"][-1]["clock_domain"] = "device-unknown"
    with pytest.raises(ConversionError, match="Invalid clock mapping"):
        validate_profile(p, development=True)


def test_nav2_goal_is_not_base_action():
    p, _ = profile_fixture()
    p["action_channels"][0].update(component="base", semantics="nav2_goal")
    with pytest.raises(
        ConversionError, match="Base action source must be requested_velocity"
    ):
        validate_profile(p, development=True)


@pytest.mark.parametrize(
    "sample,message",
    [
        ({"lift_moving": True}, "Missing lift phase"),
        (
            {
                "lift_moving": True,
                "phase": "lift_positioning",
                "base_stopped": False,
                "arms_stowed": True,
            },
            "stopped-base",
        ),
        (
            {"lift_moving": True, "phase": "lift_positioning", "base_stopped": True},
            "stowed-arms",
        ),
    ],
)
def test_lift_phase_negative(sample, message):
    with pytest.raises(ConversionError, match=message):
        validate_lift_phase([sample], stowed_pose_review="synthetic-test")


def test_placeholder_stowed_pose_rejected():
    with pytest.raises(ConversionError, match="reviewed stowed-pose"):
        validate_lift_phase([{"lift_moving": True}], stowed_pose_review=None)


def test_claimed_acceptance_cannot_register_actual_capture():
    p, _ = profile_fixture()
    p.update(origin="robot", status="accepted", approval_ref="invented")
    with pytest.raises(ConversionError, match="no owner-approved actual capture"):
        validate_profile(p)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("streams", ["malformed"], "Malformed stream"),
        ("state_channels", ["malformed"], "Malformed state"),
        ("source_commits", [], "provenance"),
    ],
)
def test_malformed_nested_profile_rejects_cleanly(field, value, message):
    p, _ = profile_fixture()
    p[field] = value
    with pytest.raises(ConversionError, match=message):
        validate_profile(p, development=True)


def test_phase_check_observes_original_measured_lift_motion():
    from types import SimpleNamespace

    from oa_embodied_ai.recording.bag import Sample, validate_recording_phases

    p, _ = profile_fixture()
    p["state_channels"].append(
        {
            "name": "lift_joint",
            "joint": "lift_joint",
            "component": "lift",
            "unit": "m",
            "stream": "state",
            "field": "position",
        }
    )
    samples = [
        Sample(t, t, SimpleNamespace(name=["lift_joint"], position=[v]))
        for t, v in [(0, 0.0), (1, 0.1)]
    ]
    with pytest.raises(ConversionError, match="reviewed stowed-pose"):
        validate_recording_phases({"state": samples}, p)
