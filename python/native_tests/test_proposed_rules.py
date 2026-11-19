# SPDX-License-Identifier: MIT
"""Proposed conversion rules (19 Nov 2026), on messages recorded in Gazebo run 44448676604.

The fixture holds recorded messages copied from the bag of openamr-upperbody-sw run 44448676604 (Baseline 1, opt-in
AI capture, sim-oracle, simulation time): the first second of the four cameras (stamps and shape), /arm/joint_states
and the held gripper command; controller-state messages; the three lift commands with the lift state around them;
/cmd_vel around its two gaps; samples just outside the model limits. Where a test needs a stream the fixture does not
hold, the test says that the stream is synthetic. Simulation evidence only; every rule here is Proposed.
"""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from oa_embodied_ai.conversion.validation import ConversionError, normalize_vector
from oa_embodied_ai.recording import profile as profiles
from oa_embodied_ai.recording.bag import (
    Sample,
    align,
    extract,
    validate_recording_phases,
    validate_source_samples,
)
from oa_embodied_ai.recording.profile import (
    PROPOSED_CAPTURE_PROFILES,
    capture_contract_digest,
    validate_profile,
)

# Looked up per test, so that each test fails on its own when its rule is reverted.
check_declarations = lambda profile: profiles.check_declarations(profile)  # noqa: E731

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/tests"))
FIXTURE = ROOT / "python/tests/fixtures/baseline1_run44448676604_excerpts.json"
PROFILE = ROOT / PROPOSED_CAPTURE_PROFILES["openamrobot-baseline1-gazebo-sim-oracle-v1"]["path"]
CAMERAS = {
    "/left_wrist_camera/image": "observation.images.camera_wrist_left",
    "/right_wrist_camera/image": "observation.images.camera_wrist_right",
    "/head_camera/rgbd/image": "observation.images.head",
    "/base_camera/rgbd/image": "observation.images.base",
}


def recorded():
    return json.loads(FIXTURE.read_text())


def baseline1():
    return json.loads(PROFILE.read_text())


def header(row):
    sec, nsec = divmod(row["stamp_ns"], 1_000_000_000)
    return SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nsec), frame_id=row.get("frame_id", ""))


def sample(row, **fields):
    """A recorded row as the converter sees a deserialized message."""
    message = SimpleNamespace(**fields)
    if "stamp_ns" in row:
        message.header = header(row)
    time = row.get("stamp_ns", row["receipt_ns"])
    return Sample(time, row["receipt_ns"], message)


def joint_state(row):
    return sample(row, name=row["name"], position=row["position"])


def image(row):
    import numpy as np

    data = np.zeros(row["height"] * row["width"] * 3, dtype=np.uint8)
    return sample(row, height=row["height"], width=row["width"], step=row["width"] * 3,
                  encoding=row["encoding"], data=data)


def controller(row):
    point = lambda p: SimpleNamespace(positions=p["positions"], velocities=p["velocities"])
    return sample(row, joint_names=row["joint_names"], reference=point(row["reference"]),
                  feedback=point(row["feedback"]))


def twist(row):
    return sample(row, linear=SimpleNamespace(x=row["linear"][0], y=row["linear"][1], z=row["linear"][2]),
                  angular=SimpleNamespace(x=row["angular"][0], y=row["angular"][1], z=row["angular"][2]))


def subset_profile():
    """Grippers and the four cameras of the Proposed Baseline 1 profile, as recorded."""
    full = baseline1()
    profile = copy.deepcopy(full)
    profile["components"] = ["left_gripper", "right_gripper"]
    keep = lambda c: c["component"] in profile["components"]
    profile["state_channels"] = [c for c in full["state_channels"] if keep(c)]
    profile["action_channels"] = [c for c in full["action_channels"] if keep(c)]
    for c in profile["state_channels"] + profile["action_channels"]:
        c.pop("range_tolerance", None)
    used = {c["stream"] for c in profile["state_channels"] + profile["action_channels"]}
    profile["streams"] = [s for s in full["streams"] if s.get("camera_key") or s["id"] in used]
    profile["state_names"] = [c["name"] for c in profile["state_channels"]]
    profile["action_names"] = [c["name"] for c in profile["action_channels"]]
    for key in ("component_contract", "phase_evidence", "vector_segments", "episode_origin"):
        profile.pop(key, None)
    profile["proposed_rules"] = []
    return profile


def subset_data():
    data = recorded()
    ids = {s["topic"]: s["id"] for s in baseline1()["streams"]}
    result = {ids[t]: [image(r) for r in data[t]] for t in CAMERAS}
    result["arm_joint_states"] = [joint_state(r) for r in data["/arm/joint_states"]]
    result["build1_capture_gripper_command"] = [joint_state(r) for r in data["/build1/capture/gripper_command"]]
    return result


def template():
    from oa_embodied_ai.dataset_qa.contract import c9_asset
    from oa_embodied_ai.dataset_qa.strict_io import load_data

    return load_data(c9_asset("schemas/yaml/c9-lerobot-dataset-valid.yaml"))


# Rule gate -----------------------------------------------------------------------------------------------------------

def test_a_declared_proposed_rule_must_be_enabled_by_name():
    from capture_fixture import profile_fixture

    profile, _ = profile_fixture()
    profile["proposed_rules"] = ["episode_origin"]
    profile["episode_origin"] = "first_complete_grid"
    with pytest.raises(ConversionError, match="enable it explicitly with --enable-proposed episode_origin"):
        validate_profile(profile, development=True)
    assert validate_profile(profile, development=True, enabled=["episode_origin"]) is profile
    with pytest.raises(ConversionError, match="Unknown Proposed rule"):
        validate_profile(profile, development=True, enabled=["episode_origin", "anything"])
    profile["proposed_rules"] = []
    with pytest.raises(ConversionError, match="needs the Proposed rule 'episode_origin'"):
        validate_profile(profile, development=True, enabled=["episode_origin"])


def test_the_baseline1_profile_declares_only_listed_rules():
    profile = baseline1()
    check_declarations(profile)
    assert set(profile["proposed_rules"]) <= set(profiles.PROPOSED_RULES)


def test_a_proposed_profile_converts_only_when_enabled_and_bound(tmp_path, monkeypatch):
    from oa_embodied_ai.recording import profile as module
    from oa_embodied_ai.training.inputs import file_sha256

    capture = baseline1()
    contract = {"component_profile_version": 1, "c9_contract_version": 1, "status": "proposed",
                "channels": [{k: c.get(k) for k in ("name", "component", "unit", "range")} for c in capture["state_channels"]]}
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    capture["component_contract"] = {"path": str(path), "sha256": file_sha256(path)}
    entry = dict(PROPOSED_CAPTURE_PROFILES[capture["profile_id"]], sha256=capture_contract_digest(capture))
    monkeypatch.setitem(module.PROPOSED_CAPTURE_PROFILES, capture["profile_id"], entry)
    rules = capture["proposed_rules"]
    with pytest.raises(ConversionError, match="Unapproved profile"):
        validate_profile(capture, enabled=rules)
    assert validate_profile(capture, enabled=rules + ["proposed_capture_profile"]) is capture
    changed = copy.deepcopy(capture)
    changed["max_skew_ns"] += 1
    with pytest.raises(ConversionError, match="differs from its Proposed binding"):
        validate_profile(changed, enabled=rules + ["proposed_capture_profile"])


# 1. Episode origin ---------------------------------------------------------------------------------------------------

def test_without_the_origin_rule_the_sidecar_refuses_pre_origin_samples():
    from oa_embodied_ai.conversion.native import sidecar_for

    profile = subset_profile()
    frames = align(subset_data(), profile)
    assert frames[0]["time_ns"] == 6_400_000_000
    with pytest.raises(ConversionError, match="precedes episode origin"):
        sidecar_for(profile, frames, template())


def test_first_complete_grid_drops_and_counts_the_leading_pre_origin_frame():
    from oa_embodied_ai.conversion.native import sidecar_for

    profile = subset_profile()
    profile["proposed_rules"] = ["episode_origin"]
    profile["episode_origin"] = "first_complete_grid"
    report = {}
    frames = align(subset_data(), profile, report=report)
    origin = report["episode_origin"]
    # latest first sample: the left wrist camera at 6.4 s, on the 200 ms grid; the frame at 6.4 s selects the head and
    # base frames of 6.334 s and the gripper command of 6.375 s (recorded), so it is dropped and counted
    assert origin["origin_ns"] == origin["latest_first_sample_ns"] == 6_400_000_000
    assert origin["pre_origin_frames_dropped"] == 1
    assert origin["first_dropped_frame_offsets_ns"]["head_camera_rgbd_image"] == -66_000_000
    assert origin["first_dropped_frame_offsets_ns"]["build1_capture_gripper_command"] == -25_000_000
    assert [f["time_ns"] for f in frames] == [6_600_000_000, 6_800_000_000]
    sidecar = sidecar_for(profile, frames, template())
    capture = sidecar["openamrobot"]
    end = capture["recording"]["end_timestamp_ns"]
    for row in capture["samples"]:
        for item in row["streams"].values():
            assert 0 <= item["selected_timestamp_ns"] <= end
            assert 0 <= item["host_receipt"]["mapped_authoritative_ns"] <= end
    assert capture["samples"][0]["authoritative_timestamp_ns"] == 200_000_000


# 2. and 3. Arm commands from the controller reference; commands versus measured states ---------------------------

def test_arm_commands_are_read_from_the_recorded_controller_reference():
    data = recorded()
    channels = [c for c in baseline1()["action_channels"] if c["stream"].endswith("controller_state")]
    for topic, side in (("/left_arm_controller/controller_state", "left"),
                        ("/right_arm_controller/controller_state", "right")):
        for row in data[topic]:
            message = controller(row)
            for c in (c for c in channels if c["joint"].startswith(f"openarm_{side}_")):
                assert extract(message, c) == row["reference"]["positions"][row["joint_names"].index(c["joint"])]
                legacy = {k: v for k, v in c.items() if k != "names_field"}
                with pytest.raises(ConversionError, match="missing or duplicate joint"):
                    extract(message, legacy)


def test_command_semantics_keeps_commands_and_measured_states_apart():
    profile = baseline1()
    check_declarations(profile)
    wrong = copy.deepcopy(profile)
    wrong["action_channels"][0]["semantics"] = "requested_velocity"
    with pytest.raises(ConversionError, match="must declare what it commands"):
        check_declarations(wrong)
    wrong = copy.deepcopy(profile)
    wrong["state_channels"][0]["field"] = "reference.positions"
    with pytest.raises(ConversionError, match="must be a measured value"):
        check_declarations(wrong)
    wrong = copy.deepcopy(profile)
    wrong["action_channels"][0]["stream"] = "arm_joint_states"
    with pytest.raises(ConversionError, match="disjoint source streams"):
        check_declarations(wrong)


def test_the_sidecar_states_what_action_and_observation_are():
    from oa_embodied_ai.conversion.native import sidecar_for

    profile = subset_profile()
    profile["proposed_rules"] = ["episode_origin"]
    profile["episode_origin"] = "first_complete_grid"
    frames = align(subset_data(), profile)
    before = sidecar_for(profile, frames, template())
    assert before["openamrobot"]["action_semantics"] == template()["openamrobot"]["action_semantics"]
    profile["proposed_rules"].append("command_semantics")
    after = sidecar_for(profile, frames, template())
    semantics = after["openamrobot"]["action_semantics"]
    assert semantics["observation_relationship"].startswith("action[t] is the latest requested command issued at or before")
    meanings = [d["meaning"] for d in after["native_lerobot"]["features"]["action"]["dimensions"]]
    assert meanings == ["requested command (held_gripper_goal)"] * 2
    assert {d["meaning"] for d in after["native_lerobot"]["features"]["observation.state"]["dimensions"]} == {"measured state"}


# 4. Several source streams per vector --------------------------------------------------------------------------------

def test_a_declared_multi_source_vector_composes_its_sidecar_stream():
    from oa_embodied_ai.conversion.native import sidecar_for

    profile = subset_profile()
    profile["proposed_rules"] = ["episode_origin"]
    profile["episode_origin"] = "first_complete_grid"
    data = subset_data()
    # The right finger state is read from the recorded gripper command stream here: a second source for the state
    # vector, to exercise the composition (command_semantics, which forbids this, is not listed).
    right = next(c for c in profile["state_channels"] if c["component"] == "right_gripper")
    right["stream"] = "build1_capture_gripper_command"
    frames = align(data, profile)
    with pytest.raises(ConversionError, match="needs one acquisition source"):
        sidecar_for(profile, frames, template())
    profile["proposed_rules"].append("vector_segments")
    profile["vector_segments"] = {"state": [
        {"stream": "arm_joint_states", "names": [profile["state_names"][0]]},
        {"stream": "build1_capture_gripper_command", "names": [profile["state_names"][1]]}]}
    check_declarations(profile)
    sidecar = sidecar_for(profile, frames, template())
    origin = frames[0]["episode_origin_ns"]
    for frame, row in zip(frames, sidecar["openamrobot"]["samples"], strict=True):
        item = row["streams"]["robot_state"]
        parts = [frame["selected"][s] for s in ("arm_joint_states", "build1_capture_gripper_command")]
        assert item["source_frame_id"] == "|".join(
            f"{s}:{p.time_ns}" for s, p in zip(("arm_joint_states", "build1_capture_gripper_command"), parts, strict=True))
        assert item["selected_timestamp_ns"] == min(p.time_ns for p in parts) - origin
        assert item["host_receipt"]["raw"] == min(p.receipt_ns for p in parts)
    profile["vector_segments"]["state"].reverse()
    with pytest.raises(ConversionError, match="in vector order"):
        check_declarations(profile)


def test_a_sample_selected_again_keeps_its_sequence_id_and_is_not_fresh():
    from oa_embodied_ai.conversion.native import sidecar_for

    profile = subset_profile()
    profile["proposed_rules"] = ["episode_origin"]
    profile["episode_origin"] = "first_complete_grid"
    frames = align(subset_data(), profile)
    frames[1]["selected"]["arm_joint_states"] = frames[0]["selected"]["arm_joint_states"]
    rows = sidecar_for(profile, frames, template())["openamrobot"]["samples"]
    first, second = (r["streams"]["robot_state"] for r in rows)
    assert second["sequence_id"] == first["sequence_id"] == 0
    assert (first["health"], second["health"]) == ("fresh", "stale")


# 5. Lift command: stop index to metres -------------------------------------------------------------------------------

def test_recorded_lift_stop_indices_map_to_the_declared_targets():
    channel = next(c for c in baseline1()["action_channels"] if c["name"] == "lift_joint")
    commands = [r["data"] for r in recorded()["/sil/lift/command"]]
    assert commands == [1, 1, 0]
    meta = lambda: {"representation": "qpos", "names": ["lift_joint"], "units": ["stop_index"]}
    values = [normalize_vector([v], meta(), [channel], profile="p", components=["lift"])[0] for v in commands]
    assert values == [0.175, 0.175, 0.0]
    with pytest.raises(ConversionError, match="has no declared target"):
        normalize_vector([3], meta(), [channel], profile="p", components=["lift"])
    legacy = {k: v for k, v in channel.items() if k != "stop_targets"}
    with pytest.raises(ConversionError, match="Unsupported unit conversion"):
        normalize_vector([1], meta(), [legacy], profile="p", components=["lift"])


# 6. Tolerance at model limits ----------------------------------------------------------------------------------------

def test_recorded_samples_just_outside_the_limits_are_kept_and_counted():
    limits = recorded()["limits"]
    profile = baseline1()
    arm = [c for c in profile["state_channels"] if c["stream"] == "arm_joint_states"]
    lift = [c for c in profile["state_channels"] if c["name"] == "lift_joint"]
    cases = [("arm_joint_states", arm, limits["/arm/joint_states"]), ("joint_states", lift, limits["/joint_states"])]
    for stream, channels, rows in cases:
        sub = {"profile_id": "p", "streams": [], "state_channels": channels, "action_channels": []}
        data = {stream: [joint_state(r) for r in rows]}
        flags = {}
        validate_source_samples(data, sub, flags)
        assert flags and all(v == len(rows) or v <= len(rows) for v in flags.values())
        for name in flags:
            c = next(c for c in channels if c["name"] == name)
            for r in rows:
                value = r["position"][r["name"].index(c["joint"])]
                assert c["range"][0] - 0.001 <= value <= c["range"][1] + 0.001
        strict = copy.deepcopy(sub)
        for c in strict["state_channels"]:
            c.pop("range_tolerance")
        with pytest.raises(ConversionError, match="outside"):
            validate_source_samples(data, strict)
        tight = copy.deepcopy(sub)
        for c in tight["state_channels"]:
            c["range_tolerance"] = 0.0001
        with pytest.raises(ConversionError, match="outside"):
            validate_source_samples(data, tight)


def test_a_tolerated_value_is_not_clamped():
    row = recorded()["limits"]["/joint_states"][0]
    channel = next(c for c in baseline1()["state_channels"] if c["name"] == "lift_joint")
    measured = row["position"][row["name"].index("lift_joint")]
    flags = {}
    value = normalize_vector([measured], {"representation": "qpos", "names": ["lift_joint"], "units": ["m"]},
                             [channel], profile="p", components=["lift"], flags=flags)[0]
    assert value == measured < 0 and flags == {"lift_joint": 1}


# 7. Held commands ----------------------------------------------------------------------------------------------------

def synthetic_state(start, end, period=20_000_000):
    """SYNTHETIC measured stream (not in the fixture): a 50 Hz JointState with one joint at 0."""
    rows = [{"stamp_ns": t, "receipt_ns": t, "frame_id": "base_link", "name": ["j"], "position": [0.0]}
            for t in range(start, end + 1, period)]
    return [joint_state(r) for r in rows]


def base_profile(hold=None):
    stream = {"id": "cmd_vel", "component": "base", "topic": "/cmd_vel", "type": "geometry_msgs/msg/Twist",
              "frame_id": "base_link", "clock_domain": "ros.sim_time", "timestamp_source": "bag"}
    if hold:
        stream["hold"] = hold
    state = {"id": "state", "component": "base", "topic": "/s", "type": "sensor_msgs/msg/JointState",
             "frame_id": "base_link", "clock_domain": "ros.sim_time", "timestamp_source": "header"}
    return {"profile_id": "p", "fps": 5, "max_skew_ns": 100_000_000, "components": ["base"],
            "streams": [state, stream],
            "state_channels": [{"name": "j", "component": "base", "unit": "rad", "source_unit": "rad",
                                "field": "position", "stream": "state", "joint": "j"}],
            "action_channels": [{"name": n, "component": "base", "unit": u, "source_unit": u, "field": f,
                                 "stream": "cmd_vel", "semantics": "requested_velocity"}
                                for n, u, f in (("base_linear_x", "m/s", "linear.x"), ("base_angular_z", "rad/s", "angular.z"))],
            "state_names": ["j"], "action_names": ["base_linear_x", "base_angular_z"]}


def gap_data(gap):
    rows = gap["before"] + gap["after"]
    return {"state": synthetic_state(rows[0]["receipt_ns"], rows[-1]["receipt_ns"]),
            "cmd_vel": [twist(r) for r in rows]}


def test_a_zero_velocity_request_is_held_across_the_recorded_gap():
    gap = recorded()["/cmd_vel"]["gaps"][1]  # 37.82 s to 112.72 s: Nav2 published nothing, last request zero
    assert gap["after"][0]["receipt_ns"] - gap["before"][-1]["receipt_ns"] == 74_900_000_000
    with pytest.raises(ConversionError, match="Missing/stale frame for stream 'cmd_vel'"):
        align(gap_data(gap), base_profile())
    report = {}
    frames = align(gap_data(gap), base_profile({"rule": "while_zero"}), report=report)
    assert report["held_commands"]["cmd_vel"]["held_frames"] > 300
    assert all(f["action"] == [0.0, 0.0] for f in frames[:-1])
    moving = copy.deepcopy(gap)
    moving["before"][-1]["angular"][2] = -0.15
    for row in moving["before"]:
        if row["receipt_ns"] == moving["before"][-1]["receipt_ns"]:
            row["angular"][2] = -0.15
    with pytest.raises(ConversionError, match="not zero beyond max_skew_ns"):
        align(gap_data(moving), base_profile({"rule": "while_zero"}))


def lift_profile(hold):
    full = baseline1()
    command = next(s for s in full["streams"] if s["id"] == "sil_lift_command")
    state = next(s for s in full["streams"] if s["id"] == "joint_states")
    command = {k: v for k, v in command.items() if k != "hold"}
    if hold:
        command["hold"] = hold
    state_channel = next(c for c in full["state_channels"] if c["name"] == "lift_joint")
    action = next(c for c in full["action_channels"] if c["name"] == "lift_joint")
    return {"profile_id": "p", "fps": 5, "max_skew_ns": 100_000_000, "components": ["lift"],
            "streams": [state, command], "state_channels": [state_channel], "action_channels": [action],
            "state_names": ["lift_joint"], "action_names": ["lift_joint"]}


def lift_data(window=0):
    data = recorded()
    rows = data["/joint_states"][window]["rows"]
    commands = [r for r in data["/sil/lift/command"] if rows[0]["stamp_ns"] - 1 <= r["receipt_ns"] <= rows[-1]["stamp_ns"]]
    return {"joint_states": [joint_state(r) for r in rows],
            "sil_lift_command": [sample(r, data=r["data"]) for r in commands]}


def test_the_lift_target_is_held_and_stands_in_before_the_first_command():
    hold = baseline1()["streams"][[s["id"] for s in baseline1()["streams"]].index("sil_lift_command")]["hold"]
    data = lift_data(0)  # 40.34 s to 41.02 s: the first lift command (stop 1) at 40.524 s
    with pytest.raises(ConversionError, match="No overlapping nonempty episode interval"):
        align(data, lift_profile(None))  # one command message: without a hold it bounds the episode
    report = {}
    frames = align(lift_data(0), lift_profile(hold), report=report)
    held = report["held_commands"]["sil_lift_command"]
    assert held["stand_in_frames"] >= 1 and held["held_frames"] >= 1
    for frame in frames:
        expected = 0.175 if frame["time_ns"] >= 40_524_000_000 else 0.0
        assert frame["action"] == [expected]
        assert frame["state"] == [0.0]  # measured: the lift has not moved yet
    wrong = copy.deepcopy(hold)
    wrong["before_first"]["field_value"] = 1
    with pytest.raises(ConversionError, match="differs from the first measured"):
        align(lift_data(0), lift_profile(wrong))


# 8. Lift phase evidence ----------------------------------------------------------------------------------------------

def phase_profile(**evidence):
    profile = lift_profile(None)
    profile["state_channels"] = [dict(profile["state_channels"][0])]
    full = baseline1()
    arm = [c for c in full["state_channels"] if c["component"] in ("left_arm", "right_arm")]
    profile["state_channels"] += [dict(c, stream="joint_states") for c in arm]
    profile["streams"].append({"id": "odom", "component": "base", "topic": "/odom", "type": "nav_msgs/msg/Odometry",
                               "frame_id": "odom", "clock_domain": "ros.sim_time", "timestamp_source": "header"})
    phase = copy.deepcopy(full["phase_evidence"])
    phase.update(evidence)
    profile["phase_evidence"] = phase
    return profile


def phase_data(window):
    data = lift_data(window)
    rows = recorded()["/joint_states"][window]["rows"]
    # SYNTHETIC odometry with zero twist; run 44448676604 measured exactly zero twist on all 393 moving lift samples
    zero = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    data["odom"] = [sample({"stamp_ns": r["stamp_ns"], "receipt_ns": r["receipt_ns"]},
                           twist=SimpleNamespace(twist=SimpleNamespace(linear=zero, angular=zero))) for r in rows]
    data["sil_lift_command"] = [sample(r, data=r["data"]) for r in recorded()["/sil/lift/command"]]
    return data


def test_a_placeholder_stowed_pose_never_passes():
    profile = phase_profile()
    del profile["phase_evidence"]["on_missing_evidence"]
    with pytest.raises(ConversionError, match="reviewed stowed-pose evidence"):
        validate_recording_phases(phase_data(1), profile)


def test_lift_phase_evidence_is_reported_from_the_recorded_commands():
    report = {}
    validate_recording_phases(phase_data(1), phase_profile(), report=report)  # 42.45 s to 43.15 s, lift rising
    assert report["status"] == "NOT_VERIFIED" and report["mode"] == "report"
    assert report["stowed_pose_reviewed"] is False
    assert report["moving_samples"] == report["in_lift_positioning"] == report["base_stopped"] > 0
    assert report["arms_within_pose_tolerance"] == 0  # the arms are near joint4 0, not in park (joint4 0.1)
    assert [(i["start_ns"], i["target"]) for i in report["lift_positioning_intervals"]] == [
        (40_524_000_000, 0.175), (42_652_000_000, 0.175), (53_263_000_000, 0.0)]
    late = {}
    validate_recording_phases(phase_data(4), phase_profile(), report=late)  # 115.7 s to 116.0 s: 0.2 mm, no command
    assert late["in_lift_positioning"] == 0 and late["moving_outside_lift_positioning"]
