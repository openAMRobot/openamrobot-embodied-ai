# SPDX-License-Identifier: MIT
"""Proposed reference_integrity on recorded messages: a 5 s slice of Gazebo run 44448676604.

The fixture (python/tests/fixtures/baseline1_slice, provenance next to it) holds the recorded
messages of the Baseline 1 profile's topics and their camera_info, bytes unchanged. Negative
cases rewrite a copy of it (one topic, one change) or change one declaration; each is refused
with a named code. The positive cases convert with the native LeRobot writer and verify.
Simulation evidence only.
"""

import copy
import hashlib
import json
import math
import shutil
from pathlib import Path

import pytest
from oa_embodied_ai.conversion import reference
from oa_embodied_ai.conversion.batch import prepare_episode
from oa_embodied_ai.conversion.reference import IntegrityError
from oa_embodied_ai.conversion.validation import ConversionError
from oa_embodied_ai.recording import profile as profiles

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "python/tests/fixtures/baseline1_slice"
PROFILE = ROOT / "configs/capture_profiles/proposed/openamrobot-baseline1-gazebo-sim-oracle-v1.json"


def baseline1_slice():
    profile = json.loads(PROFILE.read_text())
    profile["episode_id"] = "baseline1-run-44448676604-slice-37.5s-42.5s"
    profile["dataset_revision"] = "run-44448676604-slice-37.5s-42.5s"
    return profile


@pytest.fixture
def bind(tmp_path, monkeypatch):
    """Bind a (changed) profile as its own Proposed entry, with a component contract made from its
    channels, so a declaration under test is reached; returns the enabled rule names."""

    def _bind(profile):
        from oa_embodied_ai.training.inputs import file_sha256

        contract = {
            "component_profile_version": 1,
            "c9_contract_version": 1,
            "status": "proposed",
            "channels": [{k: c.get(k) for k in ("name", "component", "unit", "range")} for c in profile["state_channels"]],
        }
        text = json.dumps(contract)
        # one path per content: two conversions of one profile bind the same file
        path = tmp_path / f"contract-{hashlib.sha256(text.encode()).hexdigest()[:16]}.json"
        path.write_text(text)
        profile["component_contract"] = {"path": str(path), "sha256": file_sha256(path)}
        entry = dict(profiles.PROPOSED_CAPTURE_PROFILES[profile["profile_id"]],
                     sha256=profiles.capture_contract_digest(profile))
        monkeypatch.setitem(profiles.PROPOSED_CAPTURE_PROFILES, profile["profile_id"], entry)
        return profile["proposed_rules"] + ["proposed_capture_profile"]

    return _bind


def template():
    from oa_embodied_ai.dataset_qa.contract import c9_asset
    from oa_embodied_ai.dataset_qa.strict_io import load_data

    return load_data(c9_asset("schemas/yaml/c9-lerobot-dataset-valid.yaml"))


def prepare(bag, profile, enabled):
    return prepare_episode(bag, profile, template(), enabled=enabled)


def rewrite(tmp_path, topic, edit, name="edited"):
    """Copy the fixture, changing the messages of one topic: edit(msg, receipt) returns the
    message to write, or None to leave it out. Other topics keep their recorded bytes."""
    from rosbags.highlevel import AnyReader
    from rosbags.rosbag2 import StoragePlugin, Writer

    out = tmp_path / name
    with AnyReader([FIXTURE]) as reader, Writer(out, version=9, storage_plugin=StoragePlugin.MCAP) as writer:
        conns = {}
        for c in sorted(reader.connections, key=lambda c: c.topic):
            conns[c.id] = writer.add_connection(c.topic, c.msgtype, msgdef=c.msgdef.data, rihs01=c.digest,
                                                serialization_format=c.ext.serialization_format,
                                                offered_qos_profiles=c.ext.offered_qos_profiles)
        for c, receipt, raw in reader.messages():
            if c.topic == topic:
                msg = edit(reader.deserialize(raw, c.msgtype), receipt)
                if msg is None:
                    continue
                raw = reader.typestore.serialize_cdr(msg, c.msgtype)
            writer.write(conns[c.id], receipt, raw)
    return out


def scaled(names, factor):
    def edit(msg, _):
        msg.position = msg.position.copy()  # deserialized arrays are read-only views
        for i, name in enumerate(msg.name):
            if name in names:
                msg.position[i] *= factor
        return msg

    return edit


# Positive on the recorded slice (no writer) ------------------------------------------------------------------------


def test_the_recorded_slice_passes_every_declared_check(bind):
    profile = baseline1_slice()
    episode = prepare(FIXTURE, profile, bind(profile))
    checks = episode.reference_report["checks"]
    assert len(episode.frames) == 23
    assert {k: v["status"] for k, v in checks.items()} == dict.fromkeys(checks, "PASS")
    streams = checks["timestamps"]["streams"]
    # recorded: 10 equal header stamps on /arm/joint_states and 14 equal receipt times on /cmd_vel
    assert streams["arm_joint_states"] == {"topic": "/arm/joint_states", "bag_metadata_messages": 2415,
                                           "read": 2415, "kept": 2405, "equal_timestamp_dropped": 10}
    assert streams["cmd_vel"]["kept"] + streams["cmd_vel"]["equal_timestamp_dropped"] == 31
    assert checks["timestamps"]["grid"]["pre_origin_frames_dropped"] == 1
    assert checks["actions"]["action_changes"] >= 1
    assert checks["cameras"]["max_skew_ns"] <= profile["integrity"]["camera_sync_max_skew_ns"]
    calibration = episode.reference_report["calibration"]
    assert {v["status"] for v in calibration.values()} == {"recorded"}
    assert calibration["base_camera_rgbd_image"]["fx"] == pytest.approx(554.3827, abs=1e-4)


def test_the_sidecar_carries_recorded_hashes_not_template_values(bind):
    from oa_embodied_ai.conversion.native import sidecar_for

    profile = baseline1_slice()
    episode = prepare(FIXTURE, profile, bind(profile))
    sidecar = sidecar_for(episode.profile, episode.frames, episode.template, reference=episode.reference_report)
    text = json.dumps(sidecar)
    assert reference.ZERO_HASH not in text
    streams = sidecar["openamrobot"]["streams"]
    head = episode.reference_report["calibration"]["head_camera_rgbd_image"]["sha256"]
    assert streams["head"]["configuration_hash"] == "sha256:" + head
    assert sidecar["openamrobot"]["provenance"]["robot_configuration_hash"] == (
        "sha256:" + profile["integrity"]["robot_model"]["sha256"])


# Negative: one change each, a named refusal ------------------------------------------------------------------------


def test_duplicate_stamps_without_keep_last_are_refused(bind):
    profile = baseline1_slice()
    stream = next(s for s in profile["streams"] if s["id"] == "arm_joint_states")
    del stream["equal_timestamp_rule"]
    with pytest.raises(ConversionError, match="duplicate timestamps.*'arm_joint_states'"):
        prepare(FIXTURE, profile, bind(profile))


def test_a_channel_declared_in_degrees_is_refused(bind):
    profile = baseline1_slice()
    for kind in ("state", "action"):
        next(c for c in profile[kind + "_channels"] if c["name"] == "openarm_left_joint1")["unit"] = "degree"
    with pytest.raises(IntegrityError, match="UNITS_DEGREES: .*openarm_left_joint1.*degree"):
        prepare(FIXTURE, profile, bind(profile))


def test_degree_values_in_the_canonical_data_are_refused(bind, tmp_path):
    arms = {f"openarm_{s}_joint{i}" for s in ("left", "right") for i in range(1, 8)}
    bag = rewrite(tmp_path, "/arm/joint_states", scaled(arms, 180 / math.pi))
    profile = baseline1_slice()
    with pytest.raises(IntegrityError, match="UNITS_RANGE: state 'openarm_.*fits the range in degrees"):
        prepare(bag, profile, bind(profile))


def test_a_one_frame_action_shift_is_detected(bind):
    from oa_embodied_ai.recording.bag import read_bag

    profile = baseline1_slice()
    episode = prepare(FIXTURE, profile, bind(profile))
    data = read_bag(FIXTURE, episode.profile)
    frames = copy.deepcopy(episode.frames)
    actions = [f["action"] for f in frames]
    for frame, action in zip(frames[1:], actions[:-1]):
        frame["action"] = action
    with pytest.raises(IntegrityError, match=r"ACTION_CONVENTION: action is shifted by \+1 frame"):
        reference.check_actions(frames, data, episode.profile)


def test_missing_camera_frames_are_refused(bind, tmp_path):
    def drop(msg, receipt):
        return None if 39_000_000_000 <= receipt < 39_600_000_000 else msg

    bag = rewrite(tmp_path, "/head_camera/rgbd/image", drop)
    profile = baseline1_slice()
    with pytest.raises(IntegrityError, match="CAMERA_MISSING: .*'head_camera_rgbd_image'"):
        prepare(bag, profile, bind(profile))


def test_camera_skew_beyond_the_declared_limit_is_refused(bind):
    profile = baseline1_slice()
    profile["integrity"]["camera_sync_max_skew_ns"] = 50_000_000  # recorded maximum 67 ms
    with pytest.raises(IntegrityError, match="CAMERA_SYNC: frame .* declared limit 50000000 ns"):
        prepare(FIXTURE, profile, bind(profile))


def test_a_lift_declared_in_millimetres_is_refused(bind):
    profile = baseline1_slice()
    for kind in ("state", "action"):
        lift = next(c for c in profile[kind + "_channels"] if c["name"] == "lift_joint")
        lift["unit"], lift["range"] = "mm", [0.0, 350.0]
        if kind == "state":
            lift["source_unit"] = "mm"
        else:
            lift["stop_targets"].update(unit="mm", values=[0.0, 175.0, 350.0])
    with pytest.raises(IntegrityError, match="LIFT_UNIT: state channel 'lift_joint' is mm"):
        prepare(FIXTURE, profile, bind(profile))


def test_a_lift_stop_index_read_as_metres_is_refused(bind):
    # the recorded lift stays at 0 m in this slice, so lift positions scaled to mm stay 0 and
    # cannot be told apart here; the recorded lift command (Int8 stop index 1 at 40.524 s) can
    profile = baseline1_slice()
    lift = next(c for c in profile["action_channels"] if c["name"] == "lift_joint")
    lift["source_unit"] = "m"
    del lift["stop_targets"]
    with pytest.raises(IntegrityError, match="UNITS_RANGE: action 'lift_joint' = 1.0 m .*a stop index read as metres"):
        prepare(FIXTURE, profile, bind(profile))


def test_a_right_finger_with_the_left_finger_sign_is_refused(bind, tmp_path):
    bag = rewrite(tmp_path, "/build1/capture/gripper_command", scaled({"openarm_right_finger_joint1"}, -1))
    profile = baseline1_slice()
    with pytest.raises(IntegrityError, match="GRIPPER_SIGN: action 'openarm_right_finger_joint1'"):
        prepare(bag, profile, bind(profile))


def test_an_opening_sign_that_contradicts_the_range_is_refused(bind):
    profile = baseline1_slice()
    profile["integrity"]["gripper"]["channels"]["openarm_right_finger_joint1"]["opening_sign"] = 1
    with pytest.raises(IntegrityError, match="GRIPPER_SIGN: state channel 'openarm_right_finger_joint1' range"):
        prepare(FIXTURE, profile, bind(profile))


def test_counts_that_do_not_reconcile_with_the_bag_metadata_are_refused(bind, tmp_path):
    bag = tmp_path / "counts"
    shutil.copytree(FIXTURE, bag)
    meta = (bag / "metadata.yaml").read_text()
    assert "message_count: 2415" in meta
    (bag / "metadata.yaml").write_text(meta.replace("message_count: 2415", "message_count: 2416", 1))
    profile = baseline1_slice()
    with pytest.raises(IntegrityError, match="TIMESTAMP_RECONCILE: stream '.*': bag metadata 2416, read 2415"):
        prepare(bag, profile, bind(profile))


def test_camera_info_of_another_camera_is_refused(bind):
    profile = baseline1_slice()
    profile["integrity"]["calibration_sources"]["left_wrist_camera_image"] = "/head_camera/rgbd/camera_info"
    with pytest.raises(IntegrityError, match="CALIBRATION_MISMATCH: /head_camera/rgbd/camera_info: 640x360"):
        prepare(FIXTURE, profile, bind(profile))


def test_the_sidecar_does_not_depend_on_the_key_order_of_the_profile(bind):
    from oa_embodied_ai.conversion.native import sidecar_for

    profile = baseline1_slice()
    episode = prepare(FIXTURE, profile, bind(profile))
    first = sidecar_for(episode.profile, episode.frames, episode.template, reference=episode.reference_report)
    stored = json.loads(json.dumps(episode.profile, sort_keys=True))  # as conversion.json keeps it
    again = sidecar_for(stored, episode.frames, episode.template, reference=episode.reference_report)
    assert list(stored["source_commits"]) != list(profile["source_commits"])
    assert json.dumps(again, sort_keys=True) == json.dumps(first, sort_keys=True)


def test_a_template_hash_never_remains_in_the_sidecar(bind):
    from oa_embodied_ai.conversion.native import sidecar_for

    profile = baseline1_slice()
    profile["integrity"]["robot_model"] = {"sha256": None, "reason": "test: model hash not recorded"}
    episode = prepare(FIXTURE, profile, bind(profile))
    assert {"item": "robot_model_sha256", "reason": "test: model hash not recorded"} in episode.reference_report["not_available"]
    with pytest.raises(IntegrityError, match="TEMPLATE_HASH"):
        sidecar_for(episode.profile, episode.frames, episode.template, reference=episode.reference_report)


# Native LeRobot writer and verify ----------------------------------------------------------------------------------


def convert(bind, tmp_path, name):
    from oa_embodied_ai.conversion.native import export_native_episodes

    profile = baseline1_slice()
    episode = prepare(FIXTURE, profile, bind(profile))
    out = tmp_path / name
    export_native_episodes([episode], out)
    return out


def rebind(dataset):
    """A consistent dataset after a change: output hashes recomputed (an attacker or a writer bug)."""
    path = dataset / reference.OUTPUT_MANIFEST
    manifest = json.loads(path.read_text())
    manifest["reference"]["output_sha256"] = reference.output_hashes(dataset)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def test_a_converted_slice_verifies_and_converts_identically(bind, tmp_path):
    a = convert(bind, tmp_path, "a")
    b = convert(bind, tmp_path, "b")
    manifest = json.loads((a / reference.OUTPUT_MANIFEST).read_text())
    assert manifest["reference"]["output_sha256"] == reference.output_hashes(a) == reference.output_hashes(b)
    assert (a / reference.OUTPUT_MANIFEST).read_bytes() == (b / reference.OUTPUT_MANIFEST).read_bytes()
    result = reference.verify(a, FIXTURE)
    assert result["status"] == "PASS"
    assert result["checks"]["vectors_and_timestamps"] == {"frames": 23, "status": "PASS"}
    assert result["checks"]["robot_model"]["status"] == "NOT_RECOMPUTED"


def test_a_tampered_output_file_is_refused(bind, tmp_path):
    out = convert(bind, tmp_path, "tampered")
    video = sorted((out / "videos").rglob("*.mp4"))[0]
    data = bytearray(video.read_bytes())
    data[len(data) // 2] ^= 0xFF
    video.write_bytes(bytes(data))
    with pytest.raises(IntegrityError, match="TAMPERED_OUTPUT: .*" + video.relative_to(out).as_posix()):
        reference.verify(out, FIXTURE)


def test_shifted_actions_in_a_rehashed_dataset_are_refused(bind, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = convert(bind, tmp_path, "shifted")
    path = sorted((out / "data").rglob("*.parquet"))[0]
    table = pq.read_table(path)
    actions = table.column("action").to_pylist()
    shifted = pa.array([actions[0]] + actions[:-1], type=table.schema.field("action").type)
    pq.write_table(table.set_column(table.schema.get_field_index("action"), "action", shifted), path)
    rebind(out)
    with pytest.raises(IntegrityError, match=r"ACTION_SHIFTED: action is shifted by \+1 frame"):
        reference.verify(out, FIXTURE)


def test_a_manifest_count_that_differs_from_the_bag_is_refused(bind, tmp_path):
    out = convert(bind, tmp_path, "manifest")
    path = out / reference.OUTPUT_MANIFEST
    manifest = json.loads(path.read_text())
    manifest["episodes"][0]["stream_timestamps"]["cmd_vel"]["equal_timestamp_dropped"] -= 1
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(IntegrityError, match="MANIFEST_MISMATCH: episode stream_timestamps"):
        reference.verify(out, FIXTURE)


def test_another_bag_is_refused(bind, tmp_path):
    out = convert(bind, tmp_path, "source")
    other = tmp_path / "other"
    shutil.copytree(FIXTURE, other)
    (other / "metadata.yaml").write_text((other / "metadata.yaml").read_text() + "\n")
    with pytest.raises(IntegrityError, match="SOURCE_MISMATCH"):
        reference.verify(out, other)
