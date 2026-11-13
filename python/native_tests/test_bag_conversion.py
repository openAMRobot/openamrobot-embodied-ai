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


def test_bag_missing_declared_lift_regression(tmp_path):
    # Actual serialized JointState messages omit lift_joint while the selected
    # recording profile declares it. This isolates pre-export presence checking;
    # it does not adopt a full-body profile or invoke an exporter.
    import copy

    profile, _ = profile_fixture()
    bag = write_bag(tmp_path / "missing-lift-bag", profile)
    declared = copy.deepcopy(profile)
    declared["components"].append("lift")
    declared["state_channels"].append(
        {
            "name": "lift_joint",
            "component": "lift",
            "joint": "lift_joint",
            "unit": "m",
            "source_unit": "m",
            "stream": "state",
            "field": "position",
        }
    )
    with pytest.raises(
        ConversionError,
        match="Missing required component 'lift'.*synthetic-c9-capture-v1",
    ):
        read_bag(bag, declared)


def test_multi_episode_native_conversion_and_visual_training(tmp_path):
    import json

    import torch
    from capture_fixture import (
        bind_converted_for_synthetic_training,
        collection_fixture,
    )
    from oa_embodied_ai.conversion.batch import convert_collection
    from oa_embodied_ai.dataset_qa.episode import check_dataset
    from oa_embodied_ai.evaluation.cli import evaluate_c11
    from oa_embodied_ai.training.cli import train_act
    from oa_embodied_ai.training.config import get_profile_config
    from oa_embodied_ai.training.gate import TrainingBlockedError, verify_training_gate
    from oa_embodied_ai.training.inputs import file_sha256

    collection = collection_fixture(tmp_path / "sources")
    dataset = tmp_path / "converted"
    result = convert_collection(collection, dataset, development=True)
    assert result["native_loader"]["frames_loaded"] == 8
    assert [e["episode_id"] for e in result["episodes"]] == [
        "synthetic-bag-episode-0",
        "synthetic-bag-episode-1",
    ]
    assert [e["episode_index"] for e in result["episodes"]] == [0, 1]
    sidecars = [
        json.loads(p.read_text())
        for p in sorted((dataset / "meta/openamrobot/episodes").glob("episode-*"))
    ]
    assert [s["native_lerobot"]["task"]["task_index"] for s in sidecars] == [0, 1]
    assert [s["openamrobot"]["samples"][0]["episode_index"] for s in sidecars] == [0, 1]
    policy, manifest = bind_converted_for_synthetic_training(dataset)
    raw = check_dataset(dataset, policy)
    rejects = [f for f in raw["findings"] if f["status"] == "REJECT"]
    assert not rejects, rejects
    assert (
        raw["status"] == "NOT_VERIFIED"
        and raw["exit_code"] == 2
        and not raw["training_ready"]
    )
    with pytest.raises(TrainingBlockedError, match="Synthetic testing requires"):
        verify_training_gate(dataset, input_manifest_path=manifest, policy_path=policy)
    torch.set_num_threads(1)
    cfg = get_profile_config("lerobot-v0.6.1-bi-openarm-position")
    cfg.chunk_size, cfg.d_model, cfg.n_heads = 3, 32, 4
    cfg.enc_layers = cfg.dec_layers = 1
    cfg.dim_feedforward, cfg.image_size, cfg.dropout = 64, (32, 32), 0.0
    summary = train_act(
        dataset,
        cfg,
        output_dir=tmp_path / "training",
        smoke_test=True,
        allow_synthetic_eval=True,
        input_manifest_path=manifest,
        policy_path=policy,
    )
    evaluation = evaluate_c11(
        summary["checkpoint_path"],
        dataset,
        mode="both",
        output_report=tmp_path / "evaluation.json",
        allow_synthetic_eval=True,
        input_manifest_path=manifest,
        policy_path=policy,
    )
    assert summary["checkpoint_sha256"] == file_sha256(summary["checkpoint_path"])
    assert (
        summary["acceptance_status"]
        == evaluation["acceptance_status"]
        == "NOT_VERIFIED_PHYSICAL"
    )
    assert not set(summary["train_episodes"]) & set(summary["val_episodes"])
    assert set(summary["train_episodes"] + summary["val_episodes"]) == {0, 1}
    assert evaluation["rollout_metrics"]["total_rollouts"] == 0


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("duplicate_id", "Duplicate episode identity"),
        ("duplicate_source", "Duplicate episode identity"),
        ("revision", "incompatible dataset_revision"),
        ("units", "incompatible capture semantics"),
        ("source_changed", "changed after capture validation"),
    ],
)
def test_batch_preflight_is_atomic(tmp_path, mutation, message):
    import json

    from capture_fixture import collection_fixture
    from oa_embodied_ai.conversion.batch import prepare_episode
    from oa_embodied_ai.conversion.native import export_native_episodes
    from oa_embodied_ai.dataset_qa.strict_io import load_data

    manifest = collection_fixture(tmp_path / "sources")
    entries = json.loads(manifest.read_text())["episodes"]
    episodes = [
        prepare_episode(
            manifest.parent / e["bag"],
            load_data(manifest.parent / e["profile"]),
            load_data(manifest.parent / e["sidecar_template"]),
            development=True,
        )
        for e in entries
    ]
    if mutation == "duplicate_id":
        episodes[1].profile["episode_id"] = episodes[0].profile["episode_id"]
    elif mutation == "duplicate_source":
        episodes[1].source_bag = episodes[0].source_bag
    elif mutation == "revision":
        episodes[1].profile["dataset_revision"] = "unexpected"
    elif mutation == "units":
        episodes[1].profile["action_channels"][0]["unit"] = "rad"
    else:
        (episodes[1].source_bag / "mutation.txt").write_text("changed source")
    output = tmp_path / "new-parent" / "output"
    with pytest.raises(ConversionError, match=message):
        export_native_episodes(episodes, output)
    assert not output.parent.exists()


def test_collection_cli_relative_paths_and_validate_only(tmp_path, monkeypatch, capsys):
    from capture_fixture import collection_fixture
    from oa_embodied_ai.conversion.cli import main

    manifest = collection_fixture(tmp_path / "sources")
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "converted"
    assert (
        main(
            [
                "--collection",
                str(manifest),
                "--output",
                str(output),
                "--synthetic-development",
                "--validate-only",
            ]
        )
        == 0
    )
    assert not output.exists()
    assert (
        main(
            [
                "--collection",
                str(manifest),
                "--output",
                str(output),
                "--synthetic-development",
            ]
        )
        == 0
    )
    assert (output / "meta/openamrobot/episodes/episode-000001.json").is_file()
    assert "NOT_VERIFIED" in capsys.readouterr().out


def test_failed_native_writer_leaves_sources_and_destination_untouched(
    tmp_path, monkeypatch
):
    from capture_fixture import collection_fixture
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from oa_embodied_ai.conversion.batch import convert_collection
    from oa_embodied_ai.training.inputs import inventory

    collection = collection_fixture(tmp_path / "sources")
    before = inventory(collection.parent)

    def fail(self, *args, **kwargs):
        raise RuntimeError("injected native writer failure")

    monkeypatch.setattr(LeRobotDataset, "save_episode", fail)
    with pytest.raises(RuntimeError, match="injected native writer failure"):
        convert_collection(collection, tmp_path / "dataset", development=True)
    assert not (tmp_path / "dataset").exists()
    assert not list(tmp_path.glob("oa-convert-*"))
    assert inventory(collection.parent) == before
