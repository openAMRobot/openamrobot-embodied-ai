# SPDX-License-Identifier: MIT
"""Executed native media, model conditioning and CPU software-test regressions."""

import json
import subprocess
from unittest.mock import patch

import pytest
import torch
from oa_embodied_ai.dataset_qa.contract import c9_asset
from oa_embodied_ai.dataset_qa.strict_io import load_data
from oa_embodied_ai.training.cli import train_act
from oa_embodied_ai.training.config import (
    CANONICAL_CAMERA_KEYS,
    get_profile_config,
    load_config,
    save_config,
)
from oa_embodied_ai.training.dataset import (
    ImagePreprocessor,
    MultiCameraACTDataset,
    find_episode_parquet_files,
    partition_episodes,
)
from oa_embodied_ai.training.gate import TrainingBlockedError, verify_training_gate
from oa_embodied_ai.training.inputs import file_sha256
from oa_embodied_ai.training.model import ACTPolicy


def test_c9_profile_names_units_and_order_are_exact():
    example = load_data(c9_asset("schemas/yaml/c9-lerobot-dataset-valid.yaml"))
    cfg = get_profile_config("lerobot-v0.6.1-bi-openarm-position")
    for key in ("observation.state", "action"):
        dims = example["native_lerobot"]["features"][key]["dimensions"]
        assert cfg.channel_names == [d["name"] for d in dims]
        assert {d["unit"] for d in dims} == {"degree"}
    historical = get_profile_config("openamrobot-fixed-mast-19dof")
    assert len(historical.channel_names) == 19
    assert historical.channel_names[3:5] == ["torso_pitch", "torso_yaw"]
    assert historical.channel_names[5:12] == [
        "left_shoulder_pan",
        "left_shoulder_lift",
        "left_elbow_flex",
        "left_wrist_flex",
        "left_wrist_roll",
        "left_wrist_yaw",
        "left_gripper",
    ]
    assert all(
        "lift_joint" != name for name in cfg.channel_names + historical.channel_names
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("state_dim", 19),
        ("action_dim", 19),
        ("channel_names", ["alias"] * 16),
        ("camera_keys", CANONICAL_CAMERA_KEYS[::-1]),
        ("camera_keys", ["observation.images.camera_scene"] * 4),
        ("chunk_size", 0),
        ("eval_split", 0),
        ("vision_backbone", "unimplemented"),
    ],
)
def test_unsupported_profile_or_config_mix_is_rejected(tiny_config, field, value):
    setattr(tiny_config, field, value)
    with pytest.raises(ValueError):
        ACTPolicy(tiny_config)


def test_configuration_roundtrip_preserves_preprocessing(tiny_config, tmp_path):
    path = tmp_path / "config.yaml"
    save_config(tiny_config, path)
    loaded = load_config(path)
    assert loaded.to_dict() == tiny_config.to_dict()
    img = torch.ones(3, 64, 48)
    processed = ImagePreprocessor((32, 32))(img)
    assert processed.shape == (3, 32, 32)
    assert processed[0, 0, 0].item() == pytest.approx((1 - 0.485) / 0.229)


def test_shared_native_files_use_episode_rows_and_video_segments(
    synthetic_inputs, tiny_config
):
    root, _, _ = synthetic_inputs
    episodes = find_episode_parquet_files(root)
    assert [e.episode_index for e in episodes] == [0, 1]
    assert (
        episodes[0].path == episodes[1].path
    )  # v3 shared Parquet, no episode_* filename.
    dataset = MultiCameraACTDataset(root, episodes, tiny_config)
    assert len(dataset) == 4
    sample = dataset[2]  # First frame of episode 1, inside shared video.
    assert sample["images"].shape == (4, 3, 32, 32)
    assert sample["actions"].shape == (3, 16)
    assert sample["action_mask"].tolist() == [True, True, False]
    assert sample["actions"][2].eq(0).all()
    assert [
        dataset.samples[2]["video_map"][k][1][0] for k in tiny_config.camera_keys
    ] == [2] * 4
    assert not torch.equal(sample["images"][0], sample["images"][1])
    assert dataset[3]["action_mask"].tolist() == [True, False, False]
    assert dataset.samples[2]["episode_index"] == 1


def test_split_uses_actual_noncontiguous_episode_ids(synthetic_inputs):
    from dataclasses import replace

    root, _, _ = synthetic_inputs
    episodes = find_episode_parquet_files(root)
    episodes = [
        replace(episodes[0], episode_index=4),
        replace(episodes[1], episode_index=19),
    ]
    _, _, train, val = partition_episodes(episodes, seed=42)
    assert set(train + val) == {4, 19} and not set(train) & set(val)
    assert partition_episodes(episodes, seed=42)[2:] == (train, val)
    with pytest.raises(ValueError):
        partition_episodes(episodes[:1])


def test_missing_camera_cannot_be_replaced_by_zero_images(
    synthetic_inputs, tiny_config
):
    root, _, _ = synthetic_inputs
    video = next((root / "videos" / CANONICAL_CAMERA_KEYS[0]).rglob("*.mp4"))
    video.unlink()
    with pytest.raises(subprocess.CalledProcessError):
        MultiCameraACTDataset(root, find_episode_parquet_files(root), tiny_config)


def test_missing_or_corrupt_decode_cannot_be_hidden(synthetic_inputs, tiny_config):
    root, _, _ = synthetic_inputs
    dataset = MultiCameraACTDataset(root, find_episode_parquet_files(root), tiny_config)
    with patch(
        "oa_embodied_ai.training.dataset.extract_video_frame",
        side_effect=ValueError("decoder failed"),
    ):
        with pytest.raises(ValueError, match="decoder failed"):
            dataset[0]
    with pytest.raises(ValueError, match="requires image"):
        MultiCameraACTDataset(
            root, find_episode_parquet_files(root), tiny_config, load_images=False
        )


def test_every_camera_affects_prediction_and_receives_gradient(tiny_config):
    torch.manual_seed(42)
    model = ACTPolicy(tiny_config).eval()
    images = torch.randn(1, 4, 3, 32, 32, requires_grad=True)
    state = torch.zeros(1, 16)
    prediction, mu, logvar = model(state, images)
    assert prediction.shape == (1, 3, 16) and mu is logvar is None
    prediction.square().sum().backward()
    assert all(images.grad[0, i].abs().sum() > 0 for i in range(4))
    with torch.no_grad():
        for camera in range(4):
            altered = images.detach().clone()
            altered[:, camera] = 0
            assert not torch.allclose(model(state, altered)[0], prediction)
    with pytest.raises(ValueError, match="no state-only fallback"):
        model(state)


def test_cvae_masks_padded_actions_and_loss(tiny_config):
    model = ACTPolicy(tiny_config).eval()
    state, actions = torch.zeros(1, 16), torch.randn(1, 3, 16)
    mask = torch.tensor([[True, False, False]])
    first = model.cvae_encoder(state, actions, mask)
    actions[:, 1:] += 1000
    second = model.cvae_encoder(state, actions, mask)
    assert all(torch.allclose(a, b) for a, b in zip(first, second, strict=True))
    prediction = actions.clone()
    prediction[:, 1:] -= 9000
    loss = model.compute_loss(prediction, actions, None, None, mask)
    assert loss["loss"].item() == 0


@pytest.mark.parametrize(
    "profile", ["lerobot-v0.6.1-bi-openarm-position", "openamrobot-fixed-mast-19dof"]
)
def test_both_profiles_forward_without_dimension_adaptation(profile):
    cfg = get_profile_config(profile)
    cfg.d_model, cfg.n_heads, cfg.chunk_size = 32, 4, 2
    cfg.enc_layers = cfg.dec_layers = 1
    model = ACTPolicy(cfg).eval()
    with torch.no_grad():
        output, _, _ = model(
            torch.zeros(1, cfg.state_dim), torch.randn(1, 4, 3, 32, 32)
        )
    assert output.shape == (1, 2, cfg.action_dim)
    with pytest.raises(ValueError):
        model(torch.zeros(1, 17), torch.randn(1, 4, 3, 32, 32))


def test_synthetic_override_preserves_raw_c10_block(synthetic_inputs):
    root, policy, manifest = synthetic_inputs
    with pytest.raises(TrainingBlockedError):
        verify_training_gate(root, input_manifest_path=manifest, policy_path=policy)
    status = verify_training_gate(
        root,
        input_manifest_path=manifest,
        policy_path=policy,
        allow_synthetic_override=True,
    )
    assert status.is_synthetic_override and not status.training_ready
    assert status.quality_status == status.qa_report["status"] == "NOT_VERIFIED"
    assert status.qa_report["exit_code"] == 2
    assert status.qa_report["production_acceptance"] == "NOT_VERIFIED"


@pytest.mark.parametrize(
    "fault", ["native_data", "camera", "policy", "native_dependencies", "forged_robot"]
)
def test_real_native_failures_block_before_model_even_with_override(
    synthetic_inputs, tiny_config, rebind, tmp_path, fault
):
    from contextlib import nullcontext

    root, policy, manifest = synthetic_inputs
    if fault == "native_data":
        next((root / "data").rglob("*.parquet")).unlink()
    elif fault == "camera":
        next((root / "videos").rglob("*.mp4")).unlink()
    elif fault == "policy":
        data = json.loads(policy.read_text())
        data["approved"] = False
        policy.write_text(json.dumps(data))
    elif fault == "forged_robot":
        for path in (root / "meta/openamrobot/episodes").glob("*.json"):
            data = json.loads(path.read_text())
            data["openamrobot"]["provenance"]["data_origin"] = "actual"
            data["openamrobot"]["provenance"]["source_profile"] = "ROBOT"
            path.write_text(json.dumps(data))
    rebind(synthetic_inputs)
    dependency = (
        patch(
            "oa_embodied_ai.dataset_qa.episode.parquet",
            side_effect=ImportError("native unavailable"),
        )
        if fault == "native_dependencies"
        else nullcontext()
    )
    with dependency, patch("oa_embodied_ai.training.cli.ACTPolicy") as model:
        with pytest.raises(TrainingBlockedError) as error:
            train_act(
                root,
                tiny_config,
                output_dir=tmp_path / "run",
                allow_synthetic_eval=True,
                input_manifest_path=manifest,
                policy_path=policy,
            )
        assert error.value.exit_code == 2
        model.assert_not_called()
    assert not (tmp_path / "run/c11_training_summary.json").exists()


def test_bounded_cpu_training_and_honest_checkpoint(
    synthetic_inputs, tiny_config, tmp_path
):
    root, policy, manifest = synthetic_inputs
    summary = train_act(
        root,
        tiny_config,
        output_dir=tmp_path / "run",
        smoke_test=True,
        allow_synthetic_eval=True,
        input_manifest_path=manifest,
        policy_path=policy,
    )
    assert summary["status"] == "COMPLETED" and summary["is_synthetic_override"]
    assert summary["acceptance_status"] == "NOT_VERIFIED_PHYSICAL"
    assert summary["production_acceptance"] == "NOT_VERIFIED"
    assert summary["gate_status"] == "NOT_VERIFIED"
    assert not set(summary["train_episodes"]) & set(summary["val_episodes"])
    assert summary["checkpoint_sha256"] == file_sha256(summary["checkpoint_path"])
    assert summary["provenance"]["dependencies"]["lerobot"] is None
    markdown = (tmp_path / "run/c11_training_summary.md").read_text()
    assert (
        "NOT_VERIFIED_PHYSICAL" in markdown
        and '"acceptance_status": "VERIFIED_PHYSICAL"' not in markdown
    )


def test_every_selected_episode_requires_diagnostics(
    synthetic_inputs, tiny_config, tmp_path
):
    from oa_embodied_ai.dataset_qa import episode as c10

    root, policy, manifest = synthetic_inputs
    report = c10.check_dataset(root, policy, [0, 1])
    report["findings"] = [
        f
        for f in report["findings"]
        if not (f["episode"] == 1 and f["check_id"] == "C10_VIDEO")
    ]
    with (
        patch.object(c10, "check_dataset", return_value=report),
        patch("oa_embodied_ai.training.cli.ACTPolicy") as model,
    ):
        with pytest.raises(TrainingBlockedError):
            train_act(
                root,
                tiny_config,
                output_dir=tmp_path / "run",
                allow_synthetic_eval=True,
                input_manifest_path=manifest,
                policy_path=policy,
            )
        model.assert_not_called()


def test_registry_pass_cannot_override_native_qa(
    synthetic_inputs, tiny_config, rebind, tmp_path
):
    root, policy, manifest = synthetic_inputs
    (root / "registry.json").write_text('{"quality_status":"passed"}')
    (root / "data/chunk-000/file-000.parquet").write_bytes(b"corrupt native artifact")
    rebind(synthetic_inputs)
    with patch("oa_embodied_ai.training.cli.ACTPolicy") as model:
        with pytest.raises(TrainingBlockedError):
            train_act(
                root,
                tiny_config,
                output_dir=tmp_path / "run",
                allow_synthetic_eval=True,
                input_manifest_path=manifest,
                policy_path=policy,
            )
        model.assert_not_called()
