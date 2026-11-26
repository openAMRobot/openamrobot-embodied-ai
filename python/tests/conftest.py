# SPDX-License-Identifier: MIT
"""Pytest configuration for oa_embodied_ai test suite."""

# On Windows, torch C++ runtime must initialize before pyarrow to prevent DLL initialization collision
try:
    import torch  # noqa: F401 - initialize Windows runtime before PyArrow
except ImportError:
    pass


import json
import shutil

import pytest


@pytest.fixture(scope="session")
def synthetic_template(tmp_path_factory):
    import torch
    from c11_fixture import synthetic_dataset

    torch.set_num_threads(1)
    return synthetic_dataset(tmp_path_factory.mktemp("c11-template") / "dataset")


@pytest.fixture
def synthetic_inputs(synthetic_template, tmp_path):
    root, policy, manifest = synthetic_template
    copied = tmp_path / "dataset"
    shutil.copytree(root, copied)
    policy_copy, manifest_copy = tmp_path / policy.name, tmp_path / manifest.name
    shutil.copyfile(policy, policy_copy)
    shutil.copyfile(manifest, manifest_copy)
    return copied, policy_copy, manifest_copy


@pytest.fixture
def tiny_config():
    from oa_embodied_ai.training.config import get_profile_config

    cfg = get_profile_config("lerobot-v0.6.1-bi-openarm-position")
    cfg.chunk_size, cfg.d_model, cfg.n_heads = 3, 32, 4
    cfg.enc_layers = cfg.dec_layers = 1
    cfg.dim_feedforward, cfg.image_size = 64, (32, 32)
    cfg.dropout, cfg.device = 0.0, "cpu"
    return cfg


@pytest.fixture
def rebind():
    from oa_embodied_ai.training.inputs import file_sha256, inventory

    def update(inputs):
        root, policy, manifest_path = inputs
        manifest = json.loads(manifest_path.read_text())
        manifest.update(files_sha256=inventory(root), policy_sha256=file_sha256(policy))
        manifest_path.write_text(json.dumps(manifest))

    return update
