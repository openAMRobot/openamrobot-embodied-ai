# SPDX-License-Identifier: MIT
"""Consume exact canonical C9 bytes; no schema fork or network retrieval."""

import hashlib
import os
from pathlib import Path

from jsonschema import Draft202012Validator

from .strict_io import load_data

INTERFACES_REPOSITORY = "openAMRobot/openamrobot-interfaces"
INTERFACES_COMMIT = "87ab8a936c97ab55ee74b8988ca3d7659160ea6a"
C9_SCHEMA_SHA256 = "692e04693bc738033af1da10f98e029f342497ddaeae4792945f829335f1938e"
C9_EXAMPLE_SHA256 = "e21218b4cd4ce09969ca589cc83e72d46bfa99eee9e197195a56ec1ad22a3e28"
SCHEMA_PATH = "schemas/json/c9-lerobot-dataset.schema.json"
EXAMPLE_PATH = "schemas/yaml/c9-lerobot-dataset-valid.yaml"


def c9_asset(relative):
    root = Path(os.environ.get("OA_INTERFACES_ROOT", Path(__file__).resolve().parents[3]
                               / "external/openamrobot-interfaces"))
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Canonical C9 asset missing: {path}; check out {INTERFACES_REPOSITORY} "
                                f"at {INTERFACES_COMMIT} and set OA_INTERFACES_ROOT")
    expected = {SCHEMA_PATH: C9_SCHEMA_SHA256, EXAMPLE_PATH: C9_EXAMPLE_SHA256}[relative]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Canonical C9 asset hash mismatch: {relative}")
    return path


def load_c9_schema(path=None):
    path = Path(path) if path is not None else c9_asset(SCHEMA_PATH)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != C9_SCHEMA_SHA256:
        raise ValueError("C9 schema differs from the pinned accepted contract")
    schema = load_data(path)
    Draft202012Validator.check_schema(schema)
    return schema, {"repository": INTERFACES_REPOSITORY, "commit": INTERFACES_COMMIT,
                    "path": SCHEMA_PATH, "sha256": digest}
