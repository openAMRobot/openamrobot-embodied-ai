# SPDX-License-Identifier: MIT
"""Caller-supplied snapshot bindings, independent of C10 report claims.

Real manifests require recording/release owner evidence. Hashes establish byte
consistency, not authenticity or human approval.
"""

import hashlib
import json
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from ..dataset_qa.strict_io import load_data


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root):
    root = Path(root).resolve(strict=True)
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Snapshot contains a symbolic link: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = file_sha256(path)
        elif not path.is_dir():
            raise ValueError(f"Snapshot contains a non-regular artifact: {path}")
    if not result:
        raise ValueError("Snapshot is empty")
    return result


def json_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def load_input_manifest(path):
    manifest = load_data(Path(path))
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("manifest_version")) is not int
        or manifest["manifest_version"] != 1
    ):
        raise ValueError("Expected input manifest version 1")
    for key in ("dataset_id", "dataset_revision", "profile", "evidence"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f"Input manifest requires {key}")
    if manifest.get("data_origin") not in ("actual", "synthetic"):
        raise ValueError("Input manifest requires explicit actual/synthetic origin")
    indices = manifest.get("episode_indices")
    if (
        not isinstance(indices, list)
        or not indices
        or any(type(i) is not int or i < 0 for i in indices)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("Input manifest requires unique selected episode indices")
    files = manifest.get("files_sha256")
    if not isinstance(files, dict) or not files:
        raise ValueError("Input manifest requires complete file hashes")
    for name, digest in files.items():
        if (
            not isinstance(name, str)
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or Path(name).as_posix() != name
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError("Invalid snapshot file binding")
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("policy_sha256", ""))):
        raise ValueError("Input manifest requires exact policy SHA-256")
    return manifest


def assert_unchanged(root, manifest, policy_path):
    if inventory(root) != manifest["files_sha256"]:
        raise ValueError("Dataset bytes differ from the immutable input manifest")
    if file_sha256(policy_path) != manifest["policy_sha256"]:
        raise ValueError("Policy bytes differ from the input manifest")


@contextmanager
def bound_snapshot(dataset_path, input_manifest_path, policy_path):
    """Train from a private byte-verified copy; original input may not mutate.

    Protects normal concurrent edits. Storage authenticity/hostile same-user
    mutation still requires externally controlled immutable storage.
    """
    from .gate import TrainingBlockedError

    try:
        if input_manifest_path is None or policy_path is None:
            raise ValueError("Explicit input manifest and reviewed policy are required")
        manifest = load_input_manifest(input_manifest_path)
        manifest_digest = file_sha256(input_manifest_path)
        root = Path(dataset_path).resolve(strict=True)
        assert_unchanged(root, manifest, policy_path)
        with tempfile.TemporaryDirectory(prefix="oa-c11-snapshot-") as temporary:
            snapshot = Path(temporary) / "dataset"
            shutil.copytree(root, snapshot, symlinks=True)
            policy_copy = Path(temporary) / "policy.json"
            shutil.copyfile(policy_path, policy_copy)
            assert_unchanged(root, manifest, policy_path)
            assert_unchanged(snapshot, manifest, policy_copy)
            yield snapshot, policy_copy
            if file_sha256(input_manifest_path) != manifest_digest:
                raise ValueError("Input manifest changed during execution")
            assert_unchanged(root, manifest, policy_path)
            assert_unchanged(snapshot, manifest, policy_copy)
    except TrainingBlockedError:
        raise
    except (OSError, ValueError) as error:
        raise TrainingBlockedError(f"Input snapshot failed: {error}") from error
