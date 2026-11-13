# SPDX-License-Identifier: MIT
"""Validate an explicit capture adapter, separate from contract acceptance."""

from __future__ import annotations

from oa_embodied_ai.conversion.validation import ConversionError, require_components
from oa_embodied_ai.training.config import CANONICAL_CAMERA_KEYS

# Empty until source owners adopt an exact capture contract. Runtime input strings
# cannot grant acceptance. Registration is a reviewed code change, not a CLI flag.
APPROVED_CAPTURE_PROFILES = {}


def capture_contract_digest(profile):
    """Hash reviewed capture semantics, independently of per-episode identity."""
    from oa_embodied_ai.training.inputs import json_sha256

    fields = (
        "capture_version",
        "profile_id",
        "native_profile_id",
        "robot_configuration_id",
        "fps",
        "max_skew_ns",
        "streams",
        "components",
        "state_channels",
        "action_channels",
        "state_names",
        "action_names",
        "source_commits",
        "component_contract",
        "phase_evidence",
    )
    return json_sha256({key: profile.get(key) for key in fields})


def validate_profile(profile, *, development=False):
    if not isinstance(profile, dict) or profile.get("capture_version") != 1:
        raise ConversionError(
            "Incompatible capture schema version; expected capture_version 1"
        )
    for key in (
        "profile_id",
        "episode_id",
        "dataset_id",
        "dataset_revision",
        "task",
        "robot_configuration_id",
    ):
        if not isinstance(profile.get(key), str) or not profile[key].strip():
            raise ConversionError(f"Missing {key}")
    if profile.get("origin") not in ("synthetic", "simulation", "robot"):
        raise ConversionError("Explicit recording origin required")
    if profile.get("status") != "accepted":
        if not development or profile["origin"] != "synthetic":
            raise ConversionError(
                "Unapproved profile: only explicitly synthetic development is permitted"
            )
    if (
        profile.get("origin") != "synthetic"
        and profile.get("profile_id") not in APPROVED_CAPTURE_PROFILES
    ):
        raise ConversionError(
            "Unapproved profile: no owner-approved actual capture contract registered"
        )
    if profile.get("status") == "accepted" and not profile.get("approval_ref"):
        raise ConversionError(
            "Accepted profile requires owner approval_ref; the string alone is not approval evidence"
        )
    if profile["origin"] != "synthetic":
        reviewed = APPROVED_CAPTURE_PROFILES[profile["profile_id"]]
        if reviewed.get("sha256") != capture_contract_digest(profile) or reviewed.get(
            "approval_ref"
        ) != profile.get("approval_ref"):
            raise ConversionError(
                "Unapproved profile: capture contract differs from its reviewed binding"
            )
    if type(profile.get("fps")) is not int or profile["fps"] <= 0:
        raise ConversionError("fps must be a positive integer")
    if type(profile.get("max_skew_ns")) is not int or profile["max_skew_ns"] < 0:
        raise ConversionError(
            "Explicit nonnegative max_skew_ns required; no production tolerance default"
        )
    streams = profile.get("streams", [])
    if not isinstance(streams, list) or not streams:
        raise ConversionError("Missing capture streams")
    ids, topics, features = set(), set(), set()
    for stream in streams:
        if not isinstance(stream, dict):
            raise ConversionError("Malformed stream metadata: expected an object")
        for key in (
            "id",
            "component",
            "topic",
            "type",
            "frame_id",
            "clock_domain",
            "timestamp_source",
        ):
            if not isinstance(stream.get(key), str) or not stream[key]:
                raise ConversionError(f"Stream {stream.get('id')} missing {key}")
        if stream["id"] in ids or stream["topic"] in topics:
            raise ConversionError("Duplicate stream identity or topic")
        ids.add(stream["id"])
        topics.add(stream["topic"])
        if stream["timestamp_source"] not in ("header", "bag"):
            raise ConversionError(f"Invalid clock mapping for '{stream['id']}'")
        if stream.get("camera_key"):
            features.add(stream["camera_key"])
            if not stream.get("calibration_id"):
                raise ConversionError(f"Missing calibration ID for '{stream['id']}'")
    if features != set(CANONICAL_CAMERA_KEYS):
        raise ConversionError(
            "Missing required camera: all four canonical camera streams required"
        )
    if len({s["clock_domain"] for s in streams}) != 1:
        raise ConversionError(
            "Invalid clock mapping: mixed domains need an explicit reviewed affine adapter"
        )
    for kind in ("state", "action"):
        channels = profile.get(kind + "_channels")
        if not isinstance(channels, list) or not channels:
            raise ConversionError(f"Missing {kind} channels")
        if any(not isinstance(c, dict) for c in channels):
            raise ConversionError(f"Malformed {kind} channel metadata")
        names = [c.get("name") for c in channels]
        if any(not isinstance(n, str) or not n for n in names) or len(names) != len(
            set(names)
        ):
            raise ConversionError(f"Invalid {kind} channel names/order")
        if profile.get(kind + "_names") != names:
            raise ConversionError(f"Invalid {kind} channel ordering")
        for c in channels:
            if c.get("stream") not in ids:
                raise ConversionError(
                    f"Component '{c.get('component')}' missing {kind} topic mapping"
                )
            if not all(c.get(k) for k in ("component", "unit", "source_unit", "field")):
                raise ConversionError(f"Malformed {kind} mapping for '{c['name']}'")
            if (
                c["component"] == "base"
                and kind == "action"
                and c.get("semantics") != "requested_velocity"
            ):
                raise ConversionError(
                    "Base action source must be requested_velocity; Nav2 goals/applied telemetry are not substitutes"
                )
        require_components(
            profile.get("components"),
            {c["component"]: True for c in channels},
            profile["profile_id"],
        )
    sources = profile.get("source_commits", {})
    if (
        not isinstance(sources, dict)
        or not sources
        or any(
            not isinstance(v, str)
            or len(v) != 40
            or any(ch not in "0123456789abcdef" for ch in v)
            for v in sources.values()
        )
    ):
        raise ConversionError(
            "Source/model/schema commit provenance required (full SHA)"
        )
    if "lift" in profile["components"]:
        from pathlib import Path

        from oa_embodied_ai.dataset_qa.strict_io import load_data
        from oa_embodied_ai.training.inputs import file_sha256

        binding = profile.get("component_contract", {})
        if (
            not isinstance(binding, dict)
            or not binding.get("path")
            or not binding.get("sha256")
        ):
            raise ConversionError(
                "Lift requires an explicit canonical C9 component-contract file and SHA256 binding"
            )
        path = Path(binding["path"])
        if not path.is_file() or file_sha256(path) != binding["sha256"]:
            raise ConversionError("C9 component-contract artifact missing or modified")
        contract = load_data(path)
        if (
            contract.get("component_profile_version") != 1
            or contract.get("c9_contract_version") != 1
        ):
            raise ConversionError("Incompatible C9 component profile version")
        if contract.get("status") != "accepted" and profile["origin"] != "synthetic":
            raise ConversionError(
                "Unapproved C9 component contract cannot authorize real-data export"
            )
        expected = contract.get("channels", [])
        for kind in ("state", "action"):
            actual = profile[kind + "_channels"]
            if [
                (c.get("name"), c.get("component"), c.get("unit"), c.get("range"))
                for c in actual
            ] != [
                (c.get("name"), c.get("component"), c.get("unit"), c.get("range"))
                for c in expected
            ]:
                raise ConversionError(
                    f"{kind} channel mapping differs from canonical C9 component contract"
                )
    return profile
