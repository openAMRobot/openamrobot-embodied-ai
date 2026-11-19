# SPDX-License-Identifier: MIT
"""Validate an explicit capture adapter, separate from contract acceptance."""

from __future__ import annotations

from oa_embodied_ai.conversion.validation import ConversionError, require_components
from oa_embodied_ai.training.config import CANONICAL_CAMERA_KEYS

# Empty until source owners adopt an exact capture contract. Runtime input strings
# cannot grant acceptance. Registration is a reviewed code change, not a CLI flag.
APPROVED_CAPTURE_PROFILES = {}

# Proposed capture profiles awaiting owner review. Never consulted by
# validate_profile: an entry here grants nothing. Registering one is moving it to
# APPROVED_CAPTURE_PROFILES with status accepted and an owner approval_ref.
PROPOSED_CAPTURE_PROFILES = {
    "openamrobot-baseline1-gazebo-sim-oracle-v1": {
        "path": "configs/capture_profiles/proposed/openamrobot-baseline1-gazebo-sim-oracle-v1.json",
        "sha256": "b3cff03ca2bf2f7236e0cec984f5fee039e9e086e0a9524d6f23716e0e44ceac",
        "status": "proposed",
        "evidence": "openamr-upperbody-sw run 44448676604 (Gazebo, sim-oracle); simulation only",
    },
}


# Proposed conversion rules (19 Nov 2026), none approved. A profile that relies on
# one lists it in `proposed_rules` (hashed with the capture contract) and the
# caller enables it by name (oa-convert-bag --enable-proposed NAME); a rule that
# is declared but not enabled, or a declaration whose rule is not listed,
# refuses. `proposed_capture_profile` is caller-only: it lets a non-synthetic
# profile listed in PROPOSED_CAPTURE_PROFILES convert when its contract digest
# matches, and records that it is unapproved. Defaults are unchanged.
PROPOSED_RULES = {
    "episode_origin": "origin = first grid time with a sample of every required stream at or before it; leading frames that select earlier samples are dropped and counted",
    "controller_reference": "joint-named vectors of JointTrajectoryControllerState (joint_names + reference.positions)",
    "command_semantics": "every action channel declares what it commands; states are measured; streams disjoint",
    "vector_segments": "a state or action vector from several declared source streams, ordered segments",
    "stop_index": "an integer stop index mapped to its declared target position",
    "range_tolerance": "a declared tolerance at model limits; samples inside it are kept unchanged and counted",
    "held_command": "a command stream holds its last value between messages (until_next, or while_zero)",
    "lift_positioning_from_command": "lift_positioning intervals from each lift command to arrival at its target",
    "lift_phase_report": "lift-phase evidence recorded in the manifest instead of refusing",
}
CALLER_RULES = {
    "proposed_capture_profile": "convert a Proposed (unregistered) capture profile and its Proposed C9 component contract, bound by digest",
}
DIGEST_OPTIONAL_FIELDS = ("proposed_rules", "episode_origin", "vector_segments")


def check_proposed_rules(profile, enabled):
    """Refuse a declared Proposed rule that is not enabled by name."""
    declared = profile.get("proposed_rules", [])
    if not isinstance(declared, list) or len(set(declared)) != len(declared):
        raise ConversionError("proposed_rules must be a list of unique rule names")
    for rule in declared:
        if rule not in PROPOSED_RULES:
            raise ConversionError(f"Unknown Proposed rule '{rule}'")
        if rule not in enabled:
            raise ConversionError(
                f"Profile uses Proposed rule '{rule}' (not approved); enable it explicitly with --enable-proposed {rule}"
            )
    for rule in enabled:
        if rule not in PROPOSED_RULES and rule not in CALLER_RULES:
            raise ConversionError(f"Unknown Proposed rule '{rule}'")


def uses(profile, rule):
    return rule in profile.get("proposed_rules", [])


def require_rule(profile, rule, what):
    if not uses(profile, rule):
        raise ConversionError(
            f"{what} needs the Proposed rule '{rule}' listed in proposed_rules"
        )


ACTION_SEMANTICS = (
    "requested_velocity",
    "held_gripper_goal",
    "controller_reference",
    "lift_stop_target",
)
COMMAND_FIELD_PREFIXES = ("reference.", "output.", "error.", "desired.")


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def check_declarations(profile):
    """Every Proposed declaration needs its rule listed (and so enabled)."""
    streams = {s["id"]: s for s in profile["streams"]}
    action_streams = {c["stream"] for c in profile["action_channels"]}
    state_streams = {c["stream"] for c in profile["state_channels"]}
    if "episode_origin" in profile:
        require_rule(profile, "episode_origin", "episode_origin")
        if profile["episode_origin"] != "first_complete_grid":
            raise ConversionError("episode_origin: only first_complete_grid is defined")
    for stream in profile["streams"]:
        hold = stream.get("hold")
        if hold is None:
            continue
        require_rule(profile, "held_command", f"hold on '{stream['id']}'")
        if (
            not isinstance(hold, dict)
            or hold.get("rule") not in ("until_next", "while_zero")
            or stream["id"] not in action_streams
            or stream["id"] in state_streams
            or stream.get("camera_key")
        ):
            raise ConversionError(
                f"hold on '{stream['id']}': rule until_next or while_zero, on a command stream only"
            )
        first = hold.get("before_first")
        if first is not None and (
            not isinstance(first, dict)
            or not _number(first.get("field_value"))
            or first.get("check_state_channel") not in profile["state_names"]
            or not _number(first.get("tolerance"))
            or first["tolerance"] < 0
        ):
            raise ConversionError(
                f"hold.before_first on '{stream['id']}' needs field_value, check_state_channel and tolerance"
            )
    for kind in ("state", "action"):
        for c in profile[kind + "_channels"]:
            if "names_field" in c:
                require_rule(profile, "controller_reference", f"names_field of '{c['name']}'")
                if c["names_field"] != "joint_names" or not c.get("joint"):
                    raise ConversionError(
                        f"names_field of '{c['name']}': joint_names with a joint is defined"
                    )
            if c["source_unit"] == "stop_index":
                require_rule(profile, "stop_index", f"stop_index of '{c['name']}'")
                targets = c.get("stop_targets")
                if (
                    not isinstance(targets, dict)
                    or targets.get("unit") != c["unit"]
                    or not isinstance(targets.get("values"), list)
                    or not targets["values"]
                    or not all(_number(v) for v in targets["values"])
                    or not isinstance(targets.get("ids"), list)
                    or len(targets["ids"]) != len(targets["values"])
                    or not isinstance(targets.get("source"), str)
                    or not targets["source"]
                ):
                    raise ConversionError(
                        f"stop_index of '{c['name']}' needs stop_targets: values, ids, unit and source"
                    )
            if "range_tolerance" in c:
                require_rule(profile, "range_tolerance", f"range_tolerance of '{c['name']}'")
                if not _number(c["range_tolerance"]) or c["range_tolerance"] < 0 or c.get("range") is None:
                    raise ConversionError(
                        f"range_tolerance of '{c['name']}': a nonnegative number on a ranged channel"
                    )
    if uses(profile, "command_semantics"):
        if state_streams & action_streams:
            raise ConversionError(
                "command_semantics: state and action channels need disjoint source streams"
            )
        for c in profile["state_channels"]:
            if c["field"].startswith(COMMAND_FIELD_PREFIXES) or c.get("semantics", "measured") != "measured":
                raise ConversionError(f"State channel '{c['name']}' must be a measured value")
        for c in profile["action_channels"]:
            reference = c["field"].startswith("reference.")
            if c.get("semantics") not in ACTION_SEMANTICS or reference != (
                c["semantics"] == "controller_reference"
            ):
                raise ConversionError(
                    f"Action channel '{c['name']}' must declare what it commands ({', '.join(ACTION_SEMANTICS)})"
                )
    segments = profile.get("vector_segments")
    if segments is not None:
        require_rule(profile, "vector_segments", "vector_segments")
        if not isinstance(segments, dict) or set(segments) - {"state", "action"}:
            raise ConversionError("vector_segments: state and/or action lists")
        for kind, declared in segments.items():
            channels = {c["name"]: c for c in profile[kind + "_channels"]}
            names = []
            for segment in declared if isinstance(declared, list) else [None]:
                if (
                    not isinstance(segment, dict)
                    or set(segment) != {"stream", "names"}
                    or segment["stream"] not in streams
                    or not isinstance(segment["names"], list)
                    or not segment["names"]
                    or any(channels.get(n, {}).get("stream") != segment["stream"] for n in segment["names"])
                ):
                    raise ConversionError(
                        f"vector_segments.{kind}: each segment names its stream and that stream's channels"
                    )
                names += segment["names"]
            if names != profile[kind + "_names"]:
                raise ConversionError(
                    f"vector_segments.{kind} must list every {kind} channel once, in vector order"
                )
    evidence = profile.get("phase_evidence") or {}
    if "lift_positioning" in evidence:
        require_rule(profile, "lift_positioning_from_command", "phase_evidence.lift_positioning")
        rule = evidence["lift_positioning"]
        if (
            not isinstance(rule, dict)
            or rule.get("source") != "command_to_arrival"
            or rule.get("command_channel") not in profile["action_names"]
            or rule.get("state_channel") not in profile["state_names"]
            or not _number(rule.get("arrival_tolerance"))
            or rule["arrival_tolerance"] < 0
        ):
            raise ConversionError(
                "phase_evidence.lift_positioning: command_to_arrival with command_channel, state_channel and arrival_tolerance"
            )
    if "on_missing_evidence" in evidence:
        require_rule(profile, "lift_phase_report", "phase_evidence.on_missing_evidence")
        if evidence["on_missing_evidence"] != "report":
            raise ConversionError("phase_evidence.on_missing_evidence: only report is defined")


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
    semantics = {key: profile.get(key) for key in fields}
    # Added with the Proposed rules: hashed when present, so earlier digests hold.
    semantics.update({k: profile[k] for k in DIGEST_OPTIONAL_FIELDS if k in profile})
    return json_sha256(semantics)


def _proposed_binding(profile, enabled):
    """The Proposed entry a caller enabled, or None. Grants no approval."""
    entry = PROPOSED_CAPTURE_PROFILES.get(profile.get("profile_id"))
    if "proposed_capture_profile" not in enabled or entry is None:
        return None
    if profile.get("status") != "proposed" or entry.get("status") != "proposed":
        raise ConversionError("A Proposed capture profile must carry status proposed")
    if entry.get("sha256") != capture_contract_digest(profile):
        raise ConversionError(
            "Proposed profile: capture contract differs from its Proposed binding"
        )
    return entry


def validate_profile(profile, *, development=False, enabled=()):
    if (
        not isinstance(profile, dict)
        or type(profile.get("capture_version")) is not int
        or profile["capture_version"] != 1
    ):
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
    enabled = tuple(enabled)
    check_proposed_rules(profile, enabled)
    proposed = _proposed_binding(profile, enabled)
    if profile.get("status") != "accepted" and proposed is None:
        if not development or profile["origin"] != "synthetic":
            raise ConversionError(
                "Unapproved profile: only explicitly synthetic development is permitted"
            )
    if (
        profile.get("origin") != "synthetic"
        and profile.get("profile_id") not in APPROVED_CAPTURE_PROFILES
        and proposed is None
    ):
        raise ConversionError(
            "Unapproved profile: no owner-approved actual capture contract registered"
        )
    if profile.get("status") == "accepted" and not profile.get("approval_ref"):
        raise ConversionError(
            "Accepted profile requires owner approval_ref; the string alone is not approval evidence"
        )
    if profile["origin"] != "synthetic" and proposed is None:
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
        if "equal_timestamp_rule" in stream and stream["equal_timestamp_rule"] not in (
            "keep_last",
        ):
            raise ConversionError(
                f"Unknown equal_timestamp_rule for '{stream['id']}'; only keep_last is defined"
            )
        if stream.get("camera_key"):
            if (
                not isinstance(stream["camera_key"], str)
                or stream["camera_key"] in features
            ):
                raise ConversionError(
                    "Duplicate or malformed canonical camera feature mapping"
                )
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
            if not isinstance(c.get("stream"), str) or c["stream"] not in ids:
                raise ConversionError(
                    f"Component '{c.get('component')}' missing {kind} topic mapping"
                )
            if not all(
                isinstance(c.get(k), str) and c[k]
                for k in ("component", "unit", "source_unit", "field")
            ):
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
        if (
            contract.get("status") != "accepted"
            and profile["origin"] != "synthetic"
            and proposed is None
        ):
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
    check_declarations(profile)
    return profile
