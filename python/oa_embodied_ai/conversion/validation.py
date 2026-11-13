# SPDX-License-Identifier: MIT
"""Pre-export component and explicit-unit checks shared by source adapters."""

from __future__ import annotations

import math
from collections.abc import Mapping


class ConversionError(ValueError):
    """Ordinary invalid recording; CLI reports this without a traceback."""


COMPONENTS = {"left_arm", "right_arm", "left_gripper", "right_gripper", "lift", "base"}
CONVERSIONS = {
    ("rad", "rad"): 1.0,
    ("deg", "rad"): math.pi / 180,
    ("rad", "deg"): 180 / math.pi,
    ("deg", "deg"): 1.0,
    ("rad", "degree"): 180 / math.pi,
    ("degree", "rad"): math.pi / 180,
    ("degree", "degree"): 1.0,
    ("mm", "m"): 0.001,
    ("m", "m"): 1.0,
    ("m/s", "m/s"): 1.0,
    ("rad/s", "rad/s"): 1.0,
}


def require_components(declared, available, profile):
    """Check presence before indexing any declared component (upperbody#39)."""
    if not isinstance(declared, list) or not declared:
        raise ConversionError("Profile components must be a nonempty list")
    if any(not isinstance(c, str) for c in declared):
        raise ConversionError("Malformed component declaration: expected names")
    if len(set(declared)) != len(declared):
        raise ConversionError("Duplicate component declaration")
    for component in declared:
        if component not in COMPONENTS:
            raise ConversionError(f"Unsupported component '{component}'")
        if component not in available or available[component] is None:
            raise ConversionError(
                f"Missing required component '{component}': declared by profile {profile}, "
                f"but no valid {component} state was found in the selected recording."
            )


def normalize_vector(values, metadata, channels, *, profile, components):
    """Normalize an explicitly named source vector; never infer a mapping by size.

    Metadata includes representation, ordered names and units. Cartesian 16D
    poses cannot be interpreted as two seven-joint arms and grippers.
    """
    if (
        not isinstance(channels, list)
        or not channels
        or any(
            not isinstance(c, Mapping)
            or not all(
                isinstance(c.get(k), str) and c[k]
                for k in ("name", "component", "unit")
            )
            for c in channels
        )
    ):
        raise ConversionError(
            "Malformed channel metadata: names, components and units required"
        )
    if not isinstance(values, (list, tuple)):
        raise ConversionError("Malformed source vector: expected a list")
    if not isinstance(metadata, Mapping) or metadata.get("representation") != "qpos":
        raise ConversionError(
            "Malformed source metadata: explicit qpos representation required"
        )
    if metadata.get("names") != [c["name"] for c in channels]:
        raise ConversionError("Invalid channel ordering or missing joint mapping")
    if len(values) != len(channels):
        raise ConversionError(
            f"Wrong vector dimensions: expected {len(channels)}, got {len(values)}"
        )
    units = metadata.get("units")
    if not isinstance(units, list) or len(units) != len(channels):
        raise ConversionError(
            "Malformed source metadata: one explicit unit per channel required"
        )
    available = {c["component"]: True for c in channels}
    require_components(components, available, profile)
    output = []
    for value, source_unit, channel in zip(values, units, channels, strict=True):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ConversionError(
                f"Invalid numeric value for component '{channel['component']}'"
            )
        expected = channel.get("source_unit")
        if expected is not None and source_unit != expected:
            raise ConversionError(
                f"Unit mismatch for '{channel['name']}': expected {expected}, got {source_unit}"
            )
        if not isinstance(source_unit, str):
            raise ConversionError("Malformed source metadata: unit must be a string")
        factor = CONVERSIONS.get((source_unit, channel["unit"]))
        if factor is None:
            raise ConversionError(
                f"Unsupported unit conversion for '{channel['name']}': {source_unit} -> {channel['unit']}"
            )
        value = (
            value / 1000
            if (source_unit, channel["unit"]) == ("mm", "m")
            else value * factor
        )
        limits = channel.get("range")
        if limits is not None and not limits[0] <= value <= limits[1]:
            raise ConversionError(
                f"Component '{channel['component']}' channel '{channel['name']}' outside {limits} {channel['unit']}"
            )
        output.append(value)
    return output


def validate_lift_phase(samples, *, stowed_pose_review):
    """Data evidence only; this never enforces a robot interlock."""
    moving = [s for s in samples if s.get("lift_moving")]
    if moving and not stowed_pose_review:
        raise ConversionError(
            "Lift phase requires reviewed stowed-pose evidence; park is a placeholder"
        )
    for sample in moving:
        if sample.get("phase") != "lift_positioning":
            raise ConversionError(
                "Missing lift phase: moving lift requires lift_positioning"
            )
        if (
            sample.get("base_stopped") is not True
            or sample.get("arms_stowed") is not True
        ):
            raise ConversionError(
                "Lift phase requires stopped-base and stowed-arms evidence for every moving sample"
            )
