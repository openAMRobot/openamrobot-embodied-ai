# SPDX-License-Identifier: MIT
"""Prepare an ordered bag collection without merging or relabelling source data."""

from pathlib import Path

from oa_embodied_ai.conversion.native import ConversionEpisode, export_native_episodes
from oa_embodied_ai.conversion.validation import ConversionError
from oa_embodied_ai.dataset_qa.strict_io import load_data
from oa_embodied_ai.recording.bag import align, read_bag, validate_recording_phases
from oa_embodied_ai.recording.profile import validate_profile
from oa_embodied_ai.training.inputs import inventory


def prepare_episode(bag, profile, template, *, development=False):
    profile = validate_profile(profile, development=development)
    source_hashes = inventory(bag)
    data = read_bag(bag, profile)
    validate_recording_phases(data, profile)
    frames = align(data, profile)
    if inventory(bag) != source_hashes:
        raise ConversionError("Source bag artifacts changed during capture validation")
    return ConversionEpisode(frames, profile, template, Path(bag), source_hashes)


def convert_collection(path, output, *, development=False, validate_only=False):
    """Manifest paths resolve relative to the manifest, independent of shell cwd."""
    path = Path(path).resolve(strict=True)
    collection = load_data(path)
    if (
        not isinstance(collection, dict)
        or type(collection.get("collection_version")) is not int
        or collection["collection_version"] != 1
        or set(collection) != {"collection_version", "episodes"}
    ):
        raise ConversionError(
            "Expected collection_version 1 and an ordered episodes list"
        )
    entries = collection["episodes"]
    if not isinstance(entries, list) or not entries:
        raise ConversionError("Episode collection must be a nonempty list")
    episodes = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {
            "bag",
            "profile",
            "sidecar_template",
        }:
            raise ConversionError(
                f"Episode {i} requires bag, profile and sidecar_template paths"
            )
        if any(not isinstance(v, str) or not v for v in entry.values()):
            raise ConversionError(f"Episode {i} paths must be nonempty strings")
        resolved = {k: path.parent / v for k, v in entry.items()}
        episodes.append(
            prepare_episode(
                resolved["bag"],
                load_data(resolved["profile"]),
                load_data(resolved["sidecar_template"]),
                development=development,
            )
        )
    # Export performs cross-episode checks before creating any output. For a
    # validate-only request, exercise the same collection checks without a writer.
    if validate_only:
        from oa_embodied_ai.conversion.native import validate_collection

        validate_collection(episodes, output)
        return {
            "episodes_validated": len(episodes),
            "frames_validated": sum(len(e.frames) for e in episodes),
        }
    return export_native_episodes(episodes, output)
