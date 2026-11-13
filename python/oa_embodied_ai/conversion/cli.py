# SPDX-License-Identifier: MIT
"""Validate a ROS bag, align streams and export with the actual LeRobot writer."""

import argparse
from pathlib import Path

from jsonschema.exceptions import ValidationError

from oa_embodied_ai.conversion.native import export_native
from oa_embodied_ai.conversion.validation import ConversionError
from oa_embodied_ai.dataset_qa.strict_io import load_data
from oa_embodied_ai.recording.bag import align, read_bag, validate_recording_phases
from oa_embodied_ai.recording.profile import validate_profile
from oa_embodied_ai.training.inputs import inventory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--sidecar-template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic-development", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        profile = validate_profile(
            load_data(args.profile), development=args.synthetic_development
        )
        source_hashes = inventory(args.bag)
        data = read_bag(args.bag, profile)
        validate_recording_phases(data, profile)
        frames = align(data, profile)
        if args.validate_only:
            print(
                f"PASS: {len(frames)} structurally aligned frames; training/physical acceptance NOT_VERIFIED"
            )
            return 0
        export_native(
            frames,
            profile,
            load_data(args.sidecar_template),
            args.output,
            source_bag=args.bag,
            expected_source_sha256=source_hashes,
        )
        print(
            f"PASS: native loader verified output at {args.output}; training/physical acceptance NOT_VERIFIED"
        )
        return 0
    except (ConversionError, OSError, ImportError, ValidationError, ValueError) as exc:
        print(f"REJECT: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
