# SPDX-License-Identifier: MIT
"""Validate a ROS bag, align streams and export with the actual LeRobot writer."""

import argparse
from pathlib import Path

from jsonschema.exceptions import ValidationError

from oa_embodied_ai.conversion.batch import convert_collection, prepare_episode
from oa_embodied_ai.conversion.native import export_native
from oa_embodied_ai.conversion.validation import ConversionError
from oa_embodied_ai.dataset_qa.strict_io import load_data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, nargs="?")
    parser.add_argument(
        "--collection", type=Path, help="Ordered multi-episode collection manifest"
    )
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--sidecar-template", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic-development", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.collection:
            if args.bag or args.profile or args.sidecar_template:
                raise ConversionError(
                    "--collection cannot be mixed with single-bag arguments"
                )
            convert_collection(
                args.collection,
                args.output,
                development=args.synthetic_development,
                validate_only=args.validate_only,
            )
        else:
            if not args.bag or not args.profile or not args.sidecar_template:
                raise ConversionError(
                    "Single-bag conversion requires bag, --profile and --sidecar-template"
                )
            episode = prepare_episode(
                args.bag,
                load_data(args.profile),
                load_data(args.sidecar_template),
                development=args.synthetic_development,
            )
            if not args.validate_only:
                export_native(
                    episode.frames,
                    episode.profile,
                    episode.template,
                    args.output,
                    source_bag=episode.source_bag,
                    expected_source_sha256=episode.source_sha256,
                )
        if args.validate_only:
            print(
                "PASS: selected episodes structurally validated; training/physical acceptance NOT_VERIFIED"
            )
            return 0
        print(
            f"PASS: native loader verified output at {args.output}; training/physical acceptance NOT_VERIFIED"
        )
        return 0
    except (
        ConversionError,
        OSError,
        ImportError,
        ValidationError,
        ValueError,
        RuntimeError,
    ) as exc:
        print(f"REJECT: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
