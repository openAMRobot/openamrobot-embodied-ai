# SPDX-License-Identifier: MIT
"""Strict JSON/YAML reader extracted from interfaces tools/lint_schemas.py."""

import json
import yaml

def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate mapping key: {key}")
        result[key] = value
    return result


class UniqueLoader(yaml.SafeLoader):
    pass


def yaml_mapping(loader, node):
    return unique_pairs(loader.construct_pairs(node))


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, yaml_mapping)


def load_data(path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        value = json.loads(text, object_pairs_hook=unique_pairs,
                           parse_constant=reject_constant)
    elif path.suffix in (".yaml", ".yml"):
        value = yaml.load(text, Loader=UniqueLoader)
    else:
        raise ValueError(f"Unsupported schema/data extension: {path}")
    # YAML timestamps, non-string keys and non-finite numbers are not JSON data.
    def check(item):
        if isinstance(item, dict):
            if any(not isinstance(k, str) for k in item):
                raise ValueError("Schema data requires string keys")
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
    check(value)
    json.dumps(value, allow_nan=False)
    return value
