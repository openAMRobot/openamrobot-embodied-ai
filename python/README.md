# oa_embodied_ai

Python tooling for OpenAMRobot embodied AI. Basic runtime dependencies are PyYAML
and jsonschema. Optional `native` installs PyArrow for episode artifact inspection;
system ffmpeg/ffprobe enable media checks. No ROS or training stack is required.

| Module | Status |
| --- | --- |
| `dataset_qa` | Registry validation, C10 episode/native artifact checks and fail-closed report consumer |
| `conversion` | Planned, README only |
| `training` | Planned, README only |
| `evaluation` | Planned, README only |

```bash
python -m pip install -e "python[test]"
oa-dataset-qa datasets/registry.yaml
python -m pytest python/tests -v
```

`oa-dataset-qa` finds the schema by searching upward from the input file and from this package
for `schemas/dataset_registry.schema.json`. Pass `--schema <path>` to use another file.

Episode checks use `oa-c10-check --dataset-path … --policy … --report …` (exit
0 PASS, 1 REJECT, 2 NOT_VERIFIED). C9 schema/example stay in a pinned external
interfaces checkout. Prepare those assets before running C10 tests; see the
[C10 guide](../docs/c10/README.md) for setup, native suites, report fields and policy.
Registry `quality_status: passed` and passing synthetic tests cannot approve
training. Real episode and production acceptance remain unverified.
