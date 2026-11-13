# Bag capture and native LeRobot conversion

`oa-record-bag` is a new observational subprocess workflow, separate from the
legacy ROS recorder node (which remains a configuration-only skeleton). It
publishes no motion commands. Actual capture profiles are not registered yet;
claims of `accepted` in input JSON cannot enable an actual profile. Source-owner
adoption is required before registration. Synthetic development is explicit.

The converter checks declared components before extraction, topic/type/frame
identity, source fields, units, dimensions, clock monotonicity, all four canonical
cameras, and explicit ordering. It rejects missing/stale/reused camera frames,
uses nearest observations within the declared skew and causal commands, and
retains original timestamps in C9. Mixed clock domains need a reviewed affine
adapter and currently reject. It does not interpolate, pad, truncate or guess
mappings. It exports one episode per invocation; held-out training needs a
separately prepared multi-episode dataset.

Native output uses the real LeRobot 0.6.1 writer, finalization, video storage and
loader. Every frame is loaded to verify dimensions, vector values, episode/frame/
task indices, timestamps and decoded images. C9 v1 remains unchanged and is loaded
from its pinned canonical interfaces asset, not copied here. A supplied C9 template
binds configuration/calibration artifacts; the converter does not manufacture
physical calibration or reviewed policy evidence. Destination creation is atomic
and refuses overwrite. Source artifacts are hashed before/after conversion.

Install in a separate Python 3.12 development environment:

```bash
uv venv --python 3.12 /tmp/oa-conversion-env
uv pip install --python /tmp/oa-conversion-env/bin/python -e 'python[test,native,training,conversion,benchmark]'
source /tmp/oa-conversion-env/bin/activate
```

This inspected LeRobot version requires Torch <2.12 and Python >=3.12. It is a
development interoperability pin, not project release approval. Registry/core C10
keeps its existing Python >=3.10 support. The existing Torch 2.15 verification
environment must not be silently downgraded to test the native adapter.

After an approved capture profile exists, the workflow is:

```bash
oa-record-bag --profile reviewed-capture.json --output /data/episode-001 --seconds 30
oa-convert-bag /data/episode-001/bag --profile reviewed-capture.json --sidecar-template reviewed-c9.json --output /data/dataset-001
oa-c10-check --dataset-path /data/dataset-001 --policy reviewed-policy.json --report /data/c10-report.json
```

`--validate-only` runs bag/alignment checks without export. On this implementation,
actual capture remains BLOCKED until owners register the exact profile; these
commands describe that gated interface, not completed ROS testing. The recorder
checks the graph before starting, captures to a unique episode directory, sends
SIGINT on the requested stop/bounded duration, retains interruption logs and
manifests, and independently validates bag content after a clean recorder exit.

Reproduce the executed synthetic CDR fixture workflow from repository root:

```bash
python python/tests/capture_fixture.py /tmp/oa-synthetic-capture
python -m oa_embodied_ai.conversion.cli /tmp/oa-synthetic-capture/bag --profile /tmp/oa-synthetic-capture/profile.json --sidecar-template /tmp/oa-synthetic-capture/sidecar-template.json --output /tmp/oa-synthetic-native --synthetic-development
python -m pytest python/native_tests/test_bag_conversion.py -q
python python/tools/prove_converter_negative.py
```

The fixture writes serialized ROS 2 messages directly to a bag through rosbags.
It is **synthetic**, not Gazebo, ROS graph execution or physical recording. The
native test also runs C10: no REJECT finding; NOT_VERIFIED/exit 2 and false
training readiness remain mandatory. Existing C10 tests cover malformed native
metadata, missing Parquet, corrupted video and provenance/identity mutations.

Capture contract fields are illustrated by `python/tests/capture_fixture.py`.
Each stream declares topic, ROS type, component, frame, timestamp source and
clock domain; cameras additionally declare calibration and canonical feature
identity. Each state/action channel declares the source stream, field/joint,
ordered name and explicit source/target units. All declared streams are required;
optional streams need a separate profile. Headerless messages require an explicit
configuration frame source and receipt-time semantics. The fixture's
`/fixture/*` topics are test-only, not proposed robot topics.

Proposed canonical whole-body names/ranges live in interfaces PR #39. A lift
profile must bind that component artifact by exact SHA256; state/action mapping
must match it. Moving lift source samples (including samples between export-grid
frames) require `lift_positioning` interval coverage, measured stopped-base
odometry and a complete reviewed stowed-arm pose. The check is data validation,
not a safety interlock. It supplies neither a stowed pose nor a tolerance.
The baseline `park` pose cannot establish acceptance. Base training action must
be requested velocity; Nav2 goals or applied telemetry are rejected substitutes.

Not verified: actual robot/simulator topics, mixed-device clock calibration,
Gazebo recording, real Enactic sample conversion, full-body action approval,
physical calibration, hardware or production acceptance. Local `/opt/ros` is
absent and Docker is stopped. Model source: current upperbody 2ca7392; the model
pins platform 131e500 even though current platform main is 713574f. No claim is
made that substituting current platform main preserves simulator compatibility.

Dependencies added explicitly: LeRobot 0.6.1 with dataset extra (Apache-2.0,
https://github.com/huggingface/lerobot/tree/7e241bd630a3719a56157a497ce5d08f244784f1),
rosbags 0.11.9 (Apache-2.0, https://gitlab.com/ternaris/rosbags). Native LeRobot's
PyAV/codec dependencies retain upstream licences; no binaries are vendored.
No new ROS node, topic, launch or control contract is represented as accepted.
