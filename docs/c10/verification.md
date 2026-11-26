<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Migration verification — 24 October 2026

Destination base `56bbd36ea313f3765df51524df07ccc6b2623f47`; final head and
hosted run/artifact URLs are recorded in the draft PR to avoid a self-referential
commit hash here. Reviewed source is interfaces `34fd6de3fa9735ee550c2da4da28b48e31bc1f9d`.

Local environment: macOS arm64, Python 3.12.13, pytest 9.1.1, jsonschema 4.26.0,
PyYAML 6.0.3, PyArrow 26.0.0, system ffmpeg/ffprobe 9.0.2. Native integration
executed actual synthetic MP4 creation, decode and corruption, plus real temporary
Parquet artifacts. No real recording, training, simulation or hardware was run.

## Executed results

| Command / check | Actual result |
| --- | --- |
| `python -m pip install -e "python[test]"` | Success; also verified in a separate basic environment with no PyArrow installed |
| `python -m pytest python/tests -v` | **87 passed**, no skips: unchanged registry 9 + migrated core 63 + new handoff 15; pytest also reports 17 passing subtests |
| `python python/tools/run_verification_tests.py python/native_tests test_c10_native.py` | **34 tests OK**, zero skips; all original 31 plus 3 new provenance/fail-closed/native-report tests |
| `python python/tools/run_verification_tests.py python/native_tests test_c10_evidence.py` | **23 tests OK**, zero skips; CLI/report determinism, source/input drift and failure retention preserved |
| `python python/tools/prove_c10_negative.py` | Success: each unchanged negative test passes on current source and fails exactly once on its temporary mutation, with zero errors/skips; clock arithmetic and synthetic-evidence promotion tested |
| `python python/tools/audit_c11_reference.py ../openamrobot-c-baseline` | Exact-SHA limited probes reproduced ROBOT-labelled synthetic dispatch, BLOCKED/PASS Markdown and contradictory training gate acceptance; no full C11 execution |
| `oa-dataset-qa datasets/registry.yaml` | Exit 0, zero release entries valid; unchanged registry behavior |
| `python -m pip wheel --no-deps ./python --wheel-dir .verification/c10-migration/wheel` | Success; inspected wheel contains no copied schema or dataset/test payload |
| Installed `oa-c10-check` and built wheel module invocation outside checkout | Both return expected NOT_VERIFIED / exit 2 with false readiness; wheel uses explicit `--c9-schema` |
| Semantic source comparison | Migrated `semantics.py` byte-identical to source, SHA-256 `60089db7220eb26e311fbc3fc7fcae97ef2a9c8110cfa7d3b28e645363b534d6` |

**144 distinct tests** across basic/native/evidence suites. The basic-only
environment independently reran the same 87 tests without native dependencies;
that is a repeated run, not 87 additional distinct tests. No tests were removed
or converted to skips. An initial native migration run exposed one old module
import; it was corrected before the final successful 34-test run.

The six captured workflow command pipelines (core, native, handoff, evidence,
mutation proof and generator) were executed by the evidence regression test with
an injected Python exit 23. Each preserved exit 23 through `tee` and retained its
log. Upload configuration is checked to run even after failure. This is local
shell fault injection, not a deliberately failing hosted Actions run.

## Synthetic report

Development generator executed with a deliberately recorded dirty tree. Raw CLI
status **NOT_VERIFIED**, process/report exit **2**, `training_ready=false`,
`production_acceptance=NOT_VERIFIED`: **134 PASS + 22 NOT_VERIFIED** findings,
one synthetic episode, two native frames covered, two measured skew values of
0 ns. The helper exits 0 because it independently verifies that expected blocked
reporting path. Native input bytes stayed unchanged.

Use the [evidence command](evidence.md) on the final clean head for authoritative
revision-bound evidence. Local logs remain ignored under `.verification/c10-migration/`;
downloadable hosted artifacts belong to the current migration run, not historic
interfaces runs. Their actual availability/status is recorded in the PR.

## Not verified

No `/opt/ros/jazzy/setup.bash` locally; existing ROS tests/build were not run on
the Mac. Hosted ROS results must be distinguished from local Python/native results.
No actual robot episode, calibration accuracy, physical clock synchronization,
native LeRobot loader/writer interoperability, approved production policy,
LeRobot release pin, full baseline training/test suite or checkpoint integrity
was validated. Contributor agreement and institutional authorization are not
inferred. See the [acceptance checklist](acceptance-checklist.md).
