<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Execution evidence and owner gates — 12 November 2026

Runtime base: `a1f92f58e28a7b10f0ae5e3550151a4d3c50291a`, the documentation-only
follow-up to canonical C10/C11 draft #5. Source code is on the dependent
`satya/capture-conversion-benchmark` branch. Contract proposal:
[interfaces #39](https://github.com/openAMRobot/openamrobot-interfaces/pull/39),
head `2572fde7c2463d4ac42961394df0cb7a253f89ba`, base
`14912e18ecb6473ebd7ef4a42c4bd1870b2023f6`.

| Area | Status and evidence | Outstanding decision/evidence owner |
|---|---|---|
| Canonical C10/C11 | #5 proposed; original #3 and Marcus #4 commits are exact ancestors. Existing 166-test baseline passed. #3 review approved its code; #5 adoption remains separate. | C11 owner/software lead confirm canonical route; #3/#4 not closed |
| C9 v1 compatibility | PASS: immutable schema/example SHA256 and all registered fixtures, 3 contract tests | No v1 semantic change |
| Proposed model mapping | PASS for pinned-source comparison: 1 test; isolated 0.351 m mutation fails once. Exact arm/gripper/lift names and separate lift group checked | Combined-model owner: generated URDF/full bounds; Device Package owner: capture/gripper semantics |
| Full-body action profile | BLOCKED; profile is Proposed and commands remain unbound | Software lead plus manipulation/platform/CTL-017 owners |
| 16/17-dimensional conversion | PASS for explicit qpos descriptors, unit conversions, malformed dimensions/order/units and missing mapping; no padding/guessing | Actual Enactic sample/native metadata and gripper equivalence remain NOT_VERIFIED |
| Missing-component gate | PASS: missing lift/arms/grippers errors name component; removed guard produces exactly one intended assertion failure | Software lead reviews upperbody#39 consumer changes |
| Actual native LeRobot | PASS: synthetic serialized CDR bag -> native writer/videos/Parquet -> actual 0.6.1 loader, 4 frames x 4 cameras; dimensions/value/index/time checks | Release owner validates project/runtime release separately |
| Converted C10 dataset | PASS for software inspection; report has zero REJECT findings, NOT_VERIFIED/exit 2 and false readiness | Actual recording, reviewed policy, calibration/config evidence |
| Capture orchestration | PASS for subprocess/manifest/error unit tests; actual topic profiles unregistered | Camera/source owners and software lead approve topic/type/frame/clock bindings |
| ROS/Gazebo recording | BLOCKED locally: absent /opt/ros and stopped Docker; no emitted simulator bag claimed | Current combined-model environment and camera streams; simulation owner |
| C10 native corruption/provenance | PASS: 34 native tests, 23 evidence tests; C10 and C11 mutation controls run | Real-source acceptance remains NOT_VERIFIED |
| C11 software path | PASS: bounded visual synthetic train/backprop/checkpoint/reload/held-out offline evaluation; raw C10 remains blocked; zero executed rollouts | Marcus/software lead adopt model architecture; real closed-loop evidence separate |
| Benchmark harness | PASS for timing/deadline/scenario/shape tests and actual five-second desktop CPU run; raw samples/hash retained | Arka: 600-second A/B/C Orin NX runs; MoveIt owner: exact stack/configuration |
| Jetson fit/physical acceptance | NOT_VERIFIED; no authorized device found, no GPU/thermal/power measurements | Designated hardware/software reviewers decide on-device/remote |
| DCO | New commits signed; #5 still has five historical Satya omissions; Marcus's signatures preserved | Contributor + maintainer agree remediation without rewriting shared history |
| Hosted contract CI | PASS at interfaces #39 head: quality/build, both repository-quality jobs, c9-component-source, DCO, CLA | Owner adoption still pending |
| Hosted #5 CI | PASS at a1f92f5: C10, CPU integration, Python, ROS and CLA; DCO action_required | No gate weakened |
| Release | NOT_VERIFIED | Physical evidence, contract adoption and release sign-off |

Local evidence is retained outside tracked source under
`.verification/workstream-20261112/`. Exact commands, exit codes, dependency
versions, source hashes and artifacts from clean-head verification are generated
by `python/tools/verify_c10_c11.py`. Synthetic bags, MP4s, Parquet, checkpoints and
large logs remain ignored. CI uploads the nonempty-suite/mutation evidence and an exact-head synthetic CDR bag,
native dataset, raw NOT_VERIFIED C10 report and SHA256 manifest. The artifact
helper returns software-path success only; it preserves raw C10 exit 2.

Executed commands (repository root, Python 3.12.13 on macOS arm64):

```bash
.venv/bin/python -m pytest python/tests -q
# 215 passed + 17 subtests before final clean-head verification
.venv/bin/python python/tools/run_verification_tests.py python/native_tests test_c10_native.py
# 34 passed, no skips
.venv/bin/python python/tools/run_verification_tests.py python/native_tests test_c10_evidence.py
# 23 passed, no skips
/private/tmp/openam-native-lerobot-env/bin/python -m pytest python/native_tests/test_bag_conversion.py -q
# 3 passed, no skips; includes actual loader and C10
.venv/bin/python python/tools/prove_converter_negative.py
.venv/bin/python python/tools/prove_c10_negative.py
.venv/bin/python python/tools/prove_c11_negative.py
.venv/bin/python python/tools/run_c11_smoke.py --output-dir .verification/workstream-20261112/training-smoke
```

The native environment uses LeRobot 0.6.1 / Torch 2.11.0 / PyArrow 26.0.1;
the existing environment uses Torch 2.15.0 / PyArrow 26.0.0. Its first combined
run had 4 failures (tests incorrectly asserted LeRobot absence) and 214 passes;
that failed log is retained. Corrected tests require truthful installed-version
provenance and keep the custom model/upstream-compatibility distinction. The corrected
combined run passed 218 tests + 17 subtests, zero skips. Native
conversion CI runs the combined suite to protect coexistence.

The currently demonstrated pipeline is **synthetic CDR bag -> native LeRobot ->
C10 NOT_VERIFIED**. C11 training is separately demonstrated on the existing
multi-episode synthetic fixture. The converter emits one episode per invocation;
this is not a demonstrated captured multi-episode training pipeline, Gazebo
recording, real robot recording, rollout or physical acceptance.

Dependencies: optional inspected LeRobot dataset 0.6.1 and rosbags 0.11.9
(Apache-2.0), psutil >=6,<8 (BSD-3-Clause); benchmark reuses declared Torch,
torchvision and NumPy. No dependency of the existing core runtime is replaced.
See conversion/README and jetson-benchmark.md for sources/licences and commands.
OpenAI Codex implemented the runtime/tests/docs and verified software. Original
C10/C11 author credit and signatures are retained; no third-party implementation
is copied and no safety controller, watchdog, arbiter or hardware output changes.
