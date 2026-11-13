<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# ACT-like inference benchmark handoff

Issue: [upperbody#38](https://github.com/openAMRobot/openamr-upperbody-sw/issues/38).
The harness loads C11's custom visual ACT-like checkpoint; it does not claim
upstream LeRobot ACT compatibility. It accepts four real input tensors decoded
from an identified dataset, with strict batch/state/image/precision checks. It
never creates a zero-input fallback or sends actuator commands. Checkpoint and
observation files are required and SHA256-bound; use identical files for A/B/C.

Warmup is excluded. CUDA synchronizes at measurement boundaries. Latency uses
`perf_counter`; scheduled completion deadlines include prior loop/telemetry delay.
Reports retain every inference duration/completion in hashed JSONL and percentile summaries, observed maximum,
achieved count/elapsed frequency, missed deadlines and over-period inferences.
Telemetry samples CPU/process/RAM/CUDA allocated/reserved memory, available sysfs
GPU load and temperatures. Jetson runs capture raw tegrastats; L4T, power mode,
CUDA/PyTorch/TensorRT, model parameter count, checkpoint/input/code/config hashes,
precision, batch, camera count/resolution, observation length and chunk size are
recorded. Missing telemetry remains explicit, including unparsed throttling.
No duration, thermal or performance threshold creates physical acceptance.

Prepare an inference-only environment using the device's owner-provisioned
PyTorch/torchvision and CUDA versions. Do not replace JetPack or the power mode.
Install this repository with the existing compatible runtime and psutil. The
benchmark extra explicitly declares NumPy, Torch, torchvision and psutil; dependency
resolution must be reviewed against the provisioned device. Example:

```bash
python -m pip install --no-deps -e python
python -m pip install 'psutil>=6,<8'
python -m oa_embodied_ai.benchmarking.runtime --help
```

The checkpoint is the C11 `act_policy_last.pt` payload (safe `weights_only=True`
loading). Observations NPZ contains exactly `state` float32 [N,state_dim] and
`images` float32 [N,4,3,H,W], already preprocessed exactly as training. Do not use
a differently sized or random workload on the real device to establish fit.
The desktop example uses four decoded frames from the synthetic training smoke
fixture and an explicitly small configuration, 32x32 images and chunk size 3.

Have the hardware owner start the approved stacks separately, without enabling
policy command output. The harness requires exact expected live ROS node names
and hashes the supplied stack configurations. B/C recheck required nodes during
the run. Replace the example node list with the owner-confirmed deployed names:

```bash
python -m oa_embodied_ai.benchmarking.runtime --scenario A --checkpoint /data/act_policy_last.pt --observations /data/observations.npz --device cuda --target-hz 30 --seconds 600 --warmup 100 --output /data/benchmark-A
python -m oa_embodied_ai.benchmarking.runtime --scenario B --checkpoint /data/act_policy_last.pt --observations /data/observations.npz --device cuda --target-hz 30 --seconds 600 --warmup 100 --ros-nodes /confirmed_ros_node --ros-config /data/ros-config.yaml --output /data/benchmark-B
python -m oa_embodied_ai.benchmarking.runtime --scenario C --checkpoint /data/act_policy_last.pt --observations /data/observations.npz --device cuda --target-hz 30 --seconds 600 --warmup 100 --ros-nodes /confirmed_ros_node --moveit-nodes /confirmed_move_group --ros-config /data/ros-config.yaml --moveit-config /data/moveit-config.yaml --output /data/benchmark-C
```

30 Hz is an example argument, **not an accepted project control-rate decision**.
Use the owner-agreed target unchanged in all three scenarios. Confirm the
`workload_sha256` matches A/B/C, then inspect metrics.json, report.md and
tegrastats.log for each. A physical fit decision requires the actual 16 GB Orin
NX identification, full ten-minute sustained runs, verified stack load, matched
inputs and complete relevant memory/thermal evidence. Missing node/tool/device
information or failed measurements cannot establish acceptance.

Actual local example: `desktop-benchmark-example.json`, macOS arm64 / 32 GB RAM,
CPU / Torch 2.15, five seconds / 51 inferences, target 10 Hz, observed 10.164 Hz,
p95 8.447 ms, p99 9.104 ms, max 9.710 ms, zero missed scheduled deadlines.
This proves measurement/reporting execution for this small synthetic model;
it establishes no Jetson capability. Raw metrics retain the exact dirty-source
parent SHA and source-file hashes from execution; they are not relabelled as
clean-final-head evidence. Parameter count: 11,219,888.

Decision memo: **Orin NX workload fit NOT_VERIFIED**. No authorized Jetson access
was found. Arka supplies the real device run; the MoveIt owner supplies the
confirmed concurrent stack/configuration. Mohamed and the designated owners
review on-device versus remote inference after those reports. If the workload
fails the agreed rate/memory/thermal budget, remote inference or a larger module
are options for owner review, not conclusions from desktop timing. No watchdog,
command timeout, replay, interlock or safety behavior is changed.

Additional dependency: psutil >=6,<8, BSD-3-Clause,
https://github.com/giampaolo/psutil. Torch/torchvision/NumPy reuse the declared
training stack; no versions were silently upgraded. Exact executed development
versions are retained under ignored .verification/workstream-20261112.
