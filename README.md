# openamrobot-embodied-ai

Embodied AI for OpenAMRobot: episode capture, dataset QA, imitation-learning baselines, evaluation and policy runtime.

## Purpose

This repository holds the embodied AI work for OpenAMRobot:

- Episode capture: a ROS 2 recorder. It is a skeleton today and records nothing until the C9 capture topic contract is approved.
- Dataset QA (C10): release registry validation and read-only C9/native episode quality checks. The migration is draft; real-episode acceptance remains unverified.
- Imitation-learning baseline ACT (C11): planned.
- Evaluation (C14): offline and simulation evaluation, planned.
- Policy runtime: a ROS 2 lifecycle node that runs a policy and publishes base commands. It is a skeleton today.

Current maturity level: Experimental.

## Owners

Team: Embodied AI and Data (Division B).

| Area | Owner | Reviewer |
| --- | --- | --- |
| Repository lead and code owner | Satya Srinivas Anandh (@SatyaA27) | Mohamed Sayed (@panthera-momagdii) |
| C10 dataset QA | Satya Srinivas Anandh | Mohamed Sayed |
| C11 ACT baseline | Marcus Andrei Guinto (@4u-79) | Mohamed Sayed |
| C14 evaluation | owner: to be assigned by Alex | Mohamed Sayed |

Code owners are the lead and Mohamed, so a pull request by the lead can still be approved. One approval from either is enough.

## Layout

| Path | Content |
| --- | --- |
| `ros2/openamrobot_ai_recorder` | Recorder skeleton (ament_python). Validates its configuration, logs the interfaces commit and C9 schema reference, writes no data. |
| `ros2/openamrobot_policy_runner` | Policy runner lifecycle node skeleton (ament_python). |
| `python/oa_embodied_ai/dataset_qa` | Registry `oa-dataset-qa`, episode `oa-c10-check`, and the fail-closed C11 report consumer. |
| `python/oa_embodied_ai/conversion` | README only: capture to LeRobotDataset export, planned. |
| `python/oa_embodied_ai/training` | README only: C11 ACT baseline, planned. |
| `python/oa_embodied_ai/evaluation` | README only: C14 evaluation, planned. |
| `schemas/dataset_registry.schema.json` | Schema for one dataset release entry. |
| `datasets/registry.yaml` | Dataset release registry. |
| `tests` | Tests for the ROS 2 nodes. |
| `docs/interfaces.md` | What this repository consumes and which interfaces are pending. |

## Boundaries

This repository never contains:

- Message or schema definitions, including C9. These live in `openamrobot-interfaces`.
- URDF or frames. These live in `openamr-platform-sw` and `openamr-upperbody-sw`.
- Safety logic, stop interfaces or motor control. These live in the platform repositories.
- Command arbitration. This is `twist_mux` in `openamr-platform-sw`.
- BOM values.
- Copies of project documents.
- Recorded data or model weights.

## Data rule

Datasets and model weights live outside Git. Git holds only registry entries in `datasets/registry.yaml`, each with the commits used during capture and the sha256 of the release. `.gitignore` excludes dataset data, rosbags, `*.mcap`, `*.pt`, `*.safetensors` and `outputs/`.

## Policy safety rule

The policy runner:

- publishes only through approved command inputs: a `geometry_msgs/Twist` on the topic set by `cmd_vel_topic`, intended as a `twist_mux` input. The topic has no default, so it never points at the motor driver by default. Arm output is not implemented and `arm_output_enabled=true` is refused.
- refuses motion activation without the approved software stop interface. Its topic, type and asserted field are parameters with no default; when they are unset, the topic has no publisher, or no stop status has been received yet, activation fails.
- stops on stale observations (`observation_timeout_s`, default 0.2), inference failure or software stop, by publishing a zero Twist and deactivating.
- never downloads model weights at runtime.

## Pending interfaces

| Interface | Owner repository | Status |
| --- | --- | --- |
| C9 capture topics | `openamrobot-interfaces` | Proposed schema only; capture topics not approved. The recorder writes no data until approval. |
| Software stop | `openamr-platform-sw` (PR #35) | Being defined. The runner refuses motion activation until it is configured. |
| Arm command arbitration | `openamrobot-manipulation` | Not defined. The runner has no arm output. |

Details are in [docs/interfaces.md](docs/interfaces.md).

## Repository map

| Concern | Repository |
| --- | --- |
| Messages, services, actions, schemas, C9 contract | `openamrobot-interfaces` |
| AMR ROS 2 software: navigation, control, drivers, URDF and frames, `twist_mux`, software stop | `openamr-platform-sw` |
| AMR firmware: motor and sensor bridges, safety I/O | `openamr-platform-fw` |
| AMR mechanical, electrical, CAD, BOM | `openamr-platform-hw` |
| Arm and lift software, MoveIt, upper body URDF and frames | `openamr-upperbody-sw` |
| End effector firmware, upper body safety I/O | `openamr-upperbody-fw` |
| Lift, plates, wiring, upper body BOM | `openamr-upperbody-hw` |
| Arm framework and arm command arbitration | `openamrobot-manipulation` |
| Project documentation | `openamrobot-docs` |
| Organization policies, contribution rules, CI templates | `.github` |
| Episode capture, dataset QA, ACT baseline, evaluation, policy runtime | `openamrobot-embodied-ai` (this repository) |

## Build and test

ROS 2 Jazzy packages, built together with `openamrobot-interfaces` pinned to the commit used in CI:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --base-paths ros2 external/openamrobot-interfaces/ros2
source install/setup.bash
colcon test --base-paths ros2 external/openamrobot-interfaces/ros2 \
  --packages-select openamrobot_ai_recorder openamrobot_policy_runner
colcon test-result --verbose
python3 -m pytest tests
```

Python tooling:

Check out the pinned external C9 assets first using the [C10 guide](docs/c10/README.md).

```bash
python -m pip install -e "python[test]"
python -m pytest python/tests -v
oa-dataset-qa datasets/registry.yaml
```

For episode/native QA install `python[test,native]` and ffmpeg/ffprobe. See the
[C10 guide](docs/c10/README.md), [audit](docs/c10/integration-audit.md),
[C11 handoff](docs/c10/c11-handoff.md), and [acceptance checklist](docs/c10/acceptance-checklist.md).
A valid registry entry does not establish training readiness. Synthetic tests and
reports do not establish actual recording or production acceptance.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Every commit must be signed off under the DCO (`git commit -s`).

## License

MIT. See [LICENSE](LICENSE).
