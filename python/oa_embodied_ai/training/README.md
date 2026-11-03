<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# C11 experimental custom visual imitation

This is a custom ACT-like CVAE/ResNet18/Transformer implementation. It does not
execute upstream LeRobot ACT. Native metadata/media are read directly; upstream
LeRobot compatibility and any validated release remain NOT_VERIFIED.

Install from repository root with `python -m pip install -e 'python[test,native,training]'`
and provide ffmpeg/ffprobe. No ROS, simulator, GPU or robot is needed for tests.
The CPU-only CI uses PyTorch's CPU package index. Dependency constraints are
experimental software requirements, not an approved production release policy.

Training requires a reviewed C10 policy and independently supplied input manifest:

```bash
oa-c11-train --dataset-path /immutable/dataset --policy /reviewed/policy.json \
  --input-manifest /reviewed/input-manifest.json --config /reviewed/training.yaml \
  --output-dir /fresh/output
```

The manifest is external to the dataset, contains `manifest_version: 1`, exact
`dataset_id`, `dataset_revision`, `episode_indices`, `profile`, `data_origin`
(actual or synthetic), nonempty `evidence`, `policy_sha256` (hex digest) and
`files_sha256` (every relative file path mapped to its full hex digest).
Release/recording owners provide identity/origin/policy bindings for actual data.
Do not derive them from a QA report or use a synthetic manifest for a recording.

C11 verifies the complete snapshot, copies it privately, invokes `check_dataset`
with that policy and selection, then calls canonical `training_readiness` with
independent expected identity. Missing, contradictory, stale or incomplete QA
blocks with exit 2. Registry quality labels cannot substitute. `--qa-report`,
if explicitly supplied, is only an exact comparison with fresh QA; no report
is discovered automatically. Output must be fresh and outside the dataset.

`--allow-synthetic-eval` permits software testing only when ALL diagnostics pass
except C10_REAL_EPISODE, which must say synthetic/NOT_VERIFIED for every selected
episode. Raw C10 readiness remains false, exit 2 and NOT_VERIFIED. The flag never
permits other missing/rejected evidence. `--smoke-test` forces CPU, two epochs
and at most two batches per training/validation epoch.

Four canonical camera keys in order: head, base, camera_wrist_left,
camera_wrist_right (each prefixed `observation.images.`). No camera alias,
missing-video zero substitute, or state-only fallback exists. Native `data_path`,
`video_path`, shared-file episode IDs and video segment offsets drive loading.
Profiles require exact names/dimensions; no padding, truncation, unit coercion or
lift action is introduced. See [the mapping](../../../docs/c10/robot-profile-compatibility-v1.md).

At least two selected episodes are needed for a disjoint validation split. Target
padding is masked in both CVAE encoder and loss; validation predicts from the
prior without target-conditioned latent. Checkpoints retain configuration,
preprocessing size, actual split IDs, input manifest/QA hashes and actual code
and dependency provenance. The final checkpoint is `act_policy_last.pt`.

All software summaries remain `NOT_VERIFIED_PHYSICAL`; production acceptance
is `NOT_VERIFIED`. Training is not physical validation. A bounded, honest
synthetic evidence bundle can be generated from the source tree:

```bash
python python/tools/run_c11_smoke.py --output-dir .verification/fresh-c11-smoke
```
