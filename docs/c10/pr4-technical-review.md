<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Prepared technical review for PR #4

Target: [PR #4](https://github.com/openAMRobot/openamrobot-embodied-ai/pull/4),
`db199924543d888259fdf87c829481968e2112f3`, incorporating C10
`79ac48e0fda292ddb52596337a745f0b77a35781`. **Prepared for submission; not posted.
Recommendation: changes required; no approval or physical acceptance.**

The custom model has an image-consuming ResNet18 path, but the PR's gate and
acceptance claims are not established by its passing tests. I independently ran
the combined original suite: 114 tests and 32 subtests passed despite these issues.

- **BLOCKER:** `python/oa_embodied_ai/training/gate.py:83–126,129–198` can accept
  arbitrary/stale QA JSON, discovers a working-directory report, does not call
  canonical `training_readiness`, and omits required findings, immutable identity,
  reviewed-policy provenance and per-selected-episode coverage. Missing/empty
  findings and absent exit code do not fail the original gate. Call fresh C10
  native QA on the exact training snapshot, then the canonical consumer with
  independently supplied identity/revision/selection. Preserve exit 2.
- **BLOCKER:** `training/cli.py:227` and `evaluation/cli.py:102` infer
  VERIFIED_PHYSICAL from ROBOT/non-override metadata. Training and offline
  prediction error cannot establish physical operation. All software reports
  must remain NOT_VERIFIED_PHYSICAL / production NOT_VERIFIED.
- **BLOCKER:** `evaluation/rollout_eval.py:109–145` never calls the policy;
  target interpolation fabricates rollout successes. No executed environment
  adapter exists. Report NOT_VERIFIED with no success metric until one exists.
- **IMPORTANT:** `training/dataset.py:81,159,259` ignores native shared
  file-*.parquet layouts/video offsets and replaces missing camera frames with
  zeros. `model.py:267` has a state-only zero-token fallback, and
  `evaluation/offline_eval.py:156` invokes inference without images. Require all
  four canonical streams and native segment bindings; fail on decode errors.
- **IMPORTANT:** `config.py:43` and `docs/c11/state_action_mapping.md:62` disagree
  with C9 adapter names and degree units (including grippers). Historical 19
  has six arm joints plus a gripper per side, not seven plus a gripper. Keep the
  profiles separate; no slicing/padding/conversion or inferred lift action.
- **IMPORTANT:** `training/cli.py:111` reuses training as validation for one
  episode; `evaluation/cli.py:132` can silently evaluate all episodes.
  `model.py:126` omits a CVAE padding mask, and validation uses target-conditioned
  latent. Preserve disjoint actual episode IDs and prior-only offline validation.
- **IMPORTANT:** `evaluation/cli.py:47,177` hashes only Parquet prefixes and
  hardcodes LeRobot 0.4.1. Hash all actual input/checkpoint bytes and report actual
  installed versions. This is a custom ACT-like policy, not verified upstream ACT.

Corrections, executable regressions and the versioned per-channel matrix are
prepared on `satya/c10-c11-integration-review`, retaining both original histories.
See [audit](c11-integration-audit.md), [mapping](robot-profile-compatibility-v1.md)
and [execution evidence](c11-integration-verification.md). The correction also
adds explicit C10 per-episode policy routing because the original single-episode
real-evidence policy cannot cover both training and held-out validation episodes.

Please review/adopt/adapt the changes and decide whether to retain the custom
experimental architecture or propose upstream ACT integration separately.
Recording/release/profile owners must supply actual immutable recordings and
approved evidence. Simulator/hardware rollout execution, upstream LeRobot
interoperability, physical and production acceptance remain NOT_VERIFIED.
Keep the original PRs and integration PR draft.

AI disclosure: OpenAI Codex inspected source and review context, implemented
corrections, authored tests/documents and executed local software verification.
No reviewer approval or real recording was fabricated.
