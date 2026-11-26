<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# C11 offline evaluation and missing rollout evidence

`oa-c11-eval` consumes a strict checkpoint and reruns the same C10 gate with an
independent input manifest and reviewed policy. It loads all four cameras with
the checkpoint preprocessing configuration. Offline MAE/MSE and chunk-boundary
error are prediction metrics, never task success. Short episodes with no chunk
boundary measurement report null rather than an invented zero.

```bash
oa-c11-eval --checkpoint /run/act_policy_last.pt --dataset-path /immutable/dataset \
  --input-manifest /reviewed/input-manifest.json --policy /reviewed/policy.json \
  --mode offline --output-report /fresh/evaluation.json
```

Same-snapshot evaluation requires the exact training input manifest and disjoint
held-out episode IDs. Missing or overlapping validation selection blocks; it never
falls back to training episodes. A distinct dataset ID uses its independently
gated selection; release owners must ensure external datasets do not duplicate
training data. `--allow-synthetic-eval` has only the narrow software-test semantics
described in the training guide. Gate failures exit 2 before model evaluation.

No policy/environment rollout adapter is supplied. `--mode rollout` and `both`
return explicit NOT_VERIFIED evidence with zero executed rollouts, no success
count/rate/CI and CLI exit 2. The previous routine interpolated synthetic states
toward a target without calling the policy and was not a closed-loop evaluator.
Its fabricated success metrics have been removed. `compute_wilson_ci` is a pure
statistical helper for externally obtained counts, not rollout evidence.

JSON and Markdown come from the same report object. Actual checkpoint bytes,
full dataset inventory, code hashes and installed dependency versions are recorded;
LeRobot is null if absent. Every software report remains NOT_VERIFIED_PHYSICAL
and production NOT_VERIFIED, even for real-data offline evaluation. No hardware
or simulator acceptance is claimed.
