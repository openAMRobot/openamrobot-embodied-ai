<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# C10 → C11 integration handoff

Reference: `4u-79/openamrobot-c-baseline` main
`1390ae798f300754303136fae9b3557db025ed0d`, inspected 24 October 2026.
No baseline source licence was found among tracked files. No source, dataset or
checkpoint is copied into this repository. Reference work remains separate.

| Finding at this head | Evidence | C11 collaborator action |
| --- | --- | --- |
| Unsupported ingest becomes synthetic with requested ROBOT label | `scripts/generate_or_ingest_dataset.py:411–428`; a temporary metadata-only bag directory dispatched to the mocked synthetic generator with `source_profile=ROBOT`, ten episodes | Fail explicitly on unsupported input; always retain synthetic origin independently of source-profile labels |
| Custom policy consumes only state/action | `scripts/train_act_baseline.py:243–316`: forward takes state; ACTDataset reads only observation.state/action; camera settings/backbone are not consumed here | Scope as a state-only experiment or implement the separately approved, pinned vision/LeRobot ACT path; this migration makes no training architecture change |
| Prediction error is labelled rollout success | `scripts/evaluate_act_baseline.py:222–248`: predictions use recorded states; success is MAE <0.65 plus a recorded terminal phase | Name the metric offline prediction error; provide separately executed task outcomes and rollout evidence |
| BLOCKED becomes report PASS | `scripts/evaluate_act_baseline.py:414–421,483–487`; mocked payload with BLOCKED yields `BLOCKED | PASS` and unconditional acceptance text | Derive all JSON/Markdown statuses from the same validated gate; preserve blocks and test contradictory evidence |
| Training gate also accepts contradictory summary | `scripts/train_act_baseline.py:345–348`: OPEN **or** overall PASS; probe with BLOCKED/PASS returns true | Use the C10 handoff decision and require consistent complete evidence; no OR fallback |
| Evidence provenance disagrees | `reports/c11_evidence_summary.md` checkpoint prefix `26703aa8…` versus `reports/c11_eval_report.json` `068fd441…`; JSON Torch 2.14.1+cpu differs from requirements 2.5.1 | Regenerate one consistent report package from actual execution with matching checkpoint/environment hashes |

The first and fourth findings from the current issue #18 review remain present;
the second and third are confirmed by source inspection. Camera keys in current
`meta/info.json` now match C9; historical aliases still appear in integrity code
and tracked video paths. The fixed-mast profile and lift proposal are separated
in documentation. These fixes do not establish native C9 sidecar compatibility,
approved thresholds/release pins or actual recording acceptance.

Reproduce the limited probes without Torch or writing to the baseline:

```bash
python python/tools/audit_c11_reference.py ../openamrobot-c-baseline \
  > .verification/c10-migration/baseline-probes.json
```

The helper requires that exact clean SHA. It executes the named functions from
the reference checkout with mocks, temporary outputs and postponed annotations.
It generates no recording and runs no full ingestion, training, evaluation,
checkpoint verification, simulation or hardware. It is diagnostic evidence, not
a C11 test-suite or pipeline pass. No full baseline test run is claimed.

## Minimal consumer interface

Use `oa_embodied_ai.dataset_qa.episode.check_dataset` on the immutable training
input, then `dataset_qa.training_readiness(report, dataset_id=…, dataset_revision=…,
episode_indices=…)`. Registry metadata may locate a release but cannot open the
training gate. Report version 1 uses additive identity/provenance fields documented
in the [runtime guide](README.md); legacy/incomplete reports fail closed.

The consumer requires consistent PASS / integer exit 0 / training_ready true,
nonempty all-PASS findings with required diagnostic coverage for every selected
episode, canonical contract and reviewed-policy provenance, exact dataset revision
and episode selection, actual origin and passing real-episode evidence. REJECT,
NOT_VERIFIED, BLOCKED, malformed reports, missing diagnostics and synthetic origin
remain blocked. Status precedence preserves REJECT. A passing synthetic test job
is reporting-path evidence, never recording acceptance.

The consumer does not authenticate arbitrary JSON or human approval claims. C11
should rerun the checker before loading the immutable input, retain its checker
commit/report hash/policy identity/native revision, and avoid reusing stale reports
after any input mutation. Pinning/authentication of external storage and release
approval remain owner responsibilities; no new release policy is introduced.

Integration collaborator: the C11 owner. Software-lead review is required for the
handoff and accepted profiles; C10 does not change or replace C11/C14 code.
