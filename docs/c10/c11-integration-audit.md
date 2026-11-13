<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# C10/C11 independent integration audit

2 November 2026. Repository `openAMRobot/openamrobot-embodied-ai`.
C10 PR #3: `79ac48e0fda292ddb52596337a745f0b77a35781`.
C11 PR #4: `db199924543d888259fdf87c829481968e2112f3`.
Both GitHub PR refs were fetched and their head SHAs compared with GitHub's PR
metadata. PR #4 already descends from the exact C10 head. An isolated worktree,
branch `satya/c10-c11-integration-review`, retains both histories; no merge
conflict or contributor-branch rewrite was needed. Original C10 checkout was
clean and remains on its original branch. Other workspace repositories were
read-only; existing ignored evidence was retained.

Mohammad's PR #3 approval covers the C10 software migration, not recording or
production acceptance. His 27 October profile review requires separate 16/19
profiles and owner agreement on ordered fields/units/sources. Issue #18's
24 October baseline review concerns the earlier separate baseline repository;
its findings were independently checked against this new PR's actual source.

**Scoped corrections are implemented. Real recording and production acceptance
remain NOT_VERIFIED.** No approved recording/release/model profile was supplied.
The original combined Python suite passed 114 tests and 32 subtests despite the
failures below; those tests were insufficient evidence of safe integration.

All paths below are relative to repository root; abbreviated `training/` and
`evaluation/` paths mean `python/oa_embodied_ai/`. Original line references
refer to the exact PR #4 SHA above; corrected references refer to this branch.

## Requirement matrix

| Requirement | Original implementation | Evidence | Gap | Correction implemented | Executable test |
| --- | --- | --- | --- | --- | --- |
| Canonical C10 decision | Independent summary-field gate | Original `training/gate.py:129,151,177` | No required findings, identity, policy or coverage checks | `training/gate.py:127` reruns C10 then calls canonical `training_readiness`; C10 canonical decision unchanged | `test_c11_gate.py::test_complete_canonical_decision_is_consumed`, startup fault matrix |
| Exact immutable input | Loads arbitrary existing JSON, including working-directory report | Original `gate.py:83,99` | QA can be stale, unrelated or bypassed | Explicit external input manifest + policy; full file hashes and private snapshot, `training/inputs.py:50,97`; optional old report must exactly match fresh QA | Missing/stale report, input/checker/sidecar/policy mutation tests |
| All selected episodes | QA selection unrelated to loader's discovery | Original `training/cli.py:84`, `dataset.py:69` | Gate may cover only one of several loaded episodes | Manifest selection independently drives checker, training/validation and actual native episode IDs | Every-selected-episode diagnostics and noncontiguous split tests |
| No registry substitute | Local QA JSON can be a partial PASS envelope | Original `gate.py:129` | Metadata acceptance mistaken for native QA | Only fresh native QA decision; registry fields never open gate | Registry-only and native-corruption/registry-PASS tests |
| Synthetic exception | Accepts incomplete fabricated PASS reports labelled SYNTHETIC | Original `gate.py:210`, original training test | Bypasses unrelated integrity and origin failures | Real C10 stays blocked; testing exception requires ALL checks PASS except per-episode synthetic `C10_REAL_EPISODE` NOT_VERIFIED | Synthetic raw-decision, missing artifacts, unapproved policy and unavailable native-dependency tests |
| Physical acceptance | Non-override training/offline evaluation becomes VERIFIED_PHYSICAL | Original `training/cli.py:227`, `evaluation/cli.py:102` | Dataset QA/training cannot prove physical execution | All software outputs remain NOT_VERIFIED_PHYSICAL; production NOT_VERIFIED; shared JSON/Markdown renderer | CPU training/evaluation status and report-agreement tests |
| Four cameras required | Missing/decode-failed frames become zeros | Original `dataset.py:259`, `model.py:267` | State-only path is presented as visual experiment | Native camera descriptors/segments required, decode errors propagate, no model fallback | Missing video/decode tests; each camera changes output and receives gradient |
| Native v3 layout | Only episode_*.parquet and guessed MP4 paths | Original `dataset.py:81,159` | Ignores shared file-*.parquet/segment bindings | Native metadata, data_path, episode filtering and video_path + half-open timestamp offsets | Two episodes share one Parquet and four shared MP4 files; offset 2 verified |
| Episode isolation | Single episode reused for validation | Original `training/cli.py:111`, `evaluation/cli.py:132` | Leakage and silent evaluation fallback | At least two episodes; disjoint actual IDs; same snapshot uses held-out IDs, missing IDs block | Split and missing/overlapping/out-of-range checkpoint tests |
| Action padding | Loss masked but CVAE encoder still sees padded actions | Original `model.py:126` | Padding contaminates latent distribution | CVAE key padding mask; validation uses prior without target-conditioned latent | Encoder padding invariance and masked loss tests |
| Rollout evidence | Mock target interpolation ignores policy | Original `rollout_eval.py:109,145` | Generates success counts without policy/environment execution | Removed fake execution; unavailable adapter reports zero executed, no success rate/count/CI | Missing-adapter test and offline/both/rollout evaluations |
| Reproducibility | Hardcoded LeRobot 0.4.1; partial first-4096-byte hash | Original `evaluation/cli.py:47,177` | Fabricated dependency provenance/incomplete content identity | Actual installed versions (null if absent), all-byte path-bound hashes, source/config/manifest/QA evidence, actual checkpoint SHA | Checkpoint byte-hash, config/preprocessing roundtrip, report/environment assertions |
| Profile compatibility | C9 names/units replaced by guessed ROS-style names | Original `config.py:43`, `docs/c11/state_action_mapping.md:62` | Degree/radian/gripper and joint-count errors | Exact C9 names retained; historical 19 unchanged; no width adaptation; [versioned mapping](robot-profile-compatibility-v1.md) | Hash-pinned C9 name/unit comparison, both forward shapes, unsupported mixes |

## Findings and disposition

- **BLOCKER — corrected locally:** competing gate accepted incomplete evidence
  (`training/gate.py:127`; original lines 129–198). Fresh QA and canonical
  readiness now bind exact caller identity/revision/selection and policy. Exit
  code 2 is preserved for blocked training and evaluation.
- **BLOCKER — corrected locally:** physical acceptance was inferred from absence
  of a synthetic flag (`training/cli.py:311`, `evaluation/cli.py:122`). Software
  completion now never grants physical/production acceptance, including real-data
  offline runs. Raw C10 NOT_VERIFIED is preserved in synthetic software evidence.
- **BLOCKER — corrected locally:** rollout success was unrelated to policy
  (`evaluation/rollout_eval.py:56`). No simulator or hardware rollout adapter is
  supplied, so rollout acceptance is explicitly NOT_VERIFIED. The previous
  numerical mock-controller routine is removed, not relabelled as simulation.
- **IMPORTANT — corrected locally:** native artifact ingestion, missing cameras,
  target leakage, padding, incomplete hashes and invented environment versions
  (`training/dataset.py:81,138,198`, `training/model.py:142,219,284`,
  `evaluation/offline_eval.py:179`, `training/evidence.py:13`). Tests execute real
  temporary synthetic MP4 decoding and Torch forward/backward passes on CPU.
- **IMPORTANT — owner decision required:** model is custom ACT-like CVAE + shared
  randomly initialized ResNet18 + Transformer decoder (`training/model.py:41,199`).
  It neither imports nor executes upstream LeRobot ACT. Global pooled camera
  tokens differ from upstream spatial conditioning. This branch adds explicit
  camera identity embeddings and removes implicit state-only inference. C11
  owner must accept this experimental architecture or propose a separately
  reviewed upstream migration, dependency pin and preprocessing/checkpoint plan.
  Checkpoints from the prior architecture are not silently loaded: strict
  state-dict/config validation rejects missing camera-identity parameters.
- **IMPORTANT — owner decision required:** historical 19-channel physical mapping
  has six arm joints and a gripper per side, not seven joints and a gripper
  (`training/config.py:18`; [mapping](robot-profile-compatibility-v1.md)). None
  of the historical numeric limits is promoted to reviewed training policy.
- **MINOR — corrected locally:** image-size configuration was lost in nested
  YAML serialization and final checkpoint called “best” without selection
  (`training/config.py:158,213`, `training/cli.py:278`). Image preprocessing now
  roundtrips and checkpoint is named `act_policy_last.pt`.
- **VERIFIED:** C10 canonical consumer and pinned C9 schema are unchanged; the checker gains
  strict additive per-episode reviewed-policy routing for multi-episode input;
  C11 invokes the canonical API; each camera affects predictions; missing evidence
  stops startup; the bounded synthetic CPU software pipeline completes.
- **NOT_VERIFIED:** actual recording, authentic human policy approvals, validated
  LeRobot release, native upstream LeRobot interoperability, combined-model ROS
  tests in this environment, executed closed-loop rollouts, hardware and production.

## Trust and operational boundaries

The input manifest is a new **local consumer input**, not a C9 schema revision.
It must be supplied by the recording/release owner from an authenticated immutable
snapshot for actual data. A report cannot supply its own expected identity. The
manifest lists every file path and SHA-256, exact dataset ID/revision, selected
episodes, profile, independently known origin, policy hash and evidence reference.
No automatic real-manifest generation or approval is introduced.

Full inventories detect normal input edits, added/removed files, stale policy and
changed bytes anywhere in Parquet, videos, sidecars or configuration artifacts.
Training/evaluation use a temporary private copy and recheck source/snapshot/policy;
training success is published only after snapshot context exit. Symbolic links
are refused. Private copying can require substantial temporary disk space for
real datasets; streaming from authenticated immutable storage is future work.
This is not protection against hostile same-user mutation or proof that an owner
has truthfully classified a dataset. Hashes authenticate consistency, not people,
origins, storage or physical measurements. Real deployment needs external access
control and release-owner approval.

Video alignment follows the same native frame ordinal within C10's validated
half-open segment. It does not establish physical camera exposure synchronization;
that remains the recording policy and C10 timing evidence. Offline evaluation
streams image decoding; vectors/prediction buffers remain episode-sized. Same
snapshot evaluation uses held-out episodes only; a distinct dataset ID may use
its independently gated selection. Content overlap across separately released
datasets must also be excluded by the release owner.

The original C10 policy accepted real-episode evidence for only one episode.
C11's held-out split needs at least two. The checker now supports explicit
`real_episodes` entries keyed by canonical decimal index; the legacy `real_episode`
form is retained. Both forms together or malformed/mismatched entries reject.
Missing selected-episode evidence stays blocked. Seven added policy-routing tests
use fabricated unit metadata and never generate a validated recording report.
The policy extension requires C10/reviewer adoption alongside the consumer fix.

## Owner dependencies

1. C11 owner: review/adopt/adapt these corrections and decide custom versus
   upstream ACT; confirm checkpoint migration and experimental dependencies.
2. C10 + recording owners: supply an actual immutable native snapshot, exact
   episode selection, authenticated origin and reviewed policy/artifact evidence.
3. Release owner: approve a validated LeRobot version/commit and environment;
   test-environment versions and synthetic release objects are not approvals.
4. Recording/calibration owners: approve measured timing, latency, freshness,
   repeated-image criteria, calibration residual conventions and exact hashes.
5. C11 + combined-model owners: confirm per-channel measured/commanded sources,
   units/frames and robot configuration; approve lift observation storage without
   inventing a lift action channel. Software lead reviews the attached mapping.
6. Evaluation/runtime owners: implement an actual policy/environment rollout
   adapter and supply executed outcome evidence before claiming rollout success.
7. Maintainers: keep PRs #3/#4 and this integration draft until owner adoption;
   real-recording and production acceptance are separate later decisions.

See [execution evidence](c11-integration-verification.md),
[PR #4 review draft](pr4-technical-review.md), and
[PR #3 status draft](pr3-integration-status.md). Those comments are prepared for
coordination; they have not been submitted to either contributor's thread.
