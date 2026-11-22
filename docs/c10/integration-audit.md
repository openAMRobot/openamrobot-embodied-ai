<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# C10 integration audit — 24 October 2026

Audit before implementation. Destination base `56bbd36ea313f3765df51524df07ccc6b2623f47`;
interfaces PR #25 head `34fd6de3fa9735ee550c2da4da28b48e31bc1f9d` (still open/draft);
accepted C9 merge `fa7c438e33bd807c174fa0747119d7d627faa3bf`, current interfaces main
`d048fb84394fc3ae59e3007994a65dc1291f46ac`; baseline `1390ae798f300754303136fae9b3557db025ed0d`;
governance `b1c8700036671812c57028901d7b6ffce4ff9e35`.
No open destination PRs at audit time. Governance issue #54 and recorded decision
EMBODIED-AI-REPO supersede the original runtime-location uncertainty in PR #25.
Neither approves numeric policies, release pins or real recordings.

| Area | Destination | Reviewed C10 source / overlap | Minimal migration / missing evidence |
| --- | --- | --- | --- |
| Registry | YAML release entries, JSON Schema, duplicate IDs, CLI, nine tests | Release provenance checks operate on recording artifacts, not registry entries | Retain validator/API/exit codes unchanged; document that `quality_status: passed` does not verify an episode |
| C9 structure | Registry holds C9 reference only | Read-only sidecar checker, duplicate/nonfinite input protection | Package checker and strict loader; consume pinned canonical schema externally, never define another C9 schema |
| Native artifacts | None | Native task/episode/features, Parquet dtype/coverage, video decode/count/pixel evidence | Reuse tested implementation; optional PyArrow and system ffmpeg/ffprobe; no LeRobot training stack |
| Semantics | None | 18 semantic families: mapping, synchronization, sequence/freshness/FPS, vectors/profiles/bounds, actions, outcomes/events | Retain exact decimal clock arithmetic and diagnostic identifiers; no guessed thresholds |
| Provenance | Release metadata only | Source, configuration/calibration byte hashes, release and real-episode evidence | Reuse; add report identity/source context and fail-closed downstream consumer |
| Fixtures/tests | Nine registry tests | 63 core, 31 native, 23 evidence tests; synthetic fixture and arithmetic revert proof | Port without weakening assertions; keep native suites separately runnable, no dependency-based skips |
| Evidence/CI | Independent Python/ROS jobs; no episode artifacts | CLI manifest, exact revision/hash checks, failure-preserving logs/upload | Retain evidence behavior under Python tooling; retain separate basic and native jobs |
| C11 | Planned modules only | Baseline has separate validators and training gate | Reference-only audit; no baseline source copied; expose one C10 API/report handoff |

## Compatibility and owner dependencies

* C9 camera keys are wrist left/right, `head`, `base`. Current baseline metadata has
  aligned keys, but its validators retain historical aliases. No alias profile is approved here.
* C9 arms-only 16-value degree profile, historical fixed-mast 19-channel profile, and
  lift geometry remain separate. Exact ordered names, units, semantics, reference frames,
  source profiles and bounds need owner evidence; a channel count cannot select a profile.
* Baseline `lerobot==0.4.1` and 30 ms are provisional baseline settings, not project
  acceptance policy. C9 inspected v0.6.1 is also not a validated pin.
* Native sampling timestamps do not prove acquisition synchronization. C10 recomputes
  explicit clock mappings and observation skew; action latency has separate semantics.
* Missing native artifacts/policy/calibration/source evidence remain NOT_VERIFIED;
  verified contradictions remain REJECT. A synthetic reporting-path pass cannot open training.
* Adoption, release/profile/timing policy, real-episode acceptance and operational integration
  remain owner decisions. No real episode supplied. Runtime repository ownership is resolved.

## Reuse and licence boundary

Port source and tests from interfaces PR #25 under MIT, preserving SPDX notices;
adapt package imports, contract lookup, paths and CI only where needed. Extract the
duplicate-key/nonfinite loader from its existing MIT schema linter, without ROS code.
Documentation derived from PR #25 is CC BY 4.0 under organization policy.
Baseline has no tracked source licence at the audited head: inspect only, copy no code
or datasets/checkpoints. Its eight-fixture corruption harness remains a separate reference.
The baseline's current ingestion/training/evaluation findings are recorded in the C11 handoff.

Sources: [ownership issue](https://github.com/openAMRobot/.github/issues/54),
[C9 acceptance and current baseline review](https://github.com/openAMRobot/openamrobot-interfaces/issues/18),
[C9 PR](https://github.com/openAMRobot/openamrobot-interfaces/pull/16),
[C10 source and reviews](https://github.com/openAMRobot/openamrobot-interfaces/pull/25).
