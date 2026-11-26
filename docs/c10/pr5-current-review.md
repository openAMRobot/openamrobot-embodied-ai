<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Canonical C10/C11 review, 22 November 2026

## Review sequence

| PR | Head | Disposition |
|---|---|---|
| [#3 C10 migration](https://github.com/openAMRobot/openamrobot-embodied-ai/pull/3) | `79ac48e` | Exact ancestor of #4 and #5. Retain its [software approval and evidence](https://github.com/openAMRobot/openamrobot-embodied-ai/pull/3#pullrequestreview-5640887450). Mark superseded and close after the owners confirm #5. |
| [#4 C11 baseline](https://github.com/openAMRobot/openamrobot-embodied-ai/pull/4) | `db19992` | Both C11 commits (`d09a546`, `db19992`) and their author sign-offs are exact ancestors of #5. Close as superseded after the C11 owner confirms and credit both commits in #5. |
| [#5 C10/C11 integration](https://github.com/openAMRobot/openamrobot-embodied-ai/pull/5) | `2469ab4` | Retain as the canonical review target. Request full review from the software lead and C11 owner after confirming the path in the [review thread](https://github.com/openAMRobot/openamrobot-embodied-ai/pull/5#pullrequestreview-5840396224). |
| [#6 capture and benchmark](https://github.com/openAMRobot/openamrobot-embodied-ai/pull/6) | `9643da1` | Review the capture delta on top of #5. The second parent is #5's current head. `1aa975b` has the same stable patch ID as #5's `2469ab4` and needs only one review. |
| [#7 C11 visual follow-up](https://github.com/openAMRobot/openamrobot-embodied-ai/pull/7) | `e7993d2` | Review separately on #5. It overlaps the snapshot, gate and configuration changes. Preserve four-camera C9/C10 conformance and explicit C11 camera selection when integrating it. |
| [#8 Baseline 1](https://github.com/openAMRobot/openamrobot-embodied-ai/pull/8) | `cf7f283` | Review the Proposed capture rules on top of #6. Adoption and evidence decisions are tracked in #8 and the linked interface requests. |

Each PR was open and draft at this snapshot. Embodied AI main is `2c0ad9b`.
Its contribution wording update is independent of the stack. Keep the original
PRs open until the owners acknowledge the canonical path. Carry accepted
follow-ups through the stack with ordinary merges and retarget the dependents
as their bases land.

## DCO and CLA

At `2469ab4` the C10 reference, CPU integration, Python and ROS jobs passed.
DCO is `action_required` for the five commits below. The [CLA bot](https://github.com/openAMRobot/openamrobot-embodied-ai/pull/5#issuecomment-7083381912)
reports two pending contributors following the [2.0 activation](https://github.com/openAMRobot/.github/blob/70d3266b8354fe3c4894896efcfb58845d2a1e63/CLA.md).
Each contributor handles their own required agreement acceptance.

The [DCO app](https://github.com/dcoapp/app#individual-remediation-commit-support)
supports a retroactive commit authored and signed by the original contributor.
This route is blocked until the maintainers approve it under the
[organization DCO policy](https://github.com/openAMRobot/.github/blob/70d3266b8354fe3c4894896efcfb58845d2a1e63/DCO.md)
and enable `allowRemediationCommits.individual: true` in the default branch's
`.github/dco.yml`. The file is absent on the inspected main. If approved, the
author creates one signed follow-up commit containing one line of this form
for each exact SHA, using the original author name and email.

```text
I, <original name> <original email in angle brackets>, hereby add my Signed-off-by to this commit: <full SHA>
```

The [app source](https://github.com/dcoapp/app/blob/1f76e915889f250f194c72e896957aaf4ab066f7/lib/dco.js#L136-L166)
matches the complete SHA and original author. The five SHAs in order are

```text
cee3c843c1340e07416e43da129585c3859689c6
6ebc4260b9f4fbf4cd0f004c2025d3aca7cfc387
49822d3c7e415f702cfc005a3ffe6476dbc811b6
638e5f8239012267d97669bcc1d36eb035e2e351
59bb4bf7646d39774513c2c6b8f4cf33bca3786e
```

If maintainers keep the default per-commit behavior, the fallback is a new
replacement draft from `db199924543d888259fdf87c829481968e2112f3`.
The original author cherry-picks and personally signs the five commits above
in order, then cherry-picks signed `a1f92f5` and `2469ab4`. Before integrating
the follow-up fixes, verify that the replacement tree matches `2469ab4` and
that `db19992` is an exact ancestor. The owners can then replay the
dependent changes on new branches, omitting duplicate `1aa975b`. Retargeting
the existing #6 branch to a rewritten base would retain the old DCO omissions.
The fallback requires owner coordination because it creates a replacement stack.
An unsigned `git cherry-pick --no-commit` rehearsal of those seven patches
from `db19992` reproduced exact PR #5 tree
`0ec0d54f3141768a26b6912b08fc954a202d793d`.

## Baseline and follow-up

C11 inventories the dataset before and after staging, runs C10 against the
private copy, and uses the same files and manifest selection for training or
evaluation. The proposal preserves the manifest and policy formats and detects
manifest changes before dispatch. A supplied report must equal the fresh report
as canonical JSON, including numeric and Boolean types. A passing report with
conflicting origin, production status or finding severity blocks the gate.

Exact dataset revision and selected episode indices come from the independent
input manifest. C10 uses reviewed `native_identity` for identifiers absent
from the writer's native metadata and rejects conflicting legacy fields.
The synthetic C11 exercise now uses native metadata without injected revision
or episode IDs. A known simulation or synthetic converter origin prevents a
`C10_REAL_EPISODE` pass even with a v1 `actual` sidecar and a matching policy claim.
The manifest owner's external evidence remains necessary to authenticate origin.

C11 loads the four canonical camera keys from their native video segments.
The explicit 16-channel adapter and historical fixed-mast 19-channel profiles
remain distinct. The proposed mobile-lift profile remains unsupported by C11.
A sample at index `t` targets `action[t]` through `action[t + K - 1]` with an
episode-end mask. No automatic timestamp shift occurs. A new recording action
convention needs a C11 owner decision about which command the first target
represents, with an alignment regression before adoption.

The follow-up commits carry contributor sign-off. The five historical
commits and contributor agreement still have their own acceptance gates.
The tests exercise software and synthetic data. Hardware, actual recording,
rollout, release and production evidence remain outstanding.

## Local verification

With pinned C9 assets and the C11 CPU environment, `python -m pytest -q
python/tests python/native_tests` passed 247 tests plus 29 subtests.
`python python/tools/prove_c11_negative.py` proved all three controls. A
disposable #7 worktree applied the proposal and retained its video-capture
cache cleanup in the snapshot context. Its Python suite passed 190 tests
plus 17 subtests. #7 has its own review path and new CI after integration.
