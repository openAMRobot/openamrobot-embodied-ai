<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# C10 integration and acceptance checklist

Recorded architecture: runtime C10 lives in this repository's Python package;
C9 schema remains canonical in interfaces (EMBODIED-AI-REPO, 20 October 2026).
This resolves location, not implementation adoption or production acceptance.

- [ ] C10 owner: record **adopt / adapt / reject** in the migration PR thread;
  software lead reviews the integration and reporting interface.
- [ ] Release owner: approve a validated LeRobot release/commit and dependency pins;
  inspected v0.6.1 and baseline 0.4.1 are not approval evidence.
- [ ] Recording/profile owners: approve explicit state/action and recording source
  profiles, names/order/units/meaning/reference frames and authoritative ranges;
  keep arm-only, fixed-mast and lift profiles distinct.
- [ ] Recording owners: approve timestamp mapping, synchronization, freshness,
  measured FPS, repeated-image and action-latency policies from actual evidence.
- [ ] Calibration/configuration owners: supply immutable artifacts, hashes,
  calibration residual conventions and reviewed acceptance criteria.
- [ ] Recording and C10 owners: supply an actual episode revision, native metadata,
  Parquet/videos and matching C9 sidecar; establish real-episode acceptance criteria
  and retain the resulting C10 report. **REAL_EPISODE_NOT_VERIFIED.**
- [ ] Runtime owners: agree recorder-to-C10 invocation and C10-to-C11 responsibilities,
  immutable storage/report provenance and operational deployment ownership.
- [ ] C11 owner: address the [current baseline findings](c11-handoff.md), consume
  the fail-closed interface and retain consistent evidence before training acceptance.
- [ ] Contributor/maintainers: verify applicable contributor agreement and any
  institutional authorization independently of DCO signoff.

Keep both migration and original interfaces PR draft. Any later interfaces-side
conformance extraction needs review; do not close/delete historical work early.
Synthetic test success, hosted CI and registry quality labels do not settle any
unchecked acceptance item. No hardware, real-data, training or production acceptance
is claimed by this migration.
