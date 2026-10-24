<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# C10 source and dependency provenance

Source implementation: [interfaces PR #25](https://github.com/openAMRobot/openamrobot-interfaces/pull/25),
head `34fd6de3fa9735ee550c2da4da28b48e31bc1f9d`.
Software retains MIT SPDX headers and the source repository MIT licence applies.
Documentation derived from that work retains CC BY 4.0 notices under organization policy.
This destination's LICENSE, ownership files and accepted C9 definitions are unchanged.

| Source | Destination | Adaptation |
| --- | --- | --- |
| `tools/check_c10_dataset.py` | `python/oa_embodied_ai/dataset_qa/episode.py` | Package-relative imports, canonical schema lookup, importable CLI arguments, additive report provenance/source context |
| `tools/c10_semantics.py` | `python/oa_embodied_ai/dataset_qa/semantics.py` | Byte-identical semantic implementation |
| `tools/lint_schemas.py` strict reader functions | `python/oa_embodied_ai/dataset_qa/strict_io.py` | Extracted loader only; no ROS or schema-lint runtime |
| `tools/generate_c10_evidence.py` | `python/tools/generate_c10_evidence.py` | Package module invocation, source inventory and test fixture paths |
| `tools/prove_c10_negative.py`, nonempty test runner | `python/tools/` | Package imports/paths; additional fail-closed handoff mutation proof |
| Core/native/evidence suites and fixture | `python/tests/`, `python/native_tests/` | Imports, canonical asset paths, CLI argv, workflow paths; original 63/31/23 assertions retained |
| Fixture matrix and runtime guide | `docs/c10/` | Paths, current architecture, new API, current evidence and limitations |

New: `contract.py`, `handoff.py`, handoff regressions, reference-only audit helper,
integration/acceptance documentation and migrated workflow integration. Basic
registry validator, CLI, schema, registry and its nine tests are unchanged.

Canonical C9 consumer pin: interfaces `87ab8a936c97ab55ee74b8988ca3d7659160ea6a`,
the destination's existing ROS dependency pin. Accepted schema SHA-256
`692e04693bc738033af1da10f98e029f342497ddaeae4792945f829335f1938e` and
example `e21218b4cd4ce09969ca589cc83e72d46bfa99eee9e197195a56ec1ad22a3e28`
also match current interfaces main at audit time. Assets stay external and are
verified before use. No baseline `lerobot` release is installed or treated as approved.

## Dependencies

* Existing PyYAML (MIT, [upstream](https://github.com/yaml/pyyaml)) and jsonschema
  (MIT, [upstream](https://github.com/python-jsonschema/jsonschema)) are retained,
  with no new version constraints on basic registry installs.
* Existing test extra retains pytest (MIT, [upstream](https://github.com/pytest-dev/pytest)).
* New optional `native` extra: `pyarrow>=17,<27` (Apache-2.0,
  [Apache Arrow](https://github.com/apache/arrow)), preserving the reviewed source range.
* External ffmpeg/ffprobe ([source and licence](https://ffmpeg.org/legal.html)):
  system-installed media verification tools, LGPL/GPL depending on build. No binaries
  are vendored or required for basic registry installs.
* Standard library supplies evidence, hashing and probes; no Torch, ROS, Gazebo,
  LeRobot training dependency or baseline package is added.

No baseline source licence was present at the audited head; it is an inspection
reference only. References to 16/19/lift profiles, provisional timing and inspected
releases describe distinctions, not new approvals. Material OpenAI Codex assistance
produced this migration, added tests, tooling, workflow and documentation. Source
diffs, local execution and mutation controls were reviewed during this run; human
adoption and real-data review remain outstanding. No contributor-agreement or
institutional authorization is inferred from Git identity or a successful test.
