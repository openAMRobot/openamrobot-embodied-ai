<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# C10 check-to-corrupted-fixture acceptance matrix

Generated from the named cases in `python/tests/test_c10_dataset.py` and
`python/native_tests/test_c10_native.py`. Core corruption cases remain valid against the
unchanged C9 Draft 2020-12 schema. Wrapper cases deliberately corrupt native
artifacts or exercise missing/contradictory evidence. No real recording is altered.

| Stable diagnostic | Deliberate corruption / missing-evidence case | Expected | Named unittest |
| --- | --- | --- | --- |
| `C10_ACTION_BOUNDS` | `action_range` | REJECT | `C10SemanticTests.test_reject_action_range` |
| `C10_ACTION_LATENCY` | `action_timestamp_semantics` | REJECT | `C10SemanticTests.test_reject_action_timestamp_semantics` |
| `C10_ACTION_LATENCY` | `action_value_semantics` | REJECT | `C10SemanticTests.test_reject_action_value_semantics` |
| `C10_ACTION_LATENCY` | `command_latency` | REJECT | `C10SemanticTests.test_reject_command_latency` |
| `C10_BINDING` | `binding_revision` | REJECT | `C10SemanticTests.test_reject_binding_revision` |
| `C10_BINDING` | `episode_length` | REJECT | `C10SemanticTests.test_reject_episode_length` |
| `C10_BINDING` | `external_native_identity_contradiction` | REJECT | `C10SemanticTests.test_reject_external_native_identity_contradiction` |
| `C10_BINDING` | `format_version` | REJECT | `C10SemanticTests.test_reject_format_version` |
| `C10_BINDING` | `orphan_native_episode` | REJECT | `C10SemanticTests.test_reject_orphan_native_episode` |
| `C10_COMPLETENESS` | `interrupted_recording` | REJECT | `C10SemanticTests.test_reject_interrupted_recording` |
| `C10_COMPLETENESS` | `success_with_abort_event` | REJECT | `C10SemanticTests.test_reject_success_with_abort_event` |
| `C10_COVERAGE` | `duplicate_frame` | REJECT | `C10SemanticTests.test_reject_duplicate_frame` |
| `C10_COVERAGE` | `frame_coverage` | REJECT | `C10SemanticTests.test_reject_frame_coverage` |
| `C10_COVERAGE` | `native_frame_index` | REJECT | `C10SemanticTests.test_reject_native_frame_index` |
| `C10_EVENTS` | `duplicate_event_id` | REJECT | `C10SemanticTests.test_reject_duplicate_event_id` |
| `C10_EVENTS` | `event_missing_frame` | REJECT | `C10SemanticTests.test_reject_event_missing_frame` |
| `C10_EVENTS` | `event_outside_recording` | REJECT | `C10SemanticTests.test_reject_event_outside_recording` |
| `C10_EVENTS` | `inverted_event` | REJECT | `C10SemanticTests.test_reject_inverted_event` |
| `C10_FEATURES` | `camera_fps_descriptor` | REJECT | `C10SemanticTests.test_reject_camera_fps_descriptor` |
| `C10_FEATURES` | `feature_shape` | REJECT | `C10SemanticTests.test_reject_feature_shape` |
| `C10_FEATURES` | `legacy_camera_aliases` | REJECT | `C10SemanticTests.test_reject_legacy_camera_aliases` |
| `C10_FEATURES` | `native_vector_name_order` | REJECT | `C10SemanticTests.test_reject_native_vector_name_order` |
| `C10_FEATURES` | `native_vector_shape` | REJECT | `C10SemanticTests.test_reject_native_vector_shape` |
| `C10_HEALTH` | `fresh_reused_sample` | REJECT | `C10SemanticTests.test_reject_fresh_reused_sample` |
| `C10_HEALTH` | `fresh_without_receipt` | REJECT | `C10SemanticTests.test_reject_fresh_without_receipt` |
| `C10_MAPPING` | `affine_arithmetic` | REJECT | `C10SemanticTests.test_reject_affine_arithmetic` |
| `C10_MAPPING` | `mapping_validity_interval` | REJECT | `C10SemanticTests.test_reject_mapping_validity_interval` |
| `C10_PROFILE` | `profile_dimension_order` | REJECT | `C10SemanticTests.test_reject_profile_dimension_order` |
| `C10_PROFILE` | `profile_units` | REJECT | `C10SemanticTests.test_reject_profile_units` |
| `C10_SELECTION` | `selected_time_disagrees_with_mapping` | REJECT | `C10SemanticTests.test_reject_selected_time_disagrees_with_mapping` |
| `C10_SELECTION` | `selection_source` | REJECT | `C10SemanticTests.test_reject_selection_source` |
| `C10_SEQUENCE` | `nonconsecutive_source_id_reuse` | REJECT | `C10SemanticTests.test_reject_nonconsecutive_source_id_reuse` |
| `C10_SEQUENCE` | `sequence_reset` | REJECT | `C10SemanticTests.test_reject_sequence_reset` |
| `C10_SKEW` | `reported_skew` | REJECT | `C10SemanticTests.test_reject_reported_skew` |
| `C10_STREAM_QUALITY` | `measured_source_fps` | REJECT | `C10SemanticTests.test_reject_measured_source_fps` |
| `C10_STREAM_QUALITY` | `sequence_gap` | REJECT | `C10SemanticTests.test_reject_sequence_gap` |
| `C10_STREAM_QUALITY` | `stale_observation_age` | REJECT | `C10SemanticTests.test_reject_stale_observation_age` |
| `C10_SYNC` | `declared_policy_tolerance_mismatch` | REJECT | `C10SemanticTests.test_reject_declared_policy_tolerance_mismatch` |
| `C10_SYNC` | `inter_observation_skew` | REJECT | `C10SemanticTests.test_reject_inter_observation_skew` |
| `C10_TASK` | `orphan_native_task` | REJECT | `C10SemanticTests.test_reject_orphan_native_task` |
| `C10_TASK` | `task_text` | REJECT | `C10SemanticTests.test_reject_task_text` |
| `C10_TIME_ORDER` | `descending_authoritative_time` | REJECT | `C10SemanticTests.test_reject_descending_authoritative_time` |
| `C10_TIME_ORDER` | `timestamp_order` | REJECT | `C10SemanticTests.test_reject_timestamp_order` |
| `C10_VECTOR_VALUES` | `vector_infinity` | REJECT | `C10SemanticTests.test_reject_vector_infinity` |
| `C10_VECTOR_VALUES` | `vector_nonfinite` | REJECT | `C10SemanticTests.test_reject_vector_nonfinite` |
| `C10_VECTOR_VALUES` | `vector_width` | REJECT | `C10SemanticTests.test_reject_vector_width` |
| `C10_ARTIFACT` | `absent_configuration_artifact` | NOT_VERIFIED | `C10WrapperTests.test_wrapper_absent_configuration_artifact` |
| `C10_ARTIFACT` | `artifact_bytes_changed` | REJECT | `C10WrapperTests.test_wrapper_artifact_bytes_changed` |
| `C10_ARTIFACT` | `configuration_declared_hash` | REJECT | `C10WrapperTests.test_wrapper_configuration_declared_hash` |
| `C10_ARTIFACT` | `configuration_symlink_escape` | REJECT | `C10WrapperTests.test_wrapper_configuration_symlink_escape` |
| `C10_CALIBRATION` | `calibration_identity` | REJECT | `C10WrapperTests.test_wrapper_calibration_identity` |
| `C10_CALIBRATION` | `calibration_residual_exceeds` | REJECT | `C10WrapperTests.test_wrapper_calibration_residual_exceeds` |
| `C10_CALIBRATION` | `unapproved_calibration` | NOT_VERIFIED | `C10WrapperTests.test_wrapper_unapproved_calibration` |
| `C10_INPUT` | `duplicate_json_key` | REJECT | `C10WrapperTests.test_wrapper_duplicate_json_key` |
| `C10_INPUT` | `malformed_nested_policy` | REJECT | `C10WrapperTests.test_wrapper_malformed_nested_policy` |
| `C10_NATIVE_DATA` | `native_row_range` | REJECT | `C10WrapperTests.test_wrapper_native_row_range` |
| `C10_NATIVE_DATA` | `orphan_native_episode_rows` | REJECT | `C10WrapperTests.test_wrapper_orphan_native_episode_rows` |
| `C10_NATIVE_DATA` | `truncated_parquet` | REJECT | `C10WrapperTests.test_wrapper_truncated_parquet` |
| `C10_NATIVE_METADATA` | `duplicate_native_episode` | REJECT | `C10WrapperTests.test_wrapper_duplicate_native_episode` |
| `C10_PARQUET_DTYPE` | `native_double_vectors` | REJECT | `C10WrapperTests.test_wrapper_native_double_vectors` |
| `C10_PROFILE` | `source_profile_identity` | REJECT | `C10WrapperTests.test_wrapper_source_profile_identity` |
| `C10_REAL_EPISODE` | `synthetic_is_not_real_evidence` | NOT_VERIFIED | `C10WrapperTests.test_wrapper_synthetic_is_not_real_evidence` |
| `C10_RELEASE` | `recording_release_commit` | REJECT | `C10WrapperTests.test_wrapper_recording_release_commit` |
| `C10_RELEASE` | `unapproved_release` | NOT_VERIFIED | `C10WrapperTests.test_wrapper_unapproved_release` |
| `C10_SCHEMA` | `invalid_c9_schema` | REJECT | `C10WrapperTests.test_wrapper_invalid_c9_schema` |
| `C10_VIDEO` | `decoded_video_frame_count` | REJECT | `C10WrapperTests.test_wrapper_decoded_video_frame_count` |
| `C10_VIDEO` | `frozen_decoded_pixels` | REJECT | `C10WrapperTests.test_wrapper_frozen_decoded_pixels` |

Additional explicit tests cover clean semantic PASS, independent fixture objects,
inclusive action/synchronization/clock-map boundaries, ties-to-even mapping,
exact JSON/YAML fractional clock lexemes, pending tolerance with an approved
policy, pending/unmapped clocks, honest missing/dropped slots, stale producer
labels requiring review, missing episode identity, source-profile identity,
explicit action reference offsets, unavailable native data, native metadata
totals, strict JSON and nonfinite vectors, CLI exits and unchanged dataset bytes.

`test_every_core_diagnostic_has_named_corruption` fails if a semantic diagnostic
is added without a named corruption. All CLI diagnostic families also appear
above. The acceptance matrix proves diagnostic-specific fault detection, not
independent correctness of every algorithm or physical acceptance.

`test_real_video_decode_or_explicit_unavailable` decodes a temporary synthetic
MP4 and corrupts its bytes when ffmpeg/ffprobe are available. Without them it
explicitly asserts NOT_VERIFIED and prints which branch was exercised. It does
not use unittest skip. The decoded-frame-count and frozen-pixel tests use a
deterministic subprocess double and are distinguished from actual decoding.

The revert demonstration uses `python python/tools/prove_c10_negative.py`: one unchanged
negative test passes on current code, then fails with exactly one assertion
failure after the arithmetic comparison is reverted in a temporary source copy.
The original checkout and fixtures remain intact.

The migrated proof additionally mutates the downstream readiness decision; the unchanged synthetic-origin consumer test must fail exactly once. New handoff tests cover blocked verdicts, incomplete diagnostics, policy/contract provenance and snapshot mismatches.
