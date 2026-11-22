# SPDX-License-Identifier: MIT
"""Read-only C9 cross-field checks. No physical acceptance thresholds are built in.

The caller structurally validates the sidecar first and supplies decoded native
records. An explicitly supplied policy uses ``approved: true``, ``id`` and
``evidence``; these identify the caller's reviewed policy, not an approval by
this program. Missing policy/evidence is NOT_VERIFIED, never an implicit pass.
"""

from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import math
import struct


CHECK_IDS = (
    "C10_BINDING", "C10_FEATURES", "C10_TASK", "C10_COVERAGE",
    "C10_TIME_ORDER", "C10_MAPPING", "C10_SELECTION", "C10_HEALTH",
    "C10_SEQUENCE", "C10_SKEW", "C10_SYNC", "C10_STREAM_QUALITY",
    "C10_VECTOR_VALUES", "C10_PROFILE", "C10_ACTION_BOUNDS",
    "C10_ACTION_LATENCY", "C10_EVENTS", "C10_COMPLETENESS",
)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _decimal(value):
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("nonfinite decimal")
    return result


def _mapped(raw, mapping):
    """Use decimal input values, sufficient exact precision, ties to even."""
    raw, origin, scale, offset, start, end = (
        _decimal(raw), _decimal(mapping["source_origin"]),
        _decimal(mapping["scale_ns_per_source_unit"]),
        _decimal(mapping["authoritative_origin_ns"]),
        _decimal(mapping["validity"]["source_start"]),
        _decimal(mapping["validity"]["source_end"]),
    )
    if not start <= raw <= end or scale <= 0:
        raise ValueError("outside inclusive validity or nonpositive scale")
    # Decimal's default 28 digits can lose nanoseconds at large source origins.
    with localcontext() as ctx:
        ctx.prec = sum(len(x.as_tuple().digits) + abs(x.as_tuple().exponent)
                       for x in (raw, origin, scale, offset)) + 20
        value = (offset + scale * (raw - origin)).to_integral_value(rounding=ROUND_HALF_EVEN)
    if value < 0:
        raise ValueError("negative recording time")
    return int(value)


def check_episode(sidecar, native, policy, emit):
    """Check a schema-valid sidecar against native metadata and optional rows.

    ``native`` contains info, episode, tasks (index -> instruction), and rows
    (list or None). Returns measured metrics only; diagnostic verdicts go to
    ``emit(check_id, status, observed=..., expected=..., ...)``.
    """
    binding, robot = sidecar["native_lerobot"], sidecar["openamrobot"]
    info, episode = native.get("info", {}), native.get("episode", {})
    rows, tasks = native.get("rows"), native.get("tasks", {})
    streams, samples = robot["streams"], robot["samples"]
    features, task = binding["features"], binding["task"]
    policy = policy or {}
    approved = (policy.get("approved") is True and bool(policy.get("id"))
                and bool(policy.get("evidence")))
    policy_ref = {"id": policy.get("id"), "evidence": policy.get("evidence")}

    def check(code, valid, observed, expected, **context):
        emit(code, "PASS" if valid else "REJECT", observed=observed,
             expected=expected, fix="Correct the source evidence and regenerate the recording sidecar.", **context)

    def pending(code, observed, expected, **context):
        emit(code, "NOT_VERIFIED", observed=observed, expected=expected,
             fix="Supply the missing native evidence or an explicitly reviewed policy with evidence.", **context)

    expected_binding = {"dataset_id": binding["dataset_id"],
                        "dataset_revision": binding["dataset_revision"],
                        "format_version": binding["format_version"],
                        "episode_id": binding["episode_id"],
                        "episode_index": binding["episode_index"], "length": binding["length"]}
    observed_binding = {"dataset_id": info.get("repo_id", info.get("dataset_id")),
                        "dataset_revision": info.get("dataset_revision"),
                        "format_version": info.get("codebase_version"),
                        "episode_id": episode.get("episode_id"),
                        "episode_index": episode.get("episode_index"), "length": episode.get("length")}
    identity = policy.get("native_identity", {})
    if approved and isinstance(identity, dict) and identity.get("evidence"):
        for key in ("dataset_id", "dataset_revision"):
            if key in identity:
                check("C10_BINDING", identity[key] == expected_binding[key], identity[key],
                      expected_binding[key], policy=policy_ref)
            if observed_binding[key] is None:
                observed_binding[key] = identity.get(key)
        episode_identity = identity.get("episode_ids", {}).get(str(binding["episode_index"]))
        if episode_identity is not None:
            check("C10_BINDING", episode_identity == binding["episode_id"], episode_identity,
                  binding["episode_id"], policy=policy_ref)
            observed_binding["episode_id"] = episode.get("episode_id", episode_identity)
    if "episode_id" in episode:
        expected_binding["episode_id"] = binding["episode_id"]
        observed_binding["episode_id"] = episode["episode_id"]
    missing_binding = [key for key, value in observed_binding.items() if value is None]
    known_mismatch = any(value is not None and value != expected_binding[key]
                         for key, value in observed_binding.items())
    if missing_binding and not known_mismatch:
        pending("C10_BINDING", observed_binding, expected_binding)
    else:
        check("C10_BINDING", observed_binding == expected_binding, observed_binding, expected_binding)

    check("C10_FEATURES", info.get("fps") == binding["fps"], info.get("fps"), binding["fps"], units="Hz")
    native_features = info.get("features", {})
    for key, descriptor in features.items():
        actual = native_features.get(key, {})
        expected = {"dtype": descriptor["dtype"], "shape": descriptor["shape"]}
        observed = {"dtype": actual.get("dtype"), "shape": actual.get("shape")}
        if "dimensions" in descriptor:
            names = [item["name"] for item in descriptor["dimensions"]]
            expected["names"] = names
            observed["names"] = actual.get("names")
            dimensions_valid = (len(names) == descriptor["shape"][0] and len(names) == len(set(names)))
        else:
            dimensions_valid = descriptor["fps"] == binding["fps"]
            if "fps" in actual:
                dimensions_valid &= actual["fps"] == descriptor["fps"]
            video_info = actual.get("info", {})
            if "video.fps" in video_info:
                dimensions_valid &= video_info["video.fps"] == descriptor["fps"]
            if "fps" in video_info:
                dimensions_valid &= video_info["fps"] == descriptor["fps"]
        check("C10_FEATURES", expected == observed and dimensions_valid, observed, expected, stream=key)
    feature_keys = [stream["native_feature_key"] for stream in streams.values()]
    check("C10_FEATURES", set(feature_keys) == set(features) and len(feature_keys) == len(set(feature_keys)),
          feature_keys, sorted(features))
    for name, stream in streams.items():
        if "frame" in stream:
            frame, descriptor = stream["frame"], features.get(stream["native_feature_key"], {})
            observed = {"shape": [frame["height"], frame["width"], 3], "encoding": frame["encoding"]}
            expected = {"shape": descriptor.get("shape"), "encoding": descriptor.get("encoding")}
            check("C10_FEATURES", observed == expected, observed, expected, stream=name)

    native_task = tasks.get(task["task_index"], tasks.get(str(task["task_index"])))
    check("C10_TASK", native_task == task["instruction"] and episode.get("tasks") == [task["instruction"]],
          {"instruction": native_task, "episode_tasks": episode.get("tasks")},
          {"instruction": task["instruction"], "episode_tasks": [task["instruction"]]})
    frame_indices = [sample["frame_index"] for sample in samples]
    check("C10_COVERAGE", frame_indices == list(range(binding["length"])), frame_indices,
          list(range(binding["length"])))
    start, end = robot["recording"]["start_timestamp_ns"], robot["recording"]["end_timestamp_ns"]
    references = [sample["authoritative_timestamp_ns"] for sample in samples]
    check("C10_TIME_ORDER", start == 0 and start <= end and all(start <= t <= end for t in references)
          and all(a < b for a, b in zip(references, references[1:])),
          {"start": start, "end": end, "timestamps": references},
          "recording starts at zero; strictly increasing frame references within inclusive recording bounds", units="ns")

    metrics = {"frame_count": len(samples), "streams": {}, "inter_observation_skew_ns": [], "action_latency_ns": []}
    previous = {}
    source_ids = {name: {} for name in streams}
    runs = {name: {"stale": 0, "reuse": 0} for name in streams}
    frame_times = {sample["frame_index"]: sample["authoritative_timestamp_ns"] for sample in samples}
    observations = [name for name in streams if name != "action"]
    for name in streams:
        metrics["streams"][name] = {"available_count": 0, "missing_count": 0, "dropped_count": 0,
                                   "unmapped_count": 0, "stale_count": 0, "reuse_count": 0,
                                   "sequence_gap_count": 0, "max_sequence_gap": 0,
                                   "max_stale_run": 0, "max_reuse_run": 0,
                                   "configured_fps": streams[name].get("frame", {}).get("fps"),
                                   "nominal_fps": binding["fps"],
                                   "selected_timestamps_ns": [], "freshness_ns": []}
    for sample in samples:
        frame, reference = sample["frame_index"], sample["authoritative_timestamp_ns"]
        check("C10_TASK", sample["episode_index"] == binding["episode_index"] and sample["task_index"] == task["task_index"],
              {"episode_index": sample["episode_index"], "task_index": sample["task_index"]},
              {"episode_index": binding["episode_index"], "task_index": task["task_index"]}, frame=frame)
        check("C10_COVERAGE", set(sample["streams"]) == set(streams), sorted(sample["streams"]), sorted(streams), frame=frame)
        usable = {}
        for name, metadata in streams.items():
            if name not in sample["streams"]:
                continue
            item = sample["streams"][name]
            context = {"frame": frame, "stream": name}
            health = item["health"]
            selected = item["selected_timestamp_ns"]
            source = item["selection_timestamp_source"]
            timestamps = metadata["timestamps"]
            stats = metrics["streams"][name]
            runs[name]["stale"] = runs[name]["stale"] + 1 if health == "stale" else 0
            stats["max_stale_run"] = max(stats["max_stale_run"], runs[name]["stale"])
            if health in ("missing", "dropped", "unmapped", "stale"):
                stats[health + "_count"] += 1
            absent = health in ("missing", "dropped")
            if absent:
                runs[name]["reuse"] = 0
            host, device = item["host_receipt"], item["device_timestamp"]
            health_valid = True
            if absent:
                health_valid = (item["sequence_id"] is None and item["source_frame_id"] is None
                                and selected is None and item["measured_skew_ns"] is None and source == "none"
                                and host["status"] == "unavailable" and device["status"] == "unavailable")
            else:
                health_valid = (item["sequence_id"] is not None and bool(item["source_frame_id"])
                                and host["status"] in ("valid", "pending") and source == timestamps["capture_source"])
                chosen = device if source == "device" else host
                if health == "unmapped":
                    health_valid &= (selected is None and item["measured_skew_ns"] is None
                                     and chosen["status"] == "pending" and "raw" in chosen and bool(chosen.get("reason")))
                else:
                    health_valid &= selected is not None and chosen["status"] == "valid"
            check("C10_HEALTH", health_valid, item,
                  "health, IDs and timestamp availability agree", **context)
            if health == "stale":
                pending("C10_HEALTH", item, "review of producer-reported stale evidence; the label alone is not a QA verdict", **context)
            mapped_sources = {}
            if not absent:
                for source_name, stamp in (("host_receipt", host), ("device", device)):
                    source_metadata = timestamps[source_name]
                    mapping = source_metadata.get("mapping_to_authoritative", {})
                    if stamp["status"] == "unavailable":
                        # Device acquisition timestamps can explicitly be unavailable.
                        check("C10_MAPPING", source_name == "device" and bool(stamp.get("reason")),
                              stamp, "unavailable device timestamp has a reason", **context)
                        continue
                    if stamp["status"] == "pending":
                        valid_pending = (mapping.get("status") in ("pending", "unavailable")
                                         and "raw" in stamp and bool(stamp.get("reason"))
                                         and "mapped_authoritative_ns" not in stamp)
                        check("C10_MAPPING", valid_pending, stamp, "pending mapping preserves raw source time and reason", **context)
                        if valid_pending:
                            pending("C10_MAPPING", mapping, "valid calibrated mapping", **context)
                        continue
                    try:
                        calculated = _mapped(stamp["raw"], mapping) if mapping.get("status") == "valid" else None
                        valid = (calculated is not None and calculated == stamp.get("mapped_authoritative_ns")
                                 and start <= calculated <= end)
                    except (KeyError, ValueError, InvalidOperation, TypeError, OverflowError):
                        calculated, valid = None, False
                    if source_name == "device" and source_metadata.get("availability") != "available":
                        valid = False
                    check("C10_MAPPING", valid, stamp, {"source": source_name, "recomputed_ns": calculated}, units="ns", **context)
                    if valid:
                        mapped_sources[source_name] = calculated
            if selected is not None:
                selection_valid = source == timestamps["capture_source"] and mapped_sources.get(source) == selected
                check("C10_SELECTION", selection_valid, {"source": source, "selected_ns": selected},
                      {"source": timestamps["capture_source"], "selected_ns": mapped_sources.get(source)}, units="ns", **context)
                check("C10_TIME_ORDER", start <= selected <= end, selected, {"min": start, "max": end}, units="ns", **context)
                skew = abs(selected - reference)
                check("C10_SKEW", item["measured_skew_ns"] == skew, item["measured_skew_ns"], skew, units="ns", **context)
                if selection_valid and health_valid:
                    usable[name] = selected
                    stats["available_count"] += 1
                    stats["selected_timestamps_ns"].append(selected)
                    stats["freshness_ns"].append(max(0, reference - selected))
            if not absent:
                seen_sequence = source_ids[name].get(item["source_frame_id"])
                if seen_sequence is not None:
                    check("C10_SEQUENCE", seen_sequence == item["sequence_id"],
                          {"source_frame_id": item["source_frame_id"], "sequence_id": item["sequence_id"]},
                          {"original_sequence_id": seen_sequence}, **context)
                source_ids[name][item["source_frame_id"]] = item["sequence_id"]
                prior = previous.get(name)
                if prior:
                    reuse = item["sequence_id"] == prior["sequence_id"]
                    preserved = all(item[key] == prior[key] for key in (
                        "source_frame_id", "selected_timestamp_ns", "selection_timestamp_source", "host_receipt", "device_timestamp"))
                    sequence_valid = item["sequence_id"] >= prior["sequence_id"]
                    if reuse:
                        stats["reuse_count"] += 1
                        runs[name]["reuse"] += 1
                        stats["max_reuse_run"] = max(stats["max_reuse_run"], runs[name]["reuse"])
                        sequence_valid &= preserved
                        check("C10_HEALTH", health != "fresh", health, "reused source samples must not claim fresh", **context)
                    else:
                        runs[name]["reuse"] = 0
                        gap = max(0, item["sequence_id"] - prior["sequence_id"] - 1)
                        stats["sequence_gap_count"] += gap
                        stats["max_sequence_gap"] = max(stats["max_sequence_gap"], gap)
                        sequence_valid &= item["source_frame_id"] != prior["source_frame_id"]
                    for stamp_key in ("host_receipt", "device_timestamp"):
                        a, b = prior[stamp_key].get("raw"), item[stamp_key].get("raw")
                        if a is not None and b is not None:
                            sequence_valid &= _decimal(b) >= _decimal(a)
                    if selected is not None and prior["selected_timestamp_ns"] is not None:
                        sequence_valid &= selected >= prior["selected_timestamp_ns"]
                    check("C10_SEQUENCE", sequence_valid, item,
                          "nondecreasing sequence/time; new IDs on advance; exact original evidence on reuse", **context)
                previous[name] = item
        if all(name in usable for name in observations):
            times = [usable[name] for name in observations]
            skew = max(times) - min(times)
            metrics["inter_observation_skew_ns"].append({"frame_index": frame, "value": skew})
            tolerance = policy.get("synchronization", {}).get("max_skew_ns")
            declaration = robot["synchronization"]
            if declaration["status"] == "pending_evidence":
                pending("C10_SYNC", {"skew_ns": skew, "declaration": declaration},
                        "declared tolerance backed by approved external synchronization policy", units="ns", frame=frame)
            elif approved and _finite(tolerance) and tolerance >= 0:
                check("C10_SYNC", skew <= tolerance and declaration["tolerance_ns"] == tolerance,
                      {"skew_ns": skew, "declared_tolerance_ns": declaration["tolerance_ns"]},
                      "declared tolerance matches policy; inter-observation skew within approved tolerance",
                      units="ns", threshold=tolerance, policy=policy_ref, frame=frame)
            else:
                pending("C10_SYNC", {"skew_ns": skew, "declaration": robot["synchronization"]},
                        "approved external synchronization policy", units="ns", frame=frame)
        else:
            pending("C10_SYNC", {"unavailable_streams": [name for name in observations if name not in usable]},
                    "mapped selected time for every observation stream", frame=frame)
        if "action" in usable:
            latency_policy = policy.get("action_latency", {})
            offset = latency_policy.get("reference_frame_offset") if approved else None
            if type(offset) is not int or frame + offset not in frame_times:
                pending("C10_ACTION_LATENCY", {"reference_frame_offset": offset},
                        "approved explicit reference-frame offset and an existing reference frame", frame=frame)
                continue
            reference_frame = frame + offset
            latency = usable["action"] - frame_times[reference_frame]
            metrics["action_latency_ns"].append({"frame_index": frame, "reference_frame_index": reference_frame, "value": latency})
            if (approved and all(_finite(latency_policy.get(key)) for key in ("min_ns", "max_ns"))
                    and latency_policy["min_ns"] <= latency_policy["max_ns"]):
                semantics = {key: robot["action_semantics"][key] for key in
                             ("values", "timestamp", "observation_relationship")}
                valid = (all(latency_policy.get(key) == value for key, value in semantics.items())
                         and latency_policy["min_ns"] <= latency <= latency_policy["max_ns"])
                check("C10_ACTION_LATENCY", valid, {"latency_ns": latency, **semantics},
                      latency_policy, units="ns", threshold=latency_policy, policy=policy_ref, frame=frame)
            else:
                pending("C10_ACTION_LATENCY", latency, "approved latency bounds and exact declared observation relationship", units="ns", frame=frame)
        else:
            pending("C10_ACTION_LATENCY", None, "mapped action timestamp", frame=frame)

    for name, stats in metrics["streams"].items():
        selected_times = stats["selected_timestamps_ns"]
        stats["observed_intervals_ns"] = [b - a for a, b in zip(selected_times, selected_times[1:])]
        times = sorted(set(stats["selected_timestamps_ns"]))
        stats["measured_fps"] = (len(times) - 1) * 1_000_000_000 / (times[-1] - times[0]) if len(times) > 1 else None
        stats["max_freshness_ns"] = max(stats["freshness_ns"], default=None)
        quality = policy.get("stream_quality", {}).get(name)
        if not approved or not isinstance(quality, dict) or not quality:
            pending("C10_STREAM_QUALITY", stats, "approved source FPS/freshness/gap/reuse policy", stream=name)
            continue
        checks = {"min_fps": (stats["measured_fps"], lambda a, b: a >= b),
                  "max_freshness_ns": (stats["max_freshness_ns"], lambda a, b: a <= b),
                  "max_sequence_gap": (stats["max_sequence_gap"], lambda a, b: a <= b),
                  "max_reuse": (stats["reuse_count"], lambda a, b: a <= b)}
        if set(quality) < set(checks):
            pending("C10_STREAM_QUALITY", stats, "all FPS, freshness, sequence-gap and reuse policy fields",
                    policy=policy_ref, stream=name)
            continue
        valid = (set(quality) == set(checks)
                 and all(_finite(value) and value >= 0 for value in quality.values()))
        unavailable = False
        for key, limit in quality.items():
            if key not in checks or not _finite(limit):
                continue
            value, compare = checks[key]
            if value is None:
                unavailable = True
            else:
                valid &= compare(value, limit)
        if valid and (unavailable or stats["available_count"] != len(samples)):
            pending("C10_STREAM_QUALITY", stats, quality, policy=policy_ref, stream=name)
        else:
            check("C10_STREAM_QUALITY", valid, stats, quality, threshold=quality, policy=policy_ref, stream=name)

    for key in ("observation.state", "action"):
        descriptor = features[key]
        stream_name = "robot_state" if key == "observation.state" else "action"
        actual_profile = {"profile": descriptor["profile"], "dimensions": descriptor["dimensions"],
                          "reference_frame": streams[stream_name]["reference_frame"]}
        wanted_profile = policy.get("profiles", {}).get(key)
        if approved and isinstance(wanted_profile, dict):
            check("C10_PROFILE", actual_profile == wanted_profile, actual_profile, wanted_profile, policy=policy_ref, stream=key)
        else:
            pending("C10_PROFILE", actual_profile, "approved explicit ordered dimension, unit, meaning and reference-frame profile", stream=key)

    if rows is None:
        pending("C10_VECTOR_VALUES", None, "decoded native state/action rows")
        pending("C10_ACTION_BOUNDS", None, "decoded native action rows and approved named bounds")
    else:
        check("C10_COVERAGE", [row.get("frame_index") for row in rows] == frame_indices,
              [row.get("frame_index") for row in rows], frame_indices)
        for row in rows:
            frame = row.get("frame_index")
            check("C10_TASK", row.get("episode_index") == binding["episode_index"] and row.get("task_index") == task["task_index"],
                  {"episode_index": row.get("episode_index"), "task_index": row.get("task_index")},
                  {"episode_index": binding["episode_index"], "task_index": task["task_index"]}, frame=frame)
            if isinstance(frame, int):
                grid_time = frame / binding["fps"]
                float32_time = struct.unpack("f", struct.pack("f", grid_time))[0]
                check("C10_TIME_ORDER", row.get("timestamp") in (grid_time, float32_time), row.get("timestamp"),
                      {"native_grid_seconds": grid_time, "float32_seconds": float32_time}, units="s", frame=frame)
            for key in ("observation.state", "action"):
                value = row.get(key)
                valid_vector = isinstance(value, (list, tuple)) and len(value) == features[key]["shape"][0] and all(_finite(v) for v in value)
                check("C10_VECTOR_VALUES", valid_vector, value,
                      {"length": features[key]["shape"][0], "values": "finite numeric values"}, frame=frame, stream=key)
                if key != "action":
                    continue
                bounds = policy.get("action_bounds")
                if not approved or not isinstance(bounds, dict) or not bounds:
                    pending("C10_ACTION_BOUNDS", value, "approved bounds keyed by every declared action dimension and unit", frame=frame)
                    continue
                dimensions = features[key]["dimensions"]
                valid_bounds = set(bounds) == {dim["name"] for dim in dimensions}
                for dim, number in zip(dimensions, value if valid_vector else []):
                    bound = bounds.get(dim["name"], {})
                    valid_bounds &= (isinstance(bound, dict) and _finite(bound.get("min")) and _finite(bound.get("max"))
                                     and bound.get("unit") == dim["unit"] and bound["min"] <= number <= bound["max"])
                check("C10_ACTION_BOUNDS", valid_vector and valid_bounds, value, bounds, threshold=bounds, policy=policy_ref, frame=frame)

    event_ids = [event["event_id"] for event in robot["events"]]
    check("C10_EVENTS", len(event_ids) == len(set(event_ids)), event_ids, "unique event IDs")
    for event in robot["events"]:
        lower, upper = event["start_timestamp_ns"], event["end_timestamp_ns"]
        valid = start <= lower <= upper <= end and all(
            index in frame_times and lower <= frame_times[index] <= upper for index in event["linked_frame_indices"])
        check("C10_EVENTS", valid, event, "ordered event bounds and linked frame times within recording and event interval")
    outcome, recording = robot["outcome"], robot["recording"]
    reason_valid = isinstance(outcome.get("reason"), str) and bool(outcome["reason"].strip())
    complete = recording["status"] == "complete"
    consistent = (outcome["status"] == "success" or reason_valid)
    if outcome["status"] == "aborted":
        consistent &= any(event["type"] == "abort" for event in robot["events"])
    if outcome["status"] == "success":
        consistent &= not any(event["type"] == "abort" for event in robot["events"])
    check("C10_COMPLETENESS", complete and consistent,
          {"recording": recording, "outcome": outcome},
          "complete recording; non-success reason; aborted outcomes linked to abort evidence")
    return metrics
