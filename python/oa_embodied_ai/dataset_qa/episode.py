#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read-only C10 episode quality gate. See docs/c10/README.md."""

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from jsonschema import Draft202012Validator
import yaml

from .strict_io import load_data, unique_pairs
from .contract import load_c9_schema
from .semantics import check_episode



def report_value(value):
    """Keep non-finite native evidence visible in standards-compliant JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(k): report_value(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [report_value(v) for v in value]
    return value


def preserve_clock_precision(path, sidecar):
    """After existing structural validation, retain decimal clock lexemes.

    Converting a raw fractional tick through binary float can move a value
    across an exact nanosecond half-tie. Only clock arithmetic fields are
    replaced; native descriptors retain the existing loader's representations.
    """
    if path.suffix == ".json":
        exact = json.loads(path.read_text(), object_pairs_hook=unique_pairs, parse_float=Decimal)
    else:
        class DecimalLoader(yaml.SafeLoader):
            pass
        DecimalLoader.add_constructor("tag:yaml.org,2002:float",
                                      lambda loader,node: Decimal(loader.construct_scalar(node).replace("_","")))
        exact = yaml.load(path.read_text(), Loader=DecimalLoader)
    for name,stream in sidecar["openamrobot"]["streams"].items():
        precise = exact["openamrobot"]["streams"][name]["timestamps"]
        for source in ("host_receipt","device"):
            if "mapping_to_authoritative" in precise[source]:
                stream["timestamps"][source]["mapping_to_authoritative"] = precise[source]["mapping_to_authoritative"]
    for sample, precise in zip(sidecar["openamrobot"]["samples"],exact["openamrobot"]["samples"]):
        for name,item in sample["streams"].items():
            for source in ("host_receipt","device_timestamp"):
                if "raw" in item[source]:
                    item[source]["raw"] = precise["streams"][name][source]["raw"]


def inside(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def jsonl(path):
    return [json.loads(line, object_pairs_hook=unique_pairs,
                       parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            for line in path.read_text().splitlines() if line.strip()]


def parquet(path):
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    # Duplicate physical columns otherwise collapse silently in to_pylist().
    if len(set(table.column_names)) != len(table.column_names):
        raise ValueError("duplicate Parquet columns")
    return table


def native_metadata(root):
    info = load_data(inside(root, "meta/info.json"))
    if not isinstance(info, dict) or not isinstance(info.get("features"), dict):
        raise ValueError("native info must contain a features object")
    task_path = inside(root, "meta/tasks.parquet")
    if task_path.exists():
        task_rows = parquet(task_path).to_pylist()
    else:
        task_rows = jsonl(inside(root, "meta/tasks.jsonl"))
    tasks = {}
    for row in task_rows:
        index = row["task_index"]
        if index in tasks:
            raise ValueError("duplicate native task_index")
        tasks[index] = row.get("task", row.get("instruction", row.get("__index_level_0__")))
    episode_paths = sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))
    episodes = [row for path in episode_paths for row in parquet(inside(root,path)).to_pylist()] if episode_paths else jsonl(inside(root, "meta/episodes.jsonl"))
    indexes = [row["episode_index"] for row in episodes]
    if len(set(indexes)) != len(indexes):
        raise ValueError("duplicate native episode_index")
    return info, tasks, {row["episode_index"]: row for row in episodes}


def read_rows(root, info, episode, emit, known_episode_indices=None):
    try:
        template = info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
        args = dict(episode_index=episode["episode_index"],
                    episode_chunk=episode["episode_index"] // info.get("chunks_size", 1000),
                    chunk_index=episode.get("data/chunk_index", 0),
                    file_index=episode.get("data/file_index", 0))
        path = inside(root, template.format(**args))
        if not path.is_file():
            emit("C10_NATIVE_DATA", "NOT_VERIFIED", observed=str(path.relative_to(root)),
                 fix="Provide the native episode Parquet artifact.")
            return None
        table = parquet(path)
        all_rows = table.to_pylist()
        if known_episode_indices is not None:
            orphans = sorted({row.get("episode_index") for row in all_rows if row.get("episode_index") not in known_episode_indices},key=str)
            emit("C10_NATIVE_DATA", "REJECT" if orphans else "PASS",observed=orphans,
                 expected="every stored episode_index belongs to the native episode registry",
                 fix="Reconcile orphan rows in the referenced native Parquet file with immutable episode metadata.")
        rows = [row for row in all_rows if row.get("episode_index") == episode["episode_index"]]
        emit("C10_NATIVE_DATA", "PASS", observed=len(rows), expected=episode.get("length"), units="frames")
        import pyarrow as pa
        for key in ("observation.state", "action"):
            dtype = table.schema.field(key).type
            flat = (pa.types.is_list(dtype) or pa.types.is_fixed_size_list(dtype)) and pa.types.is_float32(dtype.value_type)
            emit("C10_PARQUET_DTYPE", "PASS" if flat else "REJECT", observed=str(dtype),
                 expected="list<float32>", stream=key, fix="Record native float32 vectors using the declared feature descriptor.")
        if "dataset_from_index" in episode and "dataset_to_index" in episode:
            actual = [row.get("index") for row in rows]
            expected = list(range(episode["dataset_from_index"], episode["dataset_to_index"]))
            emit("C10_NATIVE_DATA", "PASS" if actual == expected else "REJECT", observed=actual, expected=expected,
                 fix="Restore the immutable native episode row range.")
        return rows
    except ImportError:
        emit("C10_NATIVE_DATA", "NOT_VERIFIED", fix="Install optional pyarrow to inspect native Parquet artifacts.")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        emit("C10_NATIVE_DATA", "REJECT", observed=str(error), fix="Restore a readable native Parquet artifact and its metadata binding.")
    return None


def videos(root, sidecar, info, episode, policy, emit):
    for name, stream in sidecar["openamrobot"]["streams"].items():
        if "frame" not in stream:
            continue
        key = stream["native_feature_key"]
        try:
            template = info.get("video_path")
            if not template:
                raise FileNotFoundError("No native video_path descriptor")
            chunk = episode.get(f"videos/{key}/chunk_index")
            file_index = episode.get(f"videos/{key}/file_index")
            # v3 shared containers must have explicit per-episode timestamp offsets.
            start = episode.get(f"videos/{key}/from_timestamp")
            end = episode.get(f"videos/{key}/to_timestamp")
            shared = "{file_index" in template
            if shared and (chunk is None or file_index is None or start is None or end is None):
                raise FileNotFoundError("Missing v3 video chunk/file/time range binding")
            path = inside(root, template.format(video_key=key, chunk_index=chunk, file_index=file_index,
                                               episode_index=episode["episode_index"],
                                               episode_chunk=episode["episode_index"] // info.get("chunks_size", 1000)))
            if not path.is_file():
                raise FileNotFoundError(str(path.relative_to(root)))
            if not shutil.which("ffprobe"):
                raise FileNotFoundError("ffprobe is unavailable; video decode/count was not performed")
            proc = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
                                   "-show_entries", "frame=best_effort_timestamp_time,width,height", "-of", "json", str(path)],
                                  capture_output=True, text=True, timeout=60)
            if proc.returncode or proc.stderr.strip():
                raise ValueError(proc.stderr or "ffprobe decode failed")
            all_frames = json.loads(proc.stdout)["frames"]
            positions = list(range(len(all_frames)))
            if shared:
                positions = [i for i,f in enumerate(all_frames) if start <= float(f["best_effort_timestamp_time"]) < end]
            frames = [all_frames[i] for i in positions]
            shape = stream["frame"]
            valid = len(frames) == episode["length"] and all(f["width"] == shape["width"] and f["height"] == shape["height"] for f in frames)
            emit("C10_VIDEO", "PASS" if valid else "REJECT", stream=name, observed={"frames":len(frames)},
                 expected={"frames":episode["length"],"width":shape["width"],"height":shape["height"]},
                 units="frames,pixels", fix="Restore the recorded video segment and native frame binding.")
            # Source IDs cannot prove pixel freshness. Identical images are not
            # automatically bad (a static scene is legitimate); policy decides.
            content_policy = policy.get("video_content", {}).get(name, {})
            limit = content_policy.get("max_identical_run")
            reviewed = policy.get("approved") is True and bool(policy.get("id")) and bool(policy.get("evidence"))
            if not reviewed or type(limit) is not int or limit < 1 or not shutil.which("ffmpeg"):
                emit("C10_VIDEO", "NOT_VERIFIED", stream=name,
                     observed="pixel-content freshness was not measured",
                     expected="decoded image-content evidence with reviewed static-scene policy",
                     fix="Supply ffmpeg and a reviewed video_content max_identical_run policy; source IDs alone do not prove fresh pixels.")
            elif valid:
                decoded = subprocess.run(["ffmpeg","-v","error","-i",str(path),"-map","0:v:0","-f","framemd5","-"],
                                         capture_output=True,text=True,timeout=60)
                if decoded.returncode or decoded.stderr.strip():
                    raise ValueError(decoded.stderr or "ffmpeg content decode failed")
                hashes = [line.split(",")[-1].strip() for line in decoded.stdout.splitlines() if line and not line.startswith("#")]
                if len(hashes) != len(all_frames):
                    raise ValueError("decoded image hashes and frame metadata count disagree")
                run, maximum, previous = 0, 0, None
                for i in positions:
                    digest = hashes[i]
                    run = run + 1 if digest == previous else 1
                    previous = digest
                    maximum = max(maximum,run)
                emit("C10_VIDEO", "PASS" if maximum <= limit else "REJECT", stream=name,
                     observed={"max_identical_run":maximum},expected=content_policy,units="frames",
                     threshold=limit,policy=policy.get("id"),fix="Review repeated decoded images against the recorded scene and approved freshness policy.")
        except FileNotFoundError as error:
            emit("C10_VIDEO", "NOT_VERIFIED", stream=name, observed=str(error), fix="Provide native video artifacts, metadata offsets and ffprobe.")
        except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
            emit("C10_VIDEO", "REJECT", stream=name, observed=str(error), fix="Restore a decodable video and consistent frame range.")


def evidence(root, sidecar, policy, emit):
    am = sidecar["openamrobot"]
    approved = policy.get("approved") is True and bool(policy.get("id")) and bool(policy.get("evidence"))
    artifacts = policy.get("artifacts", []) if approved else []
    source = policy.get("source_profile", {}) if approved else {}
    declared_profile = am["provenance"]["source_profile"]
    if not source:
        emit("C10_PROFILE", "NOT_VERIFIED", observed=declared_profile,
             fix="Supply the reviewed source-profile identity and immutable serialized profile artifact.")
    else:
        emit("C10_PROFILE", "PASS" if source.get("id") == declared_profile else "REJECT",
             observed=declared_profile, expected=source.get("id"),
             fix="Use the recording's actual approved source-profile identity.")
        verify_artifact(root, source.get("artifact", {}), source.get("artifact", {}).get("sha256"), emit, "source_profile")
    references = [("robot_configuration", None, am["provenance"]["robot_configuration_hash"])]
    references += [("stream_configuration", name, stream["configuration_hash"]) for name,stream in am["streams"].items()]
    for kind, name, declared in references:
        matches = [a for a in artifacts if a.get("kind") == kind and a.get("stream") == name]
        if not matches:
            emit("C10_ARTIFACT", "NOT_VERIFIED", stream=name, observed=declared, expected="immutable source bytes + reviewed hash", fix="Supply an exact-byte artifact reference in the reviewed policy.")
        for artifact in matches:
            verify_artifact(root, artifact, declared, emit, name)
    for name,stream in am["streams"].items():
        if "calibration" not in stream:
            continue
        cal = stream["calibration"]
        matches = [a for a in artifacts if a.get("kind") == "calibration" and a.get("stream") == name]
        if not matches:
            emit("C10_CALIBRATION", "NOT_VERIFIED", stream=name, observed=cal, fix="Supply immutable calibration evidence and a reviewed acceptance policy.")
        for artifact in matches:
            same = (artifact.get("id"), artifact.get("version")) == (cal["id"], cal["version"])
            emit("C10_CALIBRATION", "PASS" if same else "REJECT", stream=name,
                 observed=[cal["id"],cal["version"]], expected=[artifact.get("id"),artifact.get("version")], fix="Bind the recording to its actual versioned calibration artifact.")
            verify_artifact(root, artifact, artifact.get("sha256"), emit, name)
        # A measured residual is evidence, not automatically an approved result.
        acceptance = policy.get("calibration", {}).get(name, {}) if approved else {}
        residual = cal["residual"]
        if residual["status"] != "measured" or not all(k in acceptance for k in ("max_value","unit","name","statistic","method")):
            emit("C10_CALIBRATION", "NOT_VERIFIED", stream=name, observed=residual, fix="Provide measured residuals and owner-reviewed acceptance criteria.")
        else:
            valid = all(residual.get(k) == acceptance[k] for k in ("unit","name","statistic","method")) and residual["value"] <= acceptance["max_value"]
            emit("C10_CALIBRATION", "PASS" if valid else "REJECT", stream=name, observed=residual, expected=acceptance,
                 units=acceptance["unit"], threshold=acceptance["max_value"], policy=policy.get("id"), fix="Resolve calibration evidence against the approved criterion.")
    actual = am["provenance"]["lerobot_compatibility"]["validated_release_pin"]
    release = policy.get("release") if approved else None
    recording = [s for s in am["provenance"]["software"] if s["name"].lower() == "lerobot"]
    if actual is None or not release or not recording:
        emit("C10_RELEASE", "NOT_VERIFIED", observed=actual, fix="Obtain a release-owner pin and identify the recording LeRobot version/commit.")
    else:
        valid = (actual == release.get("pin") and release.get("version") == actual["release"]
                 and release.get("commit") == actual["commit"] and actual["release"] != "unknown"
                 and all(s["version"] == actual["release"] and s["commit"] == actual["commit"] for s in recording))
        emit("C10_RELEASE", "PASS" if valid else "REJECT", observed={"pin":actual,"software":recording}, expected=release,
             fix="Reconcile recording software provenance with the approved release pin.")
    real = policy.get("real_episode", {}) if approved else {}
    real_verified = (am["provenance"]["data_origin"] == "actual" and bool(real.get("evidence"))
                     and all(real.get(key) == sidecar["native_lerobot"][key]
                             for key in ("dataset_id","dataset_revision","episode_id","episode_index")))
    emit("C10_REAL_EPISODE", "PASS" if real_verified else "NOT_VERIFIED",
         observed=am["provenance"]["data_origin"], expected="actual recording with provenance reviewed externally",
         fix="Provide a real recorded episode; synthetic fixtures establish QA behavior only.")


def verify_artifact(root, artifact, declared, emit, stream):
    try:
        path = inside(root, artifact["path"])
        if not path.is_file():
            raise FileNotFoundError(artifact["path"])
        digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        expected = artifact.get("sha256")
        emit("C10_ARTIFACT", "PASS" if digest == expected == declared else "REJECT", stream=stream,
             observed={"bytes_sha256":digest,"declared":declared}, expected=expected,
             fix="Restore the exact immutable artifact bytes and matching declared hash.")
    except FileNotFoundError as error:
        emit("C10_ARTIFACT", "NOT_VERIFIED", stream=stream, observed=str(error), fix="Make the referenced immutable artifact available under the dataset root.")
    except (OSError, ValueError, KeyError, TypeError) as error:
        emit("C10_ARTIFACT", "REJECT", stream=stream, observed=str(error), fix="Use an in-dataset artifact path with an explicit SHA-256 identity.")


def check_dataset(dataset_path, policy_path=None, episode_indices=None, *, c9_schema_path=None):
    root = Path(dataset_path).resolve()
    findings, reports = [], []
    provenance = {"c9_contract": None, "policy": None,
                  "checker_files_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                           for name in ("episode.py", "semantics.py", "strict_io.py", "contract.py", "handoff.py")}}
    context = {"episode":None, "source": None}
    def emit(check_id,status,observed=None,expected=None,units=None,threshold=None,policy=None,fix="Resolve the reported mismatch.",frame=None,stream=None):
        findings.append(dict(check_id=check_id,status=status,severity={"PASS":"INFO","REJECT":"ERROR","NOT_VERIFIED":"BLOCKER"}[status],
                             episode=context["episode"],frame=frame,stream=stream,observed=observed,expected=expected,
                             units=units,threshold=threshold,policy=policy,fix=fix,source=context["source"]))
    try:
        if not root.is_dir():
            raise FileNotFoundError("dataset path is not an available directory")
        policy = load_data(Path(policy_path)) if policy_path else {}
        if not isinstance(policy,dict):
            raise ValueError("policy must be an object")
        if policy.get("approved") is True and any(not isinstance(policy.get(key),str) or not policy[key].strip() for key in ("id","evidence")):
            raise ValueError("approved policy requires nonempty string id and evidence")
        for key in ("profiles","action_bounds","synchronization","stream_quality","action_latency",
                    "calibration","release","real_episode","native_identity","source_profile","video_content"):
            if key in policy and not isinstance(policy[key],dict):
                raise ValueError(f"policy {key} must be an object")
        for key in ("profiles","action_bounds","stream_quality","calibration","video_content"):
            if any(not isinstance(value,dict) for value in policy.get(key,{}).values()):
                raise ValueError(f"policy {key} entries must be objects")
        if "artifact" in policy.get("source_profile",{}) and not isinstance(policy["source_profile"]["artifact"],dict):
            raise ValueError("source_profile artifact must be an object")
        if "episode_ids" in policy.get("native_identity",{}) and not isinstance(policy["native_identity"]["episode_ids"],dict):
            raise ValueError("native_identity episode_ids must be an object")
        if not isinstance(policy.get("artifacts",[]),list) or any(not isinstance(a,dict) for a in policy.get("artifacts",[])):
            raise ValueError("policy artifacts must be an array of objects")
        schema, schema_provenance = load_c9_schema(c9_schema_path)
        provenance["c9_contract"] = schema_provenance
        provenance["policy"] = {"id": policy.get("id"), "evidence": policy.get("evidence"),
                                "approved": policy.get("approved") is True,
                                "sha256": hashlib.sha256(Path(policy_path).read_bytes()).hexdigest() if policy_path else None}
        validator = Draft202012Validator(schema)
        paths = sorted(p for p in (root / "meta/openamrobot/episodes").glob("episode-*") if p.suffix in (".json",".yaml",".yml"))
        if not paths:
            raise FileNotFoundError("No C9 sidecars at meta/openamrobot/episodes/episode-*.(json|yaml|yml)")
        try:
            info,tasks,episodes = native_metadata(root)
            emit("C10_NATIVE_METADATA","PASS",observed=len(episodes),units="episodes")
            measured_counts = {"total_episodes":len(episodes),"total_tasks":len(tasks),
                               "total_frames":sum(episode["length"] for episode in episodes.values())}
            for key,value in measured_counts.items():
                emit("C10_NATIVE_METADATA", "NOT_VERIFIED" if key not in info else "PASS" if info[key] == value else "REJECT",
                     observed=info.get(key),expected=value,units=key,
                     fix="Reconcile native info totals with immutable task and episode metadata.")
        except ImportError:
            info,tasks,episodes = {},{},{}
            emit("C10_NATIVE_METADATA","NOT_VERIFIED",fix="Install optional pyarrow to read native v3 metadata.")
        except (OSError,ValueError,KeyError,TypeError) as error:
            info,tasks,episodes = {},{},{}
            emit("C10_NATIVE_METADATA","REJECT",observed=str(error),fix="Provide valid native info/task/episode metadata without duplicate identities.")
        seen = set()
        for path in paths:
            context.update(episode=None, source=str(path.relative_to(root)))
            sidecar = load_data(inside(root,path))
            errors = sorted(validator.iter_errors(sidecar),key=lambda e:str(e.path))
            if errors:
                emit("C10_SCHEMA","REJECT",observed=[{"path":list(e.path),"message":e.message} for e in errors],fix="Conform the sidecar to the unchanged accepted C9 v1 schema.")
                continue
            preserve_clock_precision(path, sidecar)
            index = sidecar["native_lerobot"]["episode_index"]
            if episode_indices is not None and index not in episode_indices:
                continue
            context["episode"] = index
            emit("C10_SCHEMA","PASS",observed=str(path.relative_to(root)))
            if index in seen:
                emit("C10_NATIVE_METADATA","REJECT",observed=index,expected="one sidecar per episode",fix="Remove duplicate episode sidecars from the recording export.")
            seen.add(index)
            episode = episodes.get(index,{"episode_index":index})
            rows = read_rows(root,info,episode,emit,set(episodes)) if index in episodes else None
            native = {"info":info,"tasks":tasks,"episode":episode,"rows":rows}
            metrics = check_episode(sidecar,native,policy,emit)
            if rows is not None:
                covered = set(row.get("frame_index") for row in rows) & set(sample["frame_index"] for sample in sidecar["openamrobot"]["samples"])
                length = episode.get("length")
                metrics["native_frame_coverage"] = {
                    "native_frames_loaded":len(rows),"covered_native_frames":len(covered),
                    "expected_native_length":length,
                    "fraction":len(covered)/length if isinstance(length,int) and length > 0 else None}
            videos(root,sidecar,info,episode,policy,emit)
            evidence(root,sidecar,policy,emit)
            episode_findings = [f for f in findings if f["episode"] == index]
            episode_status = "REJECT" if any(f["status"] == "REJECT" for f in episode_findings) else "NOT_VERIFIED" if any(f["status"] == "NOT_VERIFIED" for f in episode_findings) else "PASS"
            reports.append({"episode_index":index,"status":episode_status,"metrics":metrics,
                            "binding": {key: sidecar["native_lerobot"][key] for key in
                                        ("dataset_id", "dataset_revision", "episode_id", "episode_index")},
                            "data_origin": sidecar["openamrobot"]["provenance"]["data_origin"],
                            "sidecar": {"path": str(path.relative_to(root)),
                                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}})
        context["episode"] = None
        context["source"] = None
        expected = set(episodes) if episode_indices is None else set(episode_indices)
        if expected - seen or not seen:
            emit("C10_NATIVE_METADATA","REJECT",observed=sorted(seen),expected=sorted(expected),fix="Supply exactly one C9 sidecar for every selected native episode.")
    except (FileNotFoundError,PermissionError,ImportError) as error:
        emit("C10_INPUT","NOT_VERIFIED",observed=str(error),fix="Provide the accessible dataset, policy and inspection dependencies.")
    except (OSError,ValueError,KeyError,TypeError,ArithmeticError,AttributeError,yaml.YAMLError,RecursionError) as error:
        emit("C10_INPUT","REJECT",observed=str(error),fix="Provide readable, duplicate-key-free C9/native metadata and a valid reviewed policy.")
    counts = Counter(f["status"] for f in findings)
    status = "REJECT" if counts["REJECT"] else "NOT_VERIFIED" if counts["NOT_VERIFIED"] else "PASS"
    metrics = {"episodes_checked":len(reports),"findings_by_status":dict(sorted(counts.items()))}
    skews = [value["value"] for episode in reports for value in episode["metrics"].get("inter_observation_skew_ns",[])]
    if skews:
        metrics["inter_observation_skew_ns"] = {"measured_frames":len(skews),"min":min(skews),"max":max(skews)}
    coverage = [episode["metrics"]["native_frame_coverage"] for episode in reports if "native_frame_coverage" in episode["metrics"]]
    if coverage:
        metrics["native_frame_coverage"] = {"episodes_measured":len(coverage),"native_frames_loaded":sum(c["native_frames_loaded"] for c in coverage),
                                            "covered_native_frames":sum(c["covered_native_frames"] for c in coverage)}
    return report_value({"report_version":1,"status":status,"exit_code":{"PASS":0,"REJECT":1,"NOT_VERIFIED":2}[status],
            "training_ready":status == "PASS","production_acceptance":"NOT_VERIFIED",
            "findings":findings,"episodes":reports,"provenance":provenance,
            "metrics":metrics})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--policy",type=Path)
    parser.add_argument("--episode",action="append",type=int)
    parser.add_argument("--c9-schema", type=Path, help="Canonical C9 v1 schema from openamrobot-interfaces")
    args = parser.parse_args(argv)
    if args.policy and args.report.resolve() == args.policy.resolve():
        parser.error("--report must not overwrite the reviewed policy")
    if args.c9_schema and args.report.resolve() == args.c9_schema.resolve():
        parser.error("--report must not overwrite the canonical C9 schema")
    try:
        args.report.resolve().relative_to(args.dataset_path.resolve())
    except ValueError:
        pass
    else:
        parser.error("--report must be outside the read-only dataset")
    report = check_dataset(args.dataset_path,args.policy,args.episode, c9_schema_path=args.c9_schema)
    try:
        args.report.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    except (OSError,ValueError) as error:
        print(f"C10 report could not be written: {error}",file=sys.stderr)
        return 2
    print(f"C10 {report['status']}: {len(report['episodes'])} episodes; {report['metrics']['findings_by_status']}")
    for finding in report["findings"]:
        if finding["status"] != "PASS":
            print(f"  {finding['status']} {finding['check_id']} episode={finding['episode']} frame={finding['frame']} stream={finding['stream']}: {finding['fix']}")
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
