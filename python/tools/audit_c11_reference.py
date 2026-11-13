#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read-only, exact-SHA C11 dispatch/report probes; no baseline source is vendored.

Executes selected functions from the supplied public reference checkout with mocks.
No dataset generation, ingestion, training, checkpoint loading or rollout is performed.
"""

import argparse
import ast
import json
import logging
from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest.mock import Mock

BASELINE_SHA = "1390ae798f300754303136fae9b3557db025ed0d"


def extract(root, relative, name, context):
    tree = ast.parse((root / relative).read_text(encoding="utf-8-sig"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(root / relative), "exec"), context)
    return context[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    args = parser.parse_args()
    baseline = args.baseline.resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=baseline, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=baseline, text=True).strip()
    if head != BASELINE_SHA or dirty:
        parser.error("Probe requires the clean audited baseline SHA; inspect a new head before changing this probe")
    result = {"baseline_sha": head, "scope": "mocked dispatch/report/gate functions only; no full pipeline or hardware"}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bag = root / "bag"
        bag.mkdir()
        (bag / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n")
        generator = Mock(return_value=root / "output")
        ingest = extract(baseline, "scripts/generate_or_ingest_dataset.py", "ingest_from_bag_or_directory",
                         {"Path": Path, "LOG": logging.getLogger("audit"), "generate_synthetic_dataset": generator})
        ingest(bag, root / "output", source_profile="ROBOT")
        result["synthetic_fallback_probe"] = {"source_profile": generator.call_args.kwargs["source_profile"],
                                               "num_episodes": generator.call_args.kwargs["num_episodes"]}
        payload = json.loads((baseline / "reports/c11_eval_report.json").read_text(encoding="utf-8-sig"))
        payload["provenance"]["c10_preflight_qa_gate"] = "BLOCKED"
        summary = extract(baseline, "scripts/evaluate_act_baseline.py", "generate_markdown_summary",
                          {"Path": Path, "json": json})
        output = root / "report.md"
        summary(payload, output)
        lines = output.read_text().splitlines()
        result["blocked_report_probe"] = {
            "gate_row": next(line for line in lines if "**C10 Pre-Flight QA Gate**" in line),
            "unconditional_acceptance": any("satisfies the C11 acceptance criteria" in line for line in lines)}
        qa = Mock()
        qa.run_qa.return_value = {"summary": {"training_gate": "BLOCKED", "overall_status": "PASS", "passed": 0}}
        gate = extract(baseline, "scripts/train_act_baseline.py", "enforce_preflight_gate",
                       {"Path": Path, "LOG": logging.getLogger("audit"), "json": json,
                        "shutil": shutil, "cdi": qa, "REPO_ROOT": root})
        result["contradictory_gate_probe"] = {"training_gate": "BLOCKED", "overall_status": "PASS",
                                             "training_allowed": gate(root, reports_dir=root / "reports")}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
