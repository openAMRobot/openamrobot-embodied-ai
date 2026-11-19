# SPDX-License-Identifier: MIT
"""Cut a receipt-time window of selected topics from a ROS 2 bag, bytes unchanged.

Used to make the recorded test fixtures (python/tests/fixtures/baseline1_slice):
every kept message keeps its original serialized CDR bytes, its bag receipt
time, its topic, type, message definition and type hash. Nothing is
deserialized, reordered, resampled or re-stamped. The output is an MCAP bag
with zstd chunk compression. A JSON report lists, per topic, the messages in
the source window and in the output (they must agree) and the sha256 of the
output files.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cut(source, output, start_ns, end_ns, topics):
    from rosbags.highlevel import AnyReader
    from rosbags.rosbag2 import StoragePlugin, Writer
    from rosbags.rosbag2.enums import CompressionFormat, CompressionMode

    output = Path(output)
    if output.exists():
        raise SystemExit(f"output exists; refusing overwrite: {output}")
    counts = {t: 0 for t in topics}
    with AnyReader([Path(source)]) as reader:
        selected = [c for c in reader.connections if c.topic in counts]
        missing = sorted(set(topics) - {c.topic for c in selected})
        if missing:
            raise SystemExit(f"topics missing from the source bag: {missing}")
        writer = Writer(output, version=9, storage_plugin=StoragePlugin.MCAP)
        writer.set_compression(CompressionMode.STORAGE, CompressionFormat.ZSTD)
        with writer:
            out = {}
            for conn in sorted(selected, key=lambda c: c.topic):
                out[conn.id] = writer.add_connection(
                    conn.topic,
                    conn.msgtype,
                    msgdef=conn.msgdef.data,
                    rihs01=conn.digest,
                    serialization_format=conn.ext.serialization_format,
                    offered_qos_profiles=conn.ext.offered_qos_profiles,
                )
            for conn, receipt, raw in reader.messages(
                connections=selected, start=start_ns, stop=end_ns
            ):
                writer.write(out[conn.id], receipt, raw)
                counts[conn.topic] += 1
    written = {}
    with AnyReader([output]) as reader:
        for conn in reader.connections:
            written[conn.topic] = written.get(conn.topic, 0) + conn.msgcount
    if written != counts:
        raise SystemExit(f"written counts differ from the source window: {written} != {counts}")
    return {
        "source": str(source),
        "window_receipt_ns": [start_ns, end_ns],
        "window_rule": "bag receipt time start <= t < stop (rosbags AnyReader.messages start/stop)",
        "messages_per_topic": counts,
        "output_sha256": {
            p.relative_to(output).as_posix(): sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
        "bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start-ns", type=int, required=True)
    parser.add_argument("--end-ns", type=int, required=True)
    parser.add_argument("--topic", action="append", required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.end_ns <= args.start_ns:
        raise SystemExit("--end-ns must be after --start-ns")
    report = cut(args.source, args.output, args.start_ns, args.end_ns, sorted(set(args.topic)))
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
