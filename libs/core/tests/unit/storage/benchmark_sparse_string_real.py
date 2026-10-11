"""Reproducible real-feature payload and latency comparison.

Usage:
  python tests/unit/storage/benchmark_sparse_string_real.py SOURCE.tf MAX_NODE

Prints raw payloads and complete npy directory sizes. The normal CFM
compiled corpus is not being replaced by this single-feature experiment.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import tempfile
import time
from pathlib import Path

from cfabric.io.compiler import Compiler
from cfabric.storage.sparse_string_pool import SparseStringPool
from cfabric.storage.string_pool import StringPool


def elapsed_ns(fn, nodes, repeats=3):
    times = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        checksum = 0
        for n in nodes:
            checksum += fn(n) is not None
        times.append((time.perf_counter_ns() - start) / len(nodes))
        assert checksum >= 0
    return round(statistics.median(times), 2)


def query_ms(fn, repeats=3):
    durations = []
    previous = None
    for _ in range(repeats):
        start = time.perf_counter_ns()
        nodes = fn()
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if previous is not None:
            assert nodes.tolist() == previous
        previous = nodes.tolist()
        durations.append(elapsed)
    return round(statistics.median(durations), 3), len(previous or [])


def run(tf: Path, max_node: int):
    metadata, data, is_edge, _, is_config = Compiler(str(tf.parent))._parse_tf_file(tf)
    if is_edge or is_config or metadata.get("valueType", "str") != "str":
        raise ValueError("expected a string node feature")
    if not data:
        raise ValueError("expected nonempty string feature")
    assert all(isinstance(val, str) for val in data.values())

    dense = StringPool.from_dict(data, max_node)
    sparse = SparseStringPool.from_dict(data, max_node)
    assert sparse.to_dict() == data
    assert dense.to_dict() == data
    assert sparse.get_frequency_counts() == dense.get_frequency_counts()
    existing = sorted(data)
    selected = existing[::max(1, len(existing) // 2000)]
    missing = [n for n in range(1, max_node + 1, max(1, max_node // 2000)) if n not in data]
    rng = random.Random(2410)
    probes = selected + missing
    rng.shuffle(probes)
    assert all(dense.get(n) == sparse.get(n) for n in probes)

    with tempfile.TemporaryDirectory() as d:
        old = Path(d) / "dense" / tf.stem
        new = Path(d) / "sparse" / tf.stem
        old.parent.mkdir()
        new.parent.mkdir()
        dense.save(str(old))
        sparse.save(str(new))
        from_disk = SparseStringPool.load(str(new), max_node)
        dense_from_disk = StringPool.load(str(old))
        assert from_disk.to_dict() == data
        assert dense_from_disk.to_dict() == data
        assert all(dense_from_disk.get(n) == from_disk.get(n) for n in probes)

        # The query backend always compares entire CFM node space on F.s(),
        # so also measure filter_by_value(range(1, max_node + 1)).
        first_value = data[existing[0]]
        query_nodes = range(1, max_node + 1)
        dense_matches = dense_from_disk.filter_by_value(query_nodes, first_value)
        sparse_matches = from_disk.filter_by_value(query_nodes, first_value)
        assert dense_matches.tolist() == sparse_matches.tolist()
        dense_filter_ms, matched_count = query_ms(
            lambda: dense_from_disk.filter_by_value(query_nodes, first_value)
        )
        sparse_filter_ms, _ = query_ms(
            lambda: from_disk.filter_by_value(query_nodes, first_value)
        )
        totals = (
            sum(p.stat().st_size for p in old.parent.iterdir()),
            sum(p.stat().st_size for p in new.parent.iterdir()),
        )
        measured = {
            "feature": tf.name,
            "max_node": max_node,
            "populated": len(data),
            "coverage_pct": round(100 * len(data) / max_node, 3),
            "unique_values": len(sparse.strings),
            "codes_dtype": str(sparse.codes.dtype),
            "dense_payload_bytes": dense.indices.nbytes,
            "sparse_payload_bytes": sparse.payload_nbytes,
            "dense_full_npy_bytes": totals[0],
            "sparse_full_npy_bytes": totals[1],
            "npy_saving_pct": round(100 * (1 - totals[1] / totals[0]), 2),
            "lookup_probe_count": len(probes),
            "dense_lookup_ns": elapsed_ns(dense_from_disk.get, probes),
            "sparse_lookup_ns": elapsed_ns(from_disk.get, probes),
            "sparse_scalar_index_bytes": (
                0 if from_disk._bucket_offsets is None
                else from_disk._bucket_offsets.nbytes
            ),
            "filtered_value_count": matched_count,
            "dense_filter_ms": dense_filter_ms,
            "sparse_filter_ms": sparse_filter_ms,
        }
    print("REAL_SPARSE_STRING " + json.dumps(measured, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_tf", type=Path)
    parser.add_argument("max_node", type=int)
    args = parser.parse_args()
    run(args.feature_tf, args.max_node)
