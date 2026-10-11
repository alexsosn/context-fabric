"""Real corpus multi-process *feature-only* memory benchmark.

Run on Linux:
  python tests/unit/storage/benchmark_sparse_worker_memory.py path/to/docid.tf 8285600

Reports /proc/self/smaps_rollup after mmap arrays are touched by each of N
independent SPAWN workers. Per-worker Python/import/string-vocabulary memory
is included. These numbers do NOT represent full compiled CFM RSS.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import tempfile
from pathlib import Path

import numpy as np

from cfabric.io.compiler import Compiler
from cfabric.storage.string_pool import StringPool
from cfabric.storage.sparse_string_pool import SparseStringPool


def linux_memory_kib() -> dict[str, int]:
    """Return unshared and proportional process memory from Linux smaps_rollup."""
    values = {}
    with open("/proc/self/smaps_rollup", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 2 and parts[0].endswith(":"):
                key = parts[0][:-1]
                if key in {"Rss", "Pss", "Private_Clean", "Private_Dirty", "Shared_Clean", "Shared_Dirty"}:
                    values[key] = int(parts[1])
    required = {"Rss", "Pss", "Private_Clean", "Private_Dirty"}
    if not required.issubset(values):
        raise RuntimeError("Linux smaps_rollup is incomplete")
    values["Uss"] = values["Private_Clean"] + values["Private_Dirty"]
    return values


def _worker(pipe, mode: str, prefix: str, max_node: int) -> None:
    try:
        if mode == "dense":
            storage = StringPool.load(prefix, mmap_mode="r")
            # Touch all pages in the dense index array (prevent laziness from
            # making disk-backed RSS appear artificially low).
            checksum = int(np.sum(storage.indices, dtype=np.uint64))
        elif mode == "sparse":
            storage = SparseStringPool.load(prefix, max_node=max_node, mmap_mode="r")
            checksum = int(np.sum(storage.nodes, dtype=np.uint64))
            checksum += int(np.sum(storage.codes, dtype=np.uint64))
            # Exercise the process-private lazy random-lookup block index.
            for node in range(1, 80):
                checksum += storage.get(node) is not None
        else:
            raise ValueError(mode)
        pipe.send({"pid": os.getpid(), "mode": mode, "checksum": checksum, **linux_memory_kib()})
        if pipe.recv() != "stop":
            raise RuntimeError("missing parent stop signal")
    except BaseException as error:
        pipe.send({"error": repr(error)})
        raise
    finally:
        pipe.close()


def _measure(mode: str, prefix: str, max_node: int, count: int) -> dict:
    ctx = mp.get_context("spawn")
    children = []
    for _ in range(count):
        parent_conn, child_conn = ctx.Pipe()
        process = ctx.Process(target=_worker, args=(child_conn, mode, prefix, max_node))
        process.start()
        child_conn.close()
        children.append((process, parent_conn))
    try:
        observations = [connection.recv() for _, connection in children]
        if any("error" in item for item in observations):
            raise RuntimeError(f"worker errors: {observations}")
        if len({item["pid"] for item in observations}) != count:
            raise RuntimeError("worker PID collision")
        return {
            "workers": count,
            "mode": mode,
            "total_pss_kib": sum(item["Pss"] for item in observations),
            "total_uss_kib": sum(item["Uss"] for item in observations),
            "total_rss_kib": sum(item["Rss"] for item in observations),
            "worker_memories": [
                {key: item[key] for key in ("Rss", "Pss", "Uss")} for item in observations
            ],
        }
    finally:
        for process, connection in children:
            try:
                connection.send("stop")
            except (EOFError, BrokenPipeError, OSError):
                pass
            connection.close()
            process.join(timeout=15)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


def run(tf_path: Path, max_node: int, workers: int = 3) -> None:
    if workers < 1 or workers > 8:
        raise ValueError("workers must be between 1 and 8")
    if os.name != "posix" or not Path("/proc/self/smaps_rollup").exists():
        raise RuntimeError("this benchmark requires Linux procfs")
    metadata, data, is_edge, _, is_config = Compiler(str(tf_path.parent))._parse_tf_file(tf_path)
    if is_edge or is_config or metadata.get("valueType", "str") != "str":
        raise ValueError("expected string-valued node feature")
    if not data:
        raise ValueError("expected nonempty feature")
    with tempfile.TemporaryDirectory() as tmp:
        dense_path = Path(tmp) / "dense" / tf_path.stem
        sparse_path = Path(tmp) / "sparse" / tf_path.stem
        dense_path.parent.mkdir()
        sparse_path.parent.mkdir()
        dense = StringPool.from_dict(data, max_node)
        sparse = SparseStringPool.from_dict(data, max_node)
        assert dict(sparse.items()) == dict(dense.items())
        dense.save(str(dense_path))
        sparse.save(str(sparse_path))
        for count in (1, workers):
            dense_observation = _measure("dense", str(dense_path), max_node, count)
            sparse_observation = _measure("sparse", str(sparse_path), max_node, count)
            print(
                "REAL_SPARSE_WORKERS " + json.dumps(
                    {
                        "feature": tf_path.name,
                        "max_node": max_node,
                        "filled": len(data),
                        "dense": dense_observation,
                        "sparse": sparse_observation,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("feature", type=Path)
    parser.add_argument("max_node", type=int)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    run(args.feature, args.max_node, args.workers)
