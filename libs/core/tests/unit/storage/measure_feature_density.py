"""Research-only estimator for sparse feature storage on real Text-Fabric data.

Usage:
    python tests/unit/storage/measure_feature_density.py MAX_NODE feature1.tf [feature2.tf ...]

Current CF node feature arrays allocate an int32/uint32 value or string-pool
index for every node. This script reports data-array *lower bounds* for
alternative representations; it does not implement indexed sparse lookup,
string pool tables, or count on-disk .npy headers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from cfabric.utils.helpers import setFromSpec


def measure(path: Path, max_node: int) -> dict[str, int | float | str]:
    data_started = False
    metadata: dict[str, str] = {}
    feature_type = None
    implicit_node = 1
    populated = 0
    last_node = 0
    distinct: set[str] = set()
    with path.open(encoding="utf-8") as src:
        for line_number, raw in enumerate(src, 1):
            line = raw.rstrip("\r\n")
            if not data_started:
                if feature_type is None:
                    feature_type = line
                    if line != "@node":
                        raise ValueError(f"{path}: expected @node not {line!r}")
                    continue
                if not line:
                    data_started = True
                    continue
                if line.startswith("@") and "=" in line:
                    key, value = line[1:].split("=", 1)
                    metadata[key] = value
                continue
            fields = line.split("\t")
            if len(fields) == 1:
                n_nodes = 1
                first_seen = implicit_node
                max_seen = implicit_node
                raw_value = fields[0]
            elif len(fields) == 2:
                nodes = setFromSpec(fields[0])
                if not nodes:
                    raise ValueError(f"{path}:{line_number}: empty node spec")
                n_nodes = len(nodes)
                first_seen = min(nodes)
                max_seen = max(nodes)
                raw_value = fields[1]
            else:
                raise ValueError(f"{path}:{line_number}: unexpected fields")
            if first_seen <= last_node:
                raise ValueError(
                    f"{path}:{line_number}: duplicate/unordered feature node "
                    f"{first_seen} after {last_node}"
                )
            last_node = max_seen
            implicit_node = max_seen + 1
            if max_seen > max_node:
                raise ValueError(f"{path}:{line_number}: node {max_seen} > {max_node}")
            # Numeric blank => omitted; string empty => explicitly present.
            if metadata.get("valueType", "str") == "int" and raw_value == "":
                continue
            populated += n_nodes
            distinct.add(raw_value)
    dense_bytes = 4 * max_node
    sparse_pairs = 8 * populated
    bitmap_values = (max_node + 7) // 8 + 4 * populated
    # Codes require a missing sentinel, so <=255 unique strings fit in u8.
    unique = len(distinct)
    code_width = (1 if unique <= 255 else 2 if unique <= 65535 else 4)
    dense_narrow = code_width * max_node
    return {
        "feature": path.name,
        "value_type": metadata.get("valueType", "str"),
        "max_node": max_node,
        "populated": populated,
        "coverage_pct": round(100 * populated / max_node, 3),
        "unique_encoded_values": unique,
        "dense_bytes": dense_bytes,
        "sparse_pairs_bytes": sparse_pairs,
        "bitmap_values_bytes_lower_bound": bitmap_values,
        "narrow_dense_code_bytes_lower_bound": dense_narrow,
        "candidate_code_bytes": code_width,
        # A real sparse index also needs rank/select metadata, not counted.
        "best_payload_lower_bound": min(dense_bytes, sparse_pairs, bitmap_values, dense_narrow),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("max_node", type=int)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for feature in args.paths:
        result = measure(feature, args.max_node)
        print(
            "REAL_FEATURE " + " ".join(f"{k}={v}" for k, v in result.items()),
            flush=True,
        )


if __name__ == "__main__":
    main()
