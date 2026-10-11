"""Measure lossless oslots CSR vs run-length layout on a real .tf source file.

Only raw array payloads are compared, not complete .cfm directory sizes.
Usage: python tests/unit/storage/measure_oslots_tf.py /path/to/oslots.tf
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from cfabric.storage.csr import CSRArray
from cfabric.utils.helpers import setFromSpec

from test_oslots_rle_experiment import RunLengthCSR, measure_lookup


def scan(path: Path, sample_size: int = 2048) -> None:
    rng = random.Random(20261011)
    sample = []
    seen_sources = set()
    count = 0
    total_slots = 0
    total_runs = 0
    contiguous_rows = 0
    min_source = None
    max_source = 0
    implicit_source = 1
    in_data = False

    with path.open(encoding="utf-8") as source_file:
        for line_number, raw_line in enumerate(source_file, 1):
            line = raw_line.rstrip("\r\n")
            if not in_data:
                if line == "":
                    in_data = True
                continue
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) == 2:
                sources = sorted(setFromSpec(fields[0]))
                targets = tuple(sorted(setFromSpec(fields[1])))
            elif len(fields) == 1:
                sources = [implicit_source]
                targets = tuple(sorted(setFromSpec(fields[0])))
            else:
                raise ValueError(f"unexpected edge row at {line_number}")

            if not sources:
                continue
            implicit_source = max(sources) + 1
            if not targets:
                continue

            runs = 1 + sum(
                b != a + 1 for a, b in zip(targets, targets[1:])
            )
            for node in sources:
                if node in seen_sources:
                    raise ValueError(f"duplicate source node {node} at {line_number}")
                seen_sources.add(node)
                min_source = node if min_source is None else min(min_source, node)
                max_source = max(max_source, node)
                count += 1
                total_slots += len(targets)
                total_runs += runs
                contiguous_rows += runs == 1
                # Uniform reservoir sample for timing; workload-specific estimates
                # are computed over the entire feature, not the sample.
                if len(sample) < sample_size:
                    sample.append(targets)
                else:
                    j = rng.randrange(count)
                    if j < sample_size:
                        sample[j] = targets

    if count == 0:
        raise ValueError("No oslots records found")
    assert min_source is not None
    row_count = max_source - min_source + 1
    if row_count != count:
        raise ValueError("non-contiguous source ID range; source gap handling needed")
    csr_bytes = 4 * (row_count + 1 + total_slots)
    rle_bytes = 4 * (row_count + 1 + 2 * total_runs)
    original = CSRArray.from_sequences(sample)
    alternative = RunLengthCSR.from_sequences(sample)
    for i, expected in enumerate(sample):
        assert tuple(original.get_as_tuple(i)) == expected
        assert alternative.get_as_tuple(i) == expected
    probes = [rng.randrange(len(sample)) for _ in range(1000)]
    csr_time = measure_lookup(original.get_as_tuple, probes)
    rle_time = measure_lookup(alternative.get_as_tuple, probes)
    print(
        f"REAL_OSLOTS corpus={path} source_rows={count} slot_links={total_slots}"
        f" runs={total_runs} contiguous_rows={contiguous_rows}"
        f" contiguous_pct={100 * contiguous_rows / count:.2f}"
        f" csr_bytes={csr_bytes} rle_bytes={rle_bytes}"
        f" reduction_pct={100 * (1 - rle_bytes / csr_bytes):.2f}"
        f" sampled_csr_us_per_row={csr_time:.3f}"
        f" sampled_rle_us_per_row={rle_time:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("oslots_tf", type=Path)
    arguments = parser.parse_args()
    scan(arguments.oslots_tf)
