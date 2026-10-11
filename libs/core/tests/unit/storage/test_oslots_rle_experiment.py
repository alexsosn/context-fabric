"""Research-only benchmark: CSR versus inclusive-run encoding of oslots.

Run: python -m pytest tests/unit/storage/test_oslots_rle_experiment.py -s -q
Nothing here changes production CFM compilation/loading.
"""
from __future__ import annotations

import random
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import pytest

from cfabric.storage.csr import CSRArray
from cfabric.utils.helpers import setFromSpec


class RunLengthCSR:
    """Prototype of variable-length, increasing uint32 slot IDs as inclusive runs."""

    def __init__(self, indptr: np.ndarray, bounds: np.ndarray):
        self.indptr = indptr
        self.bounds = bounds

    @classmethod
    def from_sequences(cls, rows):
        indptr = np.zeros(len(rows) + 1, dtype=np.uint32)
        bounds = []
        for i, row in enumerate(rows):
            previous = None
            for raw in row:
                value = int(raw)
                if not 0 < value < 2**32:
                    raise ValueError("slot id outside uint32 or zero")
                if previous is not None and value <= previous:
                    raise ValueError("slot IDs must be strictly increasing")
                if previous is None or value != previous + 1:
                    bounds.extend((value, value))
                else:
                    bounds[-1] = value
                previous = value
            indptr[i + 1] = len(bounds) // 2
        return cls(indptr, np.asarray(bounds, dtype=np.uint32))

    def get_as_tuple(self, i):
        if not 0 <= i < len(self.indptr) - 1:
            raise IndexError(i)
        start = int(self.indptr[i]) * 2
        end = int(self.indptr[i + 1]) * 2
        return tuple(
            slot
            for j in range(start, end, 2)
            for slot in range(int(self.bounds[j]), int(self.bounds[j + 1]) + 1)
        )

    @property
    def nbytes(self):
        return self.indptr.nbytes + self.bounds.nbytes


class AdaptiveRunCSR:
    """Per-row CSR or run storage, with one uint8 mode per row."""

    def __init__(self, indptr, data, modes):
        self.indptr, self.data, self.modes = indptr, data, modes

    @classmethod
    def from_sequences(cls, rows):
        indptr = np.zeros(len(rows) + 1, dtype=np.uint32)
        modes = np.zeros(len(rows), dtype=np.uint8)
        words = []
        for i, row in enumerate(rows):
            encoded = RunLengthCSR.from_sequences([row]).bounds
            if len(encoded) < len(row):
                modes[i] = 1
                words.extend(encoded)
            else:
                words.extend(row)
            indptr[i + 1] = len(words)
        return cls(indptr, np.asarray(words, dtype=np.uint32), modes)

    def get_as_tuple(self, i):
        if not 0 <= i < len(self.modes):
            raise IndexError(i)
        a, b = int(self.indptr[i]), int(self.indptr[i + 1])
        if not self.modes[i]:
            return tuple(self.data[a:b])
        return tuple(
            slot
            for j in range(a, b, 2)
            for slot in range(int(self.data[j]), int(self.data[j + 1]) + 1)
        )

    @property
    def nbytes(self):
        return self.indptr.nbytes + self.data.nbytes + self.modes.nbytes


def make_rows(shape, count=6000, width=64):
    rows = []
    for i in range(count):
        base = i * width * 4 + 1
        if shape == "contiguous" or (shape == "mixed" and i % 5 != 0):
            rows.append(tuple(range(base, base + width)))
        elif shape in ("fragmented", "mixed"):
            rows.append(tuple(range(base, base + width * 2, 2)))
        else:
            raise ValueError(shape)
    return rows


def compare(rows):
    baseline = CSRArray.from_sequences(rows)
    candidate = RunLengthCSR.from_sequences(rows)
    assert len(baseline) == len(candidate.indptr) - 1
    for i, expected in enumerate(rows):
        assert tuple(baseline.get_as_tuple(i)) == tuple(expected)
        assert candidate.get_as_tuple(i) == tuple(expected)
    csr_bytes = baseline.indptr.nbytes + baseline.data.nbytes
    run_bytes = candidate.nbytes
    return baseline, candidate, csr_bytes, run_bytes


def measure_lookup(get, probes):
    # Report observations; do not make CI assertions about elapsed time.
    durations = []
    for _ in range(5):
        start = perf_counter()
        consumed = sum(len(get(int(i))) for i in probes)
        assert consumed > 0
        durations.append((perf_counter() - start) * 1e6 / len(probes))
    return median(durations)


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ([], []),
        ([[]], [()]),
        ([[1], [3, 4, 5], [7, 9], []], [(1,), (3, 4, 5), (7, 9), ()]),
        ([[2**32 - 2, 2**32 - 1]], [(2**32 - 2, 2**32 - 1)]),
    ],
)
def test_lossless_edge_cases(rows, expected):
    baseline, candidate, _, _ = compare(rows)
    assert [candidate.get_as_tuple(i) for i in range(len(baseline))] == expected


@pytest.mark.parametrize("rows", [[[0]], [[2, 2]], [[2, 1]], [[2**32]]])
def test_reject_noncanonical_or_out_of_range(rows):
    with pytest.raises(ValueError):
        RunLengthCSR.from_sequences(rows)


@pytest.mark.parametrize("shape", ["contiguous", "mixed", "fragmented"])
def test_adaptive_lossless_and_storage_bound(shape):
    rows = make_rows(shape, 64, 32)
    csr, rle, csr_bytes, rle_bytes = compare(rows)
    adaptive = AdaptiveRunCSR.from_sequences(rows)
    assert [adaptive.get_as_tuple(i) for i in range(len(rows))] == rows
    assert adaptive.nbytes <= min(csr_bytes, rle_bytes) + len(rows)


def test_repository_fixture():
    root = Path(__file__).resolve().parents[3]
    tf = root / "tests" / "fixtures" / "mini_corpus" / "oslots.tf"
    data = tf.read_text(encoding="utf8").split("\n\n", 1)[1]
    edges = {}
    for line in data.splitlines():
        if not line:
            continue
        source, slots = line.split("\t")
        edges[int(source)] = tuple(sorted(setFromSpec(slots)))
    rows = [edges[n] for n in sorted(edges)]
    assert rows == [(1, 2, 3), (4, 5), (1, 2, 3, 4, 5)]
    compare(rows)


@pytest.mark.parametrize("shape", ["contiguous", "mixed", "fragmented"])
def test_synthetic_benchmark(shape):
    rows = make_rows(shape)
    baseline, candidate, csr_bytes, run_bytes = compare(rows)
    adaptive = AdaptiveRunCSR.from_sequences(rows)
    probes = [random.Random(20261011 + j).randrange(len(rows)) for j in range(1000)]
    csr_time = measure_lookup(baseline.get_as_tuple, probes)
    run_time = measure_lookup(candidate.get_as_tuple, probes)
    adaptive_time = measure_lookup(adaptive.get_as_tuple, probes)
    saving = (1 - run_bytes / csr_bytes) * 100
    print(
        f"OSLOTS_BENCH shape={shape} nodes={len(rows)} slots={sum(map(len, rows))}"
        f" runs={len(candidate.bounds) // 2} csr_bytes={csr_bytes}"
        f" run_bytes={run_bytes} adaptive_bytes={adaptive.nbytes} saving_pct={saving:.2f}"
        f" csr_us_per_row={csr_time:.3f} run_us_per_row={run_time:.3f}"
        f" adaptive_us_per_row={adaptive_time:.3f}",
        flush=True,
    )
    if shape == "fragmented":
        assert run_bytes > csr_bytes  # counterexample prevents blind adoption
    else:
        assert run_bytes < csr_bytes
