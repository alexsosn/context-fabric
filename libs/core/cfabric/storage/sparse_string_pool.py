"""Experimental sparse mmap storage for string-valued node features.

This backend is opt-in at CFM compile time. It retains the standard StringPool
vocabulary format but replaces the dense max_node-long index array with sorted
populated node IDs and corresponding dictionary codes. No sentinel: an explicit
empty string is distinct from an absent value.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

MAX_NODE_UINT32 = 2**32 - 1


class SparseStringPool:
    def __init__(
        self,
        strings: np.ndarray,
        nodes: np.ndarray,
        codes: np.ndarray,
        max_node: int,
    ) -> None:
        if len(nodes) != len(codes):
            raise ValueError("sparse node and code array lengths differ")
        if not 0 <= max_node <= MAX_NODE_UINT32:
            raise ValueError("max_node exceeds uint32 capacity")
        self.strings = strings
        self.nodes = nodes
        self.codes = codes
        self.max_node = max_node
        # Existing StringPool.get_value_index() performs a linear scan, which
        # would make large sparse vocabulary filters prohibitively expensive.
        self._lookup: dict[str, int] | None = None

    @staticmethod
    def code_dtype(n_strings: int) -> type[np.unsignedinteger[Any]]:
        if n_strings <= 256:
            return np.uint8
        if n_strings <= 65536:
            return np.uint16
        return np.uint32

    @classmethod
    def from_dict(cls, data: dict[int, str], max_node: int) -> SparseStringPool:
        if not 0 <= max_node <= MAX_NODE_UINT32:
            raise ValueError("max_node exceeds uint32 capacity")
        if any(n < 1 or n > max_node for n in data):
            raise ValueError("sparse feature node outside 1..max_node")
        keys = sorted(data)
        vocabulary = sorted(set(data.values()))
        lookup = {value: i for i, value in enumerate(vocabulary)}
        codes = np.fromiter(
            (lookup[data[node]] for node in keys),
            dtype=cls.code_dtype(len(vocabulary)),
            count=len(keys),
        )
        nodes = np.asarray(keys, dtype=np.uint32)
        return cls(np.asarray(vocabulary, dtype=object), nodes, codes, max_node)

    @classmethod
    def is_worth_using(
        cls,
        data: dict[int, str],
        max_node: int,
        min_saving_fraction: float = 0.25,
    ) -> bool:
        """Payload-only gate. Avoid sparse when size saving is marginal.

        Both formats use the same string vocabulary so only index arrays
        are compared. .npy header overhead is intentionally excluded.
        """
        if max_node <= 0 or not data:
            return False
        unique_count = len(set(data.values()))
        dtype_size = np.dtype(cls.code_dtype(unique_count)).itemsize
        sparse_bytes = len(data) * (4 + dtype_size)
        dense_bytes = max_node * 4
        return sparse_bytes <= dense_bytes * (1.0 - min_saving_fraction)

    @property
    def payload_nbytes(self) -> int:
        return self.nodes.nbytes + self.codes.nbytes

    def __len__(self) -> int:
        # The feature API expects the complete addressable node space.
        return self.max_node

    def get(self, node: int) -> str | None:
        if node < 1 or node > self.max_node or len(self.nodes) == 0:
            return None
        position = int(np.searchsorted(self.nodes, node))
        if position >= len(self.nodes) or int(self.nodes[position]) != node:
            return None
        return str(self.strings[int(self.codes[position])])

    def __getitem__(self, node: int) -> str | None:
        return self.get(node)

    def items(self) -> Iterator[tuple[int, str]]:
        for node, code in zip(self.nodes, self.codes):
            yield int(node), str(self.strings[int(code)])

    def to_dict(self) -> dict[int, str]:
        return dict(self.items())

    def get_value_index(self, value: str) -> int | None:
        if self._lookup is None:
            self._lookup = {str(v): i for i, v in enumerate(self.strings)}
        return self._lookup.get(value)

    def _query_nodes(self, nodes: Sequence[int] | range) -> np.ndarray:
        if isinstance(nodes, range):
            # .s() passes range(1, max_node + 1). It must not allocate
            # a dense node-sized array just to filter a sparse feature.
            if nodes.start == 1 and nodes.step == 1 and nodes.stop == self.max_node + 1:
                return self.nodes.astype(np.int64, copy=False)
            return np.asarray(nodes, dtype=np.int64)
        return np.asarray(nodes, dtype=np.int64)

    def _match_values(self, codes: set[int]) -> np.ndarray:
        if not codes or len(self.nodes) == 0:
            return np.empty(0, dtype=np.int64)
        mask = np.isin(self.codes, list(codes))
        return self.nodes[mask].astype(np.int64, copy=False)

    def _retain_input_order(
        self, nodes: Sequence[int] | range, matching: np.ndarray,
    ) -> np.ndarray:
        if isinstance(nodes, range) and nodes.start == 1 and nodes.step == 1 and nodes.stop == self.max_node + 1:
            return matching
        query = self._query_nodes(nodes)
        valid = (query >= 1) & (query <= self.max_node)
        query = query[valid]
        if len(query) == 0:
            return query
        return query[np.isin(query, matching)]

    def filter_by_value(self, nodes: Sequence[int] | range, value: str) -> np.ndarray:
        idx = self.get_value_index(value)
        return self._retain_input_order(nodes, self._match_values({idx} if idx is not None else set()))

    def filter_by_values(self, nodes: Sequence[int] | range, values: set[str]) -> np.ndarray:
        codes = {idx for val in values if (idx := self.get_value_index(val)) is not None}
        return self._retain_input_order(nodes, self._match_values(codes))

    def filter_has_value(self, nodes: Sequence[int] | range) -> np.ndarray:
        return self._retain_input_order(nodes, self.nodes.astype(np.int64, copy=False))

    def filter_missing_value(self, nodes: Sequence[int] | range) -> np.ndarray:
        # Returning all missing nodes necessarily costs O(max_node) for an
        # all-nodes query, exactly as with the dense backend.
        query = self._query_nodes(nodes)
        query = query[(query >= 1) & (query <= self.max_node)]
        return query[~np.isin(query, self.nodes)]

    def get_frequency_counts(self) -> dict[str, int]:
        unique, counts = np.unique(self.codes, return_counts=True)
        return {str(self.strings[int(code)]): int(count) for code, count in zip(unique, counts)}

    def save(self, prefix: str) -> None:
        np.save(f"{prefix}_strings.npy", self.strings, allow_pickle=True)
        np.save(f"{prefix}_nodes.npy", self.nodes)
        np.save(f"{prefix}_codes.npy", self.codes)

    @classmethod
    def load(cls, prefix: str, max_node: int, mmap_mode: str = "r") -> SparseStringPool:
        strings = np.load(f"{prefix}_strings.npy", allow_pickle=True)
        nodes = np.load(f"{prefix}_nodes.npy", mmap_mode=mmap_mode)
        codes = np.load(f"{prefix}_codes.npy", mmap_mode=mmap_mode)
        return cls(strings, nodes, codes, max_node)
