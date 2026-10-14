"""Validation gates for the research-only feature density scanner."""
from __future__ import annotations

import pytest
from pathlib import Path
from runpy import run_path

measure = run_path(str(Path(__file__).with_name('measure_feature_density.py')))['measure']


def test_node_feature_implicit_and_explicit_ranges(tmp_path):
    path = tmp_path / "feature.tf"
    path.write_text(
        "@node\n@valueType=str\n\n"
        "1\talpha\n"
        "2-3\tbeta\n"
        "\n"  # implicit node 4, explicitly empty string
        "6\talpha\n",
        encoding="utf-8",
    )
    r = measure(path, max_node=7)
    assert r["populated"] == 5
    assert r["unique_encoded_values"] == 3
    assert r["dense_bytes"] == 28
    assert r["sparse_pairs_bytes"] == 40
    assert r["bitmap_values_bytes_lower_bound"] == 21


def test_numeric_blank_is_missing_and_zero_is_present(tmp_path):
    path = tmp_path / "numeric.tf"
    path.write_text("@node\n@valueType=int\n\n1\t0\n\n3\t4\n", encoding="utf-8")
    r = measure(path, max_node=5)
    assert r["populated"] == 2
    assert r["unique_encoded_values"] == 2


def test_duplicates_fail_closed(tmp_path):
    path = tmp_path / "dup.tf"
    path.write_text("@node\n\n1\tA\n1\tB\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate feature node"):
        measure(path, max_node=10)


def test_unordered_distinct_nodes_are_legal(tmp_path):
    path = tmp_path / "unsorted.tf"
    path.write_text("@node\n\n3\tA\n1\tB\n2\tC\n", encoding="utf-8")
    r = measure(path, max_node=10)
    assert r["populated"] == 3
    assert r["unique_encoded_values"] == 3


def test_out_of_range_and_not_a_node_feature(tmp_path):
    path = tmp_path / "invalid.tf"
    path.write_text("@node\n\n11\tA\n", encoding="utf-8")
    with pytest.raises(ValueError, match="> 10"):
        measure(path, max_node=10)
    path.write_text("@edge\n\n1\t2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected @node"):
        measure(path, max_node=10)
