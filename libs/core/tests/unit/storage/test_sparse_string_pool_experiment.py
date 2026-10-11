"""TDD contract for optional sparse string features in CFM."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from cfabric.storage.sparse_string_pool import SparseStringPool
from cfabric.io.compiler import Compiler
from cfabric.core import Fabric


def test_roundtrip_and_full_node_space(tmp_path):
    values = {1: "aleph", 17: "", 500: "βήτα", 1000: "aleph"}
    pool = SparseStringPool.from_dict(values, max_node=1000)
    assert len(pool) == 1000
    assert pool.get(0) is None
    assert pool.get(-1) is None
    assert pool.get(2) is None
    assert pool.get(1001) is None
    assert pool.get(1) == "aleph"
    assert pool.get(17) == ""
    assert pool.get(500) == "βήτα"
    assert dict(pool.items()) == values
    pool.save(str(tmp_path / "lemma"))
    reread = SparseStringPool.load(str(tmp_path / "lemma"))
    assert isinstance(reread.nodes, np.memmap)
    assert isinstance(reread.codes, np.memmap)
    assert dict(reread.items()) == values
    assert reread.get_frequency_counts() == {"aleph": 2, "": 1, "βήτα": 1}


def test_filters_agree_with_dense_string_pool():
    from cfabric.storage.string_pool import StringPool

    mapping = {1: "A", 4: "", 8: "B", 15: "A"}
    dense = StringPool.from_dict(mapping, max_node=20)
    sparse = SparseStringPool.from_dict(mapping, max_node=20)
    inputs = [range(1, 21), [15, 3, 1, 8, 99, 0, 4], [], range(12, 19)]
    for nodes in inputs:
        for value in ("A", "", "B", "not-found"):
            assert sparse.filter_by_value(nodes, value).tolist() == dense.filter_by_value(nodes, value).tolist()
        for vals in ({"A", "B"}, set(), {""}, {"missing"}):
            assert sparse.filter_by_values(nodes, vals).tolist() == dense.filter_by_values(nodes, vals).tolist()
        assert sparse.filter_has_value(nodes).tolist() == dense.filter_has_value(nodes).tolist()
        assert sparse.filter_missing_value(nodes).tolist() == dense.filter_missing_value(nodes).tolist()
    assert sparse.get_value_index("nope") is None


def test_sparse_code_widths_and_validation():
    s = SparseStringPool.from_dict({1: "a", 7: "b"}, max_node=20)
    assert s.codes.dtype == np.dtype("uint8")
    many = SparseStringPool.from_dict({i: f"key-{i}" for i in range(1, 302)}, max_node=500)
    assert many.codes.dtype == np.dtype("uint16")
    with pytest.raises(ValueError):
        SparseStringPool.from_dict({0: "bad"}, max_node=100)
    with pytest.raises(ValueError):
        SparseStringPool.from_dict({101: "bad"}, max_node=100)


def test_empty_and_out_of_range_selection(tmp_path):
    pool = SparseStringPool.from_dict({}, max_node=5)
    pool.save(str(tmp_path / "empty"))
    loaded = SparseStringPool.load(str(tmp_path / "empty"))
    assert list(loaded.items()) == []
    assert loaded.get(1) is None
    assert loaded.filter_by_value(range(1, 6), "x").size == 0
    assert loaded.filter_missing_value(range(1, 6)).tolist() == [1, 2, 3, 4, 5]


def test_adaptive_compiler_and_fabric_api(tmp_path):
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "mini_corpus"
    root = tmp_path / "mini"
    shutil.copytree(fixture, root)
    (root / "rare_label.tf").write_text(
        "@node\n@valueType=str\n@description=sparse test\n\n1\tα\n8\tα\n",
        encoding="utf8",
    )
    assert Compiler(str(root), experimental_sparse_strings=True).compile()
    feature_dir = root / ".cfm" / "1" / "features"
    meta = json.loads((feature_dir / "rare_label_meta.json").read_text())
    assert meta["storage"] == "sparse-string-v1"
    assert (feature_dir / "rare_label_nodes.npy").exists()
    assert (feature_dir / "rare_label_codes.npy").exists()
    assert not (feature_dir / "rare_label_idx.npy").exists()

    api = Fabric(locations=str(root), silent="deep").load("rare_label")
    assert api.F.rare_label.v(1) == "α"
    assert api.F.rare_label.v(5) is None
    assert api.F.rare_label.v(8) == "α"
    assert sorted(api.F.rare_label.s("α")) == [1, 8]
    assert dict(api.F.rare_label.items()) == {1: "α", 8: "α"}
    assert api.F.rare_label.freqList() == (("α", 2),)


def test_default_compilation_unchanged(tmp_path):
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "mini_corpus"
    root = tmp_path / "mini"
    shutil.copytree(fixture, root)
    (root / "rare_label.tf").write_text("@node\n@valueType=str\n\n1\tX\n", encoding="utf8")
    assert Compiler(str(root)).compile()
    features = root / ".cfm" / "1" / "features"
    meta = json.loads((features / "rare_label_meta.json").read_text())
    assert meta.get("storage") is None
    assert (features / "rare_label_idx.npy").exists()
    assert not (features / "rare_label_nodes.npy").exists()


def test_dense_threshold_does_not_select_sparse(tmp_path):
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "mini_corpus"
    root = tmp_path / "mini"
    shutil.copytree(fixture, root)
    (root / "frequent_label.tf").write_text(
        "@node\n@valueType=str\n\n" + "".join(f"{i}\tZ\n" for i in range(1, 9)), encoding="utf8"
    )
    assert Compiler(str(root), experimental_sparse_strings=True).compile()
    features = root / ".cfm" / "1" / "features"
    meta = json.loads((features / "frequent_label_meta.json").read_text())
    assert meta.get("storage") is None
    assert (features / "frequent_label_idx.npy").exists()


def test_recompile_switches_backends_without_leaking_stale_arrays(tmp_path):
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "mini_corpus"
    root = tmp_path / "mini"
    shutil.copytree(fixture, root)
    (root / "rare_label.tf").write_text("@node\\n@valueType=str\\n\\n1\\tvalue\\n", encoding="utf8")
    output = root / ".cfm" / "1" / "features"
    prefix = output / "rare_label"
    assert Compiler(str(root), experimental_sparse_strings=True).compile()
    assert prefix.with_name("rare_label_nodes.npy").exists()
    assert Compiler(str(root)).compile()
    assert prefix.with_name("rare_label_idx.npy").exists()
    assert not prefix.with_name("rare_label_nodes.npy").exists()
    assert not prefix.with_name("rare_label_codes.npy").exists()
    assert Compiler(str(root), experimental_sparse_strings=True).compile()
    assert not prefix.with_name("rare_label_idx.npy").exists()
    assert prefix.with_name("rare_label_codes.npy").exists()
