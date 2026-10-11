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
    reread = SparseStringPool.load(str(tmp_path / "lemma"), max_node=1000)
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
    loaded = SparseStringPool.load(str(tmp_path / "empty"), max_node=5)
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
    assert (feature_dir / ".sparse-string-v1" / "rare_label" / "pool_nodes.npy").exists()
    assert (feature_dir / ".sparse-string-v1" / "rare_label" / "pool_codes.npy").exists()
    assert not (feature_dir / "rare_label_idx.npy").exists()

    api = Fabric(locations=str(root), silent="deep").load("rare_label")
    assert api.F.rare_label.v(1) == "α"
    assert api.F.rare_label.v(5) is None
    assert api.F.rare_label.v(8) == "α"
    assert sorted(api.F.rare_label.s("α")) == [1, 8]
    assert dict(api.F.rare_label.items()) == {1: "α", 8: "α"}
    assert api.F.rare_label.freqList() == (("α", 2),)
    assert api.F.rare_label.freqList({"word"}) == (("α", 1),)
    assert api.F.rare_label.freqList({"sentence"}) == (("α", 1),)
    assert api.F.rare_label.freqList({"phrase"}) == ()


def test_sparse_type_filter_frequency_avoids_unpopulated_get(monkeypatch, tmp_path):
    """Type-restricted frequency must scale with populated rows, not max_node."""
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "mini_corpus"
    root = tmp_path / "mini"
    shutil.copytree(fixture, root)
    (root / "rare_label.tf").write_text(
        "@node\\n@valueType=str\\n\\n1\\tα\\n8\\tα\\n", encoding="utf8"
    )
    assert Compiler(str(root), experimental_sparse_strings=True).compile()
    api = Fabric(locations=str(root), silent="deep").load("rare_label")
    pool = api.F.rare_label._data
    assert isinstance(pool, SparseStringPool)

    def explode(_node):
        raise AssertionError("freqList must iterate populated items, not get every node")

    monkeypatch.setattr(pool, "get", explode)
    assert api.F.rare_label.freqList({"word"}) == (("α", 1),)


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
    (root / "rare_label.tf").write_text("@node\n@valueType=str\n\n1\tvalue\n", encoding="utf8")
    output = root / ".cfm" / "1" / "features"
    prefix = output / "rare_label"
    assert Compiler(str(root), experimental_sparse_strings=True).compile()
    sparse_prefix = output / ".sparse-string-v1" / "rare_label" / "pool"
    assert sparse_prefix.with_name("pool_nodes.npy").exists()
    assert Compiler(str(root)).compile()
    assert prefix.with_name("rare_label_idx.npy").exists()
    assert not sparse_prefix.parent.exists()
    assert Compiler(str(root), experimental_sparse_strings=True).compile()
    assert sparse_prefix.with_name("pool_codes.npy").exists()
    assert json.loads((output / "rare_label_meta.json").read_text())["storage"] == "sparse-string-v1"


def test_sparse_artifacts_do_not_collide_with_other_feature_names(tmp_path):
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "mini_corpus"
    root = tmp_path / "mini"
    shutil.copytree(fixture, root)
    (root / "demo.tf").write_text("@node\n@valueType=str\n\n1\tvalue\n", encoding="utf8")
    # Old un-namespaced demo_nodes.npy conflicts with the valid integer
    # feature named demo_nodes.
    (root / "demo_nodes.tf").write_text("@node\n@valueType=int\n\n1\t123\n", encoding="utf8")
    assert Compiler(str(root), experimental_sparse_strings=True).compile()
    api = Fabric(locations=str(root), silent="deep").load("demo demo_nodes")
    assert api.F.demo.v(1) == "value"
    assert api.F.demo_nodes.v(1) == 123
    assert api.F.demo.v(2) is None


@pytest.mark.parametrize(
    ("nodes", "codes", "max_node"),
    [
        ([2, 1], [0, 0], 4),
        ([1, 1], [0, 0], 4),
        ([0], [0], 4),
        ([5], [0], 4),
        ([1], [1], 4),
    ],
)
def test_corrupt_sparse_indexes_rejected(nodes, codes, max_node):
    with pytest.raises(ValueError):
        SparseStringPool(
            np.array(["one"], dtype=object),
            np.asarray(nodes, dtype=np.uint32),
            np.asarray(codes, dtype=np.uint8),
            max_node,
        )


def test_non_integer_sparse_indexes_rejected():
    with pytest.raises(ValueError):
        SparseStringPool(
            np.array(["one"], dtype=object),
            np.array([1.5], dtype=np.float64),
            np.array([0], dtype=np.uint8),
            10,
        )


def test_bounded_scalar_index_roundtrip_across_block_edges(tmp_path):
    """Sparse index must work for far-apart nodes and uint32 boundaries."""
    highest = 2**32 - 1
    data = {
        1: "first", 255: "near", 256: "edge", 257: "next",
        1023: "", 1024: "edge", 1025: "after",
        highest - 1: "last", highest: "last",
    }
    pool = SparseStringPool.from_dict(data, max_node=highest)
    prefix = str(tmp_path / "sparse-index")
    pool.save(prefix)
    mmap_pool = SparseStringPool.load(prefix, max_node=highest)
    for candidate in (pool, mmap_pool):
        for node in (
            -1, 0, 1, 2, 254, 255, 256, 257, 258, 1022, 1023, 1024, 1025,
            1026, highest - 2, highest - 1, highest, highest + 1,
        ):
            assert candidate.get(node) == data.get(node)
        # A bounded index should be built lazily only for repeated scalar gets.
        for node in (1, 257, highest, 512, highest - 2) * 200:
            assert candidate.get(node) == data.get(node)
        assert candidate._bucket_offsets is not None
        assert candidate._bucket_offsets.nbytes <= max(len(data) * 4, 256)
        assert dict(candidate.items()) == data


def test_randomized_sparse_scalar_lookup_matches_dense():
    from cfabric.storage.string_pool import StringPool

    import random
    rng = random.Random(1989)
    for max_node, count in ((4097, 256), (28000, 43), (28000, 12000)):
        positions = rng.sample(range(1, max_node + 1), count)
        data = {p: ("", "𐎀", "foo", "β")[i % 4] for i, p in enumerate(positions)}
        dense = StringPool.from_dict(data, max_node)
        sparse = SparseStringPool.from_dict(data, max_node)
        for node in [0, max_node + 1, *positions, *rng.sample(range(1, max_node + 1), 1000)]:
            assert dense.get(node) == sparse.get(node)
