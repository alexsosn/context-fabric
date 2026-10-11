"""End-to-end correctness gate for experimental sparse CFM string features.

The same corpus is loaded independently as source .tf, dense .cfm, and sparse
.cfm. Tests compare the public F/S/L/T APIs and exact result order, not only
the storage helper methods.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from cfabric.core import Fabric
from cfabric.io.compiler import Compiler
from cfabric.storage.sparse_string_pool import SparseStringPool


@pytest.fixture(scope="module")
def triplicate_apis(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("cfm-search-parity")
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "mini_corpus"
    roots = [tmp_path / kind / "mini_corpus" for kind in ("source", "dense", "sparse")]
    for root in roots:
        root.parent.mkdir(parents=True)
        shutil.copytree(fixture, root)
        # Nodes 1 and 4 are words, node 8 is a sentence; explicit empty
        # strings and absent values must remain distinguishable.
        (root / "rare_tag.tf").write_text(
            "@node\n@valueType=str\n@description=parity corpus feature\n\n"
            "1\talpha\n4\t\n8\talpha\n",
            encoding="utf-8",
        )
    source, dense, sparse = roots
    assert Compiler(str(dense)).compile()
    assert Compiler(str(sparse), experimental_sparse_strings=True).compile()

    def load(root):
        return Fabric(locations=[str(root.parent)], modules=["mini_corpus"], silent="deep").loadAll()

    apis = (load(source), load(dense), load(sparse))
    assert all(api for api in apis)
    assert isinstance(apis[2].F.rare_tag._data, SparseStringPool)
    assert apis[0].F.rare_tag.v(4) == ""
    assert apis[2].F.rare_tag.v(4) == ""
    yield apis


def test_public_feature_results_exact_parity(triplicate_apis):
    api_source, api_dense, api_sparse = triplicate_apis
    for api in (api_dense, api_sparse):
        for n in range(0, 11):
            assert api.F.rare_tag.v(n) == api_source.F.rare_tag.v(n)
        assert dict(api.F.rare_tag.items()) == dict(api_source.F.rare_tag.items())
        assert api.F.rare_tag.data == api_source.F.rare_tag.data
        for value in ("alpha", "", "missing"):
            assert api.F.rare_tag.s(value) == api_source.F.rare_tag.s(value)
        for node_types in (None, {"word"}, {"phrase"}, {"sentence"}, {"word", "sentence"}):
            if node_types is None:
                assert api.F.rare_tag.freqList() == api_source.F.rare_tag.freqList()
            else:
                assert api.F.rare_tag.freqList(node_types) == api_source.F.rare_tag.freqList(node_types)


@pytest.mark.parametrize(
    "query",
    [
        "word rare_tag=alpha",
        "word rare_tag~^al",
        "word rare_tag*",
        "word rare_tag#",
        "word rare_tag#alpha",
        "sentence rare_tag=alpha",
        ". rare_tag=alpha",
        ". rare_tag*",
        ". rare_tag#",
        "phrase\n  word rare_tag=alpha",
        "sentence\n  word rare_tag=alpha",
        "w:word rare_tag=alpha\np:phrase\nw ]] p",
    ],
)
def test_search_exact_tuple_order_and_membership(triplicate_apis, query):
    api_source, api_dense, api_sparse = triplicate_apis
    expected = tuple(api_source.S.search(query))
    assert tuple(api_dense.S.search(query)) == expected
    assert tuple(api_sparse.S.search(query)) == expected


def test_locality_and_text_api_parity(triplicate_apis):
    source, dense, sparse = triplicate_apis
    for target in (dense, sparse):
        for n in (1, 4, 6, 7, 8):
            assert target.L.u(n) == source.L.u(n)
            assert target.L.d(n) == source.L.d(n)
            assert target.E.oslots.s(n) == source.E.oslots.s(n)
        for n in (1, 4, 8):
            assert target.T.text(n) == source.T.text(n)
