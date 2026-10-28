"""Portable public-API regressions for fixes first made in Python CF 0.5.x.

The Rust engine has a different compiled format (.cfr instead of .cfm).
Test *values* and independently reopened compiled corpora, not Python internals.
Historical fixes: alexsosn/context-fabric PRs #8 (empty features) and #10
(sparse integer values -1/-2147483648).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from cfabric import Fabric

from conftest import LOAD_MODES, open_corpus


def _case(tmp_path: Path, mini_corpus_path: Path) -> Path:
    corpus = tmp_path / "source"
    shutil.copytree(mini_corpus_path, corpus, ignore=shutil.ignore_patterns(".cf*"))
    features = {
        "empty_text": "@node\n@valueType=str\n\n",
        "empty_int": "@node\n@valueType=int\n\n",
        "empty_edge": "@edge\n\n",
        "empty_valued_edge": "@edge\n@edgeValues\n@valueType=int\n\n",
        "signed_year": (
            "@node\n@valueType=int\n\n"
            "1\t-1\n2\t-2147483648\n3\t0\n"
        ),
        "sparse_year": "@node\n@valueType=int\n\n1\t-1\n",
    }
    for name, content in features.items():
        (corpus / f"{name}.tf").write_text(content, encoding="utf-8")
    return corpus


def _assert_public_values(api) -> None:
    for feature in ("empty_text", "empty_int", "signed_year", "sparse_year"):
        assert feature in api.Fall(warp=False), feature
    for feature in ("empty_edge", "empty_valued_edge"):
        assert feature in api.Eall(warp=False), feature

    assert api.F.empty_text.v(1) is None
    assert api.F.empty_int.v(1) is None
    assert tuple(api.F.empty_text.items()) == ()
    assert tuple(api.F.empty_int.items()) == ()
    assert api.E.empty_edge.f(1) == ()
    assert api.E.empty_valued_edge.f(1) == ()
    assert api.F.signed_year.v(1) == -1
    assert api.F.signed_year.v(2) == -(2**31)
    assert api.F.signed_year.v(3) == 0
    assert dict(api.F.signed_year.items()) == {1: -1, 2: -(2**31), 3: 0}
    assert api.F.sparse_year.v(1) == -1
    assert api.F.sparse_year.v(2) is None
    assert dict(api.F.sparse_year.items()) == {1: -1}
    assert tuple(api.F.sparse_year.s(-1)) == (1,)
    assert tuple(api.F.sparse_year.s(0)) == ()


@pytest.mark.parametrize("mode", LOAD_MODES)
def test_python_fixed_values_survive_rust_load_modes(tmp_path, mini_corpus_path, mode):
    corpus = _case(tmp_path, mini_corpus_path)
    api = open_corpus(
        [corpus], mode, tmp_path,
        features="word empty_text empty_int empty_edge empty_valued_edge signed_year sparse_year",
    )
    _assert_public_values(api)


def test_fresh_process_compiled_reopen_preserves_missing_and_signed_values(
    tmp_path, mini_corpus_path
):
    corpus = _case(tmp_path, mini_corpus_path)
    compiled = tmp_path / "persist.cfr"
    fabric = Fabric(locations=str(corpus), silent="deep")
    assert fabric.compile(str(compiled), None, "deep")
    assert compiled.is_file()

    # Two genuinely separate interpreter processes: a same-object test can
    # falsely pass if it serves the precompiled in-memory Python features.
    code = """
import json, sys
from cfabric import Fabric
path, compiled = sys.argv[1:3]
api = Fabric(locations=path, silent='deep').openMapped(compiled)
assert 'empty_text' in api.Fall(warp=False)
assert 'empty_edge' in api.Eall(warp=False)
assert api.F.empty_text.v(1) is None
assert api.F.empty_int.v(1) is None
assert api.E.empty_valued_edge.f(1) == ()
print(json.dumps([api.F.signed_year.v(1), api.F.signed_year.v(2),
                  api.F.signed_year.v(3), api.F.sparse_year.v(1),
                  api.F.sparse_year.v(2), list(api.F.sparse_year.s(-1))]))
"""
    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, "-c", code, str(corpus), str(compiled)],
            check=True, capture_output=True, text=True, timeout=45,
        )
        assert json.loads(completed.stdout.strip().splitlines()[-1]) == [
            -1, -(2**31), 0, -1, None, [1]
        ]


def test_negative_and_empty_features_agree_with_text_fabric(
    tmp_path, mini_corpus_path
):
    tf = pytest.importorskip("tf.fabric")
    corpus = _case(tmp_path, mini_corpus_path)
    baseline = tf.Fabric(locations=str(corpus), silent="deep").loadAll(silent="deep")
    rust = Fabric(locations=str(corpus), silent="deep").loadAll(silent="deep")
    assert baseline is not False
    for feature in ("empty_text", "empty_int", "signed_year", "sparse_year"):
        for node in (1, 2, 3):
            assert rust.Fs(feature).v(node) == baseline.Fs(feature).v(node)
    for feature in ("empty_edge", "empty_valued_edge"):
        for node in (1, 2, 8):
            assert tuple(rust.Es(feature).f(node)) == tuple(baseline.Es(feature).f(node))
