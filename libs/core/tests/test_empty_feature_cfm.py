"""Regression for IIP native-TF empty features in compiled Context-Fabric."""

from __future__ import annotations

import json
from pathlib import Path

from cfabric.core import Fabric


def _write(root: Path, feature: str, content: str) -> None:
    (root / f"{feature}.tf").write_text(content, encoding="utf-8")


def test_empty_declared_features_are_materialized_and_loadable(tmp_path: Path) -> None:
    """Zero-value node/edge features must not become dangling CFM manifest entries."""
    _write(
        tmp_path, "otype", "@node\n@valueType=str\n@slotType=word\n\n1-2\tword\n3\tverse\n",
    )
    _write(tmp_path, "oslots", "@edge\n\n3\t1-2\n")
    _write(tmp_path, "otext", "@config\n@fmt:text-orig-full={form}\n\n")
    _write(tmp_path, "form", "@node\n@valueType=str\n\n1\talpha\n2\tbeta\n")
    _write(tmp_path, "translation_after", "@node\n@valueType=str\n\n")
    _write(tmp_path, "empty_int", "@node\n@valueType=int\n\n")
    _write(tmp_path, "zero_edge", "@edge\n\n")
    _write(tmp_path, "zero_valued_edge", "@edge\n@edgeValues\n@valueType=int\n\n")

    native = Fabric(locations=str(tmp_path), silent="deep")
    raw = native.loadAll(silent="deep")
    assert raw is not False and raw is not None
    assert raw.F.form.v(1) == "alpha"
    assert raw.F.translation_after.v(1) is None
    assert tuple(raw.E.zero_edge.f(3)) == ()
    assert raw.F.empty_int.v(1) is None
    assert tuple(raw.E.zero_valued_edge.f(3)) == ()
    assert native.compile(silent="deep")

    meta_files = list((tmp_path / ".cfm").glob("*/meta.json"))
    assert len(meta_files) == 1
    meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
    assert "translation_after" in meta["features"]["node"]
    assert "empty_int" in meta["features"]["node"]
    assert "zero_edge" in meta["features"]["edge"]
    assert "zero_valued_edge" in meta["features"]["edge"]
    root = meta_files[0].parent
    assert (root / "features/translation_after_strings.npy").is_file()
    assert (root / "features/translation_after_idx.npy").is_file()
    assert (root / "edges/zero_edge_indptr.npy").is_file()
    assert (root / "features/empty_int.npy").is_file()
    assert (root / "edges/zero_valued_edge_indptr.npy").is_file()

    fresh = Fabric(locations=str(tmp_path), silent="deep")
    loaded = fresh.loadAll(silent="deep")
    assert loaded is not False and loaded is not None
    assert fresh._loaded_from_cfm
    assert loaded.F.otype.s("word") == (1, 2)
    assert loaded.F.form.v(1) == "alpha"
    assert loaded.F.translation_after.v(1) is None
    assert tuple(loaded.E.zero_edge.f(3)) == ()
    assert loaded.F.empty_int.v(1) is None
    assert tuple(loaded.E.zero_valued_edge.f(3)) == ()
