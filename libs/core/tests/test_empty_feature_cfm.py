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



def _tiny_warp(root: Path) -> None:
    _write(root, "otype", "@node\n@valueType=str\n@slotType=word\n\n1-2\tword\n3\tverse\n")
    _write(root, "oslots", "@edge\n\n3\t1-2\n")
    _write(root, "otext", "@config\n@fmt:text-orig-full={form}\n\n")
    _write(root, "form", "@node\n@valueType=str\n\n1\talpha\n2\tbeta\n")


def test_compiled_cache_rejects_changed_source_even_when_size_is_identical(
    tmp_path: Path,
) -> None:
    """RED: a changed native .tf value must not be shadowed by stale mmap."""
    import os

    _tiny_warp(tmp_path)
    baseline = Fabric(locations=str(tmp_path), silent="deep")
    assert baseline.compile(silent="deep")
    cached = Fabric(locations=str(tmp_path), silent="deep")
    assert cached.loadAll(silent="deep").F.form.v(1) == "alpha"
    assert cached._loaded_from_cfm

    source = tmp_path / "form.tf"
    source.write_text(source.read_text(encoding="utf-8").replace("alpha", "omega"), encoding="utf-8")
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000_000))
    fresh = Fabric(locations=str(tmp_path), silent="deep")
    api = fresh.loadAll(silent="deep")
    assert not fresh._loaded_from_cfm, "changed source was silently shadowed"
    assert api.F.form.v(1) == "omega"


def test_compiled_cache_rejects_new_and_removed_feature_files(tmp_path: Path) -> None:
    _tiny_warp(tmp_path)
    baseline = Fabric(locations=str(tmp_path), silent="deep")
    assert baseline.compile(silent="deep")

    _write(tmp_path, "gloss", "@node\n\n1\tnew\n")
    new = Fabric(locations=str(tmp_path), silent="deep")
    api = new.loadAll(silent="deep")
    assert not new._loaded_from_cfm
    assert api.F.gloss.v(1) == "new"

    (tmp_path / "gloss.tf").unlink()
    (tmp_path / "form.tf").unlink()
    removed = Fabric(locations=str(tmp_path), silent="deep")
    api = removed.loadAll(silent="deep")
    assert not removed._loaded_from_cfm
    assert "form" not in api.Fall(warp=False)


def test_legacy_compiled_cache_without_source_snapshot_is_not_trusted(
    tmp_path: Path,
) -> None:
    _tiny_warp(tmp_path)
    baseline = Fabric(locations=str(tmp_path), silent="deep")
    assert baseline.compile(silent="deep")
    manifest_path = next((tmp_path / ".cfm").glob("*/meta.json"))
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data.pop("source_tf_stats", None)
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    fresh = Fabric(locations=str(tmp_path), silent="deep")
    assert fresh.loadAll(silent="deep").F.form.v(1) == "alpha"
    assert not fresh._loaded_from_cfm
