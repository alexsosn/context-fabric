"""Regression contracts for composed TF topology when .cfm caching is present."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from cfabric.core.config import CFM_VERSION
from cfabric.core.fabric import Fabric


def _feature_only_module(tmp_path: Path) -> Path:
    module = tmp_path / "module"
    module.mkdir()
    (module / "module_label.tf").write_text(
        "@node\n"
        "@valueType=str\n"
        "@description=feature-only composition cache contract\n"
        "\n"
        "one\n"
        "two\n"
        "three\n"
        "four\n"
        "five\n",
        encoding="utf-8",
    )
    return module


def _copy_base(fixtures_dir: Path, tmp_path: Path) -> Path:
    base = tmp_path / "base"
    shutil.copytree(fixtures_dir / "mini_corpus", base)
    shutil.rmtree(base / ".cfm", ignore_errors=True)
    return base


def _rewrite_module_value(module: Path, old: str, new: str) -> None:
    feature = module / "module_label.tf"
    before = feature.stat()
    feature.write_text(
        feature.read_text(encoding="utf-8").replace(old, new),
        encoding="utf-8",
    )
    # Make the cache-invalidation contract deterministic even on filesystems with
    # coarse timestamp updates.
    after = feature.stat()
    os.utime(
        feature,
        ns=(after.st_atime_ns, max(after.st_mtime_ns, before.st_mtime_ns + 1_000_000_000)),
    )


def test_base_cfm_cache_does_not_hide_second_location_feature(fixtures_dir, tmp_path):
    base = _copy_base(fixtures_dir, tmp_path)

    base_fabric = Fabric(locations=str(base), silent="deep")
    base_api = base_fabric.loadAll(silent="deep")
    assert base_api is not False
    assert base_api.F.otype.maxSlot == 5
    assert base_api.F.otype.maxNode == 8
    assert (base / ".cfm" / CFM_VERSION).is_dir(), (
        "base-only load must create the classic single-source cache"
    )

    module = _feature_only_module(tmp_path)
    composed = Fabric(locations=[str(base), str(module)], silent="deep")
    assert tuple(Path(location).resolve() for location in composed.locations) == (
        base.resolve(),
        module.resolve(),
    )
    api = composed.loadAll(silent="deep")

    assert api is not False
    assert api.F.otype.maxSlot == 5
    assert api.F.otype.maxNode == 8
    assert "module_label" in api.Fall()
    assert api.Fs("module_label", warn=False).v(1) == "one"
    assert api.Fs("module_label", warn=False).v(5) == "five"
    assert (base / ".cfm" / CFM_VERSION).is_dir(), (
        "composition must not delete the reusable base cache"
    )


def test_composed_load_emits_namespaced_cache_and_second_load_reuses_it(
    fixtures_dir, tmp_path
):
    base = _copy_base(fixtures_dir, tmp_path)
    module = _feature_only_module(tmp_path)

    first = Fabric(locations=[str(base), str(module)], silent="deep")
    first_api = first.loadAll(silent="deep")

    assert first_api is not False
    composed_cache = first._detect_cfm()
    assert composed_cache is not None
    assert composed_cache.is_dir()
    assert composed_cache.parent.parent.name == "compositions"
    assert composed_cache.name == CFM_VERSION
    assert not (module / ".cfm").exists()

    second = Fabric(locations=[str(base), str(module)], silent="deep")
    second_api = second.loadAll(silent="deep")

    assert second_api is not False
    assert second._detect_cfm() == composed_cache
    assert getattr(second, "_loaded_from_cfm", False) is True
    assert second_api.Fs("module_label", warn=False).v(3) == "three"


def test_composed_cache_invalidates_when_effective_overlay_changes(fixtures_dir, tmp_path):
    base = _copy_base(fixtures_dir, tmp_path)
    module = _feature_only_module(tmp_path)

    first = Fabric(locations=[str(base), str(module)], silent="deep")
    first_api = first.loadAll(silent="deep")
    assert first_api is not False
    first_cache = first._detect_cfm()
    assert first_cache is not None

    _rewrite_module_value(module, "three", "THREE")

    changed = Fabric(locations=[str(base), str(module)], silent="deep")
    assert changed._detect_cfm() is None, "changed source fingerprint must invalidate cache"
    changed_api = changed.loadAll(silent="deep")

    assert changed_api is not False
    assert getattr(changed, "_loaded_from_cfm", False) is False
    assert changed_api.Fs("module_label", warn=False).v(3) == "THREE"
    refreshed_cache = changed._detect_cfm()
    assert refreshed_cache == first_cache

    reused = Fabric(locations=[str(base), str(module)], silent="deep")
    reused_api = reused.loadAll(silent="deep")
    assert reused_api is not False
    assert getattr(reused, "_loaded_from_cfm", False) is True
    assert reused_api.Fs("module_label", warn=False).v(3) == "THREE"


def test_composition_cache_identity_is_order_sensitive(fixtures_dir, tmp_path):
    base = _copy_base(fixtures_dir, tmp_path)
    module = _feature_only_module(tmp_path)

    forward = Fabric(locations=[str(base), str(module)], silent="deep")
    forward_api = forward.loadAll(silent="deep")
    assert forward_api is not False
    forward_cache = forward._detect_cfm()
    assert forward_cache is not None

    reverse = Fabric(locations=[str(module), str(base)], silent="deep")
    reverse_api = reverse.loadAll(silent="deep")
    assert reverse_api is not False
    reverse_cache = reverse._detect_cfm()
    assert reverse_cache is not None

    assert reverse_cache != forward_cache


def test_base_module_cfm_cache_does_not_hide_overlay_module(fixtures_dir, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    base = _copy_base(fixtures_dir, root)

    base_fabric = Fabric(locations=str(base), silent="deep")
    base_api = base_fabric.loadAll(silent="deep")
    assert base_api is not False
    assert (base / ".cfm" / CFM_VERSION).is_dir(), (
        "base module must create the classic cache precondition"
    )

    module = _feature_only_module(root)
    composed = Fabric(locations=str(root), modules=["base", "module"], silent="deep")
    assert tuple(composed.modules) == ("base", "module")
    api = composed.loadAll(silent="deep")

    assert api is not False
    assert api.F.otype.maxSlot == 5
    assert api.F.otype.maxNode == 8
    assert "module_label" in api.Fall()
    assert api.Fs("module_label", warn=False).v(3) == "three"
    assert (base / ".cfm" / CFM_VERSION).is_dir(), (
        "composition must retain the reusable base cache"
    )
    composed_cache = composed._detect_cfm()
    assert composed_cache is not None
    assert composed_cache.parent.parent.name == "compositions"
    assert not (module / ".cfm").exists()
