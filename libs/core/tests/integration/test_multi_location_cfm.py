"""Regression contract for composing TF locations when the base has a .cfm cache."""

from __future__ import annotations

import shutil
from pathlib import Path

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


def test_base_cfm_cache_does_not_hide_second_location_feature(fixtures_dir, tmp_path):
    source_base = fixtures_dir / "mini_corpus"
    base = tmp_path / "base"
    shutil.copytree(source_base, base)
    shutil.rmtree(base / ".cfm", ignore_errors=True)

    base_fabric = Fabric(locations=str(base), silent="deep")
    base_api = base_fabric.loadAll(silent="deep")
    assert base_api is not False
    assert base_api.F.otype.maxSlot == 5
    assert base_api.F.otype.maxNode == 8
    assert (base / ".cfm").is_dir(), "base-only load must create the cache precondition"

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
    assert (base / ".cfm").is_dir(), "composition must not delete the reusable base cache"
