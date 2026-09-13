"""Composition-aware CFM caching for the public Fabric class.

The classic CFM layout belongs to one source directory. This module extends
Fabric with a namespaced cache for ordered multi-location/module compositions
without weakening the legacy single-source behavior.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from cfabric.core.config import CFM_VERSION, OTEXT, OSLOTS, OTYPE
from cfabric.core.fabric import Fabric as BaseFabric
from cfabric.io.compiler import Compiler
from cfabric.utils.logging import SILENT_D, set_logging_level, silentConvert

logger = logging.getLogger(__name__)

COMPOSITION_MANIFEST = "composition.json"
COMPOSITION_MANIFEST_VERSION = 1


class Fabric(BaseFabric):
    """Fabric with safe CFM acceleration for ordered composed corpora."""

    def _composition_topology(self) -> dict[str, Any]:
        return {
            "locations": [str(Path(location).resolve()) for location in self.locations],
            "modules": list(self.modules),
        }

    def _composition_cache_path(self) -> Path:
        topology = json.dumps(
            self._composition_topology(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(topology).hexdigest()
        return (
            Path(self.locations[0])
            / ".cfm"
            / "compositions"
            / digest
            / CFM_VERSION
        )

    def _composition_manifest(self) -> dict[str, Any]:
        sources: list[dict[str, Any]] = []
        for feature_name, feature in sorted(self.features.items()):
            path = Path(feature.path)
            if path.suffix != ".tf" or not path.exists():
                continue
            resolved = path.resolve()
            stat = resolved.stat()
            sources.append(
                {
                    "feature": feature_name,
                    "path": str(resolved),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        return {
            "manifest_version": COMPOSITION_MANIFEST_VERSION,
            **self._composition_topology(),
            "sources": sources,
        }

    def _detect_cfm(self) -> Path | None:
        if self._cfm_topology_is_single_source():
            return super()._detect_cfm()
        if not self.locations:
            return None

        cfm_path = self._composition_cache_path()
        manifest_path = cfm_path / COMPOSITION_MANIFEST
        if not (cfm_path / "meta.json").exists() or not manifest_path.exists():
            return None

        try:
            stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            current_manifest = self._composition_manifest()
        except (OSError, json.JSONDecodeError):
            logger.debug("Ignoring unreadable or unstable composed CFM cache at %s", cfm_path)
            return None

        if stored_manifest != current_manifest:
            logger.debug("Ignoring stale composed CFM cache at %s", cfm_path)
            return None
        return cfm_path

    def _gather_composed_precomputed_data(self) -> dict[str, Any] | None:
        """Return complete fresh merged data, or None while a composition is partial."""
        otype_feat = self.features.get(OTYPE)
        oslots_feat = self.features.get(OSLOTS)
        if (
            otype_feat is None
            or oslots_feat is None
            or not otype_feat.dataLoaded
            or not oslots_feat.dataLoaded
            or otype_feat.data is None
            or oslots_feat.data is None
        ):
            return None

        # A composed cache must represent every effective physical data feature.
        # Before loadAll() finishes, unknown feature kinds are conservatively
        # treated as data and therefore keep compilation disabled. A source that
        # is newer than its in-memory Data object also keeps compilation disabled.
        for feature in self.features.values():
            path = Path(feature.path)
            if path.suffix != ".tf" or not path.exists():
                continue
            if feature.isConfig is True:
                continue
            if not feature.dataLoaded or feature.data is None:
                return None
            loaded_at = feature.dataLoaded
            if (
                not isinstance(loaded_at, bool)
                and isinstance(loaded_at, (int, float))
                and loaded_at < path.stat().st_mtime
            ):
                logger.debug("Skipping composed CFM: source changed after load: %s", path)
                return None

        precomputed: dict[str, Any] = {
            "otype": otype_feat.data,
            "oslots": oslots_feat.data,
            "otext_meta": (
                dict(self.features[OTEXT].metaData)
                if OTEXT in self.features and self.features[OTEXT].metaData
                else {}
            ),
        }

        feature_meta: dict[str, dict[str, str]] = {}
        for name, feature in self.features.items():
            if feature.metaData:
                feature_meta[name] = dict(feature.metaData)
        precomputed["feature_meta"] = feature_meta

        computed_features = {
            "__levels__": "levels",
            "__order__": "order",
            "__rank__": "rank",
            "__levUp__": "levUp",
            "__levDown__": "levDown",
            "__boundary__": "boundary",
        }
        for internal_name, output_name in computed_features.items():
            feature = self.features.get(internal_name)
            if feature is not None and feature.dataLoaded and feature.data is not None:
                precomputed[output_name] = feature.data

        node_features: dict[str, dict[int, Any]] = {}
        edge_features: dict[str, tuple[dict[int, Any], bool]] = {}
        for name, feature in self.features.items():
            if name in (OTYPE, OSLOTS, OTEXT):
                continue
            if name.startswith("__") and name.endswith("__"):
                continue
            if feature.isConfig or feature.method:
                continue
            if not feature.dataLoaded or feature.data is None:
                continue
            if feature.isEdge:
                edge_features[name] = (feature.data, feature.edgeValues)
            else:
                node_features[name] = feature.data

        precomputed["node_features"] = node_features
        precomputed["edge_features"] = edge_features
        return precomputed

    def compile(self, output_dir: str | None = None, silent: str = SILENT_D) -> bool:
        if self._cfm_topology_is_single_source():
            return super().compile(output_dir=output_dir, silent=silent)
        if not self.locations:
            return False

        silent = silentConvert(silent)
        set_logging_level(silent)

        try:
            source_manifest = self._composition_manifest()
            precomputed = self._gather_composed_precomputed_data()
            gathered_manifest = self._composition_manifest()
        except OSError:
            logger.debug("Skipping composed .cfm compilation because a source is unstable")
            return False

        if precomputed is None:
            logger.debug(
                "Skipping composed .cfm compilation until all effective features are loaded"
            )
            return False
        if gathered_manifest != source_manifest:
            logger.debug("Skipping composed .cfm compilation because sources changed while gathering")
            return False

        cfm_path = Path(output_dir) if output_dir is not None else self._composition_cache_path()
        manifest_path = cfm_path / COMPOSITION_MANIFEST
        try:
            manifest_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Cannot invalidate composed CFM manifest %s: %s", manifest_path, exc)
            return False

        compiler = Compiler(self.locations[0])
        result = compiler.compile(cfm_path, precomputed=precomputed)
        if not result:
            return False

        try:
            final_manifest = self._composition_manifest()
        except OSError:
            logger.warning("A source disappeared during composed CFM compilation")
            return False
        if final_manifest != source_manifest:
            logger.warning(
                "Sources changed during composed CFM compilation; leaving cache invalid"
            )
            return False

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest_path.with_suffix(".json.tmp")
        try:
            temporary.write_text(
                json.dumps(source_manifest, indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            temporary.replace(manifest_path)
        except OSError as exc:
            logger.warning("Cannot publish composed CFM manifest %s: %s", manifest_path, exc)
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            return False
        return True

    def loadAll(self, silent: str = SILENT_D):
        api = super().loadAll(silent=silent)
        if api is False or api is None:
            return api

        if (
            not self._cfm_topology_is_single_source()
            and not getattr(self, "_loaded_from_cfm", False)
        ):
            self.compile(silent=silent)
        return api
