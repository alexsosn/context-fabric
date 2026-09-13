"""Core API of Context-Fabric.

This module provides the main entry points for Context-Fabric:

- `Fabric`: Main class for loading and managing corpora
- `Api`: Runtime API with F, E, L, T, S, N, C accessors

The core API consists of:

- `N`: see `cfabric.navigation.nodes.Nodes` (walk through nodes)
- `F`: see `cfabric.features.node.NodeFeature` (retrieve feature values for nodes)
- `E`: see `cfabric.features.edge.EdgeFeature` (retrieve feature values for edges)
- `L`: see `cfabric.navigation.locality.Locality` (move between levels)
- `T`: see `cfabric.navigation.text.Text` (get the text)
- `S`: see `cfabric.search.search.Search` (search by templates)

Based on Text-Fabric by Dirk Roorda.
"""

from cfabric.core import fabric as _fabric_module
from cfabric.core.composed_cache import Fabric
from cfabric.core.api import Api
from cfabric.core.config import VERSION, NAME, BANNER, OTYPE, OSLOTS, OTEXT, WARP

# Keep the long-standing direct import path (`cfabric.core.fabric.Fabric`) aligned
# with the public class while the composition-aware cache policy is isolated in
# its own module.
_fabric_module.Fabric = Fabric

__version__ = VERSION
__all__ = [
    "Fabric",
    "Api",
    "VERSION",
    "NAME",
    "BANNER",
    "OTYPE",
    "OSLOTS",
    "OTEXT",
    "WARP",
]
