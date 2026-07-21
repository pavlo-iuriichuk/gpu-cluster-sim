"""Maps entity NamedTuple classes to the `kind` tag stored on graph nodes,
and back. Shared by `engine.topology` (to tag graph nodes) and
`engine.formats` (to reconstruct the right class on import).
"""

from typing import Dict, Type

from .gpu import GPU
from .network import LeafSwitch, RailSwitch, SpineSwitch
from .node import Node

ENTITY_KIND: Dict[Type, str] = {
    GPU: "gpu",
    Node: "node",
    LeafSwitch: "leaf",
    SpineSwitch: "spine",
    RailSwitch: "rail",
}

KIND_TO_ENTITY: Dict[str, Type] = {kind: cls for cls, kind in ENTITY_KIND.items()}
