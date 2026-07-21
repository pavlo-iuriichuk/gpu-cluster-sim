"""Directed multigraph model of a GPU cluster's fabric.

Hierarchy: GPU -> Node -> {Rail switch, Leaf switch} -> Spine switch -> Pod.
Containment (which node a GPU lives on, which leaf a node hangs off, ...) is
carried by fields on the model objects themselves (e.g. `GPU.node_id`,
`Node.leaf_id`). Graph edges represent only actual network links — NVLink
between GPUs, rail links, leaf uplinks, spine uplinks, inter-pod links — each
carrying a `LinkMetrics` attribute. A directed multigraph (rather than a
plain graph) is used because a link's two directions can have distinct
metrics (an oversubscribed leaf uplink is asymmetric) and because parallel
links (e.g. multiple NICs between the same two switches) are common.
"""

from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple

import networkx as nx

from ..models.gpu import GPU
from ..models.network import LeafSwitch, LinkMetrics, LinkType, RailSwitch, SpineSwitch
from ..models.node import Node
from ..models.registry import ENTITY_KIND

if TYPE_CHECKING:
    from .paths import PathResult

EdgeKey = Tuple[str, str, int]


class ClusterTopology:
    """Thin, typed wrapper around an `nx.MultiDiGraph`."""

    def __init__(self) -> None:
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()

    def _add_entity(self, entity_id: str, data) -> None:
        kind = ENTITY_KIND[type(data)]
        self.graph.add_node(entity_id, kind=kind, data=data)

    def add_gpu(self, gpu: GPU) -> None:
        self._add_entity(gpu.gpu_id, gpu)

    def add_node(self, node: Node) -> None:
        self._add_entity(node.node_id, node)

    def add_leaf(self, leaf: LeafSwitch) -> None:
        self._add_entity(leaf.leaf_id, leaf)

    def add_spine(self, spine: SpineSwitch) -> None:
        self._add_entity(spine.spine_id, spine)

    def add_rail(self, rail: RailSwitch) -> None:
        self._add_entity(rail.rail_id, rail)

    def add_link(
        self,
        u_id: str,
        v_id: str,
        metrics: LinkMetrics,
        *,
        bidirectional: bool = True,
        reverse_metrics: Optional[LinkMetrics] = None,
        key: Optional[int] = None,
    ) -> EdgeKey:
        """Add a link u -> v. Physical links are duplex by default, so the
        reverse edge is added too, sharing `metrics` unless `reverse_metrics`
        is given (e.g. a leaf uplink where downlink and uplink bandwidth
        differ because of oversubscription).
        """
        assigned_key = self.graph.add_edge(u_id, v_id, key=key, metrics=metrics)
        if bidirectional:
            self.graph.add_edge(v_id, u_id, key=key, metrics=reverse_metrics or metrics)
        return (u_id, v_id, assigned_key)

    def entity(self, entity_id: str):
        return self.graph.nodes[entity_id]["data"]

    def kind_of(self, entity_id: str) -> str:
        return self.graph.nodes[entity_id]["kind"]

    def entities_of_kind(self, kind: str) -> Iterator[str]:
        return (n for n, d in self.graph.nodes(data=True) if d["kind"] == kind)

    def link_metrics(self, u_id: str, v_id: str, key: int = 0) -> LinkMetrics:
        return self.graph.edges[u_id, v_id, key]["metrics"]

    def edges_of_type(self, link_type: LinkType) -> Iterator[EdgeKey]:
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            if data["metrics"].link_type is link_type:
                yield (u, v, k)

    def export(self, format_name: str, path: str) -> None:
        """Serialize to `path` using the named format (e.g. "graphml",
        "graphar"). See `engine.formats` for the registry and how to add new
        formats.
        """
        from .formats import get_format

        get_format(format_name).export(self, path)

    @classmethod
    def load(cls, format_name: str, path: str) -> "ClusterTopology":
        from .formats import get_format

        return get_format(format_name).import_(path)

    def has_path(self, src: str, dst: str) -> bool:
        from .paths import has_path

        return has_path(self, src, dst)

    def shortest_path(self, src: str, dst: str, *, weight: str = "latency") -> "PathResult":
        """The single best path between two entities (typically GPUs). See
        `engine.paths.shortest_path` for `weight` semantics.
        """
        from .paths import shortest_path

        return shortest_path(self, src, dst, weight=weight)

    def k_shortest_paths(
        self, src: str, dst: str, k: int = 3, *, weight: str = "latency"
    ) -> List["PathResult"]:
        from .paths import k_shortest_paths

        return k_shortest_paths(self, src, dst, k=k, weight=weight)

    def all_paths(self, src: str, dst: str, *, cutoff: Optional[int] = None) -> List["PathResult"]:
        from .paths import all_paths

        return all_paths(self, src, dst, cutoff=cutoff)
