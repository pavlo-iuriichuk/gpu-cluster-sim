import networkx as nx

from ...models.network import LinkMetrics
from ...models.registry import KIND_TO_ENTITY
from ..topology import ClusterTopology
from ._codec import decode_namedtuple, encode_namedtuple
from .base import TopologyFormat, register_format


class GraphMLFormat(TopologyFormat):
    """Single-file, widely-supported format. Good for interop with generic
    graph tools (Gephi, yEd, ...); has no notion of chunking, so it does not
    scale the way GraphAr does for very large clusters.
    """

    name = "graphml"

    def export(self, topology: ClusterTopology, path: str) -> None:
        g = nx.MultiDiGraph()
        for entity_id, attrs in topology.graph.nodes(data=True):
            encoded = encode_namedtuple(attrs["data"])
            encoded["kind"] = attrs["kind"]
            g.add_node(entity_id, **encoded)
        for u, v, key, attrs in topology.graph.edges(keys=True, data=True):
            g.add_edge(u, v, key=key, **encode_namedtuple(attrs["metrics"]))
        nx.write_graphml(g, path)

    def import_(self, path: str) -> ClusterTopology:
        g = nx.read_graphml(path, force_multigraph=True)
        topology = ClusterTopology()
        for entity_id, attrs in g.nodes(data=True):
            attrs = dict(attrs)
            kind = attrs.pop("kind")
            entity = decode_namedtuple(KIND_TO_ENTITY[kind], attrs)
            topology.graph.add_node(entity_id, kind=kind, data=entity)
        for u, v, key, attrs in g.edges(keys=True, data=True):
            metrics = decode_namedtuple(LinkMetrics, attrs)
            topology.graph.add_edge(u, v, key=key, metrics=metrics)
        return topology


register_format(GraphMLFormat())
