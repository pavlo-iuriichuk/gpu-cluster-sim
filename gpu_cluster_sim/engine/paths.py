"""Path queries between two graph entities (typically GPUs): shortest path,
k-shortest paths, all simple paths, plus the cost/hop-count figures a
placement scorer would need (total latency, bottleneck bandwidth, distinct
leaf/spine switches spanned, spine/inter-pod crossings — see "Placement
scoring" in docs/topology-aware-gang-scheduling.md).

`ClusterTopology.graph` is a MultiDiGraph, but networkx's k-shortest-paths
and weighted-Dijkstra implementations either don't support multigraphs
(`shortest_simple_paths`) or enumerate one path per parallel-edge
combination instead of per distinct route (`all_simple_paths`). So path
*search* runs against a collapsed simple-graph view (`_simple_view`, one
edge per (u, v) using its best parallel edge), while per-hop detail in the
result still comes from the real multigraph via `_best_edge`.
"""

from typing import List, NamedTuple, Optional, Tuple

import networkx as nx

from ..models.network import LinkMetrics, LinkType
from .topology import ClusterTopology


class PathHop(NamedTuple):
    u: str
    v: str
    key: int
    metrics: LinkMetrics


class PathResult(NamedTuple):
    path: Tuple[str, ...]  # graph vertex ids traversed, source..destination inclusive
    hops: Tuple[PathHop, ...]
    hop_count: int
    total_latency_us: float
    bottleneck_bandwidth_gbps: float
    distinct_leaf_switches: int
    distinct_spine_switches: int
    spine_uplink_hops: int
    inter_pod_hops: int


def _best_edge(topology: ClusterTopology, u: str, v: str) -> Tuple[int, LinkMetrics]:
    """Among parallel u->v edges, prefer the highest bandwidth, then the
    lowest latency."""
    parallel_edges = topology.graph[u][v]
    best_key = min(
        parallel_edges,
        key=lambda k: (-parallel_edges[k]["metrics"].bandwidth_gbps, parallel_edges[k]["metrics"].latency_us),
    )
    return best_key, parallel_edges[best_key]["metrics"]


def _simple_view(topology: ClusterTopology) -> nx.DiGraph:
    simple = nx.DiGraph()
    simple.add_nodes_from(topology.graph.nodes)
    seen = set()
    for u, v in topology.graph.edges():
        if (u, v) in seen:
            continue
        seen.add((u, v))
        _, metrics = _best_edge(topology, u, v)
        simple.add_edge(u, v, latency_us=metrics.latency_us)
    return simple


def _summarize(topology: ClusterTopology, node_path: List[str]) -> PathResult:
    hops = tuple(
        PathHop(u, v, *_best_edge(topology, u, v)) for u, v in zip(node_path, node_path[1:])
    )
    return PathResult(
        path=tuple(node_path),
        hops=hops,
        hop_count=len(hops),
        total_latency_us=sum(hop.metrics.latency_us for hop in hops),
        bottleneck_bandwidth_gbps=min((hop.metrics.bandwidth_gbps for hop in hops), default=0.0),
        distinct_leaf_switches=sum(1 for n in node_path if topology.kind_of(n) == "leaf"),
        distinct_spine_switches=sum(1 for n in node_path if topology.kind_of(n) == "spine"),
        spine_uplink_hops=sum(1 for hop in hops if hop.metrics.link_type is LinkType.SPINE_UPLINK),
        inter_pod_hops=sum(1 for hop in hops if hop.metrics.link_type is LinkType.INTER_POD),
    )


def has_path(topology: ClusterTopology, src: str, dst: str) -> bool:
    return nx.has_path(topology.graph, src, dst)


def _weight_attr(weight: str) -> Optional[str]:
    if weight == "latency":
        return "latency_us"
    if weight == "hops":
        return None
    raise ValueError(f"Unknown weight mode {weight!r}, expected 'latency' or 'hops'")


def shortest_path(topology: ClusterTopology, src: str, dst: str, *, weight: str = "latency") -> PathResult:
    """The single best path from `src` to `dst`. `weight="latency"` minimizes
    total latency (using each hop's best parallel edge); `weight="hops"`
    minimizes hop count.
    """
    simple = _simple_view(topology)
    node_path = nx.shortest_path(simple, src, dst, weight=_weight_attr(weight))
    return _summarize(topology, node_path)


def k_shortest_paths(
    topology: ClusterTopology, src: str, dst: str, k: int = 3, *, weight: str = "latency"
) -> List[PathResult]:
    """The `k` best loopless paths from `src` to `dst`, best first."""
    simple = _simple_view(topology)
    results = []
    for node_path in nx.shortest_simple_paths(simple, src, dst, weight=_weight_attr(weight)):
        if len(results) >= k:
            break
        results.append(_summarize(topology, node_path))
    return results


def all_paths(topology: ClusterTopology, src: str, dst: str, *, cutoff: Optional[int] = None) -> List[PathResult]:
    """Every simple (loopless) path from `src` to `dst`. Can be
    combinatorially large on a dense fabric — pass `cutoff` (max hop count)
    to bound it.
    """
    simple = _simple_view(topology)
    return [_summarize(topology, p) for p in nx.all_simple_paths(simple, src, dst, cutoff=cutoff)]
