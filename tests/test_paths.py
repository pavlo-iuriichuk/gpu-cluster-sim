import networkx as nx
import pytest

from gpu_cluster_sim.engine import ClusterTopology, all_paths, has_path, k_shortest_paths, shortest_path
from gpu_cluster_sim.models import LinkMetrics, LinkType, Node


def test_has_path(small_topology):
    assert has_path(
        small_topology, "pod-0-rack-leaf0-node0-gpu0", "pod-0-rack-leaf1-node0-gpu0"
    )


def test_has_path_unknown_node_raises(small_topology):
    # has_path delegates straight to networkx, which distinguishes "no such
    # entity" from "entity exists but is unreachable" by raising
    with pytest.raises(nx.NodeNotFound):
        has_path(small_topology, "pod-0-rack-leaf0-node0-gpu0", "no-such-node")


def test_same_node_is_a_single_nvlink_hop(small_topology):
    result = shortest_path(
        small_topology, "pod-0-rack-leaf0-node0-gpu0", "pod-0-rack-leaf0-node0-gpu1"
    )
    assert result.hop_count == 1
    assert result.hops[0].metrics.link_type is LinkType.NVLINK
    assert result.distinct_leaf_switches == 0
    assert result.distinct_spine_switches == 0


def test_same_leaf_different_node_goes_through_leaf_only(small_topology):
    result = shortest_path(
        small_topology, "pod-0-rack-leaf0-node0-gpu0", "pod-0-rack-leaf0-node1-gpu0"
    )
    # gpu -> node -> leaf -> node -> gpu
    assert result.hop_count == 4
    assert result.distinct_leaf_switches == 1
    assert result.spine_uplink_hops == 0


def test_different_leaf_crosses_exactly_one_spine(small_topology):
    result = shortest_path(
        small_topology, "pod-0-rack-leaf0-node0-gpu0", "pod-0-rack-leaf1-node0-gpu0"
    )
    # gpu -> node -> leaf -> spine -> leaf -> node -> gpu
    assert result.hop_count == 6
    assert result.distinct_leaf_switches == 2
    assert result.distinct_spine_switches == 1
    assert result.spine_uplink_hops == 2


def test_bottleneck_bandwidth_is_the_minimum_along_the_path(small_topology):
    result = shortest_path(
        small_topology, "pod-0-rack-leaf0-node0-gpu0", "pod-0-rack-leaf1-node0-gpu0"
    )
    slowest = min(hop.metrics.bandwidth_gbps for hop in result.hops)
    assert result.bottleneck_bandwidth_gbps == slowest


def test_weight_hops_vs_latency_agree_on_this_symmetric_fabric(small_topology):
    by_latency = shortest_path(
        small_topology, "pod-0-rack-leaf0-node0-gpu0", "pod-0-rack-leaf1-node0-gpu0", weight="latency"
    )
    by_hops = shortest_path(
        small_topology, "pod-0-rack-leaf0-node0-gpu0", "pod-0-rack-leaf1-node0-gpu0", weight="hops"
    )
    assert by_latency.hop_count == by_hops.hop_count == 6


def test_shortest_path_invalid_weight_mode_raises(small_topology):
    with pytest.raises(ValueError):
        shortest_path(small_topology, "pod-0-spine0", "pod-0-spine1", weight="bogus")


def test_k_shortest_paths_finds_both_spines(small_topology):
    # only 2 minimum-cost (6-hop) paths exist, one per spine; k=2 asks for
    # exactly those without pulling in longer alternate routes
    results = k_shortest_paths(
        small_topology, "pod-0-rack-leaf0-node0-gpu0", "pod-0-rack-leaf1-node0-gpu0", k=2
    )
    spines_used = {
        entity for r in results for entity in r.path if small_topology.kind_of(entity) == "spine"
    }
    assert spines_used == set(small_topology.entities_of_kind("spine"))
    assert all(r.hop_count == 6 for r in results)


def test_all_paths_with_cutoff_matches_spine_count(small_topology):
    results = all_paths(
        small_topology, "pod-0-rack-leaf0-node0-gpu0", "pod-0-rack-leaf1-node0-gpu0", cutoff=6
    )
    assert len(results) == len(list(small_topology.entities_of_kind("spine")))


def test_parallel_edges_use_the_best_one_for_path_search():
    # two links between the same pair of nodes (e.g. redundant NICs) --
    # path search must collapse them to the higher-bandwidth/lower-latency
    # one rather than double-counting or picking arbitrarily
    topology = ClusterTopology()
    for node_id in ("a", "b"):
        topology.add_node(
            Node(node_id=node_id, rack_id="r", leaf_id="l", gpu_ids=(), cpu_count=1, host_ram_gb=1, local_nvme_gb=1)
        )
    slow = LinkMetrics(link_type=LinkType.LEAF_UPLINK, bandwidth_gbps=100.0, latency_us=10.0)
    fast = LinkMetrics(link_type=LinkType.LEAF_UPLINK, bandwidth_gbps=400.0, latency_us=1.0)
    topology.add_link("a", "b", slow, key=0, bidirectional=False)
    topology.add_link("a", "b", fast, key=1, bidirectional=False)

    result = shortest_path(topology, "a", "b")
    assert result.hop_count == 1
    assert result.hops[0].metrics == fast
