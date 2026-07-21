import pytest

from gpu_cluster_sim.engine import ClusterTopology
from gpu_cluster_sim.models import GPU, GPUModel, LeafSwitch, LinkMetrics, LinkType, Node, RailSwitch, SpineSwitch


def test_small_topology_has_expected_shape(small_topology):
    # 2 spines, 2 leaves, 4 nodes (2/leaf), 16 GPUs (4/node)
    assert set(small_topology.entities_of_kind("spine")) == {"pod-0-spine0", "pod-0-spine1"}
    assert len(list(small_topology.entities_of_kind("leaf"))) == 2
    assert len(list(small_topology.entities_of_kind("node"))) == 4
    assert len(list(small_topology.entities_of_kind("gpu"))) == 16


def test_kind_of_and_entity_lookup(small_topology):
    assert small_topology.kind_of("pod-0-spine0") == "spine"
    assert small_topology.kind_of("pod-0-rack-leaf0") == "leaf"
    assert small_topology.kind_of("pod-0-rack-leaf0-node0") == "node"
    assert small_topology.kind_of("pod-0-rack-leaf0-node0-gpu0") == "gpu"

    gpu = small_topology.entity("pod-0-rack-leaf0-node0-gpu0")
    assert isinstance(gpu, GPU)
    assert gpu.model == GPUModel.H100
    assert gpu.node_id == "pod-0-rack-leaf0-node0"
    assert gpu.rail_index == 0


def test_leaf_spine_fabric_is_full_bipartite(small_topology):
    for leaf_id in small_topology.entities_of_kind("leaf"):
        spine_neighbors = {
            v for u, v, _ in small_topology.edges_of_type(LinkType.SPINE_UPLINK) if u == leaf_id
        }
        assert spine_neighbors == set(small_topology.entities_of_kind("spine"))


def test_gpu_has_nvlink_to_every_sibling_but_not_other_nodes(small_topology):
    node = small_topology.entity("pod-0-rack-leaf0-node0")
    gpu0 = "pod-0-rack-leaf0-node0-gpu0"
    nvlink_neighbors = {v for u, v, _ in small_topology.edges_of_type(LinkType.NVLINK) if u == gpu0}
    assert nvlink_neighbors == set(node.gpu_ids) - {gpu0}


def test_gpu_has_pcie_path_out_to_its_own_node(small_topology):
    gpu0 = "pod-0-rack-leaf0-node0-gpu0"
    pcie_neighbors = {v for u, v, _ in small_topology.edges_of_type(LinkType.PCIE) if u == gpu0}
    assert pcie_neighbors == {"pod-0-rack-leaf0-node0"}


def test_add_rail():
    topology = ClusterTopology()
    rail = RailSwitch(rail_id="rail-0", rail_index=0, pod_id="pod-0")
    topology.add_rail(rail)
    assert topology.kind_of("rail-0") == "rail"
    assert topology.entity("rail-0") == rail


def test_topology_convenience_methods_delegate_to_engine_paths(small_topology):
    assert small_topology.has_path(
        "pod-0-rack-leaf0-node0-gpu0", "pod-0-rack-leaf1-node0-gpu0"
    )

    result = small_topology.shortest_path(
        "pod-0-rack-leaf0-node0-gpu0", "pod-0-rack-leaf0-node0-gpu1"
    )
    assert result.hop_count == 1

    k_results = small_topology.k_shortest_paths(
        "pod-0-rack-leaf0-node0-gpu0", "pod-0-rack-leaf1-node0-gpu0", k=2
    )
    assert len(k_results) == 2

    all_results = small_topology.all_paths(
        "pod-0-rack-leaf0-node0-gpu0", "pod-0-rack-leaf1-node0-gpu0", cutoff=6
    )
    assert len(all_results) == 2


def test_add_link_bidirectional_by_default():
    topology = ClusterTopology()
    topology.add_leaf(
        LeafSwitch(
            leaf_id="leaf-0", pod_id="pod", rack_id="rack", downlink_ports=1, uplink_ports=1, port_bandwidth_gbps=100
        )
    )
    topology.add_spine(SpineSwitch(spine_id="spine-0", pod_id="pod", port_count=1, port_bandwidth_gbps=100))
    metrics = LinkMetrics(link_type=LinkType.SPINE_UPLINK, bandwidth_gbps=100.0, latency_us=1.0)
    topology.add_link("leaf-0", "spine-0", metrics)

    assert topology.link_metrics("leaf-0", "spine-0") == metrics
    assert topology.link_metrics("spine-0", "leaf-0") == metrics


def test_add_link_asymmetric_reverse_metrics():
    topology = ClusterTopology()
    topology.add_node(
        Node(
            node_id="node-0",
            rack_id="rack-0",
            leaf_id="leaf-0",
            gpu_ids=(),
            cpu_count=1,
            host_ram_gb=1,
            local_nvme_gb=1,
        )
    )
    topology.add_leaf(
        LeafSwitch(
            leaf_id="leaf-0", pod_id="pod", rack_id="rack-0", downlink_ports=1, uplink_ports=1, port_bandwidth_gbps=100
        )
    )
    downlink = LinkMetrics(link_type=LinkType.LEAF_UPLINK, bandwidth_gbps=400.0, latency_us=2.0, oversubscription_ratio=3.0)
    uplink = LinkMetrics(link_type=LinkType.LEAF_UPLINK, bandwidth_gbps=133.0, latency_us=2.0, oversubscription_ratio=3.0)
    topology.add_link("node-0", "leaf-0", downlink, reverse_metrics=uplink)

    assert topology.link_metrics("node-0", "leaf-0") == downlink
    assert topology.link_metrics("leaf-0", "node-0") == uplink


def test_add_link_not_bidirectional_only_adds_one_direction():
    topology = ClusterTopology()
    topology.add_gpu(
        GPU(gpu_id="gpu-a", node_id="node-0", rail_index=0, model=GPUModel.H100, memory_gb=80, nvlink_domain="node-0")
    )
    topology.add_gpu(
        GPU(gpu_id="gpu-b", node_id="node-0", rail_index=1, model=GPUModel.H100, memory_gb=80, nvlink_domain="node-0")
    )
    metrics = LinkMetrics(link_type=LinkType.NVLINK, bandwidth_gbps=900.0, latency_us=0.5)
    topology.add_link("gpu-a", "gpu-b", metrics, bidirectional=False)

    assert topology.link_metrics("gpu-a", "gpu-b") == metrics
    with pytest.raises(KeyError):
        topology.link_metrics("gpu-b", "gpu-a")
