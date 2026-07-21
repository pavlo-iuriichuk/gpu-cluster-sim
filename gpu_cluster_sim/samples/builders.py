"""Builders for realistic sample cluster topologies, used to populate
`data/`. Each builder returns a `ClusterTopology`; `populate_pod` is the
shared primitive (a "pod" = one availability zone: a leaf-spine fabric plus
the nodes/GPUs hanging off it), so a region is just several pods stitched
together with inter-pod links.
"""

from typing import Dict, List, NamedTuple

from ..engine.topology import ClusterTopology
from ..models.gpu import GPU, GPUModel
from ..models.network import LeafSwitch, LinkMetrics, LinkType, SpineSwitch
from ..models.node import Node


class RackGroup(NamedTuple):
    """One homogeneous slice of a pod: `leaf_count` racks, each with
    `nodes_per_leaf` identical nodes of `gpu_model` GPUs.
    """

    name: str
    leaf_count: int
    nodes_per_leaf: int
    gpus_per_node: int
    gpu_model: GPUModel
    gpu_memory_gb: int
    node_cpu_count: int = 96
    node_host_ram_gb: int = 2048
    node_local_nvme_gb: int = 8000


def populate_pod(
    topology: ClusterTopology,
    pod_id: str,
    spine_count: int,
    rack_groups: List[RackGroup],
    *,
    leaf_uplink_bw_gbps: float = 400.0,
    leaf_uplink_latency_us: float = 2.0,
    leaf_oversubscription_ratio: float = 3.0,
    spine_uplink_bw_gbps: float = 400.0,
    spine_uplink_latency_us: float = 1.0,
    nvlink_bw_gbps: float = 900.0,
    nvlink_latency_us: float = 0.5,
    pcie_bw_gbps: float = 400.0,
    pcie_latency_us: float = 1.0,
) -> Dict[str, List[str]]:
    """Add one pod's spines, leaves (full leaf<->spine bipartite fabric),
    nodes, and GPUs (full NVLink mesh per node) to `topology` in place.
    Returns the ids created, so callers (e.g. a multi-pod region builder)
    can wire additional links such as inter-pod backbones.
    """
    spine_ids = [f"{pod_id}-spine{i}" for i in range(spine_count)]
    for spine_id in spine_ids:
        topology.add_spine(
            SpineSwitch(spine_id=spine_id, pod_id=pod_id, port_count=64, port_bandwidth_gbps=spine_uplink_bw_gbps)
        )

    leaf_ids: List[str] = []
    node_ids: List[str] = []
    gpu_ids: List[str] = []

    for group in rack_groups:
        for leaf_idx in range(group.leaf_count):
            leaf_id = f"{pod_id}-{group.name}-leaf{leaf_idx}"
            topology.add_leaf(
                LeafSwitch(
                    leaf_id=leaf_id,
                    pod_id=pod_id,
                    rack_id=leaf_id,  # one leaf (ToR switch) per rack
                    downlink_ports=group.nodes_per_leaf,
                    uplink_ports=spine_count,
                    port_bandwidth_gbps=leaf_uplink_bw_gbps,
                    oversubscription_ratio=leaf_oversubscription_ratio,
                )
            )
            leaf_ids.append(leaf_id)
            for spine_id in spine_ids:
                topology.add_link(
                    leaf_id,
                    spine_id,
                    LinkMetrics(
                        link_type=LinkType.SPINE_UPLINK,
                        bandwidth_gbps=spine_uplink_bw_gbps,
                        latency_us=spine_uplink_latency_us,
                        oversubscription_ratio=leaf_oversubscription_ratio,
                    ),
                )

            for node_idx in range(group.nodes_per_leaf):
                node_id = f"{leaf_id}-node{node_idx}"
                gpu_ids_for_node = tuple(f"{node_id}-gpu{i}" for i in range(group.gpus_per_node))
                topology.add_node(
                    Node(
                        node_id=node_id,
                        rack_id=leaf_id,
                        leaf_id=leaf_id,
                        gpu_ids=gpu_ids_for_node,
                        cpu_count=group.node_cpu_count,
                        host_ram_gb=group.node_host_ram_gb,
                        local_nvme_gb=group.node_local_nvme_gb,
                    )
                )
                node_ids.append(node_id)
                topology.add_link(
                    node_id,
                    leaf_id,
                    LinkMetrics(
                        link_type=LinkType.LEAF_UPLINK,
                        bandwidth_gbps=leaf_uplink_bw_gbps,
                        latency_us=leaf_uplink_latency_us,
                    ),
                )

                for gpu_idx, gpu_id in enumerate(gpu_ids_for_node):
                    topology.add_gpu(
                        GPU(
                            gpu_id=gpu_id,
                            node_id=node_id,
                            rail_index=gpu_idx,
                            model=group.gpu_model,
                            memory_gb=group.gpu_memory_gb,
                            nvlink_domain=node_id,
                        )
                    )
                    gpu_ids.append(gpu_id)
                    # the GPU's own path out to the node's NIC/uplink, without
                    # which it could only ever reach its NVLink siblings
                    topology.add_link(
                        gpu_id,
                        node_id,
                        LinkMetrics(
                            link_type=LinkType.PCIE,
                            bandwidth_gbps=pcie_bw_gbps,
                            latency_us=pcie_latency_us,
                        ),
                    )

                for a in gpu_ids_for_node:
                    for b in gpu_ids_for_node:
                        if a != b:
                            topology.add_link(
                                a,
                                b,
                                LinkMetrics(
                                    link_type=LinkType.NVLINK,
                                    bandwidth_gbps=nvlink_bw_gbps,
                                    latency_us=nvlink_latency_us,
                                ),
                                bidirectional=False,
                            )

    return {"spine_ids": spine_ids, "leaf_ids": leaf_ids, "node_ids": node_ids, "gpu_ids": gpu_ids}


def build_single_az(pod_id: str = "az-usw2-1") -> ClusterTopology:
    """One AZ, one GPU model, several leaves behind several spines: 4
    spines x 6 leaves x 4 nodes x 8 H100s = 192 GPUs.
    """
    topology = ClusterTopology()
    populate_pod(
        topology,
        pod_id,
        spine_count=4,
        rack_groups=[
            RackGroup(
                name="h100-rack",
                leaf_count=6,
                nodes_per_leaf=4,
                gpus_per_node=8,
                gpu_model=GPUModel.H100,
                gpu_memory_gb=80,
            )
        ],
    )
    return topology


def build_datacenter_mixed_gpus(pod_id: str = "dc-central-1") -> ClusterTopology:
    """One data center, heterogeneous hardware generations sitting side by
    side (a common real-world state: H100s for new training, A100s still in
    rotation, A6000s for inference/dev): 120 H100 + 120 A100 + 40 A6000 =
    280 GPUs across 8 leaves behind 4 spines.
    """
    topology = ClusterTopology()
    populate_pod(
        topology,
        pod_id,
        spine_count=4,
        rack_groups=[
            RackGroup(
                name="h100-rack",
                leaf_count=3,
                nodes_per_leaf=5,
                gpus_per_node=8,
                gpu_model=GPUModel.H100,
                gpu_memory_gb=80,
            ),
            RackGroup(
                name="a100-rack",
                leaf_count=3,
                nodes_per_leaf=5,
                gpus_per_node=8,
                gpu_model=GPUModel.A100,
                gpu_memory_gb=80,
            ),
            RackGroup(
                name="a6000-rack",
                leaf_count=2,
                nodes_per_leaf=5,
                gpus_per_node=4,
                gpu_model=GPUModel.A6000,
                gpu_memory_gb=48,
                node_cpu_count=64,
                node_host_ram_gb=1024,
                node_local_nvme_gb=4000,
            ),
        ],
    )
    return topology


def build_multi_az_region(region_id: str = "us-west", az_count: int = 3) -> ClusterTopology:
    """A region made of several AZs, each an independent leaf-spine pod (2
    spines x 4 leaves x 3 nodes x 8 H100s = 96 GPUs/AZ), stitched together
    by an inter-pod backbone: one spine per AZ meshed to one spine per every
    other AZ, at WAN-like bandwidth/latency/oversubscription.
    """
    topology = ClusterTopology()
    az_spine_ids: List[List[str]] = []
    for i in range(az_count):
        pod_id = f"{region_id}-az{i + 1}"
        info = populate_pod(
            topology,
            pod_id,
            spine_count=2,
            rack_groups=[
                RackGroup(
                    name="h100-rack",
                    leaf_count=4,
                    nodes_per_leaf=3,
                    gpus_per_node=8,
                    gpu_model=GPUModel.H100,
                    gpu_memory_gb=80,
                )
            ],
        )
        az_spine_ids.append(info["spine_ids"])

    for i in range(len(az_spine_ids)):
        for j in range(i + 1, len(az_spine_ids)):
            topology.add_link(
                az_spine_ids[i][0],
                az_spine_ids[j][0],
                LinkMetrics(
                    link_type=LinkType.INTER_POD,
                    bandwidth_gbps=100.0,
                    latency_us=500.0,
                    oversubscription_ratio=6.0,
                ),
            )

    return topology
