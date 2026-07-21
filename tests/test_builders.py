from gpu_cluster_sim.models import GPUModel, LinkType
from gpu_cluster_sim.samples import (
    build_datacenter_mixed_gpus,
    build_multi_az_region,
    build_single_az,
)


def test_build_single_az_shape():
    topology = build_single_az()
    assert len(list(topology.entities_of_kind("spine"))) == 4
    assert len(list(topology.entities_of_kind("leaf"))) == 6
    assert len(list(topology.entities_of_kind("node"))) == 24
    assert len(list(topology.entities_of_kind("gpu"))) == 192
    assert all(
        topology.entity(g).model == GPUModel.H100 for g in topology.entities_of_kind("gpu")
    )


def test_single_az_leaf_spine_is_full_bipartite():
    topology = build_single_az()
    spines = set(topology.entities_of_kind("spine"))
    for leaf_id in topology.entities_of_kind("leaf"):
        neighbors = {v for u, v, _ in topology.edges_of_type(LinkType.SPINE_UPLINK) if u == leaf_id}
        assert neighbors == spines


def test_build_datacenter_mixed_gpus_has_three_models_and_expected_counts():
    topology = build_datacenter_mixed_gpus()
    assert len(list(topology.entities_of_kind("node"))) == 40
    assert len(list(topology.entities_of_kind("gpu"))) == 280

    models = {}
    for gpu_id in topology.entities_of_kind("gpu"):
        model = topology.entity(gpu_id).model
        models[model] = models.get(model, 0) + 1
    assert models == {GPUModel.H100: 120, GPUModel.A100: 120, GPUModel.A6000: 40}


def test_a6000_rack_group_uses_smaller_node_shape():
    topology = build_datacenter_mixed_gpus()
    a6000_nodes = {
        topology.entity(g).node_id
        for g in topology.entities_of_kind("gpu")
        if topology.entity(g).model == GPUModel.A6000
    }
    for node_id in a6000_nodes:
        node = topology.entity(node_id)
        assert len(node.gpu_ids) == 4
        assert node.cpu_count == 64


def test_build_multi_az_region_shape_and_inter_pod_backbone():
    topology = build_multi_az_region(region_id="us-west", az_count=3)
    assert len(list(topology.entities_of_kind("node"))) == 36
    assert len(list(topology.entities_of_kind("gpu"))) == 288

    inter_pod_edges = list(topology.edges_of_type(LinkType.INTER_POD))
    # one link per unordered pair of AZs, each added bidirectionally
    assert len(inter_pod_edges) == 3 * 2  # C(3,2) pairs x 2 directions


def test_inter_pod_link_is_the_only_way_between_azs():
    topology = build_multi_az_region(region_id="us-west", az_count=2)
    from gpu_cluster_sim.engine import shortest_path

    gpu_az1 = next(g for g in topology.entities_of_kind("gpu") if "az1" in g)
    gpu_az2 = next(g for g in topology.entities_of_kind("gpu") if "az2" in g)
    result = shortest_path(topology, gpu_az1, gpu_az2)
    assert result.inter_pod_hops == 1
