import pytest

from gpu_cluster_sim.engine import ClusterTopology, available_formats


@pytest.mark.parametrize("format_name", ["graphml", "graphar"])
def test_registered_formats_available(format_name):
    assert format_name in available_formats()


@pytest.mark.parametrize("format_name", ["graphml", "graphar"])
def test_round_trip_preserves_graph_shape(small_topology, tmp_path, format_name):
    path = tmp_path / f"topology.{format_name}"
    small_topology.export(format_name, str(path))
    loaded = ClusterTopology.load(format_name, str(path))

    assert loaded.graph.number_of_nodes() == small_topology.graph.number_of_nodes()
    assert loaded.graph.number_of_edges() == small_topology.graph.number_of_edges()


@pytest.mark.parametrize("format_name", ["graphml", "graphar"])
def test_round_trip_preserves_entity_data(small_topology, tmp_path, format_name):
    path = tmp_path / f"topology.{format_name}"
    small_topology.export(format_name, str(path))
    loaded = ClusterTopology.load(format_name, str(path))

    for entity_id in ("pod-0-spine0", "pod-0-rack-leaf0", "pod-0-rack-leaf0-node0", "pod-0-rack-leaf0-node0-gpu0"):
        assert loaded.kind_of(entity_id) == small_topology.kind_of(entity_id)
        assert loaded.entity(entity_id) == small_topology.entity(entity_id)


@pytest.mark.parametrize("format_name", ["graphml", "graphar"])
def test_round_trip_preserves_link_metrics(small_topology, tmp_path, format_name):
    path = tmp_path / f"topology.{format_name}"
    small_topology.export(format_name, str(path))
    loaded = ClusterTopology.load(format_name, str(path))

    original = small_topology.link_metrics("pod-0-rack-leaf0-node0", "pod-0-rack-leaf0")
    reloaded = loaded.link_metrics("pod-0-rack-leaf0-node0", "pod-0-rack-leaf0")
    assert reloaded == original


def test_get_format_rejects_unknown_name():
    from gpu_cluster_sim.engine.formats import get_format

    with pytest.raises(ValueError):
        get_format("not-a-real-format")
