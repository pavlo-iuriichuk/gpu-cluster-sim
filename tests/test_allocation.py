import pytest

from gpu_cluster_sim.engine import AllocationLedger
from gpu_cluster_sim.models import (
    GPUModel,
    LocalityConstraint,
    NodeHealth,
    QoSClass,
)


def _set_node(topology, node_id, **overrides):
    """Test-only helper: replace a node's model with one or more fields
    overridden, e.g. `_set_node(topology, "node-0", host_ram_gb=1)`.
    """
    node = topology.entity(node_id)
    topology.graph.nodes[node_id]["data"] = node._replace(**overrides)


def test_try_allocate_same_leaf_grants_and_ranks_gpus_by_node_then_rail(small_topology, make_job):
    ledger = AllocationLedger()
    job = make_job("job-1", nodes=2, gpus_per_node=4, locality_constraint=LocalityConstraint.SAME_LEAF)

    allocation = ledger.try_allocate(small_topology, job)

    assert allocation is not None
    assert len(allocation.gpu_ids) == 8
    assert len(allocation.node_ids) == 2
    leaves = {small_topology.entity(n).leaf_id for n in allocation.node_ids}
    assert len(leaves) == 1
    assert allocation.qos_class == QoSClass.BEST_EFFORT

    # gpu_ids must be grouped by node (no interleaving) and rail-ordered within a node
    node_sequence = [small_topology.entity(g).node_id for g in allocation.gpu_ids]
    seen = []
    for node_id in node_sequence:
        if not seen or seen[-1] != node_id:
            seen.append(node_id)
    assert len(seen) == len(set(node_sequence))  # each node's GPUs form one contiguous block

    for node_id in set(node_sequence):
        rails = [small_topology.entity(g).rail_index for g in allocation.gpu_ids if small_topology.entity(g).node_id == node_id]
        assert rails == sorted(rails)


def test_try_allocate_is_all_or_nothing_when_it_does_not_fit(small_topology, make_job):
    ledger = AllocationLedger()
    # only 2 leaves x 2 nodes each = 4 nodes total; ask for more nodes than exist on any one leaf
    job = make_job("job-huge", nodes=3, gpus_per_node=4, locality_constraint=LocalityConstraint.SAME_LEAF)

    allocation = ledger.try_allocate(small_topology, job)

    assert allocation is None
    assert ledger.allocation_for_job("job-huge") is None
    assert ledger.free_gpu_count(small_topology) == 16  # untouched


def test_try_allocate_double_booking_same_job_id_raises(small_topology, make_job):
    ledger = AllocationLedger()
    job = make_job("job-1", nodes=1, gpus_per_node=4)
    assert ledger.try_allocate(small_topology, job) is not None

    with pytest.raises(ValueError):
        ledger.try_allocate(small_topology, job)


def test_gpus_cannot_be_double_allocated(small_topology, make_job):
    ledger = AllocationLedger()
    job_a = make_job("job-a", nodes=2, gpus_per_node=4, locality_constraint=LocalityConstraint.SAME_LEAF)
    job_b = make_job("job-b", nodes=2, gpus_per_node=4, locality_constraint=LocalityConstraint.SAME_LEAF)
    job_c = make_job("job-c", nodes=1, gpus_per_node=4, locality_constraint=LocalityConstraint.SAME_LEAF)

    alloc_a = ledger.try_allocate(small_topology, job_a)
    alloc_b = ledger.try_allocate(small_topology, job_b)
    assert alloc_a is not None and alloc_b is not None
    assert set(alloc_a.gpu_ids).isdisjoint(alloc_b.gpu_ids)

    # cluster is now fully allocated (16/16 gpus)
    assert ledger.free_gpu_count(small_topology) == 0
    assert ledger.try_allocate(small_topology, job_c) is None


def test_same_spine_locality_allows_crossing_leaves(small_topology, make_job):
    ledger = AllocationLedger()
    # 4 nodes total, both leaves share both spines -> SAME_SPINE can use all 4 nodes
    job = make_job("job-1", nodes=4, gpus_per_node=4, locality_constraint=LocalityConstraint.SAME_SPINE)

    allocation = ledger.try_allocate(small_topology, job)
    assert allocation is not None
    assert len(allocation.node_ids) == 4


def test_best_effort_locality_has_no_leaf_or_spine_constraint(small_topology, make_job):
    ledger = AllocationLedger()
    job = make_job("job-1", nodes=4, gpus_per_node=4, locality_constraint=LocalityConstraint.BEST_EFFORT)
    allocation = ledger.try_allocate(small_topology, job)
    assert allocation is not None
    assert len(allocation.node_ids) == 4


def test_gpu_model_mismatch_is_never_selected(small_topology, make_job):
    ledger = AllocationLedger()
    job = make_job("job-1", nodes=1, gpus_per_node=4, gpu_model=GPUModel.A100)
    assert ledger.try_allocate(small_topology, job) is None


def test_unhealthy_node_is_skipped(small_topology, make_job):
    # only one node per leaf remains healthy; degrade every node on leaf1 and
    # one node on leaf0 so only a single healthy node (4 GPUs) is left
    for node_id in list(small_topology.entities_of_kind("node")):
        if node_id != "pod-0-rack-leaf0-node0":
            _set_node(small_topology, node_id, health=NodeHealth.DEGRADED)

    ledger = AllocationLedger()
    job = make_job("job-1", nodes=1, gpus_per_node=4, locality_constraint=LocalityConstraint.BEST_EFFORT)
    allocation = ledger.try_allocate(small_topology, job)
    assert allocation is not None
    assert allocation.node_ids == ("pod-0-rack-leaf0-node0",)

    job2 = make_job("job-2", nodes=1, gpus_per_node=4, locality_constraint=LocalityConstraint.BEST_EFFORT)
    assert ledger.try_allocate(small_topology, job2) is None


def test_shape_hint_gpu_count_mismatch_raises(small_topology, make_job):
    from gpu_cluster_sim.models import ShapeHint

    ledger = AllocationLedger()
    bad_job = make_job("job-1", nodes=2, gpus_per_node=4)._replace(shape_hint=ShapeHint(nodes=3, gpus_per_node=4))
    with pytest.raises(ValueError):
        ledger.try_allocate(small_topology, bad_job)


def test_release_unknown_job_raises():
    ledger = AllocationLedger()
    with pytest.raises(KeyError):
        ledger.release("no-such-job")


def test_job_for_gpu(small_topology, make_job):
    ledger = AllocationLedger()
    job = make_job("job-1", nodes=1, gpus_per_node=4)
    allocation = ledger.try_allocate(small_topology, job)

    for gpu_id in allocation.gpu_ids:
        assert ledger.job_for_gpu(gpu_id) == "job-1"
    assert ledger.job_for_gpu("some-other-gpu") is None


def test_snapshot_reflects_current_state(small_topology, make_job):
    ledger = AllocationLedger()
    job = make_job("job-1", nodes=1, gpus_per_node=4)
    allocation = ledger.try_allocate(small_topology, job)

    snapshot = ledger.snapshot()
    assert allocation in snapshot.allocations
    assert set(gpu_id for gpu_id, job_id in snapshot.gpu_to_job) == set(allocation.gpu_ids)
    assert all(job_id == "job-1" for _, job_id in snapshot.gpu_to_job)

    ledger.release("job-1")
    empty_snapshot = ledger.snapshot()
    assert empty_snapshot.allocations == ()
    assert empty_snapshot.gpu_to_job == ()


def test_gpu_memory_below_requirement_is_never_selected(small_topology, make_job):
    ledger = AllocationLedger()
    job = make_job("job-1", nodes=1, gpus_per_node=4, memory_per_gpu_gb=9999)
    assert ledger.try_allocate(small_topology, job) is None


@pytest.mark.parametrize(
    "node_override",
    [
        {"host_ram_gb": 1},
        {"local_nvme_gb": 1},
        {"cpu_count": 1},
    ],
)
def test_insufficient_node_resources_excludes_the_node(small_topology, make_job, node_override):
    for node_id in list(small_topology.entities_of_kind("node")):
        _set_node(small_topology, node_id, **node_override)

    ledger = AllocationLedger()
    job = make_job(
        "job-1", nodes=1, gpus_per_node=4, host_ram_gb=256, local_nvme_gb=1000, cpu_per_gpu=8,
        locality_constraint=LocalityConstraint.BEST_EFFORT,
    )
    assert ledger.try_allocate(small_topology, job) is None


def test_same_spine_locality_fails_when_nothing_fits(small_topology, make_job):
    ledger = AllocationLedger()
    # 4 nodes total exist; ask for more than the whole fabric can provide
    job = make_job("job-1", nodes=5, gpus_per_node=4, locality_constraint=LocalityConstraint.SAME_SPINE)
    assert ledger.try_allocate(small_topology, job) is None
