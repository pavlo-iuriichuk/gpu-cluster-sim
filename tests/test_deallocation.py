from gpu_cluster_sim.engine import (
    AllocationLedger,
    cancel_job,
    complete_job,
    history_by_reason,
    preempt_job,
    release_on_demand,
)
from gpu_cluster_sim.models import DeallocationReason


def test_each_named_wrapper_tags_the_correct_reason(small_topology, make_job):
    ledger = AllocationLedger()
    jobs = {
        "job-completed": complete_job,
        "job-cancelled": cancel_job,
        "job-preempted": preempt_job,
        "job-manual": release_on_demand,
    }
    allocations = {}
    for job_id in jobs:
        job = make_job(job_id, nodes=1, gpus_per_node=4)
        allocations[job_id] = ledger.try_allocate(small_topology, job)
        assert allocations[job_id] is not None

    rec_completed = complete_job(ledger, "job-completed", timestamp=100.0)
    rec_cancelled = cancel_job(ledger, "job-cancelled", timestamp=101.0)
    rec_preempted = preempt_job(ledger, "job-preempted", grace_period_s=30.0, timestamp=102.0)
    rec_manual = release_on_demand(ledger, "job-manual", timestamp=103.0)

    assert rec_completed.reason == DeallocationReason.COMPLETED
    assert rec_cancelled.reason == DeallocationReason.CANCELLED
    assert rec_preempted.reason == DeallocationReason.PREEMPTED
    assert rec_preempted.grace_period_s == 30.0
    assert rec_manual.reason == DeallocationReason.MANUAL

    for job_id, record in [
        ("job-completed", rec_completed),
        ("job-cancelled", rec_cancelled),
        ("job-preempted", rec_preempted),
        ("job-manual", rec_manual),
    ]:
        assert record.gpu_ids == allocations[job_id].gpu_ids
        assert record.node_ids == allocations[job_id].node_ids
        assert ledger.allocation_for_job(job_id) is None
        assert all(ledger.is_free(g) for g in record.gpu_ids)


def test_held_duration_computed_from_timestamps(small_topology, make_job):
    ledger = AllocationLedger()
    job = make_job("job-1", nodes=1, gpus_per_node=4)
    ledger.try_allocate(small_topology, job, timestamp=1000.0)
    record = preempt_job(ledger, "job-1", grace_period_s=15.0, timestamp=1090.0)
    assert record.held_duration_s == 90.0
    assert record.grace_period_s == 15.0


def test_history_by_reason_filters_correctly(small_topology, make_job):
    ledger = AllocationLedger()
    for i in range(3):
        job = make_job(f"job-{i}", nodes=1, gpus_per_node=4)
        ledger.try_allocate(small_topology, job, timestamp=0.0)

    complete_job(ledger, "job-0", timestamp=10.0)
    preempt_job(ledger, "job-1", timestamp=10.0)
    preempt_job(ledger, "job-2", timestamp=10.0)

    assert len(ledger.history()) == 3
    preempted = history_by_reason(ledger, DeallocationReason.PREEMPTED)
    assert {r.job_id for r in preempted} == {"job-1", "job-2"}
    assert history_by_reason(ledger, DeallocationReason.CANCELLED) == ()


def test_released_gpus_can_be_reallocated(small_topology, make_job):
    ledger = AllocationLedger()
    job = make_job("job-1", nodes=1, gpus_per_node=4)
    ledger.try_allocate(small_topology, job)
    assert ledger.free_gpu_count(small_topology) == 12

    complete_job(ledger, "job-1")
    assert ledger.free_gpu_count(small_topology) == 16

    job_again = make_job("job-1", nodes=1, gpus_per_node=4)
    alloc_again = ledger.try_allocate(small_topology, job_again)
    assert alloc_again is not None
    assert ledger.free_gpu_count(small_topology) == 12
