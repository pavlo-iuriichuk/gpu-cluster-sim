from decimal import Decimal

from gpu_cluster_sim.engine import (
    AllocationLedger,
    CheckpointLedger,
    complete_job,
    expected_data_loss_s,
    preempt_job,
    preemption_probability,
    unprotected_work_s,
)
from gpu_cluster_sim.models import QoSClass


def test_preemption_probability_none_without_history():
    ledger = AllocationLedger()
    assert preemption_probability(ledger) is None
    assert preemption_probability(ledger, QoSClass.PRODUCTION) is None


def test_preemption_probability_overall_and_by_qos(small_topology, make_job):
    ledger = AllocationLedger()
    for i in range(3):
        job = make_job(f"job-{i}", nodes=1, gpus_per_node=4, qos_class=QoSClass.BEST_EFFORT)
        ledger.try_allocate(small_topology, job, timestamp=0.0)
    for i, action in enumerate([preempt_job, preempt_job, complete_job]):
        action(ledger, f"job-{i}", timestamp=10.0)

    assert preemption_probability(ledger) == 2 / 3
    assert preemption_probability(ledger, QoSClass.BEST_EFFORT) == 2 / 3
    assert preemption_probability(ledger, QoSClass.PRODUCTION) is None


def test_unprotected_work_uses_last_checkpoint_when_present():
    ckpt_ledger = CheckpointLedger()
    ckpt_ledger.save_checkpoint("job-1", step=0, size_gb=Decimal("1"), duration_s=Decimal("1"), timestamp=1200.0)
    assert unprotected_work_s(ckpt_ledger, "job-1", allocated_at=0.0, now=1450.0) == 250.0


def test_unprotected_work_falls_back_to_allocated_at_when_never_checkpointed():
    ckpt_ledger = CheckpointLedger()
    assert unprotected_work_s(ckpt_ledger, "job-never", allocated_at=100.0, now=500.0) == 400.0


def test_unprotected_work_never_negative():
    ckpt_ledger = CheckpointLedger()
    ckpt_ledger.save_checkpoint("job-1", step=0, size_gb=Decimal("1"), duration_s=Decimal("1"), timestamp=1000.0)
    assert unprotected_work_s(ckpt_ledger, "job-1", allocated_at=0.0, now=500.0) == 0.0


def test_expected_data_loss_combines_probability_and_unprotected_work(small_topology, make_job):
    alloc_ledger = AllocationLedger()
    ckpt_ledger = CheckpointLedger()

    job = make_job("job-train", nodes=1, gpus_per_node=4, qos_class=QoSClass.BEST_EFFORT)
    alloc_ledger.try_allocate(small_topology, job, timestamp=0.0)
    ckpt_ledger.save_checkpoint("job-train", step=0, size_gb=Decimal("1"), duration_s=Decimal("1"), timestamp=1200.0)

    # build preemption history from other same-QoS jobs
    for i in range(2):
        filler = make_job(f"filler-{i}", nodes=1, gpus_per_node=4, qos_class=QoSClass.BEST_EFFORT)
        alloc_ledger.try_allocate(small_topology, filler, timestamp=0.0)
    preempt_job(alloc_ledger, "filler-0", timestamp=10.0)
    complete_job(alloc_ledger, "filler-1", timestamp=10.0)

    p = preemption_probability(alloc_ledger, QoSClass.BEST_EFFORT)
    unprotected = unprotected_work_s(ckpt_ledger, "job-train", allocated_at=0.0, now=1450.0)
    expected = expected_data_loss_s(
        alloc_ledger, ckpt_ledger, "job-train", allocated_at=0.0, now=1450.0, qos_class=QoSClass.BEST_EFFORT
    )
    assert expected == p * unprotected


def test_expected_data_loss_none_without_release_history():
    alloc_ledger = AllocationLedger()
    ckpt_ledger = CheckpointLedger()
    assert (
        expected_data_loss_s(alloc_ledger, ckpt_ledger, "job-1", allocated_at=0.0, now=100.0) is None
    )
