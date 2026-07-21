from decimal import Decimal

import pytest

from gpu_cluster_sim.engine import CheckpointLedger, optimal_checkpoint_interval_s


def test_save_and_load_latest_checkpoint():
    ledger = CheckpointLedger()
    ledger.save_checkpoint("job-1", step=0, size_gb=Decimal("40"), duration_s=Decimal("20"), timestamp=0.0)
    ledger.save_checkpoint("job-1", step=1000, size_gb=Decimal("40"), duration_s=Decimal("20"), timestamp=300.0)

    latest = ledger.load_checkpoint("job-1")
    assert latest.step == 1000
    assert latest.checkpoint_id == "job-1-ckpt1"
    assert latest.size_gb == Decimal("40")
    assert latest.storage_uri == "checkpoints/job-1/step-1000"


def test_load_specific_checkpoint_by_id():
    ledger = CheckpointLedger()
    ledger.save_checkpoint("job-1", step=0, size_gb=Decimal("40"), duration_s=Decimal("20"))
    ledger.save_checkpoint("job-1", step=1000, size_gb=Decimal("40"), duration_s=Decimal("20"))

    first = ledger.load_checkpoint("job-1", checkpoint_id="job-1-ckpt0")
    assert first.step == 0


def test_load_checkpoint_unknown_job_raises():
    ledger = CheckpointLedger()
    with pytest.raises(KeyError):
        ledger.load_checkpoint("no-such-job")


def test_load_checkpoint_unknown_id_raises():
    ledger = CheckpointLedger()
    ledger.save_checkpoint("job-1", step=0, size_gb=Decimal("40"), duration_s=Decimal("20"))
    with pytest.raises(KeyError):
        ledger.load_checkpoint("job-1", checkpoint_id="job-1-ckpt99")


def test_latest_checkpoint_returns_none_instead_of_raising():
    ledger = CheckpointLedger()
    assert ledger.latest_checkpoint("no-such-job") is None


def test_mean_interval_and_frequency():
    ledger = CheckpointLedger()
    for i, t in enumerate([300.0, 600.0, 900.0, 1200.0]):
        ledger.save_checkpoint("job-1", step=i * 1000, size_gb=Decimal("40"), duration_s=Decimal("20"), timestamp=t)

    assert ledger.mean_checkpoint_interval_s("job-1") == 300.0
    assert ledger.checkpoint_frequency_per_hour("job-1") == pytest.approx(12.0)


def test_interval_and_frequency_none_with_fewer_than_two_checkpoints():
    ledger = CheckpointLedger()
    assert ledger.mean_checkpoint_interval_s("job-1") is None
    assert ledger.checkpoint_frequency_per_hour("job-1") is None

    ledger.save_checkpoint("job-1", step=0, size_gb=Decimal("40"), duration_s=Decimal("20"), timestamp=0.0)
    assert ledger.mean_checkpoint_interval_s("job-1") is None


def test_checkpoint_overhead_fraction_none_with_fewer_than_two_checkpoints():
    ledger = CheckpointLedger()
    assert ledger.checkpoint_overhead_fraction("job-1") is None
    ledger.save_checkpoint("job-1", step=0, size_gb=Decimal("40"), duration_s=Decimal("20"), timestamp=0.0)
    assert ledger.checkpoint_overhead_fraction("job-1") is None


def test_checkpoint_overhead_fraction_is_decimal():
    ledger = CheckpointLedger()
    for i, t in enumerate([300.0, 600.0, 900.0]):
        ledger.save_checkpoint("job-1", step=i, size_gb=Decimal("40"), duration_s=Decimal("20"), timestamp=t)

    overhead = ledger.checkpoint_overhead_fraction("job-1")
    assert isinstance(overhead, Decimal)
    assert overhead == Decimal("20") / Decimal("300")


def test_throughput_computed_from_size_and_duration():
    ledger = CheckpointLedger()
    ledger.save_checkpoint("job-1", step=0, size_gb=Decimal("40"), duration_s=Decimal("20"))
    telemetry = ledger.telemetry.latest_checkpoint_telemetry("job-1")
    assert telemetry.throughput_gbps == Decimal("40") * 8 / Decimal("20")


def test_zero_duration_checkpoint_has_zero_throughput_not_a_crash():
    ledger = CheckpointLedger()
    ledger.save_checkpoint("job-1", step=0, size_gb=Decimal("40"), duration_s=Decimal("0"))
    telemetry = ledger.telemetry.latest_checkpoint_telemetry("job-1")
    assert telemetry.throughput_gbps == Decimal(0)


def test_optimal_checkpoint_interval_matches_youngs_formula():
    mtbf_s = 3600.0 * 8
    checkpoint_cost_s = 20.0
    result = optimal_checkpoint_interval_s(mtbf_s=mtbf_s, checkpoint_cost_s=checkpoint_cost_s)
    assert result == pytest.approx((2 * checkpoint_cost_s * mtbf_s) ** 0.5)
