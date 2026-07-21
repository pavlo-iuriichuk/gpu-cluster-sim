from decimal import Decimal

from gpu_cluster_sim.engine import TelemetryStore
from gpu_cluster_sim.models import CheckpointTelemetry, LinkTelemetry, NodeHealth, NodeTelemetry


def test_node_telemetry_latest_and_history():
    store = TelemetryStore()
    assert store.latest_node("node-0") is None

    reading_1 = NodeTelemetry(
        gpu_util_pct=Decimal("50.0"),
        hbm_util_pct=Decimal("40.0"),
        ecc_errors=0,
        temperature_c=Decimal("60.0"),
        power_draw_w=Decimal("400.0"),
    )
    reading_2 = NodeTelemetry(
        gpu_util_pct=Decimal("90.0"),
        hbm_util_pct=Decimal("70.0"),
        ecc_errors=1,
        temperature_c=Decimal("65.0"),
        power_draw_w=Decimal("420.0"),
        health=NodeHealth.DEGRADED,
    )
    store.record_node("node-0", reading_1, timestamp=1.0)
    store.record_node("node-0", reading_2, timestamp=2.0)

    assert store.latest_node("node-0") == reading_2
    assert [t for t, _ in store.node_history("node-0")] == [1.0, 2.0]


def test_link_telemetry_keyed_by_edge_and_parallel_key():
    store = TelemetryStore()
    store.record_link("a", "b", 0, LinkTelemetry(utilization=Decimal("0.5")))
    store.record_link("a", "b", 1, LinkTelemetry(utilization=Decimal("0.9")))

    assert store.latest_link("a", "b", 0).utilization == Decimal("0.5")
    assert store.latest_link("a", "b", 1).utilization == Decimal("0.9")
    assert store.latest_link("a", "b", 2) is None


def test_aggregate_node_metric_returns_decimal_for_measured_fields():
    store = TelemetryStore()
    store.record_node(
        "node-0",
        NodeTelemetry(
            gpu_util_pct=Decimal("80.0"), hbm_util_pct=Decimal("1"), ecc_errors=0,
            temperature_c=Decimal("1"), power_draw_w=Decimal("1"),
        ),
    )
    store.record_node(
        "node-1",
        NodeTelemetry(
            gpu_util_pct=Decimal("40.0"), hbm_util_pct=Decimal("1"), ecc_errors=2,
            temperature_c=Decimal("1"), power_draw_w=Decimal("1"),
        ),
    )

    avg_util = store.aggregate_node_metric(["node-0", "node-1"], "gpu_util_pct")
    assert avg_util == Decimal("60.0")
    assert isinstance(avg_util, Decimal)

    max_ecc = store.aggregate_node_metric(["node-0", "node-1"], "ecc_errors", agg=max)
    assert max_ecc == 2
    assert isinstance(max_ecc, int)


def test_aggregate_ignores_missing_readings_and_returns_none_if_all_missing():
    store = TelemetryStore()
    store.record_node(
        "node-0",
        NodeTelemetry(
            gpu_util_pct=Decimal("10"), hbm_util_pct=Decimal("1"), ecc_errors=0,
            temperature_c=Decimal("1"), power_draw_w=Decimal("1"),
        ),
    )
    assert store.aggregate_node_metric(["node-0", "node-missing"], "gpu_util_pct") == Decimal("10")
    assert store.aggregate_node_metric(["node-missing"], "gpu_util_pct") is None


def test_link_history():
    store = TelemetryStore()
    store.record_link("a", "b", 0, LinkTelemetry(utilization=Decimal("0.1")), timestamp=1.0)
    store.record_link("a", "b", 0, LinkTelemetry(utilization=Decimal("0.2")), timestamp=2.0)
    assert [t for t, _ in store.link_history("a", "b", 0)] == [1.0, 2.0]
    assert store.link_history("a", "b", 99) == []


def test_aggregate_link_metric():
    store = TelemetryStore()
    store.record_link("a", "b", 0, LinkTelemetry(utilization=Decimal("0.2")))
    store.record_link("c", "d", 0, LinkTelemetry(utilization=Decimal("0.8")))
    avg = store.aggregate_link_metric([("a", "b", 0), ("c", "d", 0)], "utilization")
    assert avg == Decimal("0.5")


def test_checkpoint_telemetry_recording_and_aggregation():
    store = TelemetryStore()
    store.record_checkpoint(
        "job-1", CheckpointTelemetry(duration_s=Decimal("10"), size_gb=Decimal("40"), throughput_gbps=Decimal("32"))
    )
    store.record_checkpoint(
        "job-2", CheckpointTelemetry(duration_s=Decimal("20"), size_gb=Decimal("40"), throughput_gbps=Decimal("16"))
    )

    assert store.latest_checkpoint_telemetry("job-1").duration_s == Decimal("10")
    assert len(store.checkpoint_telemetry_history("job-1")) == 1
    avg_duration = store.aggregate_checkpoint_metric(["job-1", "job-2"], "duration_s")
    assert avg_duration == Decimal("15")
