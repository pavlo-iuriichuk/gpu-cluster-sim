from decimal import Decimal
from typing import NamedTuple

from .node import NodeHealth


class LinkTelemetry(NamedTuple):
    """A point-in-time reading for a link. `utilization` is the fraction
    (0..1) of the link's `LinkMetrics.bandwidth_gbps` currently in use.
    Measured readings are `Decimal`, not `float` — telemetry values get
    compared, aggregated, and thresholded, and binary float error has no
    place in "is utilization above 0.8."
    """

    utilization: Decimal
    queue_depth: int = 0
    error_count: int = 0
    flapped: bool = False


class NodeTelemetry(NamedTuple):
    """A point-in-time reading for a node's GPUs/host. Straggler and health
    signals live here so the scheduler's feedback loop can act on silent
    degradation, not just hard failure.
    """

    gpu_util_pct: Decimal
    hbm_util_pct: Decimal
    ecc_errors: int
    temperature_c: Decimal
    power_draw_w: Decimal
    health: NodeHealth = NodeHealth.HEALTHY


class CheckpointTelemetry(NamedTuple):
    """A point-in-time reading of one checkpoint write's cost — feeds both
    the checkpointing-overhead calculation (time spent checkpointing vs.
    training) and the optimal-interval estimate in `engine.checkpointing`.
    """

    duration_s: Decimal
    size_gb: Decimal
    throughput_gbps: Decimal
