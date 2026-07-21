from typing import NamedTuple

from .node import NodeHealth


class LinkTelemetry(NamedTuple):
    """A point-in-time reading for a link. `utilization` is the fraction
    (0..1) of the link's `LinkMetrics.bandwidth_gbps` currently in use.
    """

    utilization: float
    queue_depth: int = 0
    error_count: int = 0
    flapped: bool = False


class NodeTelemetry(NamedTuple):
    """A point-in-time reading for a node's GPUs/host. Straggler and health
    signals live here so the scheduler's feedback loop can act on silent
    degradation, not just hard failure.
    """

    gpu_util_pct: float
    hbm_util_pct: float
    ecc_errors: int
    temperature_c: float
    power_draw_w: float
    health: NodeHealth = NodeHealth.HEALTHY


class CheckpointTelemetry(NamedTuple):
    """A point-in-time reading of one checkpoint write's cost — feeds both
    the checkpointing-overhead calculation (time spent checkpointing vs.
    training) and the optimal-interval estimate in `engine.checkpointing`.
    """

    duration_s: float
    size_gb: float
    throughput_gbps: float
