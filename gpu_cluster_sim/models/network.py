from enum import Enum
from typing import NamedTuple


class LinkType(str, Enum):
    NVLINK = "nvlink"  # GPU <-> GPU, same node (NVSwitch)
    PCIE = "pcie"  # GPU <-> its own node, i.e. the GPU's path out to the node's NIC/uplink
    RAIL = "rail"  # GPU <-> rail switch, cross-node, same rail index
    LEAF_UPLINK = "leaf_uplink"  # node <-> leaf switch
    SPINE_UPLINK = "spine_uplink"  # leaf <-> spine
    INTER_POD = "inter_pod"  # spine <-> spine, across pods


class LeafSwitch(NamedTuple):
    leaf_id: str
    pod_id: str
    rack_id: str
    downlink_ports: int
    uplink_ports: int
    port_bandwidth_gbps: float
    oversubscription_ratio: float = 1.0


class SpineSwitch(NamedTuple):
    spine_id: str
    pod_id: str
    port_count: int
    port_bandwidth_gbps: float


class RailSwitch(NamedTuple):
    """Aggregates one GPU index (rail_index) across every node in a pod, so
    an all-reduce can stay rail-local instead of crossing the leaf/spine
    fabric.
    """

    rail_id: str
    rail_index: int
    pod_id: str


class LinkMetrics(NamedTuple):
    """Static characteristics of a link. Time-varying state (utilization,
    queue depth, errors) is intentionally not stored here — NamedTuples are
    immutable, and telemetry changes far more often than topology does. See
    `engine.telemetry.TelemetryStore` for that.
    """

    link_type: LinkType
    bandwidth_gbps: float
    latency_us: float
    oversubscription_ratio: float = 1.0
