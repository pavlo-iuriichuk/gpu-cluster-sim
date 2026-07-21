from enum import Enum
from typing import NamedTuple, Tuple


class NodeHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DRAINING = "draining"
    DOWN = "down"


class Node(NamedTuple):
    """A host with one or more GPUs. `leaf_id` fixes which leaf switch it
    hangs off, which in turn determines same-leaf vs. same-spine placement
    cost for anything scheduled on this node.
    """

    node_id: str
    rack_id: str
    leaf_id: str
    gpu_ids: Tuple[str, ...]
    cpu_count: int
    host_ram_gb: int
    local_nvme_gb: int
    health: NodeHealth = NodeHealth.HEALTHY
