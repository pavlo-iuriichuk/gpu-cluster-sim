from enum import Enum
from typing import NamedTuple, Tuple


class DeallocationReason(str, Enum):
    COMPLETED = "completed"  # job finished normally
    CANCELLED = "cancelled"  # withdrawn before it finished
    PREEMPTED = "preempted"  # evicted to make room for higher-priority work
    MANUAL = "manual"  # operator/on-demand release outside the job's own lifecycle


class DeallocationRecord(NamedTuple):
    """What happened when a job's gang was released, and why. `grace_period_s`
    is carried through for preemptions specifically — see "Preemption" in
    docs/topology-aware-gang-scheduling.md — so a billing layer can exclude
    it from what the tenant is charged for.
    """

    job_id: str
    gpu_ids: Tuple[str, ...]
    node_ids: Tuple[str, ...]
    reason: DeallocationReason
    allocated_at: float
    released_at: float
    held_duration_s: float
    grace_period_s: float = 0.0
