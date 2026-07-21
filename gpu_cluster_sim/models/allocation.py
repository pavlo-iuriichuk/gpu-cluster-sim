from typing import NamedTuple, Tuple

from .job import QoSClass


class Allocation(NamedTuple):
    """A granted, atomic gang allocation — every running job holds exactly
    one of these, or none at all; there is no partial-allocation state (see
    "Why gang scheduling is non-negotiable" in
    docs/topology-aware-gang-scheduling.md). `gpu_ids` is rank-ordered, not
    just a set: rank i is the i-th collective rank, grouped by node and
    ordered by rail index within each node, so a ring all-reduce stays
    NVLink-local as long as possible before crossing the fabric. `qos_class`
    is carried over from the `JobRequest` so preemption risk can later be
    estimated per QoS class from release history.
    """

    job_id: str
    gpu_ids: Tuple[str, ...]
    node_ids: Tuple[str, ...]
    allocated_at: float
    qos_class: QoSClass


class LedgerSnapshot(NamedTuple):
    """An immutable point-in-time copy of an `AllocationLedger`, safe to
    hand to telemetry/reporting code without it changing underfoot.
    """

    gpu_to_job: Tuple[Tuple[str, str], ...]
    allocations: Tuple[Allocation, ...]
