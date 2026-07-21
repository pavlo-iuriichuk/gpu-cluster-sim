"""Preemption-probability and data-loss-risk estimation, built from
`AllocationLedger` release history and `CheckpointLedger` state — see
"Preemption" in docs/topology-aware-gang-scheduling.md. This estimates risk
from what has already happened (empirical release history); it does not
predict future node failures.
"""

from typing import Optional

from ..models.deallocation import DeallocationReason
from ..models.job import QoSClass
from .allocation import AllocationLedger
from .checkpointing import CheckpointLedger


def preemption_probability(ledger: AllocationLedger, qos_class: Optional[QoSClass] = None) -> Optional[float]:
    """Empirical P(a released job's release reason was PREEMPTED),
    optionally conditioned on QoS class. `None` if there's no release
    history yet to estimate from.
    """
    records = ledger.history()
    if qos_class is not None:
        records = tuple(r for r in records if r.qos_class == qos_class)
    if not records:
        return None
    preempted = sum(1 for r in records if r.reason is DeallocationReason.PREEMPTED)
    return preempted / len(records)


def unprotected_work_s(
    checkpoint_ledger: CheckpointLedger, job_id: str, allocated_at: float, now: float
) -> float:
    """How much work would be lost if `job_id` were preempted right now
    with no checkpoint-aware grace period: time since its last checkpoint,
    or its entire runtime so far if it has never checkpointed.
    """
    latest = checkpoint_ledger.latest_checkpoint(job_id)
    baseline = latest.created_at if latest is not None else allocated_at
    return max(0.0, now - baseline)


def expected_data_loss_s(
    allocation_ledger: AllocationLedger,
    checkpoint_ledger: CheckpointLedger,
    job_id: str,
    allocated_at: float,
    now: float,
    *,
    qos_class: Optional[QoSClass] = None,
) -> Optional[float]:
    """Expected lost work, in seconds: P(preempted) x (time since last
    checkpoint). `None` if there isn't enough release history yet to
    estimate P(preempted) — distinct from a confident estimate of zero.
    """
    p = preemption_probability(allocation_ledger, qos_class)
    if p is None:
        return None
    return p * unprotected_work_s(checkpoint_ledger, job_id, allocated_at, now)
