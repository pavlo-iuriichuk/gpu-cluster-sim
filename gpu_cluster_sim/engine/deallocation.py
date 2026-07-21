"""Symmetric counterpart to `engine.allocation`: named ways to release a
job's gang, each tagging the resulting `DeallocationRecord` with *why* —
the ledger itself has no way to know that, only the caller (scheduler,
operator, or job lifecycle event) does. All four are thin wrappers around
`AllocationLedger.release`.
"""

from typing import Optional, Tuple

from ..models.deallocation import DeallocationReason, DeallocationRecord
from .allocation import AllocationLedger


def release_on_demand(
    ledger: AllocationLedger, job_id: str, *, timestamp: Optional[float] = None
) -> DeallocationRecord:
    """Free the gang outside of the job's own lifecycle — e.g. an operator
    draining a node group.
    """
    return ledger.release(job_id, DeallocationReason.MANUAL, timestamp=timestamp)


def complete_job(ledger: AllocationLedger, job_id: str, *, timestamp: Optional[float] = None) -> DeallocationRecord:
    """The job finished normally."""
    return ledger.release(job_id, DeallocationReason.COMPLETED, timestamp=timestamp)


def cancel_job(ledger: AllocationLedger, job_id: str, *, timestamp: Optional[float] = None) -> DeallocationRecord:
    """The job was withdrawn before it finished."""
    return ledger.release(job_id, DeallocationReason.CANCELLED, timestamp=timestamp)


def preempt_job(
    ledger: AllocationLedger,
    job_id: str,
    *,
    grace_period_s: float = 0.0,
    timestamp: Optional[float] = None,
) -> DeallocationRecord:
    """The job was evicted to make room for higher-priority work.
    `grace_period_s` is how long it got to reach a checkpoint boundary
    before teardown (see "Preemption" in
    docs/topology-aware-gang-scheduling.md) — carried on the record so a
    billing layer can exclude it from what the tenant is charged for.
    """
    return ledger.release(
        job_id, DeallocationReason.PREEMPTED, grace_period_s=grace_period_s, timestamp=timestamp
    )


def history_by_reason(ledger: AllocationLedger, reason: DeallocationReason) -> Tuple[DeallocationRecord, ...]:
    return ledger.history(reason=reason)
