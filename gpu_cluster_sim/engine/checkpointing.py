"""Per-job checkpoint ledger: save a checkpoint, load the latest (or a
specific) one back, and derive how often a job is actually checkpointing.
Checkpoint write cost is recorded into a `TelemetryStore` (`self.telemetry`)
as `CheckpointTelemetry`, which is what `mean_checkpoint_interval_s` and the
overhead/optimal-interval math below are built on.
"""

import math
import time
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from ..models.checkpoint import Checkpoint
from ..models.telemetry import CheckpointTelemetry
from .telemetry import TelemetryStore


class CheckpointLedger:
    def __init__(self) -> None:
        self._checkpoints: Dict[str, List[Checkpoint]] = {}
        self.telemetry = TelemetryStore()

    def save_checkpoint(
        self,
        job_id: str,
        step: int,
        size_gb: Decimal,
        duration_s: Decimal,
        *,
        storage_uri: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> Checkpoint:
        """Record a new checkpoint for `job_id`. `duration_s` is how long
        the write took — needed to estimate checkpointing overhead and the
        optimal interval, not just to log the event.
        """
        created_at = timestamp if timestamp is not None else time.time()
        existing = self._checkpoints.setdefault(job_id, [])
        checkpoint = Checkpoint(
            job_id=job_id,
            checkpoint_id=f"{job_id}-ckpt{len(existing)}",
            step=step,
            created_at=created_at,
            size_gb=size_gb,
            storage_uri=storage_uri or f"checkpoints/{job_id}/step-{step}",
        )
        existing.append(checkpoint)

        throughput_gbps = (size_gb * 8 / duration_s) if duration_s > 0 else Decimal(0)
        self.telemetry.record_checkpoint(
            job_id,
            CheckpointTelemetry(duration_s=duration_s, size_gb=size_gb, throughput_gbps=throughput_gbps),
            timestamp=created_at,
        )
        return checkpoint

    def load_checkpoint(self, job_id: str, checkpoint_id: Optional[str] = None) -> Checkpoint:
        """Return the checkpoint a resumed job would load from: the latest
        one by default, or a specific `checkpoint_id`. Raises `KeyError` if
        the job has none (or that particular id doesn't exist) — there is
        nothing sensible to resume from otherwise.
        """
        checkpoints = self._checkpoints.get(job_id, [])
        if not checkpoints:
            raise KeyError(f"no checkpoints stored for job {job_id!r}")
        if checkpoint_id is None:
            return checkpoints[-1]
        for checkpoint in checkpoints:
            if checkpoint.checkpoint_id == checkpoint_id:
                return checkpoint
        raise KeyError(f"no checkpoint {checkpoint_id!r} for job {job_id!r}")

    def latest_checkpoint(self, job_id: str) -> Optional[Checkpoint]:
        """Same as `load_checkpoint`, but `None` instead of raising —
        for callers (like risk estimation) that need to handle "never
        checkpointed" as a normal case, not an error.
        """
        checkpoints = self._checkpoints.get(job_id)
        return checkpoints[-1] if checkpoints else None

    def checkpoints_for_job(self, job_id: str) -> Tuple[Checkpoint, ...]:
        return tuple(self._checkpoints.get(job_id, ()))

    def mean_checkpoint_interval_s(self, job_id: str) -> Optional[float]:
        """Average wall-clock time between consecutive checkpoints. `None`
        if fewer than two checkpoints exist yet to measure an interval from.
        """
        checkpoints = self.checkpoints_for_job(job_id)
        if len(checkpoints) < 2:
            return None
        deltas = [b.created_at - a.created_at for a, b in zip(checkpoints, checkpoints[1:])]
        return sum(deltas) / len(deltas)

    def checkpoint_frequency_per_hour(self, job_id: str) -> Optional[float]:
        interval_s = self.mean_checkpoint_interval_s(job_id)
        if not interval_s:
            return None
        return 3600.0 / interval_s

    def checkpoint_overhead_fraction(self, job_id: str) -> Optional[Decimal]:
        """Fraction of wall-clock time spent writing checkpoints rather
        than training: mean(checkpoint duration) / mean(interval between
        checkpoints). `None` if there isn't enough history for either half.
        """
        interval_s = self.mean_checkpoint_interval_s(job_id)
        if not interval_s:
            return None
        durations = [telemetry.duration_s for _, telemetry in self.telemetry.checkpoint_telemetry_history(job_id)]
        if not durations:
            return None
        mean_duration_s = sum(durations) / len(durations)
        # interval_s comes from Checkpoint.created_at (float, like every
        # other timestamp in the codebase) — convert via str() to avoid
        # picking up float's binary-fraction noise in the Decimal result.
        return mean_duration_s / Decimal(str(interval_s))


def optimal_checkpoint_interval_s(mtbf_s: float, checkpoint_cost_s: float) -> float:
    """Young's approximation for the checkpoint interval that minimizes
    expected wasted compute: sqrt(2 x checkpoint_cost x MTBF). A standard
    first-order result in HPC fault tolerance (see "Failure handling" in
    docs/topology-aware-gang-scheduling.md); it ignores restart cost
    (Daly's refinement adds that term) for simplicity.
    """
    return math.sqrt(2 * checkpoint_cost_s * mtbf_s)
