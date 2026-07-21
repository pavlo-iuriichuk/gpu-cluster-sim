from decimal import Decimal
from typing import NamedTuple


class Checkpoint(NamedTuple):
    """A saved checkpoint for one job. `storage_uri` models where the
    checkpoint's bytes would live in a real system (e.g. object storage) —
    this simulator tracks checkpoint metadata and cadence, not actual
    tensor payloads. `size_gb` is `Decimal`, matching the same measured
    value as recorded in `CheckpointTelemetry`.
    """

    job_id: str
    checkpoint_id: str
    step: int
    created_at: float
    size_gb: Decimal
    storage_uri: str
