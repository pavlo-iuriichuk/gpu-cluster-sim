from typing import NamedTuple


class Checkpoint(NamedTuple):
    """A saved checkpoint for one job. `storage_uri` models where the
    checkpoint's bytes would live in a real system (e.g. object storage) —
    this simulator tracks checkpoint metadata and cadence, not actual
    tensor payloads.
    """

    job_id: str
    checkpoint_id: str
    step: int
    created_at: float
    size_gb: float
    storage_uri: str
