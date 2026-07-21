from enum import Enum
from typing import NamedTuple


class GPUModel(str, Enum):
    H100 = "H100"
    A100 = "A100"
    A6000 = "A6000"


class GPU(NamedTuple):
    """A single accelerator. `rail_index` is its position within the node
    (e.g. 0..7) — GPUs sharing a rail_index across nodes hang off the same
    rail switch, which is what makes rail-aligned placement possible.
    """

    gpu_id: str
    node_id: str
    rail_index: int
    model: GPUModel
    memory_gb: int
    nvlink_domain: str
