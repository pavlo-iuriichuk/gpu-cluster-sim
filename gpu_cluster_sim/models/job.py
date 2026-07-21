from enum import Enum
from typing import NamedTuple, Optional, Tuple

from .gpu import GPUModel


class LocalityConstraint(str, Enum):
    SAME_LEAF = "same_leaf"
    SAME_SPINE = "same_spine"
    BEST_EFFORT = "best_effort"


class CollectivePattern(str, Enum):
    RING = "ring"
    TREE = "tree"
    ALL_TO_ALL = "all2all"


class QoSClass(str, Enum):
    PRODUCTION = "production"
    BATCH = "batch"
    BEST_EFFORT = "best_effort"


class ShapeHint(NamedTuple):
    """e.g. (nodes=8, gpus_per_node=8) for a 64-GPU job — distinct from just
    requesting 64 loose GPUs, since it tells the placer the preferred shape.
    """

    nodes: int
    gpus_per_node: int


class JobRequest(NamedTuple):
    job_id: str
    gpu_count: int
    shape_hint: ShapeHint
    gpu_model: GPUModel
    memory_per_gpu_gb: int
    cpu_per_gpu: int
    host_ram_gb: int
    local_nvme_gb: int
    locality_constraint: LocalityConstraint
    collective_pattern: CollectivePattern
    qos_class: QoSClass
    max_queue_time_s: Optional[float] = None
    preemptible: bool = False
    min_gpu_count: Optional[int] = None  # elastic jobs accept [min_gpu_count, gpu_count]
    affinity: Tuple[str, ...] = ()
    anti_affinity: Tuple[str, ...] = ()
