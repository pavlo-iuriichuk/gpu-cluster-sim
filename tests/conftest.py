import pytest

from gpu_cluster_sim.engine import ClusterTopology
from gpu_cluster_sim.models import (
    CollectivePattern,
    GPUModel,
    JobRequest,
    LocalityConstraint,
    QoSClass,
    ShapeHint,
)
from gpu_cluster_sim.samples import RackGroup, populate_pod


@pytest.fixture
def small_topology() -> ClusterTopology:
    """2 spines x 2 leaves x 2 nodes x 4 H100 GPUs = 16 GPUs, full leaf<->spine
    bipartite fabric, NVLink mesh + PCIe uplink per node. Small enough to be a
    fast, deterministic fixture; built with the same `populate_pod` primitive
    used for the sample topologies under data/.
    """
    topology = ClusterTopology()
    populate_pod(
        topology,
        "pod-0",
        spine_count=2,
        rack_groups=[
            RackGroup(
                name="rack",
                leaf_count=2,
                nodes_per_leaf=2,
                gpus_per_node=4,
                gpu_model=GPUModel.H100,
                gpu_memory_gb=80,
            )
        ],
    )
    return topology


@pytest.fixture
def make_job():
    """Factory for a `JobRequest` with sensible defaults, so each test only
    overrides what it actually cares about.
    """

    def _make_job(
        job_id: str,
        nodes: int = 1,
        gpus_per_node: int = 4,
        *,
        gpu_model: GPUModel = GPUModel.H100,
        memory_per_gpu_gb: int = 80,
        cpu_per_gpu: int = 8,
        host_ram_gb: int = 256,
        local_nvme_gb: int = 1000,
        locality_constraint: LocalityConstraint = LocalityConstraint.BEST_EFFORT,
        collective_pattern: CollectivePattern = CollectivePattern.RING,
        qos_class: QoSClass = QoSClass.BEST_EFFORT,
    ) -> JobRequest:
        return JobRequest(
            job_id=job_id,
            gpu_count=nodes * gpus_per_node,
            shape_hint=ShapeHint(nodes=nodes, gpus_per_node=gpus_per_node),
            gpu_model=gpu_model,
            memory_per_gpu_gb=memory_per_gpu_gb,
            cpu_per_gpu=cpu_per_gpu,
            host_ram_gb=host_ram_gb,
            local_nvme_gb=local_nvme_gb,
            locality_constraint=locality_constraint,
            collective_pattern=collective_pattern,
            qos_class=qos_class,
        )

    return _make_job
