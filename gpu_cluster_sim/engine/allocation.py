"""Gang allocation: given a `JobRequest`, either reserve every GPU it needs
right now or reserve nothing — never a partial set (see "Why gang
scheduling is non-negotiable" in docs/topology-aware-gang-scheduling.md).

`AllocationLedger` is the runtime bookkeeping (which GPU belongs to which
job), kept separate from `ClusterTopology` the same way `TelemetryStore` is:
topology is static structure, the ledger is what changes as jobs come and
go. Placement here is a straightforward feasibility search — filter free
GPUs by model/memory/node health, group by locality constraint, prefer
fewer distinct leaves — not the full scored bin-packing optimizer from
"Placement scoring" in the design doc (fragmentation delta, rail alignment,
risk, reuse bonus); this answers "can the gang fit, and where," not "what
is the best possible placement across the whole cluster."
"""

import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from ..models.allocation import Allocation, LedgerSnapshot
from ..models.deallocation import DeallocationReason, DeallocationRecord
from ..models.gpu import GPU
from ..models.job import JobRequest, LocalityConstraint
from ..models.network import LinkType
from ..models.node import NodeHealth
from .topology import ClusterTopology


class AllocationLedger:
    def __init__(self) -> None:
        self._gpu_to_job: Dict[str, str] = {}
        self._allocations: Dict[str, Allocation] = {}
        self._history: List[DeallocationRecord] = []

    def is_free(self, gpu_id: str) -> bool:
        return gpu_id not in self._gpu_to_job

    def job_for_gpu(self, gpu_id: str) -> Optional[str]:
        return self._gpu_to_job.get(gpu_id)

    def allocation_for_job(self, job_id: str) -> Optional[Allocation]:
        return self._allocations.get(job_id)

    def free_gpu_count(self, topology: ClusterTopology) -> int:
        return sum(1 for gpu_id in topology.entities_of_kind("gpu") if self.is_free(gpu_id))

    def try_allocate(
        self, topology: ClusterTopology, job: JobRequest, *, timestamp: Optional[float] = None
    ) -> Optional[Allocation]:
        """Attempt to reserve all of `job`'s GPUs at once. Returns the new
        `Allocation` on success, or `None` if the gang doesn't fit right
        now — the ledger is left completely unchanged in that case.
        """
        if job.job_id in self._allocations:
            raise ValueError(f"job {job.job_id!r} already has an allocation")

        gpu_ids = _select_gpus(topology, self, job)
        if gpu_ids is None:
            return None

        node_ids = tuple(dict.fromkeys(topology.entity(gpu_id).node_id for gpu_id in gpu_ids))
        allocation = Allocation(
            job_id=job.job_id,
            gpu_ids=gpu_ids,
            node_ids=node_ids,
            allocated_at=timestamp if timestamp is not None else time.time(),
            qos_class=job.qos_class,
        )
        for gpu_id in gpu_ids:
            self._gpu_to_job[gpu_id] = job.job_id
        self._allocations[job.job_id] = allocation
        return allocation

    def release(
        self,
        job_id: str,
        reason: DeallocationReason = DeallocationReason.MANUAL,
        *,
        grace_period_s: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> DeallocationRecord:
        """Free every GPU `job_id` holds. `reason` and (for preemption)
        `grace_period_s` are the caller's to supply — the ledger has no way
        to know *why* a job is being torn down; see `engine.deallocation`
        for named wrappers (`complete_job`, `cancel_job`, `preempt_job`,
        `release_on_demand`).
        """
        allocation = self._allocations.pop(job_id, None)
        if allocation is None:
            raise KeyError(f"no allocation held for job {job_id!r}")
        for gpu_id in allocation.gpu_ids:
            del self._gpu_to_job[gpu_id]

        released_at = timestamp if timestamp is not None else time.time()
        record = DeallocationRecord(
            job_id=allocation.job_id,
            gpu_ids=allocation.gpu_ids,
            node_ids=allocation.node_ids,
            reason=reason,
            qos_class=allocation.qos_class,
            allocated_at=allocation.allocated_at,
            released_at=released_at,
            held_duration_s=released_at - allocation.allocated_at,
            grace_period_s=grace_period_s,
        )
        self._history.append(record)
        return record

    def history(self, *, reason: Optional[DeallocationReason] = None) -> Tuple[DeallocationRecord, ...]:
        if reason is None:
            return tuple(self._history)
        return tuple(r for r in self._history if r.reason is reason)

    def snapshot(self) -> LedgerSnapshot:
        return LedgerSnapshot(
            gpu_to_job=tuple(sorted(self._gpu_to_job.items())),
            allocations=tuple(self._allocations.values()),
        )


def _select_gpus(topology: ClusterTopology, ledger: AllocationLedger, job: JobRequest) -> Optional[Tuple[str, ...]]:
    nodes_needed, gpus_per_node = job.shape_hint
    if nodes_needed * gpus_per_node != job.gpu_count:
        raise ValueError(
            f"job {job.job_id!r}: shape_hint {job.shape_hint} does not multiply out to gpu_count {job.gpu_count}"
        )

    free_by_node = _free_gpus_by_node(topology, ledger, job)
    candidate_nodes = [node_id for node_id, gpus in free_by_node.items() if len(gpus) >= gpus_per_node]

    chosen_nodes = _choose_nodes(topology, candidate_nodes, nodes_needed, job.locality_constraint)
    if chosen_nodes is None:
        return None

    gpu_ids: List[str] = []
    for node_id in chosen_nodes:
        gpu_ids.extend(gpu.gpu_id for gpu in free_by_node[node_id][:gpus_per_node])
    return tuple(gpu_ids)


def _free_gpus_by_node(
    topology: ClusterTopology, ledger: AllocationLedger, job: JobRequest
) -> Dict[str, List[GPU]]:
    """Free GPUs matching the job's model/memory need, on nodes healthy
    enough and roomy enough for the job, grouped by node and ordered by
    rail index (so slicing the first `gpus_per_node` gives a rail-aligned,
    rank-ordered subset).
    """
    by_node: Dict[str, List[GPU]] = defaultdict(list)
    for node_id in topology.entities_of_kind("node"):
        if not _node_meets_job(topology, node_id, job):
            continue
        node = topology.entity(node_id)
        for gpu_id in node.gpu_ids:
            gpu = topology.entity(gpu_id)
            if gpu.model != job.gpu_model:
                continue
            if gpu.memory_gb < job.memory_per_gpu_gb:
                continue
            if not ledger.is_free(gpu_id):
                continue
            by_node[node_id].append(gpu)
    for gpus in by_node.values():
        gpus.sort(key=lambda gpu: gpu.rail_index)
    return by_node


def _node_meets_job(topology: ClusterTopology, node_id: str, job: JobRequest) -> bool:
    node = topology.entity(node_id)
    if node.health is not NodeHealth.HEALTHY:
        return False
    if node.host_ram_gb < job.host_ram_gb:
        return False
    if node.local_nvme_gb < job.local_nvme_gb:
        return False
    if node.cpu_count < job.cpu_per_gpu * job.shape_hint.gpus_per_node:
        return False
    return True


def _choose_nodes(
    topology: ClusterTopology,
    candidate_nodes: List[str],
    nodes_needed: int,
    locality: LocalityConstraint,
) -> Optional[List[str]]:
    if locality is LocalityConstraint.SAME_LEAF:
        by_leaf: Dict[str, List[str]] = defaultdict(list)
        for node_id in candidate_nodes:
            by_leaf[topology.entity(node_id).leaf_id].append(node_id)
        for node_ids in by_leaf.values():
            if len(node_ids) >= nodes_needed:
                return sorted(node_ids)[:nodes_needed]
        return None

    if locality is LocalityConstraint.SAME_SPINE:
        leaf_to_spines = _leaf_to_spines(topology)
        by_spine: Dict[str, Set[str]] = defaultdict(set)
        for node_id in candidate_nodes:
            leaf_id = topology.entity(node_id).leaf_id
            for spine_id in leaf_to_spines.get(leaf_id, ()):
                by_spine[spine_id].add(node_id)
        for node_ids in by_spine.values():
            if len(node_ids) >= nodes_needed:
                return sorted(node_ids)[:nodes_needed]
        return None

    if len(candidate_nodes) >= nodes_needed:
        return sorted(candidate_nodes)[:nodes_needed]
    return None


def _leaf_to_spines(topology: ClusterTopology) -> Dict[str, Set[str]]:
    mapping: Dict[str, Set[str]] = defaultdict(set)
    for u, v, _ in topology.edges_of_type(LinkType.SPINE_UPLINK):
        leaf_id, spine_id = (u, v) if topology.kind_of(u) == "leaf" else (v, u)
        mapping[leaf_id].add(spine_id)
    return mapping
