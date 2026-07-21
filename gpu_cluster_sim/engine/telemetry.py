"""Telemetry storage and querying, kept separate from the topology graph.

`ClusterTopology` holds static structure (which links exist, their nominal
bandwidth/latency). `TelemetryStore` holds the time-varying readings for
nodes and links — recorded as a history so aggregations (mean utilization
across a leaf's uplinks, current health mix across a rack, ...) can be
computed over either the latest snapshot or a time window.
"""

import time
from collections import defaultdict
from decimal import Decimal
from statistics import mean
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from ..models.telemetry import CheckpointTelemetry, LinkTelemetry, NodeTelemetry

EdgeKey = Tuple[str, str, int]
NodeReading = Tuple[float, NodeTelemetry]
LinkReading = Tuple[float, LinkTelemetry]
CheckpointReading = Tuple[float, CheckpointTelemetry]
# Telemetry fields are either a measured Decimal or a discrete int count
# (ecc_errors, queue_depth, error_count) — aggregation works over either.
Numeric = Union[int, Decimal]


class TelemetryStore:
    def __init__(self) -> None:
        self._node_history: Dict[str, List[NodeReading]] = defaultdict(list)
        self._link_history: Dict[EdgeKey, List[LinkReading]] = defaultdict(list)
        self._checkpoint_history: Dict[str, List[CheckpointReading]] = defaultdict(list)

    def record_node(
        self, node_id: str, telemetry: NodeTelemetry, timestamp: Optional[float] = None
    ) -> None:
        self._node_history[node_id].append((timestamp if timestamp is not None else time.time(), telemetry))

    def record_link(
        self,
        u_id: str,
        v_id: str,
        key: int,
        telemetry: LinkTelemetry,
        timestamp: Optional[float] = None,
    ) -> None:
        edge_key = (u_id, v_id, key)
        self._link_history[edge_key].append((timestamp if timestamp is not None else time.time(), telemetry))

    def record_checkpoint(
        self, job_id: str, telemetry: CheckpointTelemetry, timestamp: Optional[float] = None
    ) -> None:
        self._checkpoint_history[job_id].append((timestamp if timestamp is not None else time.time(), telemetry))

    def latest_node(self, node_id: str) -> Optional[NodeTelemetry]:
        history = self._node_history.get(node_id)
        return history[-1][1] if history else None

    def latest_link(self, u_id: str, v_id: str, key: int = 0) -> Optional[LinkTelemetry]:
        history = self._link_history.get((u_id, v_id, key))
        return history[-1][1] if history else None

    def node_history(self, node_id: str) -> List[NodeReading]:
        return list(self._node_history.get(node_id, []))

    def link_history(self, u_id: str, v_id: str, key: int = 0) -> List[LinkReading]:
        return list(self._link_history.get((u_id, v_id, key), []))

    def latest_checkpoint_telemetry(self, job_id: str) -> Optional[CheckpointTelemetry]:
        history = self._checkpoint_history.get(job_id)
        return history[-1][1] if history else None

    def checkpoint_telemetry_history(self, job_id: str) -> List[CheckpointReading]:
        return list(self._checkpoint_history.get(job_id, []))

    def aggregate_node_metric(
        self,
        node_ids: Iterable[str],
        field: str,
        agg: Callable[[Sequence[Numeric]], Numeric] = mean,
    ) -> Optional[Numeric]:
        """e.g. aggregate_node_metric(nodes_in_rack, "temperature_c", max)"""
        values = [
            getattr(reading, field)
            for reading in (self.latest_node(nid) for nid in node_ids)
            if reading is not None
        ]
        return agg(values) if values else None

    def aggregate_link_metric(
        self,
        edge_keys: Iterable[EdgeKey],
        field: str,
        agg: Callable[[Sequence[Numeric]], Numeric] = mean,
    ) -> Optional[Numeric]:
        """e.g. aggregate_link_metric(topology.edges_of_type(LinkType.SPINE_UPLINK), "utilization")"""
        values = [
            getattr(reading, field)
            for reading in (self.latest_link(*edge_key) for edge_key in edge_keys)
            if reading is not None
        ]
        return agg(values) if values else None

    def aggregate_checkpoint_metric(
        self,
        job_ids: Iterable[str],
        field: str,
        agg: Callable[[Sequence[Numeric]], Numeric] = mean,
    ) -> Optional[Numeric]:
        """e.g. aggregate_checkpoint_metric(job_ids, "duration_s")"""
        values = [
            getattr(reading, field)
            for reading in (self.latest_checkpoint_telemetry(jid) for jid in job_ids)
            if reading is not None
        ]
        return agg(values) if values else None
