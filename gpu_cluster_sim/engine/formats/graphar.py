"""A GraphAr-inspired layout: a `graph.yml` metadata file plus one Parquet
chunk per vertex kind and per edge link-type, under `vertex/<kind>/` and
`edge/<link_type>/`. This mirrors Apache GraphAr's chunked, columnar layout
(vertices/edges partitioned by type, properties as columns) but is a
simplified, project-local reader/writer — there is no maintained pure-Python
GraphAr package to depend on, and reproducing the full spec (CSR/CSC
adjacency chunking, offset tables, cross-language metadata) is out of scope
here. Do not expect files written by this format to load in the official
Apache GraphAr toolchain.
"""

import os
from collections import defaultdict
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from ...models.network import LinkMetrics
from ...models.registry import KIND_TO_ENTITY
from ..topology import ClusterTopology
from ._codec import decode_namedtuple, encode_namedtuple
from .base import TopologyFormat, register_format

_CHUNK_FILE = "chunk0.parquet"


class GraphArFormat(TopologyFormat):
    name = "graphar"

    def export(self, topology: ClusterTopology, path: str) -> None:
        vertex_dir = os.path.join(path, "vertex")
        edge_dir = os.path.join(path, "edge")
        os.makedirs(vertex_dir, exist_ok=True)
        os.makedirs(edge_dir, exist_ok=True)

        rows_by_kind: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for entity_id, attrs in topology.graph.nodes(data=True):
            row = encode_namedtuple(attrs["data"])
            row["id"] = entity_id
            rows_by_kind[attrs["kind"]].append(row)

        vertices_meta = {}
        for kind, rows in rows_by_kind.items():
            kind_dir = os.path.join(vertex_dir, kind)
            os.makedirs(kind_dir, exist_ok=True)
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, os.path.join(kind_dir, _CHUNK_FILE))
            vertices_meta[kind] = {
                "properties": [c for c in table.column_names if c != "id"],
                "chunk_files": [_CHUNK_FILE],
            }

        rows_by_link_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for u, v, key, attrs in topology.graph.edges(keys=True, data=True):
            metrics: LinkMetrics = attrs["metrics"]
            row = encode_namedtuple(metrics)
            row["src"] = u
            row["dst"] = v
            row["key"] = key
            rows_by_link_type[metrics.link_type.value].append(row)

        edges_meta = {}
        for link_type, rows in rows_by_link_type.items():
            type_dir = os.path.join(edge_dir, link_type)
            os.makedirs(type_dir, exist_ok=True)
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, os.path.join(type_dir, _CHUNK_FILE))
            edges_meta[link_type] = {
                "properties": [c for c in table.column_names if c not in ("src", "dst", "key")],
                "chunk_files": [_CHUNK_FILE],
            }

        metadata = {
            "format": "graphar-lite/v1",
            "vertices": vertices_meta,
            "edges": edges_meta,
        }
        with open(os.path.join(path, "graph.yml"), "w") as f:
            yaml.safe_dump(metadata, f, sort_keys=False)

    def import_(self, path: str) -> ClusterTopology:
        with open(os.path.join(path, "graph.yml")) as f:
            metadata = yaml.safe_load(f)

        topology = ClusterTopology()
        for kind, info in metadata["vertices"].items():
            entity_cls = KIND_TO_ENTITY[kind]
            for chunk_file in info["chunk_files"]:
                table = pq.read_table(os.path.join(path, "vertex", kind, chunk_file))
                for row in table.to_pylist():
                    entity_id = row.pop("id")
                    entity = decode_namedtuple(entity_cls, row)
                    topology.graph.add_node(entity_id, kind=kind, data=entity)

        for link_type, info in metadata["edges"].items():
            for chunk_file in info["chunk_files"]:
                table = pq.read_table(os.path.join(path, "edge", link_type, chunk_file))
                for row in table.to_pylist():
                    u = row.pop("src")
                    v = row.pop("dst")
                    key = row.pop("key")
                    metrics = decode_namedtuple(LinkMetrics, row)
                    topology.graph.add_edge(u, v, key=key, metrics=metrics)
        return topology


register_format(GraphArFormat())
