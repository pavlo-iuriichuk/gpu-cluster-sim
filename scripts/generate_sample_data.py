#!/usr/bin/env python3
"""Regenerate the sample topologies under data/.

    python3 scripts/generate_sample_data.py
"""

import os
import shutil

from gpu_cluster_sim.samples import (
    build_datacenter_mixed_gpus,
    build_multi_az_region,
    build_single_az,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

SAMPLES = {
    "single_az": build_single_az,
    "multi_az_region": build_multi_az_region,
    "datacenter_mixed_gpus": build_datacenter_mixed_gpus,
}


def main() -> None:
    for name, builder in SAMPLES.items():
        topology = builder()
        sample_dir = os.path.join(DATA_DIR, name)
        os.makedirs(sample_dir, exist_ok=True)

        graphar_dir = os.path.join(sample_dir, "topology.graphar")
        shutil.rmtree(graphar_dir, ignore_errors=True)

        topology.export("graphml", os.path.join(sample_dir, "topology.graphml"))
        topology.export("graphar", graphar_dir)

        gpu_count = sum(1 for _ in topology.entities_of_kind("gpu"))
        node_count = sum(1 for _ in topology.entities_of_kind("node"))
        print(
            f"{name}: {node_count} nodes, {gpu_count} GPUs, "
            f"{topology.graph.number_of_nodes()} graph vertices, "
            f"{topology.graph.number_of_edges()} graph edges -> {sample_dir}"
        )


if __name__ == "__main__":
    main()
