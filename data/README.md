# Sample topologies

Generated fixtures for the gang-scheduling simulator, built from
`gpu_cluster_sim.samples.builders` and exported in both formats supported by
`ClusterTopology.export`/`ClusterTopology.load` (see
`gpu_cluster_sim/engine/formats/`).

Each sample directory contains:
- `topology.graphml` — single-file GraphML
- `topology.graphar/` — GraphAr-inspired chunked layout (`graph.yml` +
  Parquet under `vertex/<kind>/` and `edge/<link_type>/`)

Regenerate all of them with:

```
PYTHONPATH=. python3 scripts/generate_sample_data.py
```

## single_az/

One availability zone, one GPU generation: 4 spines x 6 leaves (racks) x 4
nodes x 8 H100s = **192 GPUs**, 24 nodes. Every leaf uplinks to every spine
(full bipartite fabric), every node NVLink-meshes its own 8 GPUs. Load with
`ClusterTopology.load("graphml", "data/single_az/topology.graphml")`.

## multi_az_region/

A region of 3 independent AZs (`us-west-az1/2/3`), each its own leaf-spine
pod (2 spines x 4 leaves x 3 nodes x 8 H100s = 96 GPUs/AZ, **288 GPUs**
total). One spine per AZ is meshed to one spine in every other AZ via
`INTER_POD` links at WAN-like characteristics (100 Gbps, 500us latency, 6x
oversubscription) to represent a regional backbone connecting AZs — much
worse than any single AZ's internal fabric, as it would be in practice.

## datacenter_mixed_gpus/

One data center with three hardware generations racked side by side, the
way a real fleet accumulates over time: 120 H100 + 120 A100 + 40 A6000 =
**280 GPUs** across 8 leaves behind 4 spines. The A6000 rack group models a
smaller, cheaper node shape (4 GPUs/node instead of 8, less host RAM/NVMe)
to reflect that these are typically inference/dev boxes, not training
nodes.
