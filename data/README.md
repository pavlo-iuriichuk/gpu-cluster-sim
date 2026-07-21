# Sample data

## Topologies

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

### single_az/

One availability zone, one GPU generation: 4 spines x 6 leaves (racks) x 4
nodes x 8 H100s = **192 GPUs**, 24 nodes. Every leaf uplinks to every spine
(full bipartite fabric), every node NVLink-meshes its own 8 GPUs. Load with
`ClusterTopology.load("graphml", "data/single_az/topology.graphml")`.

### multi_az_region/

A region of 3 independent AZs (`us-west-az1/2/3`), each its own leaf-spine
pod (2 spines x 4 leaves x 3 nodes x 8 H100s = 96 GPUs/AZ, **288 GPUs**
total). One spine per AZ is meshed to one spine in every other AZ via
`INTER_POD` links at WAN-like characteristics (100 Gbps, 500us latency, 6x
oversubscription) to represent a regional backbone connecting AZs — much
worse than any single AZ's internal fabric, as it would be in practice.

### datacenter_mixed_gpus/

One data center with three hardware generations racked side by side, the
way a real fleet accumulates over time: 120 H100 + 120 A100 + 40 A6000 =
**280 GPUs** across 8 leaves behind 4 spines. The A6000 rack group models a
smaller, cheaper node shape (4 GPUs/node instead of 8, less host RAM/NVMe)
to reflect that these are typically inference/dev boxes, not training
nodes.

## Quota / rate-limit policies

Sample org -> team -> user policies for `QuotaPolicy`
(`gpu_cluster_sim.engine.quota_policy`), each a plain YAML file exported via
the `yaml` format in `gpu_cluster_sim/engine/quota_formats/` (the same
pluggable-format registry pattern as the topology formats above — a new
config format is one class + `register_format(...)`, see `base.py` there).
Enforcement (`QuotaLedger`, `RateLimiter` in `gpu_cluster_sim/engine/quotas.py`)
walks the org/team/user ancestor chain for every tenant, so these three
samples are chosen to show the range of what that produces in practice.

Regenerate all of them with:

```
PYTHONPATH=. python3 scripts/generate_sample_quotas.py
```

### quotas/flat_single_team.yaml

The simplest valid policy: one org, one team, three users, and a single
quota — on the team (64 GPUs). No per-user quotas and no rate limits at
all, demonstrating that an unconfigured level is simply unbounded rather
than an error. Load with
`QuotaPolicy.load("yaml", "data/quotas/flat_single_team.yaml")`.

### quotas/hierarchical_borrowing.yaml

One org (256 GPUs) with two teams (research: 160, infra: 96); each user
has a generous personal ceiling, so the team quota is the actual limiter
and any one user can borrow the team's full idle quota while teammates
aren't using it (see `QuotaLedger.has_capacity` for the ancestor-chain
enforcement that makes this work). Submission rate limits are set at the
org, team, and one user level to show all three enforced at once.

### quotas/multi_org_strict_partition.yaml

Two independent orgs sharing one cluster — `org-blue` and `org-red`, each
its own root, since `QuotaPolicy` is a forest, not a single tree — with
per-user quotas that exactly sum to their team's cap (20+20=40, 8+8=16),
so there is **no** idle quota left to borrow: the strict-partitioning
alternative to preemption-based borrowing named in "Preemption" in
`docs/topology-aware-gang-scheduling.md`. Lower utilization, but every
user's share is guaranteed regardless of what teammates are doing. Rate
limits are tight at every level, the way a shared/public cluster would
need for abuse prevention.
