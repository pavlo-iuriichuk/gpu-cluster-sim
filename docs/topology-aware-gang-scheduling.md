# Topology-Aware Gang Scheduling for GPU Clusters

*Design study*

---

## 1. The problem in one paragraph

A distributed training job requests N GPUs. It cannot start until all N are available (gang constraint), and its step time is determined by the slowest communication link in the collective (topology constraint). The scheduler must therefore find a *subset* of the cluster that is (a) free, (b) large enough, and (c) shaped so that the job's communication pattern maps efficiently onto the physical fabric — while a stream of other jobs of varying sizes arrives, completes, and fragments the cluster over time.

Every interesting design decision falls out of the tension between those three requirements.

---

## 2. Why gang scheduling is non-negotiable

Bulk-synchronous training has a hard barrier at every step: all ranks participate in the all-reduce, so a missing rank stalls everyone.

Consequences if you schedule incrementally:

- **Idle burn.** 400 of 512 GPUs allocated and waiting = 400 GPUs producing nothing at full cost.
- **Deadlock.** Job A holds 300 and waits for 212; Job B holds 300 and waits for 212. Neither yields. Classic hold-and-wait.
- **Convoy effects.** Partially-allocated jobs hold resources that would let smaller jobs complete quickly.

So the allocation is atomic: reserve all N or reserve none. This turns scheduling from a per-task greedy assignment into a **bin-packing-with-shape-constraints** problem, which is where the difficulty comes from.

**Related notion worth naming in interview:** *gang* vs *coscheduling* vs *all-or-nothing with minimum viable size*. Some frameworks (elastic training, e.g. Horovod Elastic / torchelastic) accept a range `[min, max]` rather than a fixed N, which materially changes the scheduler's options — a good thing to raise proactively.

---

## 3. The topology model

Build an explicit hierarchy. Cost increases at each level:

| Level | Interconnect | Rough bandwidth character |
|---|---|---|
| GPU ↔ GPU, same node | NVLink / NVSwitch | Highest, uniform within node |
| Node ↔ Node, same leaf | IB / RoCE, 1 switch hop | High, low latency |
| Node ↔ Node, same spine | 3 hops (leaf→spine→leaf) | Contended, higher latency |
| Across pods / zones | Multiple spine hops, possibly oversubscribed | Worst |

Additional structure that matters in practice:

- **Rails.** In rail-optimized designs, GPU *i* on every node connects to the same rail switch. Rail-aligned placement lets an all-reduce run rail-local, avoiding cross-rail traffic entirely. This is the detail that separates people who have read about topology from people who have operated it.
- **Oversubscription ratio.** Leaf uplinks are frequently oversubscribed relative to downlinks; a placement that looks fine hop-wise can still contend.
- **NUMA / PCIe.** GPU-to-NIC affinity within a node. Wrong pairing forces traffic across the host's PCIe root complex or UPI link.

**Model it as:** a tree (or DAG for multi-rail) where leaves are GPUs, internal nodes are switches, and edges carry `(bandwidth, latency, current_utilization)`. Placement quality = a function over the induced subgraph.

---

## 4. The request is richer than "N GPUs"

A realistic request object:

```
JobRequest {
  gpu_count: N                    # or [min, max] for elastic
  shape_hint: (nodes, gpus_per_node)   # e.g. 8 x 8, not 64 loose
  gpu_model: H100 | A100 | ...
  memory_per_gpu, cpu_per_gpu, host_ram, local_nvme
  locality_constraint: SAME_LEAF | SAME_SPINE | BEST_EFFORT
  collective_pattern: RING | TREE | ALL2ALL   # informs cost function
  priority / QoS class
  max_queue_time (SLO)
  preemptible: bool
  affinity / anti-affinity, taints & tolerations
}
```

And the response is not a *set* but an **ordered list** — rank assignment matters. A ring all-reduce should traverse NVLink within a node before crossing the fabric, and cross the fabric a minimal number of times. Handing back `{gpu_ids}` unordered silently discards a large fraction of the achievable performance. NCCL will build its own rings from what it discovers, but it can only work with the placement you gave it.

---

## 5. Placement scoring

Placement is a *scored* decision, not a boolean. Candidate placements get a cost:

```
cost(P) = w1 * comm_cost(P)          # hop-weighted, pattern-aware
        + w2 * fragmentation_delta(P) # damage to future placeability
        + w3 * (1 - rail_alignment(P))
        + w4 * risk(P)                # co-located failure domains, node health
        - w5 * reuse_bonus(P)          # warm caches, already-pulled images
```

**`comm_cost`** — for a ring, sum the edge costs around the induced ring; for a tree, the depth-weighted cost. Simple usable proxy: number of distinct leaf switches spanned, then number of spine crossings.

**`fragmentation_delta`** — the subtle one. Choosing a placement that leaves 7 free GPUs on each of eight nodes is much worse than one leaving a clean free node, even if both have identical comm cost. A workable heuristic: score the free-pool by "largest well-shaped job still placeable" before and after.

**Selection strategy:** exact optimum is NP-hard, so in practice — enumerate candidate anchors (leaf switches with enough capacity), expand greedily within each, score, take the best of K. Bounded search, deterministic, explainable. Being able to say *"I'd take a bounded-search heuristic with an explainable score rather than an ILP, because scheduling latency is itself a cost and operators need to know why a job landed where it did"* is a strong interview answer.

---

## 6. Queueing policy — where the interviewer will push

Once placement works, the hard questions are all about the queue.

**Large-job starvation.** Pure FIFO with all-or-nothing means a 512-GPU job may never find a window while 8-GPU jobs churn through. Standard answers:
- **Reservation + backfill** (the EASY/Slurm approach): give the head-of-line large job a future reservation, then backfill smaller jobs *only if* they provably finish before the reservation starts. Requires runtime estimates, which are unreliable — discuss that.
- **Aging** — priority increases with wait time.
- **Defragmentation windows** — deliberately drain to consolidate.

**Wait vs. accept.** Take a mediocre placement now or hold out for a good one? Frame it as expected-value: `bad_placement_slowdown × expected_runtime` vs `expected_additional_queue_time`. A 15% slower placement on a 3-day job is worth ~10 hours — so waiting 2 hours for a clean shape is rational; waiting 2 days is not. Make this an explicit, tunable policy rather than an implicit constant.

**Preemption.** Do you evict a low-priority job to make room? If yes: checkpoint-aware preemption (evict at a checkpoint boundary, not mid-step), grace periods, and accounting so preempted tenants aren't billed for lost work. If no: strict partitioning by QoS class, at the cost of utilization.

**Multi-tenancy and fairness.** Quotas per tenant, hierarchical (org → team → user), with borrowing from idle quota and reclaim when the owner returns. DRF (dominant resource fairness) is the citation.

---

## 7. Failure handling

This is the half people forget, and it's where operating experience shows.

**Node dies mid-job.** The gang is broken. Options:
1. **Hold and swap** — keep the surviving allocation, pull a hot spare, restart from last checkpoint. Fast, but requires a spare pool and the spare may be badly placed topologically.
2. **Release and requeue** — clean, but the job goes back to the end of a contended queue and may wait hours.
3. **Elastic shrink** — continue at N−k if the framework supports it, rescale later.

Hot-spare pool sizing is a real design question: reserve too many and you burn capacity; too few and MTTR dominates. At scale, with per-node MTBF measured in months and job widths in the hundreds, job-level failure rates get high enough that this stops being an edge case.

**Health and draining.** Continuous GPU health checks (ECC errors, thermals, NVLink/IB link flaps, falling clocks), plus proactive drain — cordon a node, let the current job finish, remove it from the schedulable pool. Silent degradation is worse than hard failure: one GPU running 20% slow drags the entire synchronous job to its pace. Straggler detection belongs in the scheduler's feedback loop.

**Correlated failure domains.** Packing a job tightly into one rack maximizes performance and maximizes blast radius from a single PDU or ToR switch. Worth naming the tradeoff explicitly.
