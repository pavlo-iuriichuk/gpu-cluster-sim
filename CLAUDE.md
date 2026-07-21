# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A simulator for topology-aware gang scheduling on GPU clusters. The design
rationale (gang scheduling, leaf/spine/rail topology, placement scoring,
queueing policy, failure handling) lives in
`docs/topology-aware-gang-scheduling.md` — read it before making design
decisions in `engine/`, since most non-obvious choices in this codebase
trace back to a specific paragraph there.

## Commands

There is no build step or linter configured. The package is not
pip-installed anywhere; there's a `pytest.ini` with `pythonpath = .`, so
`pytest` works directly from the repo root, but any ad hoc script still
needs `PYTHONPATH=.`.

```bash
pip install -r requirements-dev.txt   # runtime deps + pytest/pytest-cov

pytest                                 # run the full suite (tests/)
pytest tests/test_allocation.py        # a single file
pytest tests/test_allocation.py::test_try_allocate_is_all_or_nothing_when_it_does_not_fit  # a single test
pytest --cov=gpu_cluster_sim --cov-report=term-missing   # coverage

# regenerate the sample topologies under data/ (single_az, multi_az_region, datacenter_mixed_gpus)
PYTHONPATH=. python3 scripts/generate_sample_data.py

# regenerate the sample quota/rate-limit policies under data/quotas/
PYTHONPATH=. python3 scripts/generate_sample_quotas.py
```

`tests/conftest.py` has two fixtures nearly every test file builds on:
`small_topology` (a fast, deterministic 2-spine/2-leaf/4-node/16-GPU
fabric built with the same `populate_pod` primitive as the real samples)
and `make_job` (a `JobRequest` factory with sensible defaults). Prefer
these over hand-building a topology or job in a new test, and over
depending on the generated files under `data/` (regenerable output, not a
fixture).

## Architecture

### `models/` vs `engine/`

`models/` holds every data type as an immutable `typing.NamedTuple` (plus
plain `str` `Enum`s for closed sets like `LinkType`, `NodeHealth`,
`QoSClass`). `engine/` holds everything stateful or behavioral: graph
construction, telemetry recording, ledgers, and query/format logic. A
model never imports from `engine/`; `engine/` classes hold or operate on
model instances. When adding a new concept, ask whether it's a fact
("what a GPU is") — goes in `models/` — or something that changes over
time or answers a query — goes in `engine/`.

`models/registry.py` maps each entity NamedTuple class (`GPU`, `Node`,
`LeafSwitch`, `SpineSwitch`, `RailSwitch`) to the string `kind` tag used to
identify it inside the topology graph and by the format (de)serializers —
this is the one place both `engine/topology.py` and `engine/formats/`
depend on for that mapping, so a new entity kind starts here.

### The topology graph (`engine/topology.py`)

`ClusterTopology` wraps a single `networkx.MultiDiGraph`. Graph *nodes* are
GPU/Node/Leaf/Spine/Rail entities, tagged with `kind` (from the registry
above) and `data` (the model instance itself). Graph *edges* represent
only actual network links — NVLink, PCIe (GPU-to-its-own-node), leaf
uplink, spine uplink, inter-pod — each carrying a `LinkMetrics` attribute.
Containment (which node a GPU lives on, which leaf a node hangs off) is
carried by fields on the model objects (`GPU.node_id`, `Node.leaf_id`),
*not* by graph edges — don't add containment edges, look up the field
instead. `engine/paths.py` builds on this graph for shortest/k-shortest/
all-paths queries between two entities (typically GPUs), returning
hop-by-hop cost plus placement-scoring proxies (distinct leaf/spine
switches spanned, spine/inter-pod crossings) — see "Placement scoring" in
the design doc for what those proxies are approximating.

### Telemetry vs static metrics, and `Decimal` vs `float`

`LinkMetrics` (on graph edges) and `Node`/`GPU`/switch fields are static,
nominal characteristics. Time-varying readings (utilization, temperature,
power draw, checkpoint write cost, ...) live in `models/telemetry.py` and
are recorded/queried through `engine/telemetry.py`'s `TelemetryStore`
(history per node/edge/job, plus `aggregate_*` helpers) — kept as a
separate store rather than mutating the graph, since `NamedTuple`s are
immutable and telemetry changes far more often than topology does.
Measured telemetry values are `Decimal`, not `float` (binary float error
has no place in "is utilization above 0.8"); discrete counts (`ecc_errors`,
`queue_depth`) stay `int`. Timestamps everywhere (`allocated_at`,
`created_at`, `record_*(timestamp=...)`) stay `float`, matching
`time.time()` — they are not telemetry readings, so don't convert them.

### Pluggable format registries (`engine/formats/`, `engine/quota_formats/`)

Both follow the same shape: an abstract base class (`TopologyFormat` /
`QuotaConfigFormat`) with `export`/`import_`, a module-level `_FORMATS`
dict, and `register_format`/`get_format`/`available_formats`. A concrete
format self-registers with `register_format(MyFormat())` at the bottom of
its own module; `<package>/__init__.py` just imports that module for the
side effect. To add a new format for either `ClusterTopology` or
`QuotaPolicy`, copy this shape — don't invent a different registration
mechanism for one and not the other.

`ClusterTopology.export`/`.load` and `QuotaPolicy.export`/`.load` import
their format package *lazily, inside the method body* rather than at
module top level. This is deliberate, not an oversight: the format modules
import the container class back (to construct/type-hint against it), so a
top-level import would be circular. Keep new cross-references between a
container and its format package lazy the same way.

`engine/formats/_codec.py` flattens model `NamedTuple`s to GraphML/Parquet-safe
primitives (enums -> `.value`, tuples -> `"|"`-joined strings) and back,
driven by the NamedTuple's own type hints — so a new field on `GPU` or
`LinkMetrics` doesn't need any format-specific serialization code.

### Ledgers (`AllocationLedger`, `CheckpointLedger`, `QuotaLedger`)

Each is stateful bookkeeping paired with, but separate from, the static
container it operates against (topology, or a `QuotaPolicy`) — the same
split as telemetry vs. topology. `AllocationLedger.try_allocate` is
all-or-nothing (see "Why gang scheduling is non-negotiable" in the design
doc): it returns a complete `Allocation` or `None`, never a partial one.
Its GPU-selection logic filters free GPUs by model/memory/node health,
then groups by `JobRequest.locality_constraint` (`SAME_LEAF`/`SAME_SPINE`/
`BEST_EFFORT`) — this is a feasibility search, not the full scored
bin-packing optimizer from "Placement scoring" in the doc
(fragmentation delta, rail alignment, risk, reuse bonus are not
implemented). `engine/deallocation.py`'s `complete_job`/`cancel_job`/
`preempt_job`/`release_on_demand` are named wrappers around
`AllocationLedger.release` that just tag *why* a job's gang was freed —
the ledger itself has no way to know that.

`QuotaLedger`/`RateLimiter` (`engine/quotas.py`) both enforce up the org ->
team -> user ancestor chain from a `QuotaPolicy` (`engine/quota_policy.py`):
a level with no configured quota/rate limit is unbounded at that level, so
a user can "borrow" more than an even split of a team's quota while
siblings are idle — the binding constraint is whichever ancestor's
aggregate usage is tightest, not a fixed per-user share. Deciding *whom*
to reclaim from is a scheduler policy decision built on top of this, not
something the ledger does itself.

`engine/risk.py` estimates preemption probability empirically from
`AllocationLedger.history()` (optionally conditioned on `QoSClass`, which
is why `Allocation`/`DeallocationRecord` carry `qos_class`) and combines it
with `CheckpointLedger` state to estimate expected data loss — it returns
`None` when there isn't enough release history yet, which callers must not
conflate with a confident zero.

### `samples/` and `data/`

`gpu_cluster_sim/samples/builders.py` has `populate_pod` as the one
reusable primitive (one AZ's leaf-spine fabric + nodes/GPUs, parameterized
by a list of `RackGroup`s so a pod can mix GPU generations); the three
`build_*` functions and the region builder are thin callers of it.
`scripts/generate_sample_data.py` and `scripts/generate_sample_quotas.py`
render those builders (and hand-built `QuotaPolicy`s) into `data/` — see
`data/README.md` for what each generated sample is shaped like and why.
Regenerate rather than hand-edit anything under `data/`.
