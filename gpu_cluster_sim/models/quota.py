from enum import Enum
from typing import NamedTuple, Optional


class TenantScope(str, Enum):
    ORG = "org"
    TEAM = "team"
    USER = "user"


class Tenant(NamedTuple):
    """One node in the org -> team -> user hierarchy. `parent_id` is None
    for an org — the root of its own tree.
    """

    tenant_id: str
    scope: TenantScope
    parent_id: Optional[str] = None


class Quota(NamedTuple):
    """A hard ceiling on concurrent GPUs for `tenant_id` and its entire
    subtree (see "Multi-tenancy and fairness" in
    docs/topology-aware-gang-scheduling.md). Enforcement checks every
    ancestor's quota too, not just the tenant's own — that's what lets a
    user "borrow" more than an even split of a team's quota while a
    sibling is idle, bounded by whichever ancestor's aggregate usage is
    tightest.
    """

    tenant_id: str
    max_gpus: int


class RateLimit(NamedTuple):
    """A token-bucket submission-rate cap for `tenant_id`: `max_submissions`
    tokens refill over `window_s`, plus `burst` extra one-off capacity.
    """

    tenant_id: str
    max_submissions: int
    window_s: float
    burst: int = 0
