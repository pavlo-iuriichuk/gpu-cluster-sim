"""Static org -> team -> user policy: who exists, and what quota/rate-limit
each tenant has. Mirrors `ClusterTopology`'s relationship to
`engine.formats`: this container holds the data, `engine.quota_formats`
holds the pluggable (de)serializers, and `.export`/`.load` here just
delegate to that registry by name.
"""

from typing import TYPE_CHECKING, Dict, Iterator, List, Optional

from ..models.quota import Quota, RateLimit, Tenant

if TYPE_CHECKING:
    from .quota_formats import QuotaConfigFormat  # noqa: F401


class QuotaPolicy:
    def __init__(self) -> None:
        self._tenants: Dict[str, Tenant] = {}
        self._quotas: Dict[str, Quota] = {}
        self._rate_limits: Dict[str, RateLimit] = {}

    def add_tenant(self, tenant: Tenant) -> None:
        if tenant.parent_id is not None and tenant.parent_id not in self._tenants:
            raise ValueError(f"tenant {tenant.tenant_id!r} references unknown parent {tenant.parent_id!r}")
        self._tenants[tenant.tenant_id] = tenant

    def add_quota(self, quota: Quota) -> None:
        if quota.tenant_id not in self._tenants:
            raise ValueError(f"quota references unknown tenant {quota.tenant_id!r}")
        if quota.max_gpus < 0:
            raise ValueError(f"quota for {quota.tenant_id!r} has negative max_gpus")
        self._quotas[quota.tenant_id] = quota

    def add_rate_limit(self, rate_limit: RateLimit) -> None:
        if rate_limit.tenant_id not in self._tenants:
            raise ValueError(f"rate limit references unknown tenant {rate_limit.tenant_id!r}")
        if rate_limit.window_s <= 0:
            raise ValueError(f"rate limit for {rate_limit.tenant_id!r} must have window_s > 0")
        self._rate_limits[rate_limit.tenant_id] = rate_limit

    def tenant(self, tenant_id: str) -> Tenant:
        return self._tenants[tenant_id]

    def tenants(self) -> Iterator[Tenant]:
        return iter(self._tenants.values())

    def quota_for(self, tenant_id: str) -> Optional[Quota]:
        return self._quotas.get(tenant_id)

    def rate_limit_for(self, tenant_id: str) -> Optional[RateLimit]:
        return self._rate_limits.get(tenant_id)

    def children_of(self, tenant_id: str) -> List[str]:
        return [t.tenant_id for t in self._tenants.values() if t.parent_id == tenant_id]

    def descendants_of(self, tenant_id: str) -> List[str]:
        """Every descendant of `tenant_id` (not including itself), e.g. all
        users under a team, or all users and teams under an org.
        """
        descendants: List[str] = []
        frontier = self.children_of(tenant_id)
        while frontier:
            descendants.extend(frontier)
            frontier = [grandchild for child in frontier for grandchild in self.children_of(child)]
        return descendants

    def ancestor_chain(self, tenant_id: str) -> List[str]:
        """`[tenant_id, parent, grandparent, ..., root]`."""
        chain = []
        current: Optional[str] = tenant_id
        while current is not None:
            chain.append(current)
            tenant = self._tenants.get(current)
            current = tenant.parent_id if tenant is not None else None
        return chain

    def export(self, format_name: str, path: str) -> None:
        from .quota_formats import get_format

        get_format(format_name).export(self, path)

    @classmethod
    def load(cls, format_name: str, path: str) -> "QuotaPolicy":
        from .quota_formats import get_format

        return get_format(format_name).import_(path)
