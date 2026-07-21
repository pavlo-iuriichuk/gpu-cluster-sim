"""Human-editable YAML representation of a `QuotaPolicy`: a flat list of
tenants, each optionally carrying its own `quota` and/or `rate_limit`. This
is the format an operator would actually hand-maintain; other formats
(e.g. JSON) can register alongside it without touching this one.
"""

from typing import Any, Dict, List

import yaml

from ...models.quota import Quota, RateLimit, Tenant, TenantScope
from ..quota_policy import QuotaPolicy
from .base import QuotaConfigFormat, register_format


class YAMLQuotaFormat(QuotaConfigFormat):
    name = "yaml"

    def export(self, policy: QuotaPolicy, path: str) -> None:
        tenants_out = []
        for tenant in policy.tenants():
            entry = {
                "tenant_id": tenant.tenant_id,
                "scope": tenant.scope.value,
                "parent_id": tenant.parent_id,
            }
            quota = policy.quota_for(tenant.tenant_id)
            if quota is not None:
                entry["quota"] = {"max_gpus": quota.max_gpus}
            rate_limit = policy.rate_limit_for(tenant.tenant_id)
            if rate_limit is not None:
                entry["rate_limit"] = {
                    "max_submissions": rate_limit.max_submissions,
                    "window_s": rate_limit.window_s,
                    "burst": rate_limit.burst,
                }
            tenants_out.append(entry)
        with open(path, "w") as f:
            yaml.safe_dump({"tenants": tenants_out}, f, sort_keys=False)

    def import_(self, path: str) -> QuotaPolicy:
        with open(path) as f:
            document = yaml.safe_load(f) or {}

        policy = QuotaPolicy()
        # a hand-edited file may list a user before its team — resolve
        # parent-before-child order rather than requiring the author to.
        for entry in _order_by_hierarchy(document.get("tenants", [])):
            tenant = Tenant(
                tenant_id=entry["tenant_id"],
                scope=TenantScope(entry["scope"]),
                parent_id=entry.get("parent_id"),
            )
            policy.add_tenant(tenant)
            if "quota" in entry:
                policy.add_quota(Quota(tenant_id=tenant.tenant_id, max_gpus=entry["quota"]["max_gpus"]))
            if "rate_limit" in entry:
                rate_limit = entry["rate_limit"]
                policy.add_rate_limit(
                    RateLimit(
                        tenant_id=tenant.tenant_id,
                        max_submissions=rate_limit["max_submissions"],
                        window_s=rate_limit["window_s"],
                        burst=rate_limit.get("burst", 0),
                    )
                )
        return policy


def _order_by_hierarchy(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reorder tenant entries so every parent appears before its children,
    regardless of the order they were written in the file.
    """
    ordered: List[Dict[str, Any]] = []
    added = set()
    remaining = list(entries)
    while remaining:
        ready, not_ready = [], []
        for entry in remaining:
            (ready if entry.get("parent_id") in (None, *added) else not_ready).append(entry)
        if not ready:
            raise ValueError(
                f"quota config has an unresolvable tenant hierarchy (missing or cyclic parents): "
                f"{[e['tenant_id'] for e in not_ready]}"
            )
        ordered.extend(ready)
        added.update(e["tenant_id"] for e in ready)
        remaining = not_ready
    return ordered


register_format(YAMLQuotaFormat())
