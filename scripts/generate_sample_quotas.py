#!/usr/bin/env python3
"""Regenerate the sample quota/rate-limit policies under data/quotas/.

    PYTHONPATH=. python3 scripts/generate_sample_quotas.py
"""

import os

from gpu_cluster_sim.engine import QuotaPolicy
from gpu_cluster_sim.models import Quota, RateLimit, Tenant, TenantScope

QUOTAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "quotas")


def build_flat_single_team() -> QuotaPolicy:
    """The simplest valid policy: one org, one team, no per-user quotas or
    rate limits at all. Demonstrates that an unconfigured level is simply
    unbounded — the team's quota is the only thing enforced.
    """
    policy = QuotaPolicy()
    policy.add_tenant(Tenant("org-startup", TenantScope.ORG))
    policy.add_tenant(Tenant("team-eng", TenantScope.TEAM, parent_id="org-startup"))
    policy.add_tenant(Tenant("user-dana", TenantScope.USER, parent_id="team-eng"))
    policy.add_tenant(Tenant("user-eve", TenantScope.USER, parent_id="team-eng"))
    policy.add_tenant(Tenant("user-frank", TenantScope.USER, parent_id="team-eng"))

    policy.add_quota(Quota("team-eng", max_gpus=64))
    return policy


def build_hierarchical_borrowing() -> QuotaPolicy:
    """One org, two teams, generous per-user ceilings — the team quota is
    the real limiter, so any one user can borrow the team's full idle
    quota while their teammates aren't using it. Rate limits set at the
    org, team, and one user level to show all three enforced at once.
    """
    policy = QuotaPolicy()
    policy.add_tenant(Tenant("org-acme", TenantScope.ORG))
    policy.add_tenant(Tenant("team-research", TenantScope.TEAM, parent_id="org-acme"))
    policy.add_tenant(Tenant("team-infra", TenantScope.TEAM, parent_id="org-acme"))
    policy.add_tenant(Tenant("user-alice", TenantScope.USER, parent_id="team-research"))
    policy.add_tenant(Tenant("user-bob", TenantScope.USER, parent_id="team-research"))
    policy.add_tenant(Tenant("user-carol", TenantScope.USER, parent_id="team-infra"))

    policy.add_quota(Quota("org-acme", max_gpus=256))
    policy.add_quota(Quota("team-research", max_gpus=160))
    policy.add_quota(Quota("team-infra", max_gpus=96))
    policy.add_quota(Quota("user-alice", max_gpus=160))
    policy.add_quota(Quota("user-bob", max_gpus=160))
    policy.add_quota(Quota("user-carol", max_gpus=96))

    policy.add_rate_limit(RateLimit("org-acme", max_submissions=50, window_s=3600.0, burst=10))
    policy.add_rate_limit(RateLimit("team-research", max_submissions=20, window_s=3600.0, burst=5))
    policy.add_rate_limit(RateLimit("user-alice", max_submissions=10, window_s=3600.0, burst=2))
    return policy


def build_multi_org_strict_partition() -> QuotaPolicy:
    """Two independent orgs sharing one cluster (a forest, not a single
    tree — `QuotaPolicy` doesn't require one root) with strict per-user
    quotas that exactly sum to their team's cap, so there is no idle quota
    left to borrow: the alternative to preemption-based borrowing named in
    "Preemption" in docs/topology-aware-gang-scheduling.md — utilization
    is lower, but every user's share is guaranteed regardless of what
    teammates are doing. Rate limits are tight at every level, the way a
    shared/public cluster would need for abuse prevention.
    """
    policy = QuotaPolicy()
    policy.add_tenant(Tenant("org-blue", TenantScope.ORG))
    policy.add_tenant(Tenant("team-blue-ml", TenantScope.TEAM, parent_id="org-blue"))
    policy.add_tenant(Tenant("user-blue-1", TenantScope.USER, parent_id="team-blue-ml"))
    policy.add_tenant(Tenant("user-blue-2", TenantScope.USER, parent_id="team-blue-ml"))

    policy.add_tenant(Tenant("org-red", TenantScope.ORG))
    policy.add_tenant(Tenant("team-red-infra", TenantScope.TEAM, parent_id="org-red"))
    policy.add_tenant(Tenant("user-red-1", TenantScope.USER, parent_id="team-red-infra"))
    policy.add_tenant(Tenant("user-red-2", TenantScope.USER, parent_id="team-red-infra"))

    policy.add_quota(Quota("org-blue", max_gpus=40))
    policy.add_quota(Quota("team-blue-ml", max_gpus=40))
    policy.add_quota(Quota("user-blue-1", max_gpus=20))
    policy.add_quota(Quota("user-blue-2", max_gpus=20))

    policy.add_quota(Quota("org-red", max_gpus=16))
    policy.add_quota(Quota("team-red-infra", max_gpus=16))
    policy.add_quota(Quota("user-red-1", max_gpus=8))
    policy.add_quota(Quota("user-red-2", max_gpus=8))

    for tenant_id, max_submissions, window_s, burst in [
        ("org-blue", 20, 3600.0, 2),
        ("team-blue-ml", 15, 3600.0, 1),
        ("user-blue-1", 5, 3600.0, 1),
        ("user-blue-2", 5, 3600.0, 1),
        ("org-red", 10, 3600.0, 1),
        ("team-red-infra", 8, 3600.0, 1),
        ("user-red-1", 3, 3600.0, 0),
        ("user-red-2", 3, 3600.0, 0),
    ]:
        policy.add_rate_limit(RateLimit(tenant_id, max_submissions=max_submissions, window_s=window_s, burst=burst))
    return policy


SAMPLES = {
    "flat_single_team": build_flat_single_team,
    "hierarchical_borrowing": build_hierarchical_borrowing,
    "multi_org_strict_partition": build_multi_org_strict_partition,
}


def main() -> None:
    os.makedirs(QUOTAS_DIR, exist_ok=True)
    for name, builder in SAMPLES.items():
        policy = builder()
        path = os.path.join(QUOTAS_DIR, f"{name}.yaml")
        policy.export("yaml", path)
        tenant_count = sum(1 for _ in policy.tenants())
        print(f"{name}: {tenant_count} tenants -> {path}")


if __name__ == "__main__":
    main()
