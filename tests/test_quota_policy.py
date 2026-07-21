import pytest

from gpu_cluster_sim.engine import QuotaPolicy
from gpu_cluster_sim.models import Quota, RateLimit, Tenant, TenantScope


@pytest.fixture
def policy():
    p = QuotaPolicy()
    p.add_tenant(Tenant("org-acme", TenantScope.ORG))
    p.add_tenant(Tenant("team-research", TenantScope.TEAM, parent_id="org-acme"))
    p.add_tenant(Tenant("team-infra", TenantScope.TEAM, parent_id="org-acme"))
    p.add_tenant(Tenant("user-alice", TenantScope.USER, parent_id="team-research"))
    p.add_tenant(Tenant("user-bob", TenantScope.USER, parent_id="team-research"))
    return p


def test_add_tenant_with_unknown_parent_raises():
    p = QuotaPolicy()
    with pytest.raises(ValueError):
        p.add_tenant(Tenant("user-x", TenantScope.USER, parent_id="no-such-team"))


def test_add_quota_for_unknown_tenant_raises():
    p = QuotaPolicy()
    with pytest.raises(ValueError):
        p.add_quota(Quota("no-such-tenant", max_gpus=10))


def test_add_quota_negative_max_gpus_raises(policy):
    with pytest.raises(ValueError):
        policy.add_quota(Quota("org-acme", max_gpus=-1))


def test_add_rate_limit_for_unknown_tenant_raises():
    p = QuotaPolicy()
    with pytest.raises(ValueError):
        p.add_rate_limit(RateLimit("no-such-tenant", max_submissions=5, window_s=60.0))


def test_add_rate_limit_non_positive_window_raises(policy):
    with pytest.raises(ValueError):
        policy.add_rate_limit(RateLimit("org-acme", max_submissions=5, window_s=0.0))


def test_children_of(policy):
    assert set(policy.children_of("org-acme")) == {"team-research", "team-infra"}
    assert set(policy.children_of("team-research")) == {"user-alice", "user-bob"}
    assert policy.children_of("user-alice") == []


def test_descendants_of(policy):
    assert set(policy.descendants_of("org-acme")) == {
        "team-research", "team-infra", "user-alice", "user-bob",
    }
    assert set(policy.descendants_of("team-research")) == {"user-alice", "user-bob"}
    assert policy.descendants_of("user-alice") == []


def test_ancestor_chain(policy):
    assert policy.ancestor_chain("user-alice") == ["user-alice", "team-research", "org-acme"]
    assert policy.ancestor_chain("org-acme") == ["org-acme"]


def test_quota_for_and_rate_limit_for_default_none(policy):
    assert policy.quota_for("user-alice") is None
    assert policy.rate_limit_for("user-alice") is None

    policy.add_quota(Quota("user-alice", max_gpus=10))
    policy.add_rate_limit(RateLimit("user-alice", max_submissions=5, window_s=60.0))
    assert policy.quota_for("user-alice") == Quota("user-alice", max_gpus=10)
    assert policy.rate_limit_for("user-alice") == RateLimit("user-alice", max_submissions=5, window_s=60.0)
