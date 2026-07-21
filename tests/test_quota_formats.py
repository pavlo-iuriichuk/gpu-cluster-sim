import pytest

from gpu_cluster_sim.engine import QuotaPolicy, available_quota_formats
from gpu_cluster_sim.models import Quota, RateLimit, Tenant, TenantScope


def test_yaml_format_registered():
    assert "yaml" in available_quota_formats()


def test_yaml_round_trip(tmp_path):
    policy = QuotaPolicy()
    policy.add_tenant(Tenant("org-acme", TenantScope.ORG))
    policy.add_tenant(Tenant("team-research", TenantScope.TEAM, parent_id="org-acme"))
    policy.add_tenant(Tenant("user-alice", TenantScope.USER, parent_id="team-research"))
    policy.add_quota(Quota("org-acme", max_gpus=100))
    policy.add_quota(Quota("team-research", max_gpus=64))
    policy.add_rate_limit(RateLimit("user-alice", max_submissions=5, window_s=60.0, burst=2))

    path = tmp_path / "policy.yaml"
    policy.export("yaml", str(path))
    loaded = QuotaPolicy.load("yaml", str(path))

    assert {t.tenant_id for t in loaded.tenants()} == {"org-acme", "team-research", "user-alice"}
    assert loaded.tenant("user-alice") == Tenant("user-alice", TenantScope.USER, parent_id="team-research")
    assert loaded.quota_for("team-research") == Quota("team-research", max_gpus=64)
    assert loaded.quota_for("user-alice") is None
    assert loaded.rate_limit_for("user-alice") == RateLimit(
        "user-alice", max_submissions=5, window_s=60.0, burst=2
    )
    assert loaded.ancestor_chain("user-alice") == ["user-alice", "team-research", "org-acme"]


def test_yaml_import_tolerates_out_of_order_tenants(tmp_path):
    text = """
tenants:
  - tenant_id: user-carol
    scope: user
    parent_id: team-ops
  - tenant_id: team-ops
    scope: team
    parent_id: org-beta
    quota:
      max_gpus: 20
  - tenant_id: org-beta
    scope: org
    parent_id: null
    quota:
      max_gpus: 50
"""
    path = tmp_path / "out_of_order.yaml"
    path.write_text(text)

    policy = QuotaPolicy.load("yaml", str(path))
    assert policy.ancestor_chain("user-carol") == ["user-carol", "team-ops", "org-beta"]
    assert policy.quota_for("org-beta") == Quota("org-beta", max_gpus=50)


def test_yaml_import_unresolvable_hierarchy_raises(tmp_path):
    text = """
tenants:
  - tenant_id: user-x
    scope: user
    parent_id: no-such-team
"""
    path = tmp_path / "bad.yaml"
    path.write_text(text)

    with pytest.raises(ValueError):
        QuotaPolicy.load("yaml", str(path))


def test_get_quota_format_rejects_unknown_name():
    from gpu_cluster_sim.engine.quota_formats import get_format

    with pytest.raises(ValueError):
        get_format("not-a-real-format")
