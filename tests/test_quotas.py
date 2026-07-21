import pytest

from gpu_cluster_sim.engine import QuotaLedger, QuotaPolicy, RateLimiter
from gpu_cluster_sim.models import Quota, RateLimit, Tenant, TenantScope


@pytest.fixture
def policy():
    p = QuotaPolicy()
    p.add_tenant(Tenant("org-acme", TenantScope.ORG))
    p.add_tenant(Tenant("team-research", TenantScope.TEAM, parent_id="org-acme"))
    p.add_tenant(Tenant("team-infra", TenantScope.TEAM, parent_id="org-acme"))
    p.add_tenant(Tenant("user-alice", TenantScope.USER, parent_id="team-research"))
    p.add_tenant(Tenant("user-bob", TenantScope.USER, parent_id="team-research"))

    p.add_quota(Quota("org-acme", max_gpus=100))
    p.add_quota(Quota("team-research", max_gpus=64))
    p.add_quota(Quota("team-infra", max_gpus=32))
    p.add_quota(Quota("user-alice", max_gpus=64))  # generous personal cap -- team is the real limiter
    return p


def test_user_can_borrow_the_whole_idle_team_quota(policy):
    ledger = QuotaLedger(policy)
    assert ledger.try_reserve("user-alice", 64) is True
    assert ledger.subtree_usage("team-research") == 64


def test_sibling_blocked_once_team_quota_is_exhausted(policy):
    ledger = QuotaLedger(policy)
    ledger.try_reserve("user-alice", 64)
    assert ledger.try_reserve("user-bob", 1) is False


def test_release_frees_quota_for_siblings(policy):
    ledger = QuotaLedger(policy)
    ledger.try_reserve("user-alice", 64)
    ledger.release("user-alice", 64)
    assert ledger.try_reserve("user-bob", 10) is True


def test_ancestor_chain_binds_at_the_tightest_level(policy):
    ledger = QuotaLedger(policy)
    ledger.try_reserve("user-bob", 10)
    ledger.try_reserve("team-infra", 32)
    # team-research room = 64 - 10 = 54; org room = 100 - 42 = 58 -> team is tighter
    assert ledger.has_capacity("user-alice", 54) is True
    assert ledger.has_capacity("user-alice", 55) is False


def test_org_level_cap_binds_when_it_is_the_tighter_one():
    policy = QuotaPolicy()
    policy.add_tenant(Tenant("org-x", TenantScope.ORG))
    policy.add_tenant(Tenant("team-x", TenantScope.TEAM, parent_id="org-x"))
    policy.add_quota(Quota("org-x", max_gpus=10))
    policy.add_quota(Quota("team-x", max_gpus=64))

    ledger = QuotaLedger(policy)
    assert ledger.has_capacity("team-x", 10) is True
    assert ledger.has_capacity("team-x", 11) is False


def test_unconfigured_level_is_unbounded():
    policy = QuotaPolicy()
    policy.add_tenant(Tenant("org-x", TenantScope.ORG))
    policy.add_tenant(Tenant("team-x", TenantScope.TEAM, parent_id="org-x"))
    # no quota anywhere -> anything fits
    ledger = QuotaLedger(policy)
    assert ledger.try_reserve("team-x", 10_000) is True


def test_try_reserve_does_not_mutate_ledger_on_failure(policy):
    ledger = QuotaLedger(policy)
    ledger.try_reserve("user-alice", 64)
    usage_before = ledger.usage("user-bob")
    assert ledger.try_reserve("user-bob", 1) is False
    assert ledger.usage("user-bob") == usage_before == 0


def test_release_never_goes_negative(policy):
    ledger = QuotaLedger(policy)
    ledger.release("user-alice", 5)
    assert ledger.usage("user-alice") == 0


class TestRateLimiter:
    @pytest.fixture
    def rl_policy(self):
        p = QuotaPolicy()
        p.add_tenant(Tenant("org-acme", TenantScope.ORG))
        p.add_tenant(Tenant("team-research", TenantScope.TEAM, parent_id="org-acme"))
        p.add_tenant(Tenant("user-alice", TenantScope.USER, parent_id="team-research"))
        p.add_tenant(Tenant("user-bob", TenantScope.USER, parent_id="team-research"))
        p.add_rate_limit(RateLimit("user-alice", max_submissions=5, window_s=60.0, burst=2))
        p.add_rate_limit(RateLimit("team-research", max_submissions=10, window_s=60.0))
        return p

    def test_burst_plus_steady_capacity_then_exhausted(self, rl_policy):
        limiter = RateLimiter(rl_policy)
        t0 = 1000.0
        allowed = [limiter.allow("user-alice", now=t0 + i * 0.001) for i in range(7)]
        assert allowed == [True] * 7  # burst(2) + max_submissions(5)
        assert limiter.allow("user-alice", now=t0 + 0.002) is False

    def test_tokens_refill_over_time(self, rl_policy):
        limiter = RateLimiter(rl_policy)
        t0 = 1000.0
        for i in range(7):
            limiter.allow("user-alice", now=t0 + i * 0.001)
        assert limiter.allow("user-alice", now=t0 + 0.002) is False
        # 5 tokens per 60s window -> ~1 token back after 15s
        assert limiter.allow("user-alice", now=t0 + 15.0) is True

    def test_team_level_bucket_is_shared_across_users(self, rl_policy):
        limiter = RateLimiter(rl_policy)
        t0 = 1000.0
        # 5 calls from alice (well under her own 7-token bucket) plus 5 from
        # bob (no personal limit) together exhaust the team-wide 10-token
        # bucket -- neither user's own limit is what blocks the next call
        for i in range(5):
            assert limiter.allow("user-alice", now=t0 + i * 0.0001) is True
        for i in range(5):
            assert limiter.allow("user-bob", now=t0 + (5 + i) * 0.0001) is True

        assert limiter.allow("user-alice", now=t0 + 0.0011) is False
        assert limiter.allow("user-bob", now=t0 + 0.0012) is False

    def test_unconfigured_tenant_is_never_limited(self):
        policy = QuotaPolicy()
        policy.add_tenant(Tenant("org-x", TenantScope.ORG))
        limiter = RateLimiter(policy)
        for i in range(100):
            assert limiter.allow("org-x", now=1000.0 + i * 0.001) is True

    def test_rejection_does_not_spend_tokens_at_other_levels(self, rl_policy):
        limiter = RateLimiter(rl_policy)
        t0 = 1000.0
        # alice's own bucket (capacity 7) empties first, consuming 7 of the
        # team-wide bucket's 10 tokens along the way -> 3 left for the team
        for i in range(7):
            assert limiter.allow("user-alice", now=t0 + i * 0.0001) is True
        # alice is now blocked by her own empty bucket; this must NOT also
        # spend one of the team's remaining 3 tokens
        assert limiter.allow("user-alice", now=t0 + 0.001) is False

        # bob shares no personal limit, only the team's -- exactly 3 should
        # still be left for him if alice's rejection didn't leak into it
        results = [limiter.allow("user-bob", now=t0 + 0.002 + i * 0.0001) for i in range(4)]
        assert results == [True, True, True, False]
