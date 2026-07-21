"""Runtime quota and rate-limit enforcement against a `QuotaPolicy`. Both
walk the org -> team -> user ancestor chain (see "Multi-tenancy and
fairness" in docs/topology-aware-gang-scheduling.md): a level with no
configured quota/rate limit is unbounded at that level, so enforcement only
blocks on levels that actually declare a limit. That's what lets a user
"borrow" more than an even split of a team's quota while siblings are idle
— the binding constraint is whichever ancestor's aggregate usage is
tightest, not a fixed per-user share. Deciding *whom* to reclaim from when
an owner returns (i.e. preemption selection) is a scheduler policy
decision built on top of this, not something the ledger does itself — see
`engine.deallocation.preempt_job` for the mechanism once that decision is
made.
"""

import time
from collections import defaultdict
from typing import Dict, Optional

from .quota_policy import QuotaPolicy


class QuotaLedger:
    def __init__(self, policy: QuotaPolicy) -> None:
        self.policy = policy
        self._usage: Dict[str, int] = defaultdict(int)

    def usage(self, tenant_id: str) -> int:
        """This tenant's own directly-attributed usage (not its subtree)."""
        return self._usage[tenant_id]

    def subtree_usage(self, tenant_id: str) -> int:
        """This tenant's own usage plus every descendant's — a quota is
        enforced against the whole subtree, the same way an org's cap
        covers every team and user under it.
        """
        return self._usage[tenant_id] + sum(
            self._usage[descendant_id] for descendant_id in self.policy.descendants_of(tenant_id)
        )

    def has_capacity(self, tenant_id: str, gpu_count: int) -> bool:
        """Whether `tenant_id` could take `gpu_count` more GPUs right now
        without exceeding its own quota or any ancestor's.
        """
        for ancestor_id in self.policy.ancestor_chain(tenant_id):
            quota = self.policy.quota_for(ancestor_id)
            if quota is None:
                continue
            if self.subtree_usage(ancestor_id) + gpu_count > quota.max_gpus:
                return False
        return True

    def try_reserve(self, tenant_id: str, gpu_count: int) -> bool:
        """Attempt to charge `gpu_count` GPUs against `tenant_id`. Returns
        whether it fit; the ledger is unchanged if it didn't.
        """
        if not self.has_capacity(tenant_id, gpu_count):
            return False
        self._usage[tenant_id] += gpu_count
        return True

    def release(self, tenant_id: str, gpu_count: int) -> None:
        self._usage[tenant_id] = max(0, self._usage[tenant_id] - gpu_count)


class RateLimiter:
    """Token bucket per tenant, checked up the same ancestor chain as
    `QuotaLedger` — a team-wide submission cap throttles all its users even
    if each user is individually under their own limit.
    """

    def __init__(self, policy: QuotaPolicy) -> None:
        self.policy = policy
        self._tokens: Dict[str, float] = {}
        self._last_refill: Dict[str, float] = {}

    def _refill(self, tenant_id: str, now: float) -> None:
        rate_limit = self.policy.rate_limit_for(tenant_id)
        if rate_limit is None:
            return
        capacity = rate_limit.max_submissions + rate_limit.burst
        last = self._last_refill.get(tenant_id, now)
        tokens = self._tokens.get(tenant_id, float(capacity))
        refill_rate = rate_limit.max_submissions / rate_limit.window_s
        self._tokens[tenant_id] = min(capacity, tokens + (now - last) * refill_rate)
        self._last_refill[tenant_id] = now

    def allow(self, tenant_id: str, *, now: Optional[float] = None) -> bool:
        """Whether a submission from `tenant_id` is allowed right now.
        Only consumes tokens if every ancestor level with a configured
        limit allows it — a rejection at one level must not silently spend
        budget at another.
        """
        now = now if now is not None else time.time()
        chain = self.policy.ancestor_chain(tenant_id)
        for ancestor_id in chain:
            self._refill(ancestor_id, now)

        limited_ancestors = [aid for aid in chain if self.policy.rate_limit_for(aid) is not None]
        if any(self._tokens.get(aid, 0.0) < 1.0 for aid in limited_ancestors):
            return False

        for ancestor_id in limited_ancestors:
            self._tokens[ancestor_id] -= 1.0
        return True
