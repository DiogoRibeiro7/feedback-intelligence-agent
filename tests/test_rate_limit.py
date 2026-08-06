from __future__ import annotations

import pytest

from feedback_intelligence_agent.rate_limit import InMemoryRateLimiter


def test_in_memory_rate_limiter_allows_requests_until_limit() -> None:
    limiter = InMemoryRateLimiter(max_requests=2, window_seconds=60, clock=lambda: 0.0)

    first = limiter.check("client")
    second = limiter.check("client")
    third = limiter.check("client")

    assert first.allowed is True
    assert first.remaining == 1
    assert second.allowed is True
    assert second.remaining == 0
    assert third.allowed is False
    assert third.reset_after_seconds == 60


def test_in_memory_rate_limiter_resets_after_window() -> None:
    now = 0.0

    def clock() -> float:
        return now

    limiter = InMemoryRateLimiter(max_requests=1, window_seconds=10, clock=clock)

    assert limiter.check("client").allowed is True
    assert limiter.check("client").allowed is False
    now = 10.0
    reset = limiter.check("client")

    assert reset.allowed is True
    assert reset.remaining == 0


def test_in_memory_rate_limiter_uses_independent_keys() -> None:
    limiter = InMemoryRateLimiter(max_requests=1, window_seconds=60, clock=lambda: 0.0)

    assert limiter.check("a").allowed is True
    assert limiter.check("b").allowed is True
    assert limiter.check("a").allowed is False


def test_in_memory_rate_limiter_validates_configuration() -> None:
    with pytest.raises(ValueError, match="max_requests"):
        InMemoryRateLimiter(max_requests=0, window_seconds=60)
    with pytest.raises(ValueError, match="window_seconds"):
        InMemoryRateLimiter(max_requests=1, window_seconds=0)
