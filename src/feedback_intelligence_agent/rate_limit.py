"""Small in-memory fixed-window rate limiter for API deployments."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, Field


class RateLimitDecision(BaseModel):
    """Result of one rate-limit check."""

    allowed: bool
    limit: int = Field(ge=1)
    remaining: int = Field(ge=0)
    reset_after_seconds: int = Field(ge=0)


@dataclass
class _Window:
    started_at: float
    count: int


class InMemoryRateLimiter:
    """Fixed-window request limiter keyed by caller identity."""

    def __init__(
        self,
        *,
        max_requests: int,
        window_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a limiter with a shared request budget per key."""
        if max_requests < 1:
            raise ValueError("max_requests must be at least 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be greater than 0")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._clock = clock
        self._windows: dict[str, _Window] = {}

    def check(self, key: str) -> RateLimitDecision:
        """Consume one request from the key's current window."""
        now = self._clock()
        clean_key = key.strip() or "anonymous"
        window = self._windows.get(clean_key)
        if window is None or now - window.started_at >= self.window_seconds:
            window = _Window(started_at=now, count=0)
            self._windows[clean_key] = window

        reset_after = _reset_after(window.started_at, now, self.window_seconds)
        if window.count >= self.max_requests:
            return RateLimitDecision(
                allowed=False,
                limit=self.max_requests,
                remaining=0,
                reset_after_seconds=reset_after,
            )

        window.count += 1
        return RateLimitDecision(
            allowed=True,
            limit=self.max_requests,
            remaining=max(0, self.max_requests - window.count),
            reset_after_seconds=reset_after,
        )


def _reset_after(started_at: float, now: float, window_seconds: float) -> int:
    remaining = max(0.0, window_seconds - (now - started_at))
    return max(1, int(remaining + 0.999))
