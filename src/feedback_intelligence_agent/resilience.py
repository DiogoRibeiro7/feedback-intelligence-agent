"""Retry, timeout, and circuit-breaker wrappers for LLM providers."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Literal

from feedback_intelligence_agent.llm import LLMProvider, LLMProviderError, ProviderCapabilities
from feedback_intelligence_agent.schemas import SearchResult

CircuitState = Literal["closed", "open", "half_open"]


@dataclass(frozen=True)
class ResiliencePolicy:
    """Runtime policy for remote LLM calls."""

    max_attempts: int = 3
    timeout_seconds: float = 30.0
    backoff_seconds: float = 0.25
    circuit_failure_threshold: int = 3
    circuit_recovery_seconds: float = 30.0

    def __post_init__(self) -> None:
        """Validate policy values early."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.backoff_seconds < 0:
            raise ValueError("backoff_seconds must be non-negative")
        if self.circuit_failure_threshold < 1:
            raise ValueError("circuit_failure_threshold must be at least 1")
        if self.circuit_recovery_seconds <= 0:
            raise ValueError("circuit_recovery_seconds must be positive")


@dataclass
class CircuitBreaker:
    """In-memory circuit-breaker state for one provider instance."""

    failure_count: int = 0
    state: CircuitState = "closed"
    opened_at: float | None = None

    def before_call(self, *, now: float, recovery_seconds: float) -> None:
        """Reject calls while open, or move to half-open after recovery."""
        if self.state != "open":
            return
        if self.opened_at is None or now - self.opened_at >= recovery_seconds:
            self.state = "half_open"
            return
        remaining = round(recovery_seconds - (now - self.opened_at), 3)
        raise LLMProviderError(f"LLM circuit breaker is open; retry after {remaining}s.")

    def record_success(self) -> None:
        """Close the circuit after any successful provider call."""
        self.failure_count = 0
        self.state = "closed"
        self.opened_at = None

    def record_failure(self, *, now: float, threshold: int) -> None:
        """Track a failed call and open the circuit when the threshold is reached."""
        self.failure_count += 1
        if self.failure_count >= threshold:
            self.state = "open"
            self.opened_at = now


class ResilientLLMProvider:
    """Wrap an LLM provider with retries, per-attempt timeout, and circuit breaking."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        policy: ResiliencePolicy,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Store the wrapped provider and mutable circuit state."""
        self.provider = provider
        self.policy = policy
        self.circuit = CircuitBreaker()
        self.capabilities: ProviderCapabilities = provider.capabilities
        self._sleep = sleeper
        self._monotonic = monotonic

    @property
    def provider_name(self) -> str:
        """Return the underlying provider class name for diagnostics."""
        return type(self.provider).__name__

    def generate(self, prompt: str, *, question: str, results: list[SearchResult]) -> str:
        """Generate text with retries and circuit-breaker protection."""
        self.circuit.before_call(
            now=self._monotonic(),
            recovery_seconds=self.policy.circuit_recovery_seconds,
        )
        last_error: LLMProviderError | None = None
        for attempt in range(1, self.policy.max_attempts + 1):
            try:
                response = self._call_with_timeout(prompt, question=question, results=results)
            except LLMProviderError as exc:
                last_error = exc
                self.circuit.record_failure(
                    now=self._monotonic(),
                    threshold=self.policy.circuit_failure_threshold,
                )
                if attempt >= self.policy.max_attempts:
                    break
                self._sleep(self.policy.backoff_seconds * attempt)
                continue
            self.circuit.record_success()
            return response
        assert last_error is not None
        raise LLMProviderError(
            f"{self.provider_name} failed after {self.policy.max_attempts} attempt(s): "
            f"{last_error}"
        ) from last_error

    def _call_with_timeout(
        self,
        prompt: str,
        *,
        question: str,
        results: list[SearchResult],
    ) -> str:
        """Run one provider call with a timeout."""
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            self.provider.generate,
            prompt,
            question=question,
            results=results,
        )
        try:
            return future.result(timeout=self.policy.timeout_seconds)
        except FutureTimeoutError as exc:
            future.cancel()
            raise LLMProviderError(
                f"{self.provider_name} timed out after {self.policy.timeout_seconds}s."
            ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
