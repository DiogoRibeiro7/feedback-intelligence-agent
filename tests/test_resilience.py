from __future__ import annotations

import time

import pytest

from feedback_intelligence_agent.llm import LLMProviderError, ProviderCapabilities
from feedback_intelligence_agent.resilience import ResiliencePolicy, ResilientLLMProvider
from feedback_intelligence_agent.schemas import SearchResult


class ScriptedLLM:
    capabilities = ProviderCapabilities(
        supports_streaming=False,
        supports_tool_calling=False,
        supports_json_mode=False,
    )

    def __init__(self, outcomes: list[str | Exception]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def generate(self, prompt: str, *, question: str, results: list[SearchResult]) -> str:
        del prompt, question, results
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class SlowLLM:
    capabilities = ProviderCapabilities(
        supports_streaming=False,
        supports_tool_calling=False,
        supports_json_mode=False,
    )

    def __init__(self, sleep_seconds: float) -> None:
        self.sleep_seconds = sleep_seconds
        self.calls = 0

    def generate(self, prompt: str, *, question: str, results: list[SearchResult]) -> str:
        del prompt, question, results
        self.calls += 1
        time.sleep(self.sleep_seconds)
        return "too late"


def test_resilient_provider_retries_transient_llm_errors() -> None:
    sleeps: list[float] = []
    provider = ScriptedLLM(
        [
            LLMProviderError("temporary outage"),
            LLMProviderError("still unavailable"),
            "ok",
        ]
    )
    resilient = ResilientLLMProvider(
        provider,
        policy=ResiliencePolicy(max_attempts=3, backoff_seconds=0.1),
        sleeper=sleeps.append,
    )

    answer = resilient.generate("prompt", question="q", results=[])

    assert answer == "ok"
    assert provider.calls == 3
    assert sleeps == [0.1, 0.2]
    assert resilient.circuit.state == "closed"


def test_resilient_provider_converts_timeout_to_provider_error() -> None:
    provider = SlowLLM(sleep_seconds=0.2)
    resilient = ResilientLLMProvider(
        provider,
        policy=ResiliencePolicy(max_attempts=1, timeout_seconds=0.01),
    )

    with pytest.raises(LLMProviderError, match="timed out"):
        resilient.generate("prompt", question="q", results=[])

    assert provider.calls == 1


def test_resilient_provider_opens_circuit_after_failures() -> None:
    now = [100.0]
    provider = ScriptedLLM(
        [
            LLMProviderError("first"),
            LLMProviderError("second"),
            "should not be called",
        ]
    )
    resilient = ResilientLLMProvider(
        provider,
        policy=ResiliencePolicy(
            max_attempts=1,
            circuit_failure_threshold=2,
            circuit_recovery_seconds=10.0,
        ),
        monotonic=lambda: now[0],
    )

    with pytest.raises(LLMProviderError, match="first"):
        resilient.generate("prompt", question="q", results=[])
    with pytest.raises(LLMProviderError, match="second"):
        resilient.generate("prompt", question="q", results=[])
    with pytest.raises(LLMProviderError, match="circuit breaker is open"):
        resilient.generate("prompt", question="q", results=[])

    assert provider.calls == 2
    assert resilient.circuit.state == "open"


def test_resilient_provider_half_open_success_closes_circuit() -> None:
    now = [100.0]
    provider = ScriptedLLM([LLMProviderError("first"), "recovered"])
    resilient = ResilientLLMProvider(
        provider,
        policy=ResiliencePolicy(
            max_attempts=1,
            circuit_failure_threshold=1,
            circuit_recovery_seconds=10.0,
        ),
        monotonic=lambda: now[0],
    )

    with pytest.raises(LLMProviderError, match="first"):
        resilient.generate("prompt", question="q", results=[])
    now[0] = 111.0

    answer = resilient.generate("prompt", question="q", results=[])

    assert answer == "recovered"
    assert provider.calls == 2
    assert resilient.circuit.state == "closed"


def test_resilience_policy_validates_values() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        ResiliencePolicy(max_attempts=0)
    with pytest.raises(ValueError, match="timeout_seconds"):
        ResiliencePolicy(timeout_seconds=0.0)
    with pytest.raises(ValueError, match="backoff_seconds"):
        ResiliencePolicy(backoff_seconds=-0.1)
    with pytest.raises(ValueError, match="circuit_failure_threshold"):
        ResiliencePolicy(circuit_failure_threshold=0)
    with pytest.raises(ValueError, match="circuit_recovery_seconds"):
        ResiliencePolicy(circuit_recovery_seconds=0.0)
