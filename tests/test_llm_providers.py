"""Tests for the multi-provider LLM abstraction.

The OpenAI-compatible and Ollama providers are exercised with mocked HTTP via
``httpx.MockTransport``. The Anthropic provider is exercised with a fake
``anthropic`` module injected into ``sys.modules``, so the tests never require
the optional SDK or network access.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from feedback_intelligence_agent.config import Settings
from feedback_intelligence_agent.factory import build_llm
from feedback_intelligence_agent.llm import (
    AnthropicLLM,
    BedrockConverseLLM,
    DeterministicLLM,
    LLMProviderError,
    OllamaLLM,
    OpenAIChatLLM,
    OpenAIResponsesLLM,
    ProviderCapabilities,
)
from feedback_intelligence_agent.resilience import ResilientLLMProvider

# ---------------------------------------------------------------------------
# Capability metadata
# ---------------------------------------------------------------------------


def test_deterministic_llm_capabilities() -> None:
    capabilities = DeterministicLLM.capabilities
    assert capabilities == ProviderCapabilities(
        supports_streaming=True,
        supports_tool_calling=False,
        supports_json_mode=False,
        max_context_tokens=None,
    )


def test_anthropic_llm_capabilities() -> None:
    capabilities = AnthropicLLM.capabilities
    assert capabilities.supports_streaming is True
    assert capabilities.supports_tool_calling is True
    assert capabilities.supports_json_mode is True
    assert capabilities.max_context_tokens == 1_000_000


def test_openai_and_ollama_capabilities() -> None:
    assert OpenAIChatLLM.capabilities.supports_tool_calling is True
    assert OpenAIChatLLM.capabilities.supports_json_mode is True
    assert OpenAIResponsesLLM.capabilities.supports_tool_calling is True
    assert OpenAIResponsesLLM.capabilities.supports_json_mode is True
    assert BedrockConverseLLM.capabilities.supports_tool_calling is True
    assert BedrockConverseLLM.capabilities.supports_json_mode is False
    assert OllamaLLM.capabilities.supports_tool_calling is False
    assert OllamaLLM.capabilities.supports_json_mode is True


# ---------------------------------------------------------------------------
# OpenAI-compatible provider (mocked HTTP)
# ---------------------------------------------------------------------------


def test_openai_chat_llm_requires_api_key_and_model() -> None:
    with pytest.raises(ValueError, match="api_key"):
        OpenAIChatLLM("", "gpt-4o-mini")
    with pytest.raises(ValueError, match="model"):
        OpenAIChatLLM("sk-test", "")


def test_openai_chat_llm_posts_to_configured_base_url() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["Authorization"]
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "the answer"}}]})

    llm = OpenAIChatLLM(
        "sk-test",
        "my-model",
        base_url="https://llm.example.com/",
        transport=httpx.MockTransport(handler),
    )
    answer = llm.generate("the prompt", question="q", results=[])

    assert answer == "the answer"
    assert captured["url"] == "https://llm.example.com/v1/chat/completions"
    assert captured["authorization"] == "Bearer sk-test"
    assert captured["payload"]["model"] == "my-model"
    assert captured["payload"]["messages"][-1] == {"role": "user", "content": "the prompt"}


def test_openai_chat_llm_converts_auth_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad key"})

    llm = OpenAIChatLLM("sk-bad", "my-model", transport=httpx.MockTransport(handler))
    with pytest.raises(LLMProviderError, match="OPENAI_API_KEY"):
        llm.generate("p", question="q", results=[])


def test_openai_chat_llm_converts_server_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "overloaded"})

    llm = OpenAIChatLLM("sk-test", "my-model", transport=httpx.MockTransport(handler))
    with pytest.raises(LLMProviderError, match="HTTP 503"):
        llm.generate("p", question="q", results=[])


def test_openai_chat_llm_reports_unreachable_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    llm = OpenAIChatLLM(
        "sk-test",
        "my-model",
        base_url="https://down.example.com",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(LLMProviderError, match="https://down.example.com"):
        llm.generate("p", question="q", results=[])


# ---------------------------------------------------------------------------
# OpenAI Responses provider (mocked HTTP)
# ---------------------------------------------------------------------------


def test_openai_responses_llm_requires_api_key_and_model() -> None:
    with pytest.raises(ValueError, match="api_key"):
        OpenAIResponsesLLM("", "gpt-4o-mini")
    with pytest.raises(ValueError, match="model"):
        OpenAIResponsesLLM("sk-test", "")


def test_openai_responses_llm_posts_to_responses_endpoint() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["Authorization"]
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "the answer"}],
                    }
                ]
            },
        )

    llm = OpenAIResponsesLLM(
        "sk-test",
        "my-model",
        base_url="https://api.example.com/",
        transport=httpx.MockTransport(handler),
    )
    answer = llm.generate("the prompt", question="q", results=[])

    assert answer == "the answer"
    assert captured["url"] == "https://api.example.com/v1/responses"
    assert captured["authorization"] == "Bearer sk-test"
    assert captured["payload"]["model"] == "my-model"
    assert captured["payload"]["input"] == "the prompt"
    assert captured["payload"]["instructions"] == "You produce concise, evidence-grounded analysis."


def test_openai_responses_llm_extracts_top_level_output_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"output_text": "compact answer"})

    llm = OpenAIResponsesLLM("sk-test", "my-model", transport=httpx.MockTransport(handler))

    assert llm.generate("p", question="q", results=[]) == "compact answer"


def test_openai_responses_llm_converts_auth_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad key"})

    llm = OpenAIResponsesLLM("sk-bad", "my-model", transport=httpx.MockTransport(handler))
    with pytest.raises(LLMProviderError, match="OPENAI_API_KEY"):
        llm.generate("p", question="q", results=[])


def test_openai_responses_llm_reports_malformed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "resp_123"})

    llm = OpenAIResponsesLLM("sk-test", "my-model", transport=httpx.MockTransport(handler))
    with pytest.raises(LLMProviderError, match="text output"):
        llm.generate("p", question="q", results=[])


def test_openai_responses_llm_reports_unreachable_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    llm = OpenAIResponsesLLM(
        "sk-test",
        "my-model",
        base_url="https://down.example.com",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(LLMProviderError, match="https://down.example.com"):
        llm.generate("p", question="q", results=[])


# ---------------------------------------------------------------------------
# AWS Bedrock provider (fake boto3 client)
# ---------------------------------------------------------------------------


class _FakeBedrockClientError(Exception):
    def __init__(self, code: str = "AccessDeniedException") -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class _FakeBedrockCoreError(Exception):
    pass


class _FakeBedrockExceptions:
    ClientError = _FakeBedrockClientError
    BotoCoreError = _FakeBedrockCoreError


class _FakeBedrockClient:
    def __init__(
        self,
        response: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response: dict[str, object] = response or {
            "output": {"message": {"content": [{"text": "bedrock answer"}]}}
        }
        self.error = error
        self.calls: list[dict[str, object]] = []

    def converse(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def _install_fake_botocore(monkeypatch: pytest.MonkeyPatch) -> None:
    botocore = type(sys)("botocore")
    exceptions = type(sys)("botocore.exceptions")
    exceptions.ClientError = _FakeBedrockClientError  # type: ignore[attr-defined]
    exceptions.BotoCoreError = _FakeBedrockCoreError  # type: ignore[attr-defined]
    botocore.exceptions = exceptions  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "botocore", botocore)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", exceptions)


def _make_fake_boto3(client: _FakeBedrockClient) -> ModuleType:
    module = type(sys)("boto3")
    module.calls = []  # type: ignore[attr-defined]

    def client_factory(service_name: str, *, region_name: str | None = None) -> object:
        module.calls.append((service_name, region_name))  # type: ignore[attr-defined]
        return client

    module.client = client_factory  # type: ignore[attr-defined]
    return module


def test_bedrock_llm_requires_model_and_valid_generation_options() -> None:
    with pytest.raises(ValueError, match="model"):
        BedrockConverseLLM(
            model="",
            client=_FakeBedrockClient(),
            exceptions_module=_FakeBedrockExceptions,
        )
    with pytest.raises(ValueError, match="max_tokens"):
        BedrockConverseLLM(
            max_tokens=0,
            client=_FakeBedrockClient(),
            exceptions_module=_FakeBedrockExceptions,
        )
    with pytest.raises(ValueError, match="temperature"):
        BedrockConverseLLM(
            temperature=-0.1,
            client=_FakeBedrockClient(),
            exceptions_module=_FakeBedrockExceptions,
        )


def test_bedrock_llm_calls_converse_api() -> None:
    client = _FakeBedrockClient(
        response={"output": {"message": {"content": [{"text": "Hello "}, {"text": "world."}]}}}
    )
    llm = BedrockConverseLLM(
        model="anthropic.claude-test-v1:0",
        region_name="eu-west-1",
        max_tokens=512,
        temperature=0.1,
        client=client,
        exceptions_module=_FakeBedrockExceptions,
    )
    answer = llm.generate("the prompt", question="q", results=[])

    assert answer == "Hello world."
    assert client.calls == [
        {
            "modelId": "anthropic.claude-test-v1:0",
            "system": [{"text": "You produce concise, evidence-grounded analysis."}],
            "messages": [{"role": "user", "content": [{"text": "the prompt"}]}],
            "inferenceConfig": {"maxTokens": 512, "temperature": 0.1},
        }
    ]


def test_bedrock_llm_reports_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_botocore(monkeypatch)
    monkeypatch.setitem(sys.modules, "boto3", None)
    with pytest.raises(LLMProviderError, match="poetry install --extras bedrock"):
        BedrockConverseLLM()


def test_bedrock_llm_converts_client_errors() -> None:
    llm = BedrockConverseLLM(
        client=_FakeBedrockClient(error=_FakeBedrockClientError("ValidationException")),
        exceptions_module=_FakeBedrockExceptions,
    )
    with pytest.raises(LLMProviderError, match="ValidationException"):
        llm.generate("p", question="q", results=[])


def test_bedrock_llm_converts_core_errors() -> None:
    llm = BedrockConverseLLM(
        client=_FakeBedrockClient(error=_FakeBedrockCoreError("no credentials")),
        exceptions_module=_FakeBedrockExceptions,
    )
    with pytest.raises(LLMProviderError, match="Bedrock Runtime"):
        llm.generate("p", question="q", results=[])


def test_bedrock_llm_reports_malformed_response() -> None:
    llm = BedrockConverseLLM(
        client=_FakeBedrockClient(response={"output": {"message": {"content": []}}}),
        exceptions_module=_FakeBedrockExceptions,
    )
    with pytest.raises(LLMProviderError, match="text output"):
        llm.generate("p", question="q", results=[])


# ---------------------------------------------------------------------------
# Ollama provider (mocked HTTP)
# ---------------------------------------------------------------------------


def test_ollama_llm_posts_to_chat_endpoint() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "local"}})

    llm = OllamaLLM(
        base_url="http://localhost:11434/",
        model="llama3.2",
        transport=httpx.MockTransport(handler),
    )
    answer = llm.generate("the prompt", question="q", results=[])

    assert answer == "local"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["payload"]["model"] == "llama3.2"
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["messages"][-1] == {"role": "user", "content": "the prompt"}


def test_ollama_llm_reports_unreachable_server_with_base_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    llm = OllamaLLM(base_url="http://localhost:11434", transport=httpx.MockTransport(handler))
    with pytest.raises(LLMProviderError, match="http://localhost:11434"):
        llm.generate("p", question="q", results=[])


def test_ollama_llm_converts_http_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    llm = OllamaLLM(model="missing-model", transport=httpx.MockTransport(handler))
    with pytest.raises(LLMProviderError, match="missing-model"):
        llm.generate("p", question="q", results=[])


def test_ollama_llm_requires_base_url_and_model() -> None:
    with pytest.raises(ValueError, match="base_url"):
        OllamaLLM(base_url="")
    with pytest.raises(ValueError, match="model"):
        OllamaLLM(model="")


# ---------------------------------------------------------------------------
# Anthropic provider (fake SDK module)
# ---------------------------------------------------------------------------


class _FakeTextBlock:
    def __init__(self, text: str, block_type: str = "text") -> None:
        self.type = block_type
        self.text = text


def _make_fake_anthropic(
    responder: Callable[[dict[str, object]], object],
) -> ModuleType:
    """Build a stand-in for the ``anthropic`` module with a scripted client."""

    class AuthenticationError(Exception):
        pass

    class APIConnectionError(Exception):
        pass

    class APIStatusError(Exception):
        status_code = 500

    class _Messages:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            self.calls.append(kwargs)
            return responder(kwargs)

    class Anthropic:
        def __init__(self, *, api_key: str | None = None) -> None:
            self.api_key = api_key
            self.messages = _Messages()

    module = type(sys)("anthropic")
    module.Anthropic = Anthropic  # type: ignore[attr-defined]
    module.AuthenticationError = AuthenticationError  # type: ignore[attr-defined]
    module.APIConnectionError = APIConnectionError  # type: ignore[attr-defined]
    module.APIStatusError = APIStatusError  # type: ignore[attr-defined]
    return module


def test_anthropic_llm_requires_api_key_and_model() -> None:
    with pytest.raises(ValueError, match="api_key"):
        AnthropicLLM("")
    with pytest.raises(ValueError, match="model"):
        AnthropicLLM("sk-ant-test", "")


def test_anthropic_llm_reports_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(LLMProviderError, match="poetry install --extras anthropic"):
        AnthropicLLM("sk-ant-test")


def test_anthropic_llm_extracts_text_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    def responder(kwargs: dict[str, object]) -> object:
        message = type("Message", (), {})()
        message.content = [
            _FakeTextBlock("ignored", block_type="thinking"),
            _FakeTextBlock("Hello "),
            _FakeTextBlock("world."),
        ]
        return message

    fake = _make_fake_anthropic(responder)
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    llm = AnthropicLLM("sk-ant-test")
    answer = llm.generate("the prompt", question="q", results=[])

    assert answer == "Hello world."
    assert llm.model == "claude-opus-4-8"
    client = llm._client
    assert client.api_key == "sk-ant-test"
    call = client.messages.calls[0]
    assert call["model"] == "claude-opus-4-8"
    assert call["max_tokens"] == 1024
    assert call["messages"] == [{"role": "user", "content": "the prompt"}]
    assert "system" in call


def test_anthropic_llm_converts_sdk_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    state: dict[str, Exception] = {}

    def responder(kwargs: dict[str, object]) -> object:
        raise state["error"]

    fake = _make_fake_anthropic(responder)
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    llm = AnthropicLLM("sk-ant-test")

    state["error"] = fake.AuthenticationError("bad key")
    with pytest.raises(LLMProviderError, match="ANTHROPIC_API_KEY"):
        llm.generate("p", question="q", results=[])

    state["error"] = fake.APIConnectionError("no network")
    with pytest.raises(LLMProviderError, match="connect"):
        llm.generate("p", question="q", results=[])

    status_error = fake.APIStatusError("overloaded")
    status_error.status_code = 529
    state["error"] = status_error
    with pytest.raises(LLMProviderError, match="HTTP 529"):
        llm.generate("p", question="q", results=[])


# ---------------------------------------------------------------------------
# Configuration-driven factory
# ---------------------------------------------------------------------------


def test_build_llm_defaults_to_local_deterministic_provider() -> None:
    settings = Settings(llm_provider="local")
    assert isinstance(build_llm(settings), DeterministicLLM)


def test_build_llm_openai_requires_api_key() -> None:
    settings = Settings(llm_provider="openai", OPENAI_API_KEY="")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_llm(settings)


def test_build_llm_openai_uses_configured_base_url() -> None:
    settings = Settings(
        llm_provider="openai",
        OPENAI_API_KEY="sk-test",
        OPENAI_MODEL="my-model",
        OPENAI_BASE_URL="https://llm.example.com",
    )
    llm = build_llm(settings)
    assert isinstance(llm, ResilientLLMProvider)
    assert isinstance(llm.provider, OpenAIChatLLM)
    assert llm.provider.base_url == "https://llm.example.com"
    assert llm.provider.model == "my-model"


def test_build_llm_openai_responses_requires_api_key() -> None:
    settings = Settings(llm_provider="openai_responses", OPENAI_API_KEY="")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_llm(settings)


def test_build_llm_openai_responses_uses_configured_base_url() -> None:
    settings = Settings(
        llm_provider="openai_responses",
        OPENAI_API_KEY="sk-test",
        OPENAI_MODEL="my-model",
        OPENAI_BASE_URL="https://api.example.com",
    )
    llm = build_llm(settings)
    assert isinstance(llm, ResilientLLMProvider)
    assert isinstance(llm.provider, OpenAIResponsesLLM)
    assert llm.provider.base_url == "https://api.example.com"
    assert llm.provider.model == "my-model"


def test_build_llm_anthropic_requires_api_key() -> None:
    settings = Settings(llm_provider="anthropic", ANTHROPIC_API_KEY="")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        build_llm(settings)


def test_build_llm_anthropic_uses_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _make_fake_anthropic(lambda kwargs: None)
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    settings = Settings(
        llm_provider="anthropic",
        ANTHROPIC_API_KEY="sk-ant-test",
        ANTHROPIC_MODEL="claude-haiku-4-5",
    )
    llm = build_llm(settings)
    assert isinstance(llm, ResilientLLMProvider)
    assert isinstance(llm.provider, AnthropicLLM)
    assert llm.provider.model == "claude-haiku-4-5"


def test_build_llm_bedrock_uses_configured_model_and_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeBedrockClient()
    fake_boto3 = _make_fake_boto3(client)
    _install_fake_botocore(monkeypatch)
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    settings = Settings(
        llm_provider="bedrock",
        AWS_BEDROCK_MODEL="anthropic.claude-test-v1:0",
        AWS_REGION="eu-west-1",
        AWS_BEDROCK_MAX_TOKENS=256,
        AWS_BEDROCK_TEMPERATURE=0.3,
    )
    llm = build_llm(settings)
    assert isinstance(llm, ResilientLLMProvider)
    assert isinstance(llm.provider, BedrockConverseLLM)
    assert llm.provider.model == "anthropic.claude-test-v1:0"
    assert llm.provider.region_name == "eu-west-1"
    assert llm.provider.max_tokens == 256
    assert llm.provider.temperature == 0.3
    assert fake_boto3.calls == [("bedrock-runtime", "eu-west-1")]  # type: ignore[attr-defined]


def test_build_llm_ollama_uses_configured_endpoint() -> None:
    settings = Settings(
        llm_provider="ollama",
        OLLAMA_BASE_URL="http://gpu-box:11434",
        OLLAMA_MODEL="mistral",
    )
    llm = build_llm(settings)
    assert isinstance(llm, ResilientLLMProvider)
    assert isinstance(llm.provider, OllamaLLM)
    assert llm.provider.base_url == "http://gpu-box:11434"
    assert llm.provider.model == "mistral"


def test_build_llm_can_disable_remote_resilience_wrapper() -> None:
    settings = Settings(
        llm_provider="openai",
        OPENAI_API_KEY="sk-test",
        llm_resilience_enabled=False,
    )
    llm = build_llm(settings)
    assert isinstance(llm, OpenAIChatLLM)


def test_settings_read_llm_resilience_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FEEDBACK_AGENT_LLM_RESILIENCE_ENABLED", "false")
    monkeypatch.setenv("FEEDBACK_AGENT_LLM_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("FEEDBACK_AGENT_LLM_RETRY_MAX_ATTEMPTS", "4")
    monkeypatch.setenv("FEEDBACK_AGENT_LLM_RETRY_BACKOFF_SECONDS", "0.75")
    monkeypatch.setenv("FEEDBACK_AGENT_LLM_CIRCUIT_FAILURE_THRESHOLD", "5")
    monkeypatch.setenv("FEEDBACK_AGENT_LLM_CIRCUIT_RECOVERY_SECONDS", "60")

    settings = Settings()

    assert settings.llm_resilience_enabled is False
    assert settings.llm_timeout_seconds == 12.5
    assert settings.llm_retry_max_attempts == 4
    assert settings.llm_retry_backoff_seconds == 0.75
    assert settings.llm_circuit_failure_threshold == 5
    assert settings.llm_circuit_recovery_seconds == 60


def test_settings_reject_unknown_provider_listing_options() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings(llm_provider="bogus")
    message = str(exc_info.value)
    for option in ("local", "openai", "openai_responses", "anthropic", "bedrock", "ollama"):
        assert option in message
