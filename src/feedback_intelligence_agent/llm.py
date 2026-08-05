"""LLM provider abstraction.

The deterministic local provider is the default and keeps the project fully
runnable without API keys. Optional remote providers (OpenAI-compatible,
Anthropic, AWS Bedrock, Ollama) plug into the same :class:`LLMProvider`
protocol and are selected through configuration (see ``factory.build_llm``).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Protocol, cast

import httpx

from feedback_intelligence_agent.citations import build_citations, citation_marker
from feedback_intelligence_agent.schemas import SearchResult

_PROVIDER_SYSTEM_PROMPT = "You produce concise, evidence-grounded analysis."


class LLMProviderError(RuntimeError):
    """Raised when a remote LLM provider fails at runtime.

    Wraps transport failures, HTTP errors, and SDK exceptions in one project
    error type with actionable messages, so callers never need to depend on a
    specific provider's exception hierarchy.
    """


@dataclass(frozen=True)
class ProviderCapabilities:
    """Capability metadata advertised by an LLM provider."""

    supports_streaming: bool
    supports_tool_calling: bool
    supports_json_mode: bool
    max_context_tokens: int | None = None


class LLMProvider(Protocol):
    """Protocol for text generation providers."""

    capabilities: ProviderCapabilities

    def generate(self, prompt: str, *, question: str, results: list[SearchResult]) -> str:
        """Generate a response from a grounded prompt."""


class BedrockRuntimeClient(Protocol):
    """Small protocol for the Bedrock Runtime client methods used here."""

    def converse(self, **kwargs: object) -> dict[str, Any]:
        """Call AWS Bedrock Runtime's Converse API."""


class BedrockExceptionTypes(Protocol):
    """Exception classes exposed by ``botocore.exceptions``."""

    ClientError: type[Exception]
    BotoCoreError: type[Exception]


class DeterministicLLM:
    """Local deterministic fallback used for tests and demos.

    This class does not pretend to be a general LLM. It produces a structured,
    evidence-driven answer from retrieved chunks. That makes the repository fully
    runnable without network access or paid inference APIs.
    """

    capabilities = ProviderCapabilities(
        supports_streaming=True,  # The provider can simulate token streaming.
        supports_tool_calling=False,
        supports_json_mode=False,
        max_context_tokens=None,
    )

    ACTION_LIBRARY = {
        "onboarding": (
            "Create a clearer onboarding checklist with owners, milestones, and escalation rules."
        ),
        "setup": (
            "Add proactive support when implementation or setup exceeds the expected timeline."
        ),
        "dashboard": "Expose a dashboard that shows progress, blockers, and next best actions.",
        "pricing": (
            "Review pricing communication and explain value by segment "
            "before renewal conversations."
        ),
        "integration": (
            "Prioritise the most requested integrations and publish expected delivery timelines."
        ),
        "latency": (
            "Add performance monitoring and investigate slow paths affecting high-value workflows."
        ),
        "support": (
            "Improve support triage with severity labels and clearer response-time expectations."
        ),
        "export": "Improve export reliability and add clearer error messages for failed exports.",
    }

    def generate(self, prompt: str, *, question: str, results: list[SearchResult]) -> str:
        """Generate a deterministic answer based on retrieved evidence."""
        del prompt  # The deterministic implementation uses structured inputs directly.
        if not results:
            return (
                "Answer:\nI could not find enough evidence to answer this question.\n\n"
                "Recommended actions:\n- Collect more feedback related to this topic.\n\n"
                "Citations:\n"
            )

        keywords = self._top_keywords([result.chunk.text for result in results])
        cited = build_citations(results)
        sources = ", ".join(
            f"{citation.document_id} {citation_marker(citation.citation_id)}"
            for citation in cited[:3]
        )
        issue_phrase = self._issue_phrase(question, keywords)
        lead_marker = citation_marker(cited[0].citation_id)
        answer = (
            f"The strongest signal is around {issue_phrase} {lead_marker}. "
            f"The retrieved feedback points to repeated friction in {', '.join(keywords[:4])}. "
            f"The answer is grounded in feedback sources {sources}."
        )
        actions = self._actions(keywords)
        citations = [
            f"- {citation_marker(citation.citation_id)} "
            f"{citation.document_id}: {self._quote(citation.quote)}"
            for citation in cited[:4]
        ]
        return "\n".join(
            [
                "Answer:",
                answer,
                "",
                "Recommended actions:",
                *[f"- {action}" for action in actions],
                "",
                "Citations:",
                *citations,
            ]
        )

    def _top_keywords(self, texts: list[str], *, limit: int = 8) -> list[str]:
        """Extract frequent non-trivial terms from context."""
        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "was",
            "were",
            "are",
            "our",
            "but",
            "not",
            "from",
            "into",
            "have",
            "has",
            "had",
            "very",
            "when",
            "they",
            "them",
            "too",
            "than",
            "expected",
        }
        counter: Counter[str] = Counter()
        for text in texts:
            tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", text.lower())
            counter.update(token for token in tokens if token not in stopwords)
        keywords = [token for token, _ in counter.most_common(limit)]
        return keywords or ["customer feedback"]

    def _issue_phrase(self, question: str, keywords: list[str]) -> str:
        """Build a compact issue phrase for the answer sentence."""
        lower_question = question.lower()
        for keyword in keywords:
            if keyword in lower_question:
                return keyword
        return ", ".join(keywords[:2])

    def _actions(self, keywords: list[str]) -> list[str]:
        """Map extracted keywords to recommended actions."""
        selected: list[str] = []
        for keyword in keywords:
            action = self.ACTION_LIBRARY.get(keyword)
            if action and action not in selected:
                selected.append(action)
        if len(selected) < 3:
            selected.extend(
                [
                    "Group feedback by segment and quantify how often this issue appears.",
                    "Create an owner for the highest-impact friction point "
                    "and review progress weekly.",
                    "Add follow-up instrumentation so product changes "
                    "can be measured after release.",
                ]
            )
        return selected[:3]

    def _quote(self, text: str, *, max_chars: int = 140) -> str:
        """Return a compact quote from evidence text."""
        normalised = " ".join(text.split())
        if len(normalised) <= max_chars:
            return f'"{normalised}"'
        return f'"{normalised[: max_chars - 1]}…"'


class OpenAIChatLLM:
    """Optional OpenAI-compatible chat provider.

    The implementation uses raw HTTP through ``httpx`` to keep the dependency
    tree small. ``base_url`` is configurable, so any OpenAI-compatible endpoint
    (OpenAI, Azure-compatible gateways, vLLM, LiteLLM proxies, ...) works. The
    deterministic local provider remains the default and is used in CI.
    """

    capabilities = ProviderCapabilities(
        supports_streaming=True,
        supports_tool_calling=True,
        supports_json_mode=True,
        max_context_tokens=None,
    )

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = "https://api.openai.com",
        timeout_seconds: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Validate configuration and store connection parameters."""
        if not api_key:
            raise ValueError("api_key is required for OpenAIChatLLM")
        if not model:
            raise ValueError("model is required for OpenAIChatLLM")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._transport = transport

    def generate(self, prompt: str, *, question: str, results: list[SearchResult]) -> str:
        """Generate text using the OpenAI Chat Completions API."""
        del question, results
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _PROVIDER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        try:
            with httpx.Client(timeout=self.timeout_seconds, transport=self._transport) as client:
                response = client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                raise LLMProviderError(
                    "OpenAI-compatible API authentication failed (HTTP 401). Check OPENAI_API_KEY."
                ) from exc
            raise LLMProviderError(
                "OpenAI-compatible API request to "
                f"{self.base_url} failed with HTTP {exc.response.status_code}."
            ) from exc
        except httpx.TransportError as exc:
            raise LLMProviderError(
                f"Could not reach the OpenAI-compatible API at {self.base_url}. "
                "Check the base URL and your network connection."
            ) from exc
        return str(data["choices"][0]["message"]["content"])


def _extract_openai_response_text(data: dict[str, Any]) -> str:
    """Extract assistant text from an OpenAI Responses API response payload."""
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text

    output = data.get("output")
    text_parts: list[str] = []
    refusal_parts: list[str] = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
                if part.get("type") == "refusal" and isinstance(part.get("refusal"), str):
                    refusal_parts.append(part["refusal"])
    if text_parts:
        return "".join(text_parts)
    if refusal_parts:
        return "".join(refusal_parts)
    raise LLMProviderError("OpenAI Responses API response did not include text output.")


class OpenAIResponsesLLM:
    """Optional provider backed by OpenAI's Responses API.

    This provider targets ``POST /v1/responses`` and is separate from
    :class:`OpenAIChatLLM`, because many OpenAI-compatible gateways support Chat
    Completions without implementing Responses.
    """

    capabilities = ProviderCapabilities(
        supports_streaming=True,
        supports_tool_calling=True,
        supports_json_mode=True,
        max_context_tokens=None,
    )

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = "https://api.openai.com",
        timeout_seconds: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Validate configuration and store connection parameters."""
        if not api_key:
            raise ValueError("api_key is required for OpenAIResponsesLLM")
        if not model:
            raise ValueError("model is required for OpenAIResponsesLLM")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._transport = transport

    def generate(self, prompt: str, *, question: str, results: list[SearchResult]) -> str:
        """Generate text using the OpenAI Responses API."""
        del question, results
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "instructions": _PROVIDER_SYSTEM_PROMPT,
            "input": prompt,
            "temperature": 0.2,
        }
        try:
            with httpx.Client(timeout=self.timeout_seconds, transport=self._transport) as client:
                response = client.post(
                    f"{self.base_url}/v1/responses",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                raise LLMProviderError(
                    "OpenAI Responses API authentication failed (HTTP 401). "
                    "Check OPENAI_API_KEY."
                ) from exc
            raise LLMProviderError(
                "OpenAI Responses API request to "
                f"{self.base_url} failed with HTTP {exc.response.status_code}."
            ) from exc
        except httpx.TransportError as exc:
            raise LLMProviderError(
                f"Could not reach the OpenAI Responses API at {self.base_url}. "
                "Check the base URL and your network connection."
            ) from exc
        return _extract_openai_response_text(data)


def _import_anthropic() -> ModuleType:
    """Import the optional ``anthropic`` SDK, with an actionable error if absent."""
    try:
        import anthropic
    except ImportError as exc:
        raise LLMProviderError(
            "The 'anthropic' package is required for the Anthropic provider. "
            "Install it with: poetry install --extras anthropic "
            "(or: pip install anthropic)."
        ) from exc
    module: ModuleType = anthropic
    return module


def _import_boto3() -> ModuleType:
    """Import optional ``boto3`` with an actionable Bedrock setup message."""
    try:
        import boto3
    except ImportError as exc:
        raise LLMProviderError(
            "The 'boto3' package is required for the AWS Bedrock provider. "
            "Install it with: poetry install --extras bedrock "
            "(or: pip install boto3)."
        ) from exc
    module: ModuleType = boto3
    return module


def _import_botocore_exceptions() -> BedrockExceptionTypes:
    """Import botocore exception classes used by the Bedrock provider."""
    try:
        import botocore.exceptions as exceptions
    except ImportError as exc:
        raise LLMProviderError(
            "The 'botocore' package is required for the AWS Bedrock provider. "
            "Install it with: poetry install --extras bedrock "
            "(or: pip install boto3)."
        ) from exc
    return cast(BedrockExceptionTypes, exceptions)


class AnthropicLLM:
    """Optional Anthropic provider backed by the official ``anthropic`` SDK.

    The SDK is an optional dependency (poetry extra ``anthropic``) and is
    imported lazily, so the default local path never requires it.
    """

    DEFAULT_MODEL = "claude-opus-4-8"

    capabilities = ProviderCapabilities(
        supports_streaming=True,
        supports_tool_calling=True,
        supports_json_mode=True,
        max_context_tokens=1_000_000,
    )

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        *,
        max_tokens: int = 1024,
    ) -> None:
        """Validate configuration and construct the Anthropic client."""
        if not api_key:
            raise ValueError("api_key is required for AnthropicLLM")
        if not model:
            raise ValueError("model is required for AnthropicLLM")
        self._anthropic = _import_anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self._client = self._anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, *, question: str, results: list[SearchResult]) -> str:
        """Generate text using the Anthropic Messages API."""
        del question, results
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_PROVIDER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
        except self._anthropic.AuthenticationError as exc:
            raise LLMProviderError(
                "Anthropic authentication failed. Check ANTHROPIC_API_KEY."
            ) from exc
        except self._anthropic.APIConnectionError as exc:
            raise LLMProviderError(
                "Could not connect to the Anthropic API. Check your network connection."
            ) from exc
        except self._anthropic.APIStatusError as exc:
            raise LLMProviderError(
                f"Anthropic API request failed with HTTP {exc.status_code}."
            ) from exc
        # Response content is a list of typed blocks; keep only text blocks.
        parts = [block.text for block in message.content if block.type == "text"]
        return "".join(parts)


class BedrockConverseLLM:
    """Optional AWS Bedrock provider backed by the Converse API.

    Credentials are resolved by the standard AWS SDK chain: environment
    variables, shared profiles, SSO, IAM roles, or workload identity. The
    provider imports ``boto3`` lazily, so local tests and demos do not need AWS
    dependencies installed.
    """

    DEFAULT_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"

    capabilities = ProviderCapabilities(
        supports_streaming=True,
        supports_tool_calling=True,
        supports_json_mode=False,
        max_context_tokens=None,
    )

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        region_name: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        client: BedrockRuntimeClient | None = None,
        exceptions_module: BedrockExceptionTypes | None = None,
    ) -> None:
        """Validate configuration and construct a Bedrock Runtime client."""
        if not model:
            raise ValueError("model is required for BedrockConverseLLM")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1 for BedrockConverseLLM")
        if temperature < 0.0:
            raise ValueError("temperature must be non-negative for BedrockConverseLLM")
        self.model = model
        self.region_name = region_name or None
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._exceptions = exceptions_module or _import_botocore_exceptions()
        if client is not None:
            self._client = client
            return
        boto3 = _import_boto3()
        self._client = cast(
            BedrockRuntimeClient,
            boto3.client("bedrock-runtime", region_name=self.region_name),
        )

    def generate(self, prompt: str, *, question: str, results: list[SearchResult]) -> str:
        """Generate text using AWS Bedrock Runtime's Converse API."""
        del question, results
        try:
            response = self._client.converse(
                modelId=self.model,
                system=[{"text": _PROVIDER_SYSTEM_PROMPT}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={
                    "maxTokens": self.max_tokens,
                    "temperature": self.temperature,
                },
            )
        except self._exceptions.ClientError as exc:
            response = getattr(exc, "response", {})
            error = response.get("Error", {}) if isinstance(response, dict) else {}
            code = str(error.get("Code", "unknown")) if isinstance(error, dict) else "unknown"
            raise LLMProviderError(
                "AWS Bedrock Converse request failed "
                f"({code}). Check AWS credentials, region, and model access."
            ) from exc
        except self._exceptions.BotoCoreError as exc:
            raise LLMProviderError(
                "Could not call AWS Bedrock Runtime. Check AWS credentials, "
                "region configuration, and network access."
            ) from exc
        return self._extract_text(response)

    def _extract_text(self, response: dict[str, Any]) -> str:
        """Extract assistant text from a Bedrock Converse response."""
        output = response.get("output")
        if not isinstance(output, dict):
            raise LLMProviderError("AWS Bedrock response did not include output.")
        message = output.get("message")
        if not isinstance(message, dict):
            raise LLMProviderError("AWS Bedrock response did not include a message.")
        content = message.get("content")
        if not isinstance(content, list):
            raise LLMProviderError("AWS Bedrock response did not include message content.")
        parts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        if not parts:
            raise LLMProviderError("AWS Bedrock response did not include text output.")
        return "".join(parts)


class OllamaLLM:
    """Optional Ollama-compatible local chat provider.

    Talks to a local Ollama server (default ``http://localhost:11434``) through
    its ``/api/chat`` endpoint with ``httpx``. No API key is required.
    """

    capabilities = ProviderCapabilities(
        supports_streaming=True,
        supports_tool_calling=False,
        supports_json_mode=True,
        max_context_tokens=None,
    )

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout_seconds: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Validate configuration and store connection parameters."""
        if not base_url:
            raise ValueError("base_url is required for OllamaLLM")
        if not model:
            raise ValueError("model is required for OllamaLLM")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._transport = transport

    def generate(self, prompt: str, *, question: str, results: list[SearchResult]) -> str:
        """Generate text using the Ollama chat API."""
        del question, results
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": _PROVIDER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            with httpx.Client(timeout=self.timeout_seconds, transport=self._transport) as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            raise LLMProviderError(
                f"Ollama request to {self.base_url} failed with "
                f"HTTP {exc.response.status_code}. "
                f"Make sure the model '{self.model}' is pulled (ollama pull {self.model})."
            ) from exc
        except httpx.TransportError as exc:
            raise LLMProviderError(
                f"Could not reach the Ollama server at {self.base_url}. "
                "Make sure Ollama is running (ollama serve) or set OLLAMA_BASE_URL."
            ) from exc
        return str(data["message"]["content"])
