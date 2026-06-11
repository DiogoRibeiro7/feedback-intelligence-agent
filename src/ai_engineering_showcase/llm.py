"""LLM provider abstraction."""

from __future__ import annotations

import re
from collections import Counter
from typing import Protocol

import httpx

from ai_engineering_showcase.schemas import SearchResult


class LLMProvider(Protocol):
    """Protocol for text generation providers."""

    def generate(self, prompt: str, *, question: str, results: list[SearchResult]) -> str:
        """Generate a response from a grounded prompt."""


class DeterministicLLM:
    """Local deterministic fallback used for tests and demos.

    This class does not pretend to be a general LLM. It produces a structured,
    evidence-driven answer from retrieved chunks. That makes the repository fully
    runnable without network access or paid inference APIs.
    """

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
        sources = ", ".join(result.chunk.source_id for result in results[:3])
        issue_phrase = self._issue_phrase(question, keywords)
        answer = (
            f"The strongest signal is around {issue_phrase}. "
            f"The retrieved feedback points to repeated friction in {', '.join(keywords[:4])}. "
            f"The answer is grounded in feedback sources {sources}."
        )
        actions = self._actions(keywords)
        citations = [
            f"- {result.chunk.source_id}: {self._quote(result.chunk.text)}"
            for result in results[:4]
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

    The implementation uses raw HTTP through `httpx` to keep the dependency tree
    small. The deterministic local provider remains the default and is used in CI.
    """

    def __init__(self, api_key: str, model: str, *, timeout_seconds: float = 30.0) -> None:
        if not api_key:
            raise ValueError("api_key is required for OpenAIChatLLM")
        if not model:
            raise ValueError("model is required for OpenAIChatLLM")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

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
                {"role": "system", "content": "You produce concise, evidence-grounded analysis."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        return str(data["choices"][0]["message"]["content"])
