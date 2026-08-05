"""Hallucination checks for evidence-grounded answers.

The default check is deterministic and local: answer sentences must have enough
content-token overlap with retrieved evidence. An optional LLM-as-judge wrapper
can be injected for deployments that want semantic review on top of the lexical
gate without changing the agent orchestration.
"""

from __future__ import annotations

import json
import re
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from feedback_intelligence_agent.llm import LLMProvider
from feedback_intelligence_agent.schemas import SearchResult

HallucinationRisk = Literal["low", "medium", "high"]
JudgeLabel = Literal["supported", "unsupported", "uncertain"]

_STOPWORDS = {
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
    "answer",
    "grounded",
    "feedback",
    "sources",
    "retrieved",
    "points",
    "repeated",
    "friction",
    "strongest",
    "signal",
    "around",
    "citation",
    "citations",
}


class EvidenceOverlap(BaseModel):
    """Lexical evidence-overlap result for one answer."""

    score: float = Field(ge=0.0, le=1.0)
    supported_sentences: int = Field(ge=0)
    total_sentences: int = Field(ge=0)
    unsupported_sentences: list[str] = Field(default_factory=list)
    sentence_threshold: float = Field(ge=0.0, le=1.0)
    pass_threshold: float = Field(ge=0.0, le=1.0)
    passed: bool


class JudgeVerdict(BaseModel):
    """Optional LLM-as-judge verdict for answer support."""

    label: JudgeLabel
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    provider: str


class HallucinationCheck(BaseModel):
    """Combined hallucination check exposed in diagnostics and evaluation."""

    risk: HallucinationRisk
    hallucination_detected: bool
    evidence_overlap: EvidenceOverlap
    judge: JudgeVerdict | None = None


class HallucinationJudge(Protocol):
    """Protocol for semantic answer-support judges."""

    def judge(
        self,
        answer: str,
        *,
        question: str,
        results: list[SearchResult],
    ) -> JudgeVerdict:
        """Return a semantic support verdict for an answer."""


class EvidenceOverlapHallucinationChecker:
    """Combine deterministic evidence overlap with an optional semantic judge."""

    def __init__(
        self,
        *,
        sentence_threshold: float = 0.5,
        pass_threshold: float = 0.75,
        judge: HallucinationJudge | None = None,
    ) -> None:
        """Configure lexical thresholds and an optional LLM-as-judge."""
        if not 0.0 <= sentence_threshold <= 1.0:
            raise ValueError("sentence_threshold must be between 0 and 1")
        if not 0.0 <= pass_threshold <= 1.0:
            raise ValueError("pass_threshold must be between 0 and 1")
        self.sentence_threshold = sentence_threshold
        self.pass_threshold = pass_threshold
        self.judge = judge

    def check(
        self,
        answer: str,
        *,
        question: str,
        results: list[SearchResult],
    ) -> HallucinationCheck:
        """Run overlap and optional judge checks for an answer."""
        overlap = evidence_overlap_score(
            answer,
            [result.chunk.text for result in results],
            sentence_threshold=self.sentence_threshold,
            pass_threshold=self.pass_threshold,
        )
        verdict = (
            self.judge.judge(answer, question=question, results=results) if self.judge else None
        )
        risk = _risk(overlap, verdict)
        return HallucinationCheck(
            risk=risk,
            hallucination_detected=risk == "high",
            evidence_overlap=overlap,
            judge=verdict,
        )


class LLMHallucinationJudge:
    """LLM-as-judge wrapper that expects a small JSON verdict."""

    def __init__(self, llm: LLMProvider) -> None:
        """Use an existing LLM provider for semantic support checks."""
        self.llm = llm

    def judge(
        self,
        answer: str,
        *,
        question: str,
        results: list[SearchResult],
    ) -> JudgeVerdict:
        """Ask the provider whether the answer is supported by retrieved evidence."""
        response = self.llm.generate(
            _judge_prompt(answer, results),
            question=question,
            results=results,
        )
        return _parse_judge_response(response, provider=type(self.llm).__name__)


def evidence_overlap_score(
    answer: str,
    context_texts: list[str],
    *,
    sentence_threshold: float = 0.5,
    pass_threshold: float = 0.75,
) -> EvidenceOverlap:
    """Score how many answer sentences are lexically supported by evidence."""
    sentences = _sentences(answer)
    if not sentences or not context_texts:
        return EvidenceOverlap(
            score=0.0,
            supported_sentences=0,
            total_sentences=len(sentences),
            unsupported_sentences=sentences,
            sentence_threshold=sentence_threshold,
            pass_threshold=pass_threshold,
            passed=False,
        )

    context_tokens = _content_tokens(" ".join(context_texts))
    supported = 0
    unsupported: list[str] = []
    scored = 0
    for sentence in sentences:
        tokens = _content_tokens(sentence)
        if not tokens:
            continue
        scored += 1
        overlap = len(tokens.intersection(context_tokens)) / len(tokens)
        if overlap >= sentence_threshold:
            supported += 1
        else:
            unsupported.append(sentence)
    score = supported / scored if scored else 0.0
    return EvidenceOverlap(
        score=round(score, 4),
        supported_sentences=supported,
        total_sentences=scored,
        unsupported_sentences=unsupported,
        sentence_threshold=sentence_threshold,
        pass_threshold=pass_threshold,
        passed=score >= pass_threshold,
    )


def _risk(overlap: EvidenceOverlap, verdict: JudgeVerdict | None) -> HallucinationRisk:
    """Combine lexical and optional judge signals into a compact risk label."""
    if verdict is not None and verdict.label == "unsupported":
        return "high"
    if not overlap.passed:
        return "high"
    if verdict is not None and verdict.label == "uncertain":
        return "medium"
    if overlap.unsupported_sentences:
        return "medium"
    return "low"


def _judge_prompt(answer: str, results: list[SearchResult]) -> str:
    """Render a compact support-check prompt for LLM-as-judge providers."""
    evidence = "\n".join(
        f"- {result.chunk.source_id}: {' '.join(result.chunk.text.split())}" for result in results
    )
    return (
        "Decide whether the answer is fully supported by the evidence. "
        "Return only JSON with label, confidence, and reason. "
        "label must be one of: supported, unsupported, uncertain.\n\n"
        f"Evidence:\n{evidence}\n\nAnswer:\n{answer}"
    )


def _parse_judge_response(response: str, *, provider: str) -> JudgeVerdict:
    """Parse a JSON judge response with conservative fallback behavior."""
    payload = _extract_json_object(response)
    raw_label = str(payload.get("label", "uncertain")).lower()
    if raw_label == "supported":
        label: JudgeLabel = "supported"
    elif raw_label == "unsupported":
        label = "unsupported"
    else:
        label = "uncertain"
    confidence = payload.get("confidence", 0.0)
    if not isinstance(confidence, int | float):
        confidence = 0.0
    reason = str(payload.get("reason", "judge did not provide a reason"))
    return JudgeVerdict(
        label=label,
        confidence=max(0.0, min(float(confidence), 1.0)),
        reason=reason,
        provider=provider,
    )


def _extract_json_object(response: str) -> dict[str, object]:
    """Extract the first JSON object from an LLM response."""
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if match is None:
            return {}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return payload if isinstance(payload, dict) else {}


def _sentences(text: str) -> list[str]:
    """Split text into answer sentences, excluding empty fragments."""
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def _content_tokens(text: str) -> set[str]:
    """Lowercased alphanumeric tokens with stopwords and citation IDs removed."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
    return {token for token in tokens if token not in _STOPWORDS and not token.startswith("fb-")}
