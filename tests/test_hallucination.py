from __future__ import annotations

import pytest

from feedback_intelligence_agent.hallucination import (
    EvidenceOverlapHallucinationChecker,
    LLMHallucinationJudge,
    evidence_overlap_score,
)
from feedback_intelligence_agent.llm import DeterministicLLM
from feedback_intelligence_agent.schemas import DocumentChunk, SearchResult


class JudgeLLM:
    capabilities = DeterministicLLM.capabilities

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, question: str, results: list[SearchResult]) -> str:
        self.prompts.append(prompt)
        assert question
        assert results
        return self.response


def make_results() -> list[SearchResult]:
    return [
        SearchResult(
            chunk=DocumentChunk(
                chunk_id="fb-1::chunk-0",
                source_id="fb-1",
                text="Onboarding checklist was unclear and setup took too long.",
                metadata={},
            ),
            score=0.9,
        )
    ]


def test_evidence_overlap_passes_supported_answer() -> None:
    result = evidence_overlap_score(
        "Onboarding checklist setup took too long.",
        ["Onboarding checklist was unclear and setup took too long."],
    )

    assert result.passed is True
    assert result.score == 1.0
    assert result.unsupported_sentences == []


def test_evidence_overlap_flags_unsupported_sentence() -> None:
    result = evidence_overlap_score(
        "Onboarding checklist was unclear. Pricing doubled overnight.",
        ["Onboarding checklist was unclear and setup took too long."],
    )

    assert result.passed is False
    assert result.score == pytest.approx(0.5)
    assert result.unsupported_sentences == ["Pricing doubled overnight."]


def test_checker_combines_overlap_with_llm_judge() -> None:
    llm = JudgeLLM('{"label": "unsupported", "confidence": 0.91, "reason": "pricing is absent"}')
    checker = EvidenceOverlapHallucinationChecker(judge=LLMHallucinationJudge(llm))

    result = checker.check(
        "Onboarding checklist was unclear.",
        question="Why is onboarding slow?",
        results=make_results(),
    )

    assert result.hallucination_detected is True
    assert result.risk == "high"
    assert result.judge is not None
    assert result.judge.label == "unsupported"
    assert result.judge.confidence == 0.91
    assert "Evidence:" in llm.prompts[0]


def test_llm_judge_invalid_response_is_uncertain() -> None:
    judge = LLMHallucinationJudge(JudgeLLM("not json"))

    verdict = judge.judge(
        "Onboarding checklist was unclear.",
        question="Why?",
        results=make_results(),
    )

    assert verdict.label == "uncertain"
    assert verdict.confidence == 0.0
