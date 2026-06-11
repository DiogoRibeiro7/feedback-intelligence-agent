"""Agent orchestration for the AI engineering showcase."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ai_engineering_showcase.citations import build_citations
from ai_engineering_showcase.llm import LLMProvider
from ai_engineering_showcase.prompts import build_grounded_prompt
from ai_engineering_showcase.retrieval import Retriever
from ai_engineering_showcase.schemas import AgentAnswer, Citation, SearchResult
from ai_engineering_showcase.telemetry import log_event


@dataclass(frozen=True)
class RouteRule:
    """Rule used by the lightweight query router."""

    name: str
    patterns: tuple[str, ...]


ROUTE_RULES = (
    RouteRule("onboarding", ("onboarding", "setup", "implementation", "checklist")),
    RouteRule("product_feedback", ("feature", "integration", "dashboard", "export", "bug")),
    RouteRule("support", ("support", "ticket", "response", "help")),
    RouteRule("risk_analysis", ("churn", "risk", "unhappy", "complain", "renewal")),
)


class FeedbackInsightAgent:
    """Evidence-grounded feedback intelligence agent."""

    def __init__(self, query_engine: Retriever, llm: LLMProvider) -> None:
        """Wire the retriever (dense, lexical, or hybrid) to the LLM provider."""
        self.query_engine = query_engine
        self.llm = llm

    def answer(self, question: str, *, top_k: int = 4) -> AgentAnswer:
        """Answer a user question using retrieved feedback as evidence."""
        if not question.strip():
            raise ValueError("question cannot be empty")
        route = self.route(question)
        results = self._retrieve(question, route=route, top_k=top_k)
        prompt = build_grounded_prompt(question, results, route=route)
        raw_response = self.llm.generate(prompt, question=question, results=results)
        parsed = self._parse_response(raw_response)
        citations = build_citations(results)
        confidence = self._confidence(results, citations)

        answer = AgentAnswer(
            question=question,
            answer=parsed["answer"],
            recommended_actions=parsed["actions"],
            citations=citations,
            route=route,
            confidence=confidence,
            diagnostics={
                "retrieved_chunks": len(results),
                "max_score": max((result.score for result in results), default=0.0),
                "min_score": min((result.score for result in results), default=0.0),
            },
        )
        log_event(
            "agent_answer_created",
            {
                "route": route,
                "top_k": top_k,
                "citations": len(citations),
                "confidence": confidence,
            },
        )
        return answer

    def route(self, question: str) -> str:
        """Classify a question into a stable route for observability and prompts."""
        lower_question = question.lower()
        for rule in ROUTE_RULES:
            if any(pattern in lower_question for pattern in rule.patterns):
                return rule.name
        return "general_insight"

    def _retrieve(self, question: str, *, route: str, top_k: int) -> list[SearchResult]:
        """Retrieve and rerank candidates with lightweight domain-aware signals.

        The first stage uses vector similarity. The second stage adds simple lexical
        and metadata signals. This mirrors a common production pattern: keep the
        retriever generic, then rerank with features that reflect the product domain.
        """
        candidate_k = max(top_k * 4, top_k)
        candidates = self.query_engine.search(question, top_k=candidate_k)
        scored: list[SearchResult] = []
        for result in candidates:
            adjusted_score = self._combined_score(question, route, result)
            scored.append(SearchResult(chunk=result.chunk, score=adjusted_score))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def _combined_score(self, question: str, route: str, result: SearchResult) -> float:
        """Combine vector, lexical, route, and metadata features."""
        question_terms = self._tokens(question)
        metadata = result.chunk.metadata
        searchable = " ".join(
            [
                result.chunk.text,
                str(metadata.get("customer_segment", "")),
                str(metadata.get("channel", "")),
                str(metadata.get("rating", "")),
            ]
        ).lower()
        overlap = 0.0
        if question_terms:
            overlap = sum(1 for term in question_terms if term in searchable) / len(question_terms)

        route_keywords = next((rule.patterns for rule in ROUTE_RULES if rule.name == route), ())
        route_hits = sum(1 for keyword in route_keywords if keyword in searchable)
        route_score = min(route_hits / 3.0, 1.0)

        segment_score = 0.0
        segment = str(metadata.get("customer_segment", "")).lower()
        if segment and segment.replace("_", " ") in question.lower():
            segment_score = 1.0

        low_rating_score = 0.0
        risk_terms = {"unhappy", "complain", "complaints", "risk", "churn", "bad", "poor"}
        if question_terms.intersection(risk_terms) and int(metadata.get("rating", 5)) <= 2:
            low_rating_score = 1.0

        return round(
            result.score
            + 0.22 * overlap
            + 0.18 * route_score
            + 0.08 * segment_score
            + 0.06 * low_rating_score,
            6,
        )

    def _tokens(self, text: str) -> set[str]:
        """Return compact query tokens used by reranking."""
        stopwords = {
            "what",
            "which",
            "where",
            "when",
            "why",
            "how",
            "are",
            "the",
            "for",
            "with",
            "and",
            "from",
            "should",
            "could",
            "would",
            "customers",
            "customer",
        }
        return {
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", text.lower())
            if token not in stopwords
        }

    def _parse_response(self, raw_response: str) -> dict[str, list[str] | str]:
        """Parse a sectioned LLM response into answer and action fields."""
        answer = self._section(raw_response, "Answer", ["Recommended actions", "Citations"])
        action_text = self._section(raw_response, "Recommended actions", ["Citations"])
        actions = [
            re.sub(r"^[-*]\s*", "", line).strip()
            for line in action_text.splitlines()
            if line.strip()
        ]
        return {
            "answer": answer.strip() or raw_response.strip(),
            "actions": actions[:5],
        }

    def _section(self, text: str, heading: str, next_headings: list[str]) -> str:
        """Extract a simple markdown-like section from text."""
        start_pattern = re.compile(rf"{re.escape(heading)}\s*:\s*", flags=re.IGNORECASE)
        start_match = start_pattern.search(text)
        if not start_match:
            return ""
        start = start_match.end()
        end = len(text)
        for next_heading in next_headings:
            next_pattern = re.compile(
                rf"\n\s*{re.escape(next_heading)}\s*:\s*", flags=re.IGNORECASE
            )
            next_match = next_pattern.search(text, pos=start)
            if next_match:
                end = min(end, next_match.start())
        return text[start:end].strip()

    def _confidence(self, results: list[SearchResult], citations: list[Citation]) -> float:
        """Compute a simple confidence score from retrieval quality and citations."""
        if not results or not citations:
            return 0.0
        top_score = max(result.score for result in results)
        citation_factor = min(len(citations) / 3.0, 1.0)
        score = max(top_score, 0.0) * 0.75 + citation_factor * 0.25
        return round(min(score, 1.0), 3)
