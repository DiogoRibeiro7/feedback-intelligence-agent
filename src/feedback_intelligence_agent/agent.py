"""Agent orchestration for the feedback intelligence agent."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from feedback_intelligence_agent.citations import build_citations
from feedback_intelligence_agent.guardrails import (
    SAFE_REFUSAL,
    GuardrailDecision,
    check_context,
    check_input,
    is_suspicious_context,
)
from feedback_intelligence_agent.llm import LLMProvider
from feedback_intelligence_agent.memory import (
    ConversationMemory,
    ConversationStore,
    ConversationTurn,
    DeterministicQueryRewriter,
    QueryRewrite,
    QueryRewriter,
    new_conversation_id,
)
from feedback_intelligence_agent.output_parser import parse_llm_output
from feedback_intelligence_agent.prompts import build_grounded_prompt
from feedback_intelligence_agent.reranking import DeterministicJudgeReranker, Reranker
from feedback_intelligence_agent.retrieval import Retriever
from feedback_intelligence_agent.schemas import (
    AgentAnswer,
    Citation,
    MetadataFilters,
    SearchResult,
    ToolRunRecord,
)
from feedback_intelligence_agent.telemetry import Telemetry, log_event
from feedback_intelligence_agent.tools import ToolError, ToolRegistry, ToolRouter


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

    def __init__(
        self,
        query_engine: Retriever,
        llm: LLMProvider,
        *,
        telemetry: Telemetry | None = None,
        tools: ToolRegistry | None = None,
        query_rewriter: QueryRewriter | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        """Wire the retriever (dense, lexical, or hybrid) to the LLM provider.

        Args:
            query_engine: Retriever used to gather evidence.
            llm: Text generation provider.
            telemetry: Optional telemetry emitter; defaults to a no-op instance.
            tools: Optional registry of local tools; when omitted the agent
                answers every question with the plain RAG flow.
            query_rewriter: Optional rewriter that converts follow-up questions
                into standalone questions when conversation history is passed
                to :meth:`answer`; defaults to the deterministic local rewriter.
            reranker: Optional candidate reranker; defaults to the deterministic
                local judge reranker.
        """
        self.query_engine = query_engine
        self.llm = llm
        self.telemetry = telemetry or Telemetry()
        self.tools = tools
        self.tool_router = ToolRouter(tools) if tools is not None else None
        self.query_rewriter = query_rewriter or DeterministicQueryRewriter()
        self.reranker = reranker or DeterministicJudgeReranker()

    def answer(
        self,
        question: str,
        *,
        top_k: int = 4,
        history: Sequence[ConversationTurn] | None = None,
        filters: MetadataFilters | None = None,
    ) -> AgentAnswer:
        """Answer a user question using retrieved feedback as evidence.

        Two deterministic guardrail gates protect the run: the input gate
        refuses unsafe questions before retrieval, and the context gate drops
        retrieved chunks carrying injection-style content before generation.

        When ``history`` (previous conversation turns) is supplied, follow-up
        questions are first rewritten into standalone questions using the last
        turn, and the rewritten question drives routing, retrieval, and
        generation. The full history is never sent to retrieval. Without
        ``history`` the single-turn behaviour is unchanged.
        """
        correlation_id = self.telemetry.new_correlation_id()
        input_decision = check_input(question)
        if not input_decision.allowed:
            return self._refused_answer(
                question, input_decision, top_k=top_k, correlation_id=correlation_id
            )
        rewrite: QueryRewrite | None = None
        effective_question = question
        if history:
            rewrite = self.query_rewriter.rewrite(question, history[-1])
            effective_question = rewrite.rewritten
        route = self.route(effective_question)
        run_metadata = {
            "route": route,
            "top_k": top_k,
            "question_chars": len(question),
            "guardrail_allowed": True,
        }
        if rewrite is not None:
            run_metadata["query_rewritten"] = rewrite.was_rewritten
            run_metadata["rewrite_strategy"] = rewrite.strategy
        with self.telemetry.span(
            "agent_run_started",
            "agent_run_finished",
            correlation_id=correlation_id,
            metadata=run_metadata,
        ) as run_span:
            results = self._retrieve(
                effective_question,
                route=route,
                top_k=top_k,
                correlation_id=correlation_id,
                filters=filters,
            )
            context_decision = check_context([result.chunk.text for result in results])
            context_chunks_dropped = 0
            if not context_decision.allowed:
                safe_results = [
                    result for result in results if not is_suspicious_context(result.chunk.text)
                ]
                context_chunks_dropped = len(results) - len(safe_results)
                results = safe_results
                run_span["guardrail_context_dropped"] = context_chunks_dropped
            tool_record = self._maybe_run_tool(
                effective_question, results, correlation_id=correlation_id
            )
            run_span["tool_used"] = tool_record.tool_name if tool_record is not None else None
            run_span["tool_status"] = (
                tool_record.status if tool_record is not None else "not_selected"
            )
            prompt = build_grounded_prompt(effective_question, results, route=route)
            with self.telemetry.span(
                "llm_call_started",
                "llm_call_finished",
                correlation_id=correlation_id,
                metadata={
                    "provider": getattr(self.llm, "provider_name", type(self.llm).__name__),
                    "prompt_chars": len(prompt),
                },
            ) as llm_span:
                raw_response = self.llm.generate(
                    prompt, question=effective_question, results=results
                )
                llm_span["response_chars"] = len(raw_response)
            with self.telemetry.span(
                "response_parsing_started",
                "response_parsing_finished",
                correlation_id=correlation_id,
                metadata={"response_chars": len(raw_response)},
            ) as parsing_span:
                parsed = parse_llm_output(raw_response)
                parsing_span["output_format"] = parsed.output_format
                parsing_span["repair_applied"] = parsed.repair_applied
                parsing_span["validation_error"] = parsed.validation_error
            citations = build_citations(results)
            confidence = self._confidence(results, citations)
            run_span["citations"] = len(citations)
            run_span["confidence"] = confidence
            run_span["retrieved_chunks"] = len(results)

        answer_text = parsed.payload.answer
        if tool_record is not None:
            if tool_record.status == "ok":
                answer_text = (
                    f"{answer_text}\n\n"
                    f"Tool insight ({tool_record.tool_name}): {tool_record.summary}"
                )
            else:
                answer_text = f"{answer_text}\n\nNote: {tool_record.summary}"
        diagnostics: dict[str, object] = {
            "retrieved_chunks": len(results),
            "max_score": max((result.score for result in results), default=0.0),
            "min_score": min((result.score for result in results), default=0.0),
            "guardrail_context_dropped": context_chunks_dropped,
            "metadata_filters": filters.model_dump(mode="json") if filters is not None else None,
            "reranker": type(self.reranker).__name__,
            "output_format": parsed.output_format,
            "output_repair_applied": parsed.repair_applied,
            "output_validation_error": parsed.validation_error,
            "tool_used": (
                tool_record.tool_name
                if tool_record is not None and tool_record.status == "ok"
                else None
            ),
        }
        if rewrite is not None:
            diagnostics["query_rewritten"] = rewrite.was_rewritten
            diagnostics["rewrite_strategy"] = rewrite.strategy
            diagnostics["retrieval_question"] = rewrite.rewritten
        answer = AgentAnswer(
            question=question,
            answer=answer_text,
            recommended_actions=parsed.payload.recommended_actions,
            citations=citations,
            route=route,
            confidence=confidence,
            guardrail=input_decision,
            tool_run=tool_record,
            diagnostics=diagnostics,
        )
        log_event(
            "agent_answer_created",
            {
                "route": route,
                "top_k": top_k,
                "citations": len(citations),
                "confidence": confidence,
                "tool_used": tool_record.tool_name if tool_record is not None else None,
            },
        )
        return answer

    def chat(
        self,
        message: str,
        *,
        store: ConversationStore,
        conversation_id: str | None = None,
        top_k: int = 4,
        filters: MetadataFilters | None = None,
    ) -> tuple[AgentAnswer, str]:
        """Answer a message inside a stored conversation and persist the turn.

        Loads the conversation from ``store`` (or starts a new one when
        ``conversation_id`` is ``None``), answers with the previous turns as
        context, records the new turn (user message, answer, cited document
        IDs, route, and confidence), and saves the conversation back.

        Returns:
            The agent answer and the conversation identifier (newly generated
            when none was supplied).
        """
        resolved_id = conversation_id or new_conversation_id()
        memory = store.get(resolved_id) or ConversationMemory(conversation_id=resolved_id)
        answer = self.answer(message, top_k=top_k, history=memory.turns, filters=filters)
        document_ids: list[str] = []
        for citation in answer.citations:
            if citation.document_id not in document_ids:
                document_ids.append(citation.document_id)
        turn_metadata: dict[str, object] = {
            "route": answer.route,
            "confidence": answer.confidence,
        }
        retrieval_question = answer.diagnostics.get("retrieval_question")
        if retrieval_question is not None:
            turn_metadata["retrieval_question"] = retrieval_question
        memory.add_turn(
            ConversationTurn(
                user_message=message,
                assistant_answer=answer.answer,
                retrieved_document_ids=document_ids,
                metadata=turn_metadata,
            )
        )
        store.save(memory)
        return answer, resolved_id

    def _maybe_run_tool(
        self,
        question: str,
        results: list[SearchResult],
        *,
        correlation_id: str,
    ) -> ToolRunRecord | None:
        """Route the question to a local tool and run it with telemetry.

        Returns ``None`` when no tool registry is configured or no tool route
        matches (the plain RAG flow continues unchanged). Unknown explicit
        tool requests and tool failures produce a ``refused``/``error`` record
        instead of raising, so the agent always answers gracefully.
        """
        if self.tools is None or self.tool_router is None:
            return None
        selection = self.tool_router.select(question)
        if selection.status == "no_tool":
            return None
        if selection.status == "unknown_tool":
            self.telemetry.emit(
                "tool_run_refused",
                correlation_id=correlation_id,
                metadata={"requested_tool": selection.tool_name, "reason": selection.reason},
            )
            log_event("tool_request_refused", {"requested_tool": selection.tool_name})
            return ToolRunRecord(
                tool_name=selection.tool_name or "unknown",
                status="refused",
                summary=(
                    f"The requested tool '{selection.tool_name}' is not available. "
                    f"Available tools: {', '.join(self.tools.names())}. "
                    "Answered from retrieved feedback instead."
                ),
            )
        tool = self.tools.get(selection.tool_name or "")
        if tool is None:  # Defensive: the router only selects registered tools.
            return None
        payload = tool.build_payload(question, results)
        try:
            with self.telemetry.span(
                "tool_run_started",
                "tool_run_finished",
                correlation_id=correlation_id,
                metadata={"tool": tool.name, "reason": selection.reason},
            ) as tool_span:
                output = tool.execute(payload)
                tool_span["output_fields"] = len(output.model_dump())
        except ToolError as exc:
            log_event("tool_run_failed", {"tool": tool.name, "error": str(exc)})
            return ToolRunRecord(
                tool_name=tool.name,
                status="error",
                summary=f"Tool '{tool.name}' failed and was skipped: {exc}",
            )
        log_event("tool_run_succeeded", {"tool": tool.name})
        return ToolRunRecord(
            tool_name=tool.name,
            status="ok",
            summary=output.render(),
            output=output.model_dump(),
        )

    def _refused_answer(
        self,
        question: str,
        decision: GuardrailDecision,
        *,
        top_k: int,
        correlation_id: str,
    ) -> AgentAnswer:
        """Build a safe refusal answer for a question blocked by guardrails."""
        with self.telemetry.span(
            "agent_run_started",
            "agent_run_finished",
            correlation_id=correlation_id,
            metadata={
                "route": "guardrail_refusal",
                "top_k": top_k,
                "question_chars": len(question),
                "guardrail_allowed": False,
                "guardrail_severity": decision.severity,
                "guardrail_reason": decision.reason,
            },
        ) as run_span:
            run_span["citations"] = 0
            run_span["confidence"] = 0.0
        log_event(
            "agent_query_blocked",
            {"reason": decision.reason, "severity": decision.severity},
        )
        return AgentAnswer(
            question=question,
            answer=decision.suggested_response or SAFE_REFUSAL,
            recommended_actions=[],
            citations=[],
            route="guardrail_refusal",
            confidence=0.0,
            guardrail=decision,
            diagnostics={"retrieved_chunks": 0, "max_score": 0.0, "min_score": 0.0},
        )

    def route(self, question: str) -> str:
        """Classify a question into a stable route for observability and prompts."""
        lower_question = question.lower()
        for rule in ROUTE_RULES:
            if any(pattern in lower_question for pattern in rule.patterns):
                return rule.name
        return "general_insight"

    def _retrieve(
        self,
        question: str,
        *,
        route: str,
        top_k: int,
        correlation_id: str | None = None,
        filters: MetadataFilters | None = None,
    ) -> list[SearchResult]:
        """Retrieve and rerank candidates with lightweight domain-aware signals.

        The first stage uses vector similarity. The second stage adds simple lexical
        and metadata signals. This mirrors a common production pattern: keep the
        retriever generic, then rerank with features that reflect the product domain.
        """
        candidate_k = max(top_k * 4, top_k)
        correlation_id = correlation_id or self.telemetry.new_correlation_id()
        with self.telemetry.span(
            "retrieval_started",
            "retrieval_finished",
            correlation_id=correlation_id,
            metadata={
                "retriever": type(self.query_engine).__name__,
                "route": route,
                "top_k": top_k,
                "candidate_k": candidate_k,
                "metadata_filters": (
                    filters.model_dump(mode="json") if filters is not None else None
                ),
            },
        ) as span:
            candidates = self.query_engine.search(question, top_k=candidate_k)
            unfiltered_count = len(candidates)
            if filters is not None:
                candidates = [
                    result
                    for result in candidates
                    if self._matches_metadata_filters(result.chunk.metadata, filters)
                ]
            span["metadata_filtered_out"] = unfiltered_count - len(candidates)
            route_keywords = next((rule.patterns for rule in ROUTE_RULES if rule.name == route), ())
            ranked = self.reranker.rerank(
                question,
                candidates,
                top_k=top_k,
                route_keywords=route_keywords,
            )
            span["reranker"] = type(self.reranker).__name__
            span["results"] = len(ranked)
            span["max_score"] = max((result.score for result in ranked), default=0.0)
            span["min_score"] = min((result.score for result in ranked), default=0.0)
        return ranked

    def _matches_metadata_filters(
        self, metadata: dict[str, object], filters: MetadataFilters
    ) -> bool:
        """Return whether chunk metadata satisfies all requested filters."""
        if filters.customer_segment is not None:
            segment = str(metadata.get("customer_segment", "")).lower()
            if segment != filters.customer_segment.lower():
                return False
        if filters.channel is not None and str(metadata.get("channel", "")) != filters.channel:
            return False

        rating = self._metadata_rating(metadata)
        if filters.min_rating is not None and rating < filters.min_rating:
            return False
        if filters.max_rating is not None and rating > filters.max_rating:
            return False

        created_at = self._metadata_datetime(metadata.get("created_at"))
        if filters.created_after is not None:
            created_after = self._comparable_datetime(filters.created_after)
            if created_at is None or created_at < created_after:
                return False
        if filters.created_before is not None:
            created_before = self._comparable_datetime(filters.created_before)
            if created_at is None or created_at > created_before:
                return False
        return True

    def _metadata_rating(self, metadata: dict[str, object]) -> int:
        """Read a metadata rating, defaulting to the neutral upper bound."""
        value = metadata.get("rating", 5)
        if isinstance(value, int):
            return value
        if not isinstance(value, str):
            return 5
        try:
            return int(value)
        except ValueError:
            return 5

    def _metadata_datetime(self, value: object) -> datetime | None:
        """Parse an ISO datetime value from chunk metadata."""
        if isinstance(value, datetime):
            return value
        if not isinstance(value, str) or not value:
            return None
        try:
            return self._comparable_datetime(datetime.fromisoformat(value))
        except ValueError:
            return None

    def _comparable_datetime(self, value: datetime) -> datetime:
        """Normalize datetimes so metadata and API filters compare reliably."""
        if value.tzinfo is None:
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def _confidence(self, results: list[SearchResult], citations: list[Citation]) -> float:
        """Compute a simple confidence score from retrieval quality and citations."""
        if not results or not citations:
            return 0.0
        top_score = max(result.score for result in results)
        citation_factor = min(len(citations) / 3.0, 1.0)
        score = max(top_score, 0.0) * 0.75 + citation_factor * 0.25
        return round(min(score, 1.0), 3)
