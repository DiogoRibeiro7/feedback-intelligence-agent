"""Prompt construction utilities."""

from __future__ import annotations

from ai_engineering_showcase.citations import build_citations, citation_marker
from ai_engineering_showcase.schemas import SearchResult

SYSTEM_PROMPT = """You are a careful AI product analyst.
Use only the evidence provided in the context.
Return a concise answer, recommended actions, and cite the source IDs.
Cite evidence inline with bracketed markers such as [1] or [2] that refer to
the citation numbers of the context blocks below.
Do not cite any source that is not present in the context.
Do not invent customer facts that are not present in the context.
"""


def build_grounded_prompt(question: str, results: list[SearchResult], *, route: str) -> str:
    """Build a grounded prompt for an LLM provider."""
    marker_by_document = {
        citation.document_id: citation_marker(citation.citation_id)
        for citation in build_citations(results)
    }
    context_blocks = []
    for result in results:
        metadata = result.chunk.metadata
        context_blocks.append(
            "\n".join(
                [
                    f"citation: {marker_by_document[result.chunk.source_id]}",
                    f"source_id: {result.chunk.source_id}",
                    f"score: {result.score:.3f}",
                    f"segment: {metadata.get('customer_segment', 'unknown')}",
                    f"channel: {metadata.get('channel', 'unknown')}",
                    f"rating: {metadata.get('rating', 'unknown')}",
                    f"text: {result.chunk.text}",
                ]
            )
        )

    context = "\n\n---\n\n".join(context_blocks)
    return f"""{SYSTEM_PROMPT}

Route: {route}

Question:
{question}

Context:
{context}

Return the response with these sections:
Answer:
Recommended actions:
Citations:
"""
