"""Citation construction shared by the agent and the deterministic LLM provider.

Keeping the logic in one place guarantees that the bracketed markers emitted in
answer text (``[1]``, ``[2]``) always line up with the structured citation list
built from the actually retrieved chunks.
"""

from __future__ import annotations

from ai_engineering_showcase.schemas import Citation, SearchResult


def build_citations(results: list[SearchResult], *, max_quote_chars: int = 180) -> list[Citation]:
    """Build stable, ordered citations from retrieved chunks.

    Documents are deduplicated in retrieval order and assigned sequential
    1-based ``citation_id`` values, so the same retrieval always produces the
    same citation list. Only the first (highest-ranked) chunk of each document
    is quoted.
    """
    citations: list[Citation] = []
    seen: set[str] = set()
    for result in results:
        document_id = result.chunk.source_id
        if document_id in seen:
            continue
        seen.add(document_id)
        citations.append(
            Citation(
                citation_id=len(citations) + 1,
                document_id=document_id,
                chunk_id=result.chunk.chunk_id,
                source=str(result.chunk.metadata.get("channel", "feedback")),
                quote=summarize_evidence(result.chunk.text, max_chars=max_quote_chars),
                score=result.score,
            )
        )
    return citations


def citation_marker(citation_id: int) -> str:
    """Return the bracketed marker used in answer text, e.g. ``[1]``."""
    return f"[{citation_id}]"


def summarize_evidence(text: str, *, max_chars: int = 180) -> str:
    """Compact an evidence span for citation output."""
    normalised = " ".join(text.split())
    if len(normalised) <= max_chars:
        return normalised
    return f"{normalised[: max_chars - 1]}…"


def render_citations(citations: list[Citation]) -> str:
    """Render a human-readable citation block for CLI output."""
    if not citations:
        return "Citations: none (no evidence retrieved)"
    lines = ["Citations:"]
    lines.extend(
        f"  {citation_marker(citation.citation_id)} {citation.document_id} "
        f"({citation.source}, chunk {citation.chunk_id}, score {citation.score:.3f}): "
        f'"{citation.quote}"'
        for citation in citations
    )
    return "\n".join(lines)
