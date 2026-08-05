"""Text chunking utilities."""

from __future__ import annotations

from collections.abc import Iterable

from feedback_intelligence_agent.privacy import redact_pii
from feedback_intelligence_agent.schemas import DocumentChunk, FeedbackRecord


def chunk_text(text: str, *, max_words: int = 80, overlap_words: int = 16) -> list[str]:
    """Split text into overlapping word chunks.

    Args:
        text: Input text to split.
        max_words: Maximum words per chunk.
        overlap_words: Number of words to reuse between adjacent chunks.

    Returns:
        Non-empty chunks.
    """
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")

    words = text.split()
    if not words:
        return []
    if len(words) <= max_words:
        return [" ".join(words)]

    chunks: list[str] = []
    start = 0
    step = max_words - overlap_words
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step

    return chunks


def feedback_to_chunks(
    records: Iterable[FeedbackRecord],
    *,
    max_words: int = 80,
    overlap_words: int = 16,
) -> list[DocumentChunk]:
    """Convert validated feedback records into searchable document chunks.

    Args:
        records: Validated feedback records to chunk.
        max_words: Maximum words per chunk (see :func:`chunk_text`).
        overlap_words: Words shared between adjacent chunks.
    """
    chunks: list[DocumentChunk] = []
    for record in records:
        text_for_storage = redact_pii(record.text)
        chunk_texts = chunk_text(
            text_for_storage,
            max_words=max_words,
            overlap_words=overlap_words,
        )
        for index, text in enumerate(chunk_texts):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{record.feedback_id}::chunk-{index}",
                    source_id=record.feedback_id,
                    text=text,
                    metadata={
                        "customer_segment": record.customer_segment,
                        "channel": record.channel.value,
                        "rating": record.rating,
                        "created_at": record.created_at.isoformat(),
                    },
                )
            )
    return chunks
