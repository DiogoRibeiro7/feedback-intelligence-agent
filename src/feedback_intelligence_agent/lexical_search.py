"""Lexical BM25 search implemented from scratch.

The retriever complements dense hashing-embedding search: BM25 rewards exact
term matches (product names, error codes, integration names) that vector
similarity can dilute, while staying fully local and dependency-free.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from feedback_intelligence_agent.query_expansion import QueryExpander
from feedback_intelligence_agent.schemas import DocumentChunk, SearchResult

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Return lowercased alphanumeric tokens used for lexical matching."""
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


class BM25Retriever:
    """In-memory Okapi BM25 retriever over document chunks.

    The index is built once at construction time. ``texts`` allows indexing an
    enriched representation (for example chunk text plus metadata) while the
    returned :class:`SearchResult` objects keep the original chunks.
    """

    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        *,
        texts: Sequence[str] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        query_expander: QueryExpander | None = None,
    ) -> None:
        """Build the BM25 index from chunks (or enriched ``texts``) once."""
        if texts is not None and len(texts) != len(chunks):
            raise ValueError("texts must match chunks one-to-one")
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1")
        self.k1 = k1
        self.b = b
        self.query_expander = query_expander
        self._chunks = list(chunks)
        corpus_texts = texts if texts is not None else [chunk.text for chunk in self._chunks]
        corpus = [tokenize(text) for text in corpus_texts]
        self._term_frequencies: list[Counter[str]] = [Counter(tokens) for tokens in corpus]
        self._doc_lengths = [len(tokens) for tokens in corpus]
        self._avg_doc_length = sum(self._doc_lengths) / len(corpus) if corpus else 0.0
        document_frequencies: Counter[str] = Counter()
        for term_frequency in self._term_frequencies:
            document_frequencies.update(term_frequency.keys())
        total_docs = len(corpus)
        self._idf = {
            term: math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
            for term, df in document_frequencies.items()
        }

    @property
    def size(self) -> int:
        """Number of indexed chunks."""
        return len(self._chunks)

    def search(self, question: str, *, top_k: int = 4) -> list[SearchResult]:
        """Return chunks with a positive BM25 score for the question, best first."""
        if not question.strip():
            raise ValueError("question cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self._avg_doc_length == 0.0:
            return []

        retrieval_question = (
            self.query_expander.expand(question).expanded_query
            if self.query_expander is not None
            else question
        )
        query_terms = tokenize(retrieval_question)
        scored: list[tuple[float, int]] = []
        for index, (term_frequency, doc_length) in enumerate(
            zip(self._term_frequencies, self._doc_lengths, strict=True)
        ):
            score = self._score(query_terms, term_frequency, doc_length)
            if score > 0.0:
                scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            SearchResult(chunk=self._chunks[index], score=round(score, 6))
            for score, index in scored[:top_k]
        ]

    def _score(
        self, query_terms: Sequence[str], term_frequency: Counter[str], doc_length: int
    ) -> float:
        """Compute the BM25 score of one document for the query terms."""
        score = 0.0
        length_norm = 1.0 - self.b + self.b * doc_length / self._avg_doc_length
        for term in query_terms:
            frequency = term_frequency.get(term, 0)
            if frequency == 0:
                continue
            idf = self._idf.get(term, 0.0)
            score += idf * frequency * (self.k1 + 1.0) / (frequency + self.k1 * length_norm)
        return score
