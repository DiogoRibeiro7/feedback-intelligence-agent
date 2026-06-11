"""Embedding models.

This repository intentionally includes a deterministic hashing embedding model.
It keeps the demo runnable without external infrastructure while still exposing
the same interface that a production embedding provider would use.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


class EmbeddingModel(Protocol):
    """Protocol implemented by all embedding providers."""

    dim: int

    def embed(self, texts: Sequence[str]) -> npt.NDArray[np.float64]:
        """Embed a sequence of texts into a two-dimensional NumPy array."""


class HashingEmbeddingModel:
    """Simple deterministic embedding model based on feature hashing.

    The model uses unigrams and bigrams. It is not intended to outperform modern
    neural embeddings, but it is useful for local tests, deterministic demos, and
    validating retrieval pipelines without network calls.
    """

    def __init__(self, dim: int = 512) -> None:
        if not isinstance(dim, int):
            raise TypeError("dim must be an integer")
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> npt.NDArray[np.float64]:
        """Return L2-normalised vectors for the provided texts."""
        if not isinstance(texts, Sequence):
            raise TypeError("texts must be a sequence of strings")

        matrix = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row_index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError("all texts must be strings")
            for feature in self._features(text):
                column, sign = self._hash_feature(feature)
                matrix[row_index, column] += sign

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        matrix /= norms
        return matrix

    def _features(self, text: str) -> Iterable[str]:
        """Yield unigram and bigram features from text."""
        tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
        for token in tokens:
            yield f"uni:{token}"
        for left, right in zip(tokens, tokens[1:], strict=False):
            yield f"bi:{left}_{right}"

    def _hash_feature(self, feature: str) -> tuple[int, float]:
        """Hash a feature into a vector column and signed contribution."""
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        integer = int.from_bytes(digest, byteorder="big", signed=False)
        column = integer % self.dim
        sign = 1.0 if math.floor(integer / self.dim) % 2 == 0 else -1.0
        return column, sign
