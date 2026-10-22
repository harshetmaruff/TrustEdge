"""Keyword retrieval baseline.

The retriever is deliberately simple (BM25 over word tokens) so that the
comparison in the evaluation table isolates the *trust* layer rather than the
quality of the search engine.
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from rank_bm25 import BM25Okapi

from .text_utils import STOPWORDS, simple_stem, tokenize
from .types import Chunk

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

__all__ = ["BM25Retriever", "STOPWORDS", "tokenize"]


def tokenize_query(text: str) -> List[str]:
    """Lowercase, stem and drop stopwords.

    Args:
        text: The text to tokenize.

    Returns:
        The content tokens, in order.
    """
    return [
        simple_stem(token)
        for token in _TOKEN_PATTERN.findall(text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


class BM25Retriever:
    """A BM25 keyword index over a fixed chunk collection."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        """Build the index.

        Args:
            chunks: The chunks to index.
        """
        if not chunks:
            raise ValueError("BM25Retriever requires at least one chunk")
        self.chunks: List[Chunk] = list(chunks)
        self._corpus: List[List[str]] = [tokenize_query(chunk.text) for chunk in self.chunks]
        self._bm25 = BM25Okapi(self._corpus)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Return the highest scoring chunks for a query.

        Args:
            query: The natural language query.
            top_k: Maximum number of chunks to return.

        Returns:
            ``(chunk, score)`` pairs, best first. Scores are BM25 scores and
            are not normalised, so they are only meaningful relative to other
            chunks retrieved by the same query.
        """
        tokens = tokenize_query(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        results: List[Tuple[Chunk, float]] = []
        for index, score in ranked[:top_k]:
            if score <= 0:
                continue
            results.append((self.chunks[index], float(score)))
        return results
