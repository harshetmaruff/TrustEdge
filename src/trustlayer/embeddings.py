"""Local sentence embeddings used for the similarity gate.

The embedding model is a small, freely downloadable sentence-transformers
checkpoint. It runs entirely on the local machine: no text leaves the process
and no hosted inference endpoint is contacted.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .logging_utils import get_logger

logger = get_logger(__name__)


class Embedder:
    """Thin wrapper around a local sentence-transformers model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """Load the model.

        Args:
            model_name: The HuggingFace model id of the embedding model.
        """
        self.model_name = model_name
        logger.info("loading embedding model %s", model_name)
        from sentence_transformers import SentenceTransformer  # imported lazily: heavy dependency

        self._model = SentenceTransformer(model_name)
        logger.info("embedding model %s ready", model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts into L2-normalised vectors.

        Args:
            texts: The texts to encode.

        Returns:
            An array of shape ``(len(texts), embedding_dim)``.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    @property
    def dimension(self) -> int:
        """The embedding dimension of the loaded model."""
        return int(self._model.get_sentence_embedding_dimension())

    def similarities(self, query: str, texts: Sequence[str]) -> np.ndarray:
        """Cosine similarity between a query and a list of candidate texts.

        Args:
            query: The query text.
            texts: The candidate texts.

        Returns:
            One similarity per candidate, in the same order.
        """
        if not texts:
            return np.zeros((0,), dtype=np.float32)
        vectors = self.encode([query, *texts])
        query_vector = vectors[0]
        candidates = vectors[1:]
        return np.clip(candidates @ query_vector, -1.0, 1.0)
