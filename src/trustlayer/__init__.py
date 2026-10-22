"""TrustEdge: a local trust-evaluation layer for retrieval-augmented answers.

The package classifies a question against a local knowledge base as
``supported``, ``unsupported`` or ``abstain``, and always returns the exact
text it used as evidence.
"""

from __future__ import annotations

from .config import DEFAULT_CONFIG, TrustConfig
from .types import (
    Chunk,
    Document,
    Prediction,
    Question,
    QuestionCategory,
    TrustLabel,
)

__all__ = [
    "DEFAULT_CONFIG",
    "TrustConfig",
    "Chunk",
    "Document",
    "Prediction",
    "Question",
    "QuestionCategory",
    "TrustLabel",
]
