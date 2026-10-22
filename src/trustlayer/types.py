"""Shared data types for the trust-evaluation layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class TrustLabel(str, Enum):
    """The three verdicts the trust layer can return."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    ABSTAIN = "abstain"


class QuestionCategory(str, Enum):
    """The dataset category a question was authored for.

    ``EDGE_CASE`` questions still carry a ``gold_label`` so that they are
    scored like any other question; the category only describes *why* the
    question was written.
    """

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    ABSTAIN = "abstain"
    EDGE_CASE = "edge_case"


@dataclass(frozen=True)
class Chunk:
    """A retrievable span of a source document."""

    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    text: str

    def citation(self) -> str:
        """Return a human-readable citation for this chunk."""
        return f"{self.doc_title} / {self.section}"


@dataclass(frozen=True)
class Document:
    """A parsed source document."""

    doc_id: str
    title: str
    text: str
    sections: Dict[str, str]


@dataclass(frozen=True)
class Question:
    """A labelled evaluation question."""

    qid: str
    text: str
    category: QuestionCategory
    gold_label: TrustLabel
    rationale: str = ""
    reference_evidence: str = ""


@dataclass
class Prediction:
    """The output of a single system (baseline or trust layer) for a question.

    Attributes:
        label: The verdict.
        evidence: The exact text used as evidence for the verdict.
        confidence: Confidence in the verdict, in [0, 1].
        similarity: Highest question-to-evidence similarity, when available.
        nli_scores: The NLI probability distribution, when available.
        answer: The generated answer text, when available.
        reason: Human-readable justification for the verdict.
        retrieved: The chunks that were considered, best first.
    """

    label: TrustLabel
    evidence: str = ""
    confidence: float = 0.0
    similarity: float = 0.0
    nli_scores: Dict[str, float] = field(default_factory=dict)
    answer: str = ""
    reason: str = ""
    retrieved: List[Chunk] = field(default_factory=list)
