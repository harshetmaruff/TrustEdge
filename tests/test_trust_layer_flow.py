"""End-to-end tests of the trust layer using stub models.

The stubs keep the test suite fast and offline: the point is to verify the
decision flow of the layer, not the behaviour of the neural models.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pytest

from trustlayer.config import DEFAULT_CONFIG
from trustlayer.retrieval import BM25Retriever
from trustlayer.trust_layer import TrustEvaluationLayer
from trustlayer.types import Chunk, Question, QuestionCategory, TrustLabel

CHUNKS = [
    Chunk(
        chunk_id="doc#001",
        doc_id="doc",
        doc_title="Doc",
        section="1. Claims",
        text="Claims must be submitted within 14 calendar days of the delivery date.",
    ),
    Chunk(
        chunk_id="doc#002",
        doc_id="doc",
        doc_title="Doc",
        section="2. Prohibited Items",
        text="Livestock, and any consignment containing live animals, is prohibited.",
    ),
    Chunk(
        chunk_id="doc#003",
        doc_id="doc",
        doc_title="Doc",
        section="3. Zones",
        text="Express service is offered only in Zones A, B and C.",
    ),
]


class StubEmbedder:
    """Returns a fixed similarity, or one per chunk id."""

    def __init__(self, similarity_map: Dict[str, float] | None = None, default: float = 0.9) -> None:
        self.similarity_map = similarity_map or {}
        self.default = default

    def similarities(self, query: str, texts: Sequence[str]) -> np.ndarray:
        del query
        values: List[float] = []
        for text in texts:
            chunk_id = text.split()[0] if text else ""
            values.append(self.similarity_map.get(chunk_id, self.default))
        return np.asarray(values, dtype=np.float32)


class StubNLI:
    """Scores a pair with a caller supplied function."""

    def __init__(self, scorer) -> None:
        self.scorer = scorer

    def score(self, premise: str, hypothesis: str) -> Dict[str, float]:
        return self.scorer(premise, hypothesis)

    def score_pairs(self, pairs: Sequence[Tuple[str, str]]) -> List[Dict[str, float]]:
        return [self.score(premise, hypothesis) for premise, hypothesis in pairs]


def _question(qid: str, text: str) -> Question:
    return Question(
        qid=qid,
        text=text,
        category=QuestionCategory.SUPPORTED,
        gold_label=TrustLabel.SUPPORTED,
    )


def _build_layer(embedder: StubEmbedder, scorer) -> TrustEvaluationLayer:
    retriever = BM25Retriever(CHUNKS)
    return TrustEvaluationLayer(retriever, embedder, StubNLI(scorer), DEFAULT_CONFIG)


def _no_contradiction(premise: str, hypothesis: str) -> Dict[str, float]:
    return {"entailment": 0.2, "contradiction": 0.1, "neutral": 0.7}


def test_supported_when_the_evidence_contains_an_answer() -> None:
    layer = _build_layer(StubEmbedder(), _no_contradiction)
    prediction = layer.evaluate(_question("t1", "How long do I have to file a damage claim?"))
    assert prediction.label is TrustLabel.SUPPORTED
    assert "14 calendar days" in prediction.evidence
    assert prediction.confidence > 0.0


def test_unsupported_when_the_evidence_refutes_the_claim() -> None:
    def scorer(premise: str, hypothesis: str) -> Dict[str, float]:
        if "prohibited" in premise and "livestock" in hypothesis:
            return {"entailment": 0.0, "contradiction": 0.95, "neutral": 0.05}
        return _no_contradiction(premise, hypothesis)

    layer = _build_layer(StubEmbedder(), scorer)
    prediction = layer.evaluate(_question("t2", "Can a customer ship livestock?"))
    assert prediction.label is TrustLabel.UNSUPPORTED
    assert prediction.confidence == pytest.approx(0.95, abs=0.01)


def test_abstains_when_the_evidence_is_off_topic() -> None:
    embedder = StubEmbedder(similarity_map={"Claims": 0.05}, default=0.05)
    layer = _build_layer(embedder, _no_contradiction)
    prediction = layer.evaluate(_question("t3", "How long do I have to file a damage claim?"))
    assert prediction.label is TrustLabel.ABSTAIN
    assert "similarity" in prediction.reason


def test_abstains_when_the_evidence_has_no_answer_of_the_right_type() -> None:
    layer = _build_layer(StubEmbedder(), _no_contradiction)
    prediction = layer.evaluate(
        _question("t4", "How many days of storage are included in the delivery price?")
    )
    assert prediction.label is TrustLabel.ABSTAIN
    assert "does not contain an answer" in prediction.reason


def test_abstains_when_nothing_is_retrieved() -> None:
    class EmptyRetriever(BM25Retriever):
        def search(self, query: str, top_k: int = 3):
            del query, top_k
            return []

    layer = TrustEvaluationLayer(EmptyRetriever(CHUNKS), StubEmbedder(), StubNLI(_no_contradiction), DEFAULT_CONFIG)
    prediction = layer.evaluate(_question("t5", "What is the deadline?"))
    assert prediction.label is TrustLabel.ABSTAIN
    assert prediction.reason == "no context retrieved"
