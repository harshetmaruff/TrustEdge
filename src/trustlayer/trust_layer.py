"""The trust-evaluation layer.

Decision flow for a single question:

1. Retrieve candidate chunks with the keyword retriever.
2. If nothing was retrieved -> ABSTAIN (no evidence at all).
3. Score every candidate against the question with local embeddings. If the
   best similarity is below the configured floor -> ABSTAIN (the retrieved
   text is about something else).
4. Check that the evidence actually contains an answer of the type the
   question asks for. If it does not -> ABSTAIN (the corpus is silent).
5. Run the local NLI cross-encoder over the claims the question makes:
      contradiction >= threshold -> UNSUPPORTED
      otherwise                    -> SUPPORTED

The exact chunk used as evidence is always returned with the verdict.

Why a separate answerability check exists: the cross-encoder is reliable at
recognising a contradiction but weak at confirming that a passage *answers* a
question, because an answer-agnostic hypothesis is not a proposition the model
can confirm. The answerability check supplies that half of the trust logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .answerability import Answerability, check_answerability
from .baseline import lexical_overlap, select_answer_sentence
from .config import DEFAULT_CONFIG, TrustConfig
from .embeddings import Embedder
from .logging_utils import get_logger
from .nli import NLIScorer
from .retrieval import BM25Retriever
from .statements import to_claims
from .types import Chunk, Prediction, Question, TrustLabel

logger = get_logger(__name__)

# A question that asks about an obligation is answered by an obligation.
_OBLIGATION_QUESTION = re.compile(
    r"\b(must|have to|need to|required to|am i allowed|can i|may i|do i have to)\b", re.I
)
_OBLIGATION_STATEMENT = re.compile(
    r"\b(must|have to|needs? to|required|should|shall|not allowed|subject to)\b",
    re.I,
)


def _obligation_match(question: str, answer: str) -> float:
    """Score how well an answer statement matches the kind of question asked.

    Returns:
        ``1.0`` when the question asks for an obligation and the answer states
        one, ``-1.0`` when the question asks for an obligation and the answer
        does not, and ``0.0`` when the question is not about an obligation.
    """
    if not _OBLIGATION_QUESTION.search(question):
        return 0.0
    return 1.0 if _OBLIGATION_STATEMENT.search(answer) else -1.0


@dataclass
class _ScoredClaim:
    """One scored (chunk, form) pair produced during evaluation."""

    chunk: Chunk
    claim: str
    scores: Dict[str, float]
    asserted: bool = True
    similarity: float = 0.0


class TrustEvaluationLayer:
    """Evaluate a question against retrieved evidence and refuse when unsure."""

    def __init__(
        self,
        retriever: BM25Retriever,
        embedder: Embedder,
        nli: NLIScorer,
        config: TrustConfig = DEFAULT_CONFIG,
    ) -> None:
        """Wire the layer together.

        Args:
            retriever: The keyword retriever.
            embedder: The local embedding model.
            nli: The local NLI cross-encoder.
            config: The trust policy thresholds.
        """
        self.retriever = retriever
        self.embedder = embedder
        self.nli = nli
        self.config = config

    def evaluate(self, question: Question) -> Prediction:
        """Evaluate one question.

        Args:
            question: The question to evaluate.

        Returns:
            The verdict, the evidence used and the confidence behind it.
        """
        retrieved = self.retriever.search(question.text, top_k=self.config.nli_candidates)
        if not retrieved:
            logger.info("trust layer %s: ABSTAIN (no context retrieved)", question.qid)
            return Prediction(
                label=TrustLabel.ABSTAIN,
                evidence="",
                confidence=0.0,
                reason="no context retrieved",
            )

        chunks = [chunk for chunk, _ in retrieved]
        similarities = self.embedder.similarities(question.text, [chunk.text for chunk in chunks])
        order = sorted(range(len(chunks)), key=lambda index: float(similarities[index]), reverse=True)
        best_similarity = float(similarities[order[0]])

        if best_similarity < self.config.min_similarity:
            logger.info(
                "trust layer %s: ABSTAIN (similarity %.3f < %.3f)",
                question.qid,
                best_similarity,
                self.config.min_similarity,
            )
            return Prediction(
                label=TrustLabel.ABSTAIN,
                evidence=chunks[order[0]].text,
                confidence=best_similarity,
                similarity=best_similarity,
                reason=(
                    f"the retrieved text is not about the question "
                    f"(similarity {best_similarity:.3f} < {self.config.min_similarity:.3f})"
                ),
                retrieved=chunks,
            )

        candidates: List[Tuple[Chunk, float]] = [
            (chunks[index], float(similarities[index]))
            for index in order
            if float(similarities[index]) >= self.config.min_similarity
        ]

        scored = self._score_claims(question, candidates)
        contradiction = self._best(scored, "contradiction", asserted_only=True)
        neutral = self._best(scored, "neutral")
        entailment = self._best(scored, "entailment")

        if contradiction[1] >= self.config.contradiction_threshold:
            logger.info(
                "trust layer %s: UNSUPPORTED (contradiction %.3f)",
                question.qid,
                contradiction[1],
            )
            return Prediction(
                label=TrustLabel.UNSUPPORTED,
                evidence=contradiction[0].chunk.text,
                confidence=contradiction[1],
                similarity=best_similarity,
                nli_scores=dict(contradiction[0].scores),
                answer=contradiction[0].claim,
                reason=(
                    f"the retrieved evidence contradicts the claim "
                    f"({contradiction[1]:.3f} contradiction)"
                ),
                retrieved=chunks,
            )

        answerable = self._find_answerable(question, candidates)
        if answerable is None:
            logger.info("trust layer %s: ABSTAIN (no answer in the evidence)", question.qid)
            return Prediction(
                label=TrustLabel.ABSTAIN,
                evidence=candidates[0][0].text,
                confidence=neutral[1],
                similarity=best_similarity,
                nli_scores=dict(neutral[0].scores) if neutral[0] else {},
                reason=(
                    "the retrieved evidence is on topic but does not contain an answer "
                    f"of the expected type ({neutral[1]:.3f} neutral)"
                ),
                retrieved=chunks,
            )

        chunk, _ = answerable
        answer = select_answer_sentence(question.text, chunk.text)
        alignment = float(self.embedder.similarities(question.text, [answer])[0])
        logger.info(
            "trust layer %s: SUPPORTED (answer alignment %.3f, similarity %.3f)",
            question.qid,
            alignment,
            best_similarity,
        )
        return Prediction(
            label=TrustLabel.SUPPORTED,
            evidence=chunk.text,
            confidence=alignment,
            similarity=best_similarity,
            nli_scores={
                "entailment": entailment[1],
                "contradiction": contradiction[1],
                "neutral": neutral[1],
            },
            answer=answer,
            reason=(
                "the evidence contains an answer of the expected type and is a close "
                f"match for the question ({alignment:.3f} answer alignment)"
            ),
            retrieved=chunks,
        )

    def _score_claims(
        self, question: Question, candidates: Sequence[Tuple[Chunk, float]]
    ) -> List[_ScoredClaim]:
        """Score every claim the question makes against every candidate chunk."""
        claims = to_claims(question.text)
        if not claims or not candidates:
            return []
        pairs: List[Tuple[str, str]] = []
        owners: List[Tuple[Chunk, str, bool, float]] = []
        for chunk, similarity in candidates:
            for claim in claims:
                for form in claim.forms:
                    pairs.append((chunk.text, form))
                    owners.append((chunk, form, claim.asserted, similarity))
        scores = self.nli.score_pairs(pairs)
        return [
            _ScoredClaim(chunk=chunk, claim=form, scores=score, asserted=asserted, similarity=similarity)
            for (chunk, form, asserted, similarity), score in zip(owners, scores)
        ]

    @staticmethod
    def _best(
        scored: Sequence[_ScoredClaim], label: str, asserted_only: bool = False
    ) -> Tuple[Optional[_ScoredClaim], float]:
        """Return the highest scoring pair for a label and its probability."""
        pool = [item for item in scored if item.asserted] if asserted_only else list(scored)
        if not pool:
            return None, 0.0
        best = max(pool, key=lambda item: item.scores.get(label, 0.0))
        return best, best.scores.get(label, 0.0)

    def _find_answerable(
        self, question: Question, candidates: Sequence[Tuple[Chunk, float]]
    ) -> Optional[Tuple[Chunk, float]]:
        """Return the best candidate chunk that contains an answer, if any.

        Several chunks can pass the answerability check - a pricing paragraph
        can state a charge without stating the charge the question asked for.
        Two signals rank them:

        1. An obligation question ("how long do I have to file a claim") is
           answered by an obligation statement ("claims must be submitted
           within 14 days"), not by a passage that merely mentions the topic.
        2. Otherwise the answer sentence that covers the most of the
           question's content terms wins.

        The cited evidence is therefore the passage that best matches the
        question among the answerable candidates. See the Limitations section
        of the README: this is not always the passage that literally states
        the answer.
        """
        best: Optional[Tuple[Chunk, float]] = None
        best_score = (-1.0, -1.0)
        for chunk, similarity in candidates:
            result: Answerability = check_answerability(question.text, chunk.text)
            if not result.answerable:
                continue
            answer = select_answer_sentence(question.text, chunk.text)
            score = (
                _obligation_match(question.text, answer),
                lexical_overlap(question.text, answer),
            )
            logger.debug(
                "question %r is answerable from %s (score %.2f/%.2f): %s",
                question.text,
                chunk.chunk_id,
                score[0],
                score[1],
                result.reason,
            )
            if score > best_score:
                best_score = score
                best = (chunk, similarity)
        return best
