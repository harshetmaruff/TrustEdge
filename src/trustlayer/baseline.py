"""The untrustworthy baseline.

The baseline is what an assistant does when it is allowed to answer without
checking: retrieve the best matching chunk and state something anyway. It is
included to make the failure mode measurable, not because it is a good idea.
"""

from __future__ import annotations


from .chunking import split_sentences
from .logging_utils import get_logger
from .retrieval import BM25Retriever, tokenize_query
from .types import Prediction, Question, TrustLabel

logger = get_logger(__name__)


def lexical_overlap(query: str, answer: str) -> float:
    """Fraction of the query's content words that appear in the answer.

    This is the baseline's own confidence signal: it measures topical overlap
    only, which is exactly why it cannot tell a supported answer from a
    fabricated one.

    Args:
        query: The question text.
        answer: The answer text.

    Returns:
        A value in [0, 1].
    """
    query_tokens = set(tokenize_query(query))
    if not query_tokens:
        return 0.0
    answer_tokens = set(tokenize_query(answer))
    return len(query_tokens & answer_tokens) / len(query_tokens)


def select_answer_sentence(question: str, evidence: str) -> str:
    """Pick the sentence of the evidence that best matches the question.

    Args:
        question: The question text.
        evidence: The retrieved chunk.

    Returns:
        The selected sentence, or the whole chunk if it has no sentence
        boundary.
    """
    sentences = split_sentences(evidence)
    if not sentences:
        return evidence
    query_tokens = set(tokenize_query(question))
    best_sentence = sentences[0]
    best_score = -1.0
    for sentence in sentences:
        sentence_tokens = set(tokenize_query(sentence))
        score = len(query_tokens & sentence_tokens) / max(len(query_tokens), 1)
        if score > best_score:
            best_score = score
            best_sentence = sentence
    return best_sentence


def run_baseline(
    question: Question,
    retriever: BM25Retriever,
    top_k: int = 1,
) -> Prediction:
    """Answer a question from the top retrieved chunk without checking it.

    Args:
        question: The question to answer.
        retriever: The keyword retriever.
        top_k: How many chunks to retrieve.

    Returns:
        A prediction that claims SUPPORTED whenever any context was found,
        and the answer text it produced.
    """
    retrieved = retriever.search(question.text, top_k=top_k)
    if not retrieved:
        logger.warning("baseline: no context retrieved for %s", question.qid)
        return Prediction(
            label=TrustLabel.ABSTAIN,
            evidence="",
            answer="",
            confidence=0.0,
            reason="retrieval returned no context",
        )
    chunk, score = retrieved[0]
    answer = select_answer_sentence(question.text, chunk.text)
    confidence = lexical_overlap(question.text, answer)
    logger.info(
        "baseline %s: answered from %s (bm25=%.2f, overlap=%.2f)",
        question.qid,
        chunk.chunk_id,
        score,
        confidence,
    )
    return Prediction(
        label=TrustLabel.SUPPORTED,
        evidence=chunk.text,
        confidence=confidence,
        answer=answer,
        reason="answered from the top retrieved chunk without checking the evidence",
        retrieved=[chunk for chunk, _ in retrieved],
    )
