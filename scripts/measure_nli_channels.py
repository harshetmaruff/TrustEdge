"""Measure how well the local NLI model separates the three verdicts.

AGENTS.md describes a trust layer that maps ENTAILMENT -> SUPPORTED,
CONTRADICTION -> UNSUPPORTED and NEUTRAL -> ABSTAIN. This script measures
whether the local cross-encoder can actually carry that mapping on this
corpus, so the decision rule in ``trust_layer.py`` can be justified with
numbers instead of an assumption.

Two channels are measured for every question:

* ``contradiction`` over the claims the question *asserts* (polar and
  declarative claims). This is the channel the layer uses to detect a
  refuted claim.
* ``entailment`` over the question's descriptive forms. This is the channel
  AGENTS.md would use to confirm a supported answer.

Each channel is measured twice: over every retrieved chunk, and over only the
chunks that pass the relevance floor. The difference is what the floor
protects against.

Usage:
    python scripts/measure_nli_channels.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trustlayer.chunking import chunk_all  # noqa: E402
from trustlayer.config import ARTIFACTS_DIR, DEFAULT_CONFIG  # noqa: E402
from trustlayer.dataset import load_documents, load_questions  # noqa: E402
from trustlayer.embeddings import Embedder  # noqa: E402
from trustlayer.logging_utils import get_logger  # noqa: E402
from trustlayer.nli import NLIScorer  # noqa: E402
from trustlayer.retrieval import BM25Retriever  # noqa: E402
from trustlayer.statements import to_claims  # noqa: E402

logger = get_logger(__name__)


def main() -> int:
    """Measure both channels and write the result to ``artifacts/nli_channels.json``."""
    config = DEFAULT_CONFIG
    documents = load_documents()
    questions = load_questions()
    chunks = chunk_all(documents, max_chunk_words=config.max_chunk_words)
    retriever = BM25Retriever(chunks)
    embedder = Embedder(config.embedding_model)
    nli = NLIScorer(config.nli_model)

    rows: List[Dict[str, object]] = []
    for question in questions:
        retrieved = retriever.search(question.text, top_k=config.nli_candidates)
        claims = to_claims(question.text)
        similarities = embedder.similarities(
            question.text, [chunk.text for chunk, _ in retrieved]
        )
        # The pair order has to match the order the scores are consumed in.
        pairs = [
            (chunk.text, form)
            for claim in claims
            for form in claim.forms
            for chunk, _ in retrieved
        ]
        scores = nli.score_pairs(pairs)
        index = 0
        contradiction = 0.0
        contradiction_form = ""
        contradiction_unfiltered = 0.0
        entailment = 0.0
        entailment_form = ""
        for claim in claims:
            for form in claim.forms:
                for position, (chunk, _) in enumerate(retrieved):
                    score = scores[index]
                    index += 1
                    if claim.asserted and score["contradiction"] > contradiction_unfiltered:
                        contradiction_unfiltered = score["contradiction"]
                    if not claim.asserted and score["entailment"] > entailment:
                        entailment = score["entailment"]
                        entailment_form = f"{chunk.chunk_id}: {form}"
                    # The layer only scores chunks that pass the relevance floor.
                    if float(similarities[position]) < config.min_similarity:
                        continue
                    if claim.asserted and score["contradiction"] > contradiction:
                        contradiction = score["contradiction"]
                        contradiction_form = f"{chunk.chunk_id}: {form}"
        rows.append(
            {
                "qid": question.qid,
                "gold_label": question.gold_label.value,
                "category": question.category.value,
                "contradiction": round(contradiction, 4),
                "contradiction_unfiltered": round(contradiction_unfiltered, 4),
                "contradiction_form": contradiction_form,
                "entailment": round(entailment, 4),
                "entailment_form": entailment_form,
            }
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "nli_channels.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )

    logger.info("channel separation (max probability per gold label):")
    for label in ("supported", "unsupported", "abstain"):
        subset = [row for row in rows if row["gold_label"] == label]
        if not subset:
            continue
        contradictions = [row["contradiction"] for row in subset]
        unfiltered = [row["contradiction_unfiltered"] for row in subset]
        entailments = [row["entailment"] for row in subset]
        logger.info(
            "  %-12s n=%-3d contradiction %.3f-%.3f (unfiltered %.3f-%.3f) | entailment %.3f-%.3f",
            label,
            len(subset),
            min(contradictions),
            max(contradictions),
            min(unfiltered),
            max(unfiltered),
            min(entailments),
            max(entailments),
        )
    logger.info("measurements written to %s", ARTIFACTS_DIR / "nli_channels.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
