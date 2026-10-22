"""Sweep the trust thresholds and report the trade-off.

The two thresholds that matter are:

* ``min_similarity`` - the evidence floor. Raising it makes the layer refuse
  more often (fewer hallucinations, more false abstentions).
* ``contradiction_threshold`` - how strongly the evidence has to refute a
  claim before the layer calls it unsupported. Raising it makes the layer
  more cautious about accusing the corpus of contradiction.

Usage:
    python scripts/threshold_sweep.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trustlayer.baseline import run_baseline  # noqa: E402
from trustlayer.chunking import chunk_all  # noqa: E402
from trustlayer.config import ARTIFACTS_DIR, DEFAULT_CONFIG  # noqa: E402
from trustlayer.dataset import load_documents, load_questions  # noqa: E402
from trustlayer.embeddings import Embedder  # noqa: E402
from trustlayer.logging_utils import get_logger  # noqa: E402
from trustlayer.nli import NLIScorer  # noqa: E402
from trustlayer.reporting import build_evaluation_frame, compute_metrics  # noqa: E402
from trustlayer.retrieval import BM25Retriever  # noqa: E402
from trustlayer.trust_layer import TrustEvaluationLayer  # noqa: E402

logger = get_logger(__name__)

SIMILARITY_GRID = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
CONTRADICTION_GRID = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def main() -> int:
    """Run the sweep and write the result to ``artifacts/threshold_sweep.csv``."""
    config = DEFAULT_CONFIG
    documents = load_documents()
    questions = load_questions()
    chunks = chunk_all(documents, max_chunk_words=config.max_chunk_words)
    retriever = BM25Retriever(chunks)
    embedder = Embedder(config.embedding_model)
    nli = NLIScorer(config.nli_model)
    baseline_predictions = [run_baseline(question, retriever) for question in questions]

    rows: List[Dict[str, object]] = []
    for min_similarity in SIMILARITY_GRID:
        for contradiction_threshold in CONTRADICTION_GRID:
            layer = TrustEvaluationLayer(
                retriever,
                embedder,
                nli,
                replace(
                    config,
                    min_similarity=min_similarity,
                    contradiction_threshold=contradiction_threshold,
                ),
            )
            trust_predictions = [layer.evaluate(question) for question in questions]
            frame = build_evaluation_frame(questions, baseline_predictions, trust_predictions)
            metrics = compute_metrics(frame)
            rows.append(
                {
                    "min_similarity": min_similarity,
                    "contradiction_threshold": contradiction_threshold,
                    "accuracy": metrics["trust_layer_accuracy"],
                    "false_support_rate": metrics["trust_layer_false_support_rate"],
                    "false_abstention_rate": metrics["trust_layer_false_abstention_rate"],
                    "abstention_rate": metrics["trust_layer_abstention_rate"],
                    "unsupported_detection_rate": metrics["trust_layer_unsupported_detection_rate"],
                }
            )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    frame = pd.DataFrame(rows)
    frame.to_csv(ARTIFACTS_DIR / "threshold_sweep.csv", index=False)
    logger.info("threshold sweep written to %s", ARTIFACTS_DIR / "threshold_sweep.csv")

    for min_similarity in SIMILARITY_GRID:
        subset = frame[frame["min_similarity"] == min_similarity]
        best = subset.loc[subset["accuracy"].idxmax()]
        logger.info(
            "min_similarity=%.2f -> best contradiction_threshold=%.2f accuracy=%.2f "
            "false_support=%.2f false_abstention=%.2f",
            min_similarity,
            best["contradiction_threshold"],
            best["accuracy"],
            best["false_support_rate"],
            best["false_abstention_rate"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
