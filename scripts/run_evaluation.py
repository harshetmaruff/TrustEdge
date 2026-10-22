"""Run the baseline and the trust layer over the evaluation set.

This is the main entry point of the prototype. It writes:

    artifacts/evaluation_table.csv
    artifacts/evaluation_table.md
    artifacts/evaluation_report.md
    artifacts/metrics.json

Usage:
    python scripts/run_evaluation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trustlayer.baseline import run_baseline  # noqa: E402
from trustlayer.chunking import chunk_all  # noqa: E402
from trustlayer.config import ARTIFACTS_DIR, DEFAULT_CONFIG  # noqa: E402
from trustlayer.dataset import load_documents, load_questions  # noqa: E402
from trustlayer.embeddings import Embedder  # noqa: E402
from trustlayer.logging_utils import get_logger  # noqa: E402
from trustlayer.nli import NLIScorer  # noqa: E402
from trustlayer.reporting import (  # noqa: E402
    build_evaluation_frame,
    compute_metrics,
    format_highlights,
    format_markdown_table,
    select_highlights,
)
from trustlayer.retrieval import BM25Retriever  # noqa: E402
from trustlayer.trust_layer import TrustEvaluationLayer  # noqa: E402
from trustlayer.types import Prediction  # noqa: E402

logger = get_logger(__name__)

TABLE_COLUMNS = [
    "QID",
    "Category",
    "Question",
    "Ground_Truth",
    "Baseline_Pred",
    "Trust_Layer_Pred",
    "Evidence_Snippet",
    "Confidence_Score",
]


def main() -> int:
    """Run the full evaluation and write the artefacts."""
    config = DEFAULT_CONFIG
    documents = load_documents()
    questions = load_questions()
    logger.info("loaded %d documents and %d questions", len(documents), len(questions))

    chunks = chunk_all(documents, max_chunk_words=config.max_chunk_words)
    logger.info("built %d chunks", len(chunks))
    retriever = BM25Retriever(chunks)

    embedder = Embedder(config.embedding_model)
    nli = NLIScorer(config.nli_model)
    layer = TrustEvaluationLayer(retriever, embedder, nli, config)

    baseline_predictions: List[Prediction] = []
    trust_predictions: List[Prediction] = []
    for question in questions:
        baseline_predictions.append(run_baseline(question, retriever))
        trust_predictions.append(layer.evaluate(question))

    frame = build_evaluation_frame(questions, baseline_predictions, trust_predictions)
    metrics = compute_metrics(frame)
    highlights = select_highlights(frame)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(ARTIFACTS_DIR / "evaluation_table.csv", index=False)
    (ARTIFACTS_DIR / "evaluation_table.md").write_text(
        format_markdown_table(frame, TABLE_COLUMNS) + "\n", encoding="utf-8"
    )
    (ARTIFACTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8"
    )

    report = "\n\n".join(
        [
            "# TrustEdge evaluation report",
            "",
            "## Metrics",
            "```json",
            json.dumps(metrics, indent=2, default=str),
            "```",
            "",
            "## Evaluation table",
            format_markdown_table(frame, TABLE_COLUMNS),
            "",
            "## Highlighted examples",
            format_highlights(highlights),
            "",
        ]
    )
    (ARTIFACTS_DIR / "evaluation_report.md").write_text(report, encoding="utf-8")

    logger.info("baseline accuracy: %s", metrics["baseline_accuracy"])
    logger.info("trust layer accuracy: %s", metrics["trust_layer_accuracy"])
    logger.info("baseline false-support rate: %s", metrics["baseline_false_support_rate"])
    logger.info("trust layer false-support rate: %s", metrics["trust_layer_false_support_rate"])
    logger.info("artefacts written to %s", ARTIFACTS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
