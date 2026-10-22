"""Build the evaluation table and the summary metrics.

The table is the main artefact of the prototype: it puts the baseline and the
trust layer side by side so that the difference between "answered anyway" and
"answered only with evidence" is visible question by question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .logging_utils import get_logger
from .types import Prediction, Question, TrustLabel

logger = get_logger(__name__)

REQUIRED_COLUMNS = [
    "Question",
    "Ground_Truth",
    "Baseline_Pred",
    "Trust_Layer_Pred",
    "Evidence_Snippet",
    "Confidence_Score",
]


def _snippet(text: str, width: int = 160) -> str:
    """Shorten a piece of evidence for display in the table."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= width:
        return collapsed
    return collapsed[: width - 1].rstrip() + "…"


def build_evaluation_frame(
    questions: Sequence[Question],
    baseline_predictions: Sequence[Optional[Prediction]],
    trust_predictions: Sequence[Optional[Prediction]],
) -> pd.DataFrame:
    """Assemble the per-question evaluation table.

    Args:
        questions: The evaluated questions.
        baseline_predictions: One baseline prediction per question.
        trust_predictions: One trust-layer prediction per question.

    Returns:
        A DataFrame with the required columns plus diagnostic columns.
    """
    if not len(questions) == len(baseline_predictions) == len(trust_predictions):
        raise ValueError("questions and predictions must have the same length")

    rows: List[Dict[str, object]] = []
    for question, baseline, trust in zip(questions, baseline_predictions, trust_predictions):
        baseline_label = baseline.label.value if baseline else TrustLabel.ABSTAIN.value
        trust_label = trust.label.value if trust else TrustLabel.ABSTAIN.value
        nli_scores = (trust.nli_scores if trust else {}) or {}
        rows.append(
            {
                "QID": question.qid,
                "Category": question.category.value,
                "Question": question.text,
                "Ground_Truth": question.gold_label.value,
                "Baseline_Pred": baseline_label,
                "Baseline_Answer": _snippet(baseline.answer, 110) if baseline and baseline.answer else "",
                "Baseline_Confidence": round(baseline.confidence, 4) if baseline else 0.0,
                "Trust_Layer_Pred": trust_label,
                "Evidence_Snippet": _snippet(trust.evidence) if trust and trust.evidence else "",
                "Confidence_Score": round(trust.confidence, 4) if trust else 0.0,
                "Similarity": round(trust.similarity, 4) if trust else 0.0,
                "NLI_Entailment": round(nli_scores.get("entailment", 0.0), 4),
                "NLI_Contradiction": round(nli_scores.get("contradiction", 0.0), 4),
                "NLI_Neutral": round(nli_scores.get("neutral", 0.0), 4),
                "Trust_Layer_Reason": trust.reason if trust else "",
                "Baseline_Correct": baseline_label == question.gold_label.value,
                "Trust_Layer_Correct": trust_label == question.gold_label.value,
                "Rationale": question.rationale,
            }
        )
    frame = pd.DataFrame(rows)
    logger.info("built evaluation table with %d rows", len(frame))
    return frame


def _accuracy(frame: pd.DataFrame, prediction_column: str) -> float:
    """Fraction of questions where a prediction column matches the truth."""
    if frame.empty:
        return 0.0
    return float((frame[prediction_column] == frame["Ground_Truth"]).mean())


def _label_macro_f1(frame: pd.DataFrame, prediction_column: str) -> float:
    """Macro F1 across the three verdict labels."""
    labels = [label.value for label in TrustLabel]
    if frame.empty:
        return 0.0
    scores: List[float] = []
    for label in labels:
        true_positive = int(
            ((frame["Ground_Truth"] == label) & (frame[prediction_column] == label)).sum()
        )
        false_positive = int(
            ((frame["Ground_Truth"] != label) & (frame[prediction_column] == label)).sum()
        )
        false_negative = int(
            ((frame["Ground_Truth"] == label) & (frame[prediction_column] != label)).sum()
        )
        if true_positive == 0 and (false_positive or false_negative):
            scores.append(0.0)
            continue
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return float(sum(scores) / len(scores))


def compute_metrics(frame: pd.DataFrame) -> Dict[str, object]:
    """Compute the headline metrics for both systems.

    Args:
        frame: The evaluation table.

    Returns:
        A mapping of metric name to value.
    """
    metrics: Dict[str, object] = {}
    for name, column in (("baseline", "Baseline_Pred"), ("trust_layer", "Trust_Layer_Pred")):
        correct = frame[column] == frame["Ground_Truth"]
        metrics[f"{name}_accuracy"] = round(float(correct.mean()), 4) if not frame.empty else 0.0
        metrics[f"{name}_macro_f1"] = round(_label_macro_f1(frame, column), 4)
        metrics[f"{name}_abstention_rate"] = (
            round(float((frame[column] == TrustLabel.ABSTAIN.value).mean()), 4) if not frame.empty else 0.0
        )
        # Asserting support the corpus does not give is the failure that matters.
        metrics[f"{name}_false_support_rate"] = (
            round(
                float(
                    (
                        (frame[column] == TrustLabel.SUPPORTED.value)
                        & (frame["Ground_Truth"] != TrustLabel.SUPPORTED.value)
                    ).mean()
                ),
                4,
            )
            if not frame.empty
            else 0.0
        )
        # Abstaining when the corpus actually does answer the question.
        metrics[f"{name}_false_abstention_rate"] = (
            round(
                float(
                    (
                        (frame[column] == TrustLabel.ABSTAIN.value)
                        & (frame["Ground_Truth"] != TrustLabel.ABSTAIN.value)
                    ).mean()
                ),
                4,
            )
            if not frame.empty
            else 0.0
        )
        metrics[f"{name}_unsupported_detection_rate"] = (
            round(
                float(
                    (
                        (frame["Ground_Truth"] == TrustLabel.UNSUPPORTED.value)
                        & (frame[column] == TrustLabel.UNSUPPORTED.value)
                    ).sum()
                    / max(
                        int((frame["Ground_Truth"] == TrustLabel.UNSUPPORTED.value).sum()), 1
                    )
                ),
                4,
            )
            if not frame.empty
            else 0.0
        )
    per_category = {}
    for category, group in frame.groupby("Category"):
        per_category[category] = {
            "n": int(len(group)),
            "baseline_accuracy": round(float((group["Baseline_Pred"] == group["Ground_Truth"]).mean()), 4),
            "trust_layer_accuracy": round(float((group["Trust_Layer_Pred"] == group["Ground_Truth"]).mean()), 4),
        }
    metrics["per_category"] = per_category
    metrics["n_questions"] = int(len(frame))
    return metrics


def format_markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table.

    Args:
        frame: The table to render.
        columns: The columns to include, in order.

    Returns:
        The markdown table as a string.
    """
    if frame.empty:
        return "_no rows_"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = str(row[column]).replace("|", "\\|").replace("\n", " ")
            cells.append(value)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


@dataclass
class Highlight:
    """A single highlighted example from the evaluation table."""

    title: str
    row: pd.Series


def select_highlights(frame: pd.DataFrame) -> List[Highlight]:
    """Pick the three examples the report calls out.

    The selection looks for: a question the trust layer got right, an
    unsupported question the trust layer caught but the baseline did not, and
    an edge-case question.

    Args:
        frame: The evaluation table.

    Returns:
        The highlights, in the order correct / unsupported / edge case.
    """
    highlights: List[Highlight] = []

    correct = frame[
        (frame["Trust_Layer_Correct"]) & (frame["Ground_Truth"] == TrustLabel.SUPPORTED.value)
    ]
    if not correct.empty:
        row = correct.iloc[0]
        highlights.append(Highlight(title="1. Correctly supported with evidence", row=row))

    caught = frame[
        (frame["Ground_Truth"] == TrustLabel.UNSUPPORTED.value)
        & (frame["Trust_Layer_Pred"] == TrustLabel.UNSUPPORTED.value)
        & (~frame["Baseline_Correct"])
    ]
    if not caught.empty:
        row = caught.iloc[0]
        highlights.append(Highlight(title="2. Unsupported answer caught", row=row))
    elif not frame[frame["Ground_Truth"] == TrustLabel.UNSUPPORTED.value].empty:
        row = frame[frame["Ground_Truth"] == TrustLabel.UNSUPPORTED.value].iloc[0]
        highlights.append(Highlight(title="2. Unsupported question (not caught)", row=row))

    edge = frame[frame["Category"] == "edge_case"]
    if not edge.empty:
        row = edge.iloc[0]
        highlights.append(Highlight(title="3. Edge case", row=row))

    return highlights


def format_highlights(highlights: Sequence[Highlight]) -> str:
    """Render the highlighted examples as a readable block of text."""
    blocks: List[str] = []
    for highlight in highlights:
        row = highlight.row
        blocks.append(
            "\n".join(
                [
                    f"**{highlight.title}** (`{row['QID']}`, category `{row['Category']}`)",
                    f"- Question: {row['Question']}",
                    f"- Ground truth: `{row['Ground_Truth']}`",
                    f"- Baseline: `{row['Baseline_Pred']}` (answer: {row['Baseline_Answer'] or 'n/a'})",
                    f"- Trust layer: `{row['Trust_Layer_Pred']}` "
                    f"(confidence {row['Confidence_Score']}, similarity {row['Similarity']})",
                    f"- Evidence: {row['Evidence_Snippet'] or 'n/a'}",
                    f"- Why: {row['Trust_Layer_Reason'] or 'n/a'}",
                ]
            )
        )
    return "\n\n".join(blocks)
