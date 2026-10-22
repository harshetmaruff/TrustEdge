"""Tests for the evaluation table and the reported metrics."""

from __future__ import annotations

from typing import List

from trustlayer.reporting import (
    REQUIRED_COLUMNS,
    build_evaluation_frame,
    compute_metrics,
    format_markdown_table,
    select_highlights,
)
from trustlayer.types import Prediction, Question, QuestionCategory, TrustLabel

QUESTIONS = [
    Question("q1", "First question?", QuestionCategory.SUPPORTED, TrustLabel.SUPPORTED),
    Question("q2", "Second question?", QuestionCategory.UNSUPPORTED, TrustLabel.UNSUPPORTED),
    Question("q3", "Third question?", QuestionCategory.ABSTAIN, TrustLabel.ABSTAIN),
    Question("q4", "Fourth question?", QuestionCategory.EDGE_CASE, TrustLabel.SUPPORTED),
]


def _predictions(labels: List[TrustLabel]) -> List[Prediction]:
    return [
        Prediction(label=label, evidence="some evidence", confidence=0.9, similarity=0.8)
        for label in labels
    ]


def test_frame_has_the_required_columns() -> None:
    frame = build_evaluation_frame(
        QUESTIONS,
        _predictions([TrustLabel.SUPPORTED] * 4),
        _predictions([TrustLabel.SUPPORTED, TrustLabel.UNSUPPORTED, TrustLabel.ABSTAIN, TrustLabel.SUPPORTED]),
    )
    for column in REQUIRED_COLUMNS:
        assert column in frame.columns
    assert len(frame) == 4


def test_metrics_separate_the_two_systems() -> None:
    frame = build_evaluation_frame(
        QUESTIONS,
        _predictions([TrustLabel.SUPPORTED] * 4),
        _predictions([TrustLabel.SUPPORTED, TrustLabel.UNSUPPORTED, TrustLabel.ABSTAIN, TrustLabel.SUPPORTED]),
    )
    metrics = compute_metrics(frame)
    # The baseline claims support for every question; only the two supported
    # questions are right, so half of its answers are unsupported.
    assert metrics["baseline_accuracy"] == 0.5
    assert metrics["trust_layer_accuracy"] == 1.0
    assert metrics["baseline_false_support_rate"] == 0.5
    assert metrics["trust_layer_false_support_rate"] == 0.0
    assert metrics["per_category"]["edge_case"]["n"] == 1


def test_highlights_cover_the_three_required_cases() -> None:
    frame = build_evaluation_frame(
        QUESTIONS,
        _predictions([TrustLabel.SUPPORTED] * 4),
        _predictions([TrustLabel.SUPPORTED, TrustLabel.UNSUPPORTED, TrustLabel.ABSTAIN, TrustLabel.SUPPORTED]),
    )
    highlights = select_highlights(frame)
    titles = " ".join(highlight.title for highlight in highlights).lower()
    assert "supported" in titles
    assert "unsupported" in titles
    assert "edge case" in titles


def test_markdown_table_renders() -> None:
    frame = build_evaluation_frame(
        QUESTIONS,
        _predictions([TrustLabel.SUPPORTED] * 4),
        _predictions([TrustLabel.SUPPORTED] * 4),
    )
    table = format_markdown_table(frame, ["QID", "Ground_Truth", "Trust_Layer_Pred"])
    assert table.startswith("| QID |")
    assert len(table.splitlines()) == len(frame) + 2
