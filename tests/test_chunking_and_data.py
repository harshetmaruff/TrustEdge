"""Tests for chunking and corpus loading."""

from __future__ import annotations

from trustlayer.chunking import chunk_all, parse_document, split_paragraphs, split_sentences
from trustlayer.config import DOCUMENTS_DIR
from trustlayer.dataset import load_documents, load_questions

RAW = """# Example Title

## 1. First Section

First paragraph sentence one. First paragraph sentence two.

Second paragraph sentence three.
"""


def test_parse_document_extracts_sections() -> None:
    document = parse_document("example", "Example Title", RAW)
    assert "1. First Section" in document.sections
    assert "First paragraph sentence one." in document.sections["1. First Section"]


def test_split_paragraphs_and_sentences() -> None:
    document = parse_document("example", "Example Title", RAW)
    paragraphs = split_paragraphs(document.sections["1. First Section"])
    assert len(paragraphs) == 2
    sentences = split_sentences(paragraphs[0])
    assert sentences == ["First paragraph sentence one.", "First paragraph sentence two."]


def test_chunks_never_cross_paragraphs() -> None:
    document = parse_document("example", "Example Title", RAW)
    chunks = chunk_all([document])
    assert len(chunks) == 2
    for chunk in chunks:
        assert "First paragraph" not in chunk.text or "Second paragraph" not in chunk.text


def test_corpus_is_fictional_and_complete() -> None:
    documents = load_documents(DOCUMENTS_DIR)
    assert len(documents) == 4
    questions = load_questions()
    assert 15 <= len(questions) <= 25
    categories = {question.category.value for question in questions}
    assert categories == {"supported", "unsupported", "abstain", "edge_case"}
    # Every question must carry a scorable gold label.
    assert all(question.gold_label.value in {"supported", "unsupported", "abstain"} for question in questions)
