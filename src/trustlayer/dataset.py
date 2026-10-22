"""Loading of the local corpus and the labelled evaluation set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .chunking import parse_document
from .config import DOCUMENTS_DIR, QUESTIONS_PATH
from .types import Document, Question, QuestionCategory, TrustLabel


def load_documents(documents_dir: Path = DOCUMENTS_DIR) -> List[Document]:
    """Load every ``*.txt`` document from disk.

    The document title is taken from the leading ``#`` header so that the
    file name and the displayed title cannot drift apart.

    Args:
        documents_dir: Directory holding the generated ``.txt`` documents.

    Returns:
        The parsed documents, ordered by file name.
    """
    documents: List[Document] = []
    for path in sorted(documents_dir.glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        title = next(
            (line[2:].strip() for line in raw.splitlines() if line.startswith("# ")),
            path.stem,
        )
        documents.append(parse_document(path.stem, title, raw))
    return documents


def load_questions(path: Path = QUESTIONS_PATH) -> List[Question]:
    """Load the labelled question set from JSON Lines.

    Args:
        path: Path to ``questions.jsonl``.

    Returns:
        The questions in file order.
    """
    questions: List[Question] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            questions.append(
                Question(
                    qid=payload["qid"],
                    text=payload["question"],
                    category=QuestionCategory(payload["category"]),
                    gold_label=TrustLabel(payload["gold_label"]),
                    rationale=payload.get("rationale", ""),
                    reference_evidence=payload.get("reference_evidence", ""),
                )
            )
    return questions
