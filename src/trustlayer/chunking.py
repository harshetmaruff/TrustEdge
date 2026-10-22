"""Split source documents into retrievable chunks.

Chunks keep their section heading so that the evidence shown to a reviewer
always carries the context it came from, and chunk boundaries never cut a
sentence in half.
"""

from __future__ import annotations

import re
from typing import List, Sequence

from .types import Chunk, Document

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_SPLIT = re.compile(r"\s+")


def split_sentences(text: str) -> List[str]:
    """Split a paragraph into sentences.

    Args:
        text: The paragraph to split.

    Returns:
        The sentences in the paragraph, in order.
    """
    stripped = text.strip()
    if not stripped:
        return []
    return [part.strip() for part in _SENTENCE_SPLIT.split(stripped) if part.strip()]


def split_paragraphs(text: str) -> List[str]:
    """Split a document body into paragraphs on blank lines."""
    blocks = re.split(r"\n\s*\n", text.strip())
    return [block.strip() for block in blocks if block.strip()]


def parse_document(doc_id: str, title: str, raw_text: str) -> Document:
    """Parse a raw document into a :class:`Document` with per-section text."""
    sections = {}
    current = "Overview"
    buffer: List[str] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            if buffer:
                sections[current] = "\n".join(buffer).strip()
            current = stripped[3:].strip()
            buffer = []
        elif stripped.startswith("# "):
            continue
        else:
            buffer.append(line)
    if buffer:
        sections[current] = "\n".join(buffer).strip()
    return Document(doc_id=doc_id, title=title, text=raw_text, sections=sections)


def chunk_document(document: Document, max_chunk_words: int = 110) -> List[Chunk]:
    """Turn a document into retrieval chunks.

    Paragraphs are the primary boundary: a chunk never spans two paragraphs,
    because a paragraph in this corpus is a self-contained statement. If a
    paragraph is longer than ``max_chunk_words`` it is split on sentence
    boundaries into consecutive chunks.

    Args:
        document: The document to chunk.
        max_chunk_words: Soft upper bound on chunk length in words.

    Returns:
        The chunks of the document, in reading order.
    """
    chunks: List[Chunk] = []
    counter = 0
    for section, body in document.sections.items():
        for paragraph in split_paragraphs(body):
            sentences = split_sentences(paragraph)
            current: List[str] = []
            current_words = 0
            for sentence in sentences:
                sentence_words = len(sentence.split())
                if current and current_words + sentence_words > max_chunk_words:
                    counter += 1
                    chunks.append(
                        Chunk(
                            chunk_id=f"{document.doc_id}#{counter:03d}",
                            doc_id=document.doc_id,
                            doc_title=document.title,
                            section=section,
                            text=" ".join(current),
                        )
                    )
                    current = []
                    current_words = 0
                current.append(sentence)
                current_words += sentence_words
            if current:
                counter += 1
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}#{counter:03d}",
                        doc_id=document.doc_id,
                        doc_title=document.title,
                        section=section,
                        text=" ".join(current),
                    )
                )
    return chunks


def chunk_all(documents: Sequence[Document], max_chunk_words: int = 110) -> List[Chunk]:
    """Chunk every document and return the combined chunk list."""
    chunks: List[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, max_chunk_words=max_chunk_words))
    return chunks
