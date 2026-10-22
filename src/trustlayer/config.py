"""Central configuration for the TrustEdge trust-evaluation layer.

All tunable thresholds live here so that a reader can see the trust policy in
one place instead of hunting through the pipeline modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
QUESTIONS_PATH = DATA_DIR / "questions.jsonl"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


@dataclass(frozen=True)
class TrustConfig:
    """Thresholds and model identifiers that define the trust policy.

    Attributes:
        embedding_model: Local sentence-transformers model used for the
            similarity gate. Must be a freely downloadable open model.
        nli_model: Local cross-encoder used for entailment / contradiction.
        min_similarity: Minimum cosine similarity between the question and the
            retrieved chunk for the layer to trust the chunk as evidence.
            Below this the layer abstains before the NLI model is even run.
        entailment_threshold: Minimum entailment probability required to
            declare a question SUPPORTED.
        contradiction_threshold: Minimum contradiction probability required to
            declare a question UNSUPPORTED.
        neutral_threshold: Minimum neutral probability required to declare a
            question ABSTAIN when neither of the above is decisive.
        nli_candidates: How many retrieved chunks are passed to the NLI model.
            The chunk with the highest non-neutral NLI score wins, which lets a
            lower-ranked but decisive chunk override the top-ranked one.
        max_chunk_words: Upper bound on chunk size, in words.
    """

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    min_similarity: float = 0.35
    entailment_threshold: float = 0.50
    contradiction_threshold: float = 0.50
    neutral_threshold: float = 0.50
    nli_candidates: int = 6
    max_chunk_words: int = 110


DEFAULT_CONFIG = TrustConfig()
