"""Regenerate the fictional corpus and the labelled question set.

Usage:
    python scripts/generate_data.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trustlayer.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)


def load_generator() -> ModuleType:
    """Import ``data/generate_corpus.py`` by path, without packaging ``data``."""
    module_path = REPO_ROOT / "data" / "generate_corpus.py"
    spec = importlib.util.spec_from_file_location("trustedge_generate_corpus", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    """Write the documents and the question set to disk."""
    generator = load_generator()
    documents, questions = generator.generate_all()
    for document in documents:
        logger.info("wrote document %s", document)
    logger.info("wrote question set %s", questions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
