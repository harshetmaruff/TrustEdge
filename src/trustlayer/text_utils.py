"""Small text utilities shared by the retriever and the answerability checks.

Keeping the stemmer in one place matters: the keyword retriever and the
answerability rules have to agree on what "the same word" means, otherwise a
question about a "claim" silently fails to match a passage about "claims".
"""

from __future__ import annotations

import re
from typing import List

WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")

STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for from by with
    is are was were be been being am do does did doing have has had having will would shall
    should can could may might must it its as into about over under again further once here
    there when where why how all any both each few more most other some such no nor not only
    own same so too very s t just don now i you he she we they me him her us them my your his
    its our their what which who whom whose
    """.split()
)

# Plurals whose stem is not a simple truncation.
_IRREGULAR_PLURALS = {
    "policies": "policy",
    "companies": "company",
    "deliveries": "delivery",
    "queries": "query",
    "analyses": "analysis",
    "bases": "basis",
    "processes": "process",
    "statuses": "status",
    "warranties": "warranty",
    "entries": "entry",
    "shipments": "shipment",
    "payments": "payment",
    "days": "day",
}


def simple_stem(word: str) -> str:
    """Reduce a word to a crude stem for overlap comparisons.

    Args:
        word: The word to stem.

    Returns:
        The stem. The stemmer is deliberately naive: it only handles the
        plural and common suffix cases that this corpus contains.
    """
    lowered = word.lower()
    if lowered in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[lowered]
    if lowered.endswith("ies") and len(lowered) > 4:
        return lowered[:-3] + "y"
    if lowered.endswith("sses") and len(lowered) > 4:
        return lowered[:-2]
    if lowered.endswith("s") and not lowered.endswith("ss") and len(lowered) > 3:
        return lowered[:-1]
    for suffix in ("ing", "ed"):
        if lowered.endswith(suffix) and len(lowered) - len(suffix) >= 3:
            return lowered[: -len(suffix)]
    return lowered


def tokenize(text: str) -> List[str]:
    """Split text into lowercase word tokens."""
    return [match.group(0).lower() for match in WORD.finditer(text)]
