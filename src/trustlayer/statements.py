"""Turn a question into the declarative claims to test against the evidence.

The cross-encoder was trained on natural language inference pairs where the
hypothesis is a *statement*. A bare question is not a proposition, so every
question is rewritten into one or more declarative claims before it is scored.

The rewrite is rule based: no model is used, so it is deterministic,
inspectable and free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Auxiliary verbs that start a polar (yes/no) question.
_POLAR = re.compile(
    r"^(do|does|did|is|are|was|were|can|could|will|would|should|shall|may|must|has|have|had)\s+(.+)$",
    re.I,
)

# Leading subordinate clause, e.g. "If I return it after 20 days, ...".
_SUBORDINATE = re.compile(
    r"^(?:if|when|while|after|before|assuming|given|provided|supposing)\b.*?,\s*",
    re.I,
)

# A leading conditional together with the clause it governs.
_CONDITIONAL = re.compile(
    r"^(if|when|while|after|before|assuming|given|provided|supposing)\b(.*?),\s*(.+)$",
    re.I,
)

# Coordinated clauses inside a single question, e.g. "X and when Y".
_COORDINATION = re.compile(r"\s+and\s+", re.I)

# The part of a question that names what is being asked about.
_FOCUS_PATTERNS = (
    r"^what (?:is|are|was|were) (?:the )?(.*)$",
    r"^what (?:does|do|did) (.*)$",
    r"^how many (.*)$",
    r"^how much (.*)$",
    r"^how long (?:is|are|was|were)?\s*(.*)$",
    r"^how often (.*)$",
    r"^when (?:is|are|was|were|does|do|did)?\s*(.*)$",
    r"^where (?:is|are|was|were)?\s*(.*)$",
    r"^who (?:is|are|was|were)?\s*(.*)$",
    r"^which (.*)$",
    r"^why (.*)$",
)


@dataclass(frozen=True)
class Claim:
    """One claim taken from a question, with the forms used to score it.

    Attributes:
        text: The claim as the question states it.
        asserted: Whether the claim asserts something. A wh-question asks
            something and asserts nothing, so it can never be *contradicted*;
            only a polar or declarative question can.
        forms: The phrasings handed to the NLI cross-encoder.
    """

    text: str
    asserted: bool
    forms: Tuple[str, ...]


def split_claims(question: str, strip_subordinate: bool = True) -> List[str]:
    """Split a question into the individual claims it makes.

    A question can embed more than one claim: "Do I get a refund and when is it
    paid?" asserts two separate things, and scoring them together hides which
    one the evidence addresses.

    Args:
        question: The raw question text.
        strip_subordinate: Whether to drop a leading conditional clause. The
            answerability rules want it dropped (the value in a conditional is
            part of the scenario, not something the corpus has to state), while
            the contradiction rules want to keep it (the condition decides
            whether the claim is true).

    Returns:
        The individual claim fragments.
    """
    text = question.strip()
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    claims: List[str] = []
    for sentence in sentences:
        stripped = sentence.strip()
        if strip_subordinate:
            stripped = _SUBORDINATE.sub("", stripped).strip()
        stripped = stripped.rstrip("?").strip()
        parts = [part.strip() for part in _COORDINATION.split(stripped) if part.strip()]
        claims.extend(parts)
    return claims or [text]


def focus_phrase(question: str) -> str:
    """Return the part of the question that names what is being asked about."""
    cleaned = question.strip().rstrip("?").strip()
    for pattern in _FOCUS_PATTERNS:
        match = re.match(pattern, cleaned, re.I)
        if match:
            return match.group(1).strip()
    return cleaned


def _normalise_np(noun_phrase: str) -> str:
    """Tidy a noun phrase so it can be dropped into a template."""
    cleaned = noun_phrase.strip().strip(",.;:")
    if cleaned.lower().startswith("the "):
        cleaned = cleaned[4:]
    return cleaned


def _wh_forms(claim: str) -> Tuple[str, ...]:
    """Build the forms used for a question that asks rather than asserts."""
    noun_phrase = _normalise_np(focus_phrase(claim))
    if not noun_phrase:
        return (claim,)
    return (
        f"The {noun_phrase} is as described.",
        f"The text describes {noun_phrase}.",
    )


def _polar_forms(claim: str) -> Tuple[str, ...]:
    """Build the forms used for a yes/no question.

    ``The text states that ...`` was measured to separate a true claim from a
    refuted one far more reliably than the bare question or a bare assertion.
    """
    match = _POLAR.match(claim)
    if not match:
        return (claim,)
    remainder = match.group(2).strip().rstrip("?")
    return (
        f"The text states that {remainder}.",
        f"It is the case that {remainder}.",
    )


def _conditional_forms(fragment: str) -> Optional[Tuple[str, ...]]:
    """Build the forms for a claim that is governed by a leading conditional.

    The condition has to stay attached: "I get a full refund" is false in
    general, while "if I return an unopened appliance I get a full refund" is
    true for exactly the case the corpus describes.
    """
    match = _CONDITIONAL.match(fragment)
    if not match:
        return None
    condition = f"{match.group(1)}{match.group(2)}".strip()
    main = match.group(3).strip()
    polar = _POLAR.match(main)
    if polar:
        main = polar.group(2).strip()
    claim = f"{condition}, {main}."
    stated = f"The text states that {claim[0].lower() + claim[1:]}"
    return (claim, stated)


def to_claims(question: str) -> List[Claim]:
    """Convert a question into the declarative claims to test against evidence.

    Args:
        question: The raw question text.

    Returns:
        One :class:`Claim` per thing the question says. A declarative claim
        that the question simply states is kept verbatim, because the point of
        such a claim is to check whether the corpus contradicts it.
    """
    claims: List[Claim] = []
    for fragment in split_claims(question, strip_subordinate=False):
        cleaned = fragment.strip()
        if not cleaned:
            continue
        conditional = _conditional_forms(cleaned)
        if conditional is not None:
            claim = Claim(text=cleaned, asserted=True, forms=conditional)
        else:
            polar = _POLAR.match(cleaned)
            if polar:
                claim = Claim(text=cleaned, asserted=True, forms=_polar_forms(cleaned))
            elif re.match(r"^(what|which|who|whose|when|where|why|how)\b", cleaned, re.I):
                claim = Claim(text=cleaned, asserted=False, forms=_wh_forms(cleaned))
            else:
                stated = cleaned[0].lower() + cleaned[1:]
                claim = Claim(
                    text=cleaned,
                    asserted=True,
                    forms=(cleaned, f"The text states that {stated}."),
                )
        if claim.text not in [existing.text for existing in claims]:
            claims.append(claim)
    return claims
