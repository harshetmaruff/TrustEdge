"""Answerability checks.

The NLI cross-encoder is good at spotting a *contradiction* but weak at
recognising that a passage contains the answer to a question: an
answer-agnostic hypothesis is not a proposition the model can confirm, so the
entailment probability stays high for any passage that merely shares the
question's vocabulary. Measured on this corpus, see the README.

This module supplies the missing half of the trust logic: it decides whether
the retrieved text actually contains an answer *of the type the question asks
for*. Every check here is rule based and inspectable. It does not claim to be
a semantic model; it is a conservative filter that has to be right about
abstaining far more often than it is right about answering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from .logging_utils import get_logger
from .statements import split_claims
from .text_utils import STOPWORDS, simple_stem, tokenize

logger = get_logger(__name__)

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")
_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")

# Words that follow a number but do not count the thing the question asked about.
_NON_COUNT_UNITS = {
    "credits",
    "percent",
    "per",
    "and",
    "or",
    "of",
    "the",
    "a",
    "an",
    "to",
    "in",
    "for",
    "working",
    "calendar",
    "consecutive",
    "times",
    "hz",
}

_TIME_UNITS = {
    "second",
    "seconds",
    "minute",
    "minutes",
    "hour",
    "hours",
    "day",
    "days",
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",
    "working",
    "calendar",
}

_FREQUENCY_WORDS = {
    "each",
    "every",
    "per",
    "monthly",
    "weekly",
    "daily",
    "annually",
    "quarterly",
    "period",
    "periods",
    "cycle",
    "cycles",
}

_TIME_EXPRESSIONS = re.compile(
    r"\b(?:"
    r"january|february|march|april|may|june|july|august|september|october|november|december"
    r"|\d{1,2}(?:st|nd|rd|th)|(?:19|20)\d{2}|\b\d{1,2}:\d{2}\b"
    r"|working days?|calendar days?|months?|years?|weeks?|hours?|minutes?"
    r")\b",
    re.I,
)

_ROLE_WORDS = {
    "candidate",
    "candidates",
    "fellow",
    "fellows",
    "adviser",
    "advisers",
    "assessor",
    "assessors",
    "panel",
    "agent",
    "agents",
    "council",
    "office",
    "supervisor",
    "researcher",
    "researchers",
    "visitor",
    "visitors",
}

_PLACE_WORDS = {
    "zone",
    "zones",
    "laboratory",
    "laboratories",
    "lab",
    "centre",
    "center",
    "region",
    "regions",
    "corridor",
    "corridors",
    "city",
    "coastal",
    "inland",
    "metro",
}

# Measured quantities and the units that can answer them.
_UNIT_FAMILIES: Dict[str, Set[str]] = {
    "temperature": {"degree", "degrees", "celsius", "kelvin", "centigrade"},
    "percentage": {"percent", "percentage"},
    "accuracy": {"percent", "percentage"},
    "error": {"percent", "percentage"},
    "rate": {"hz", "hertz"},
    "time": {"day", "days", "week", "weeks", "month", "months", "year", "years", "hour", "hours", "minute", "minutes"},
    "duration": {"day", "days", "week", "weeks", "month", "months", "year", "years", "hour", "hours", "minute", "minutes"},
    "charge": {"credit", "credits"},
    "cost": {"credit", "credits"},
    "fee": {"credit", "credits"},
    "price": {"credit", "credits"},
    "value": {"credit", "credits", "percent"},
    "weight": {"kg", "gram", "grams", "g"},
    "length": {"mm", "cm", "metre", "metres", "meter", "meters", "inch", "inches"},
    "voltage": {"volt", "volts"},
    "current": {"amp", "amps", "ampere", "amperes"},
    "speed": {"km", "mph", "knot", "knots"},
    "pressure": {"bar", "psi", "pascal"},
    "flow": {"litre", "litres", "liter", "liters"},
}

# Words that must not be treated as an entity answer on their own.
_TITLE_NOISE = {
    "doctoral",
    "visiting",
    "microgrant",
    "fellowship",
    "graduate",
    "postgraduate",
    "senior",
    "junior",
    "main",
    "primary",
    "general",
    "standard",
    "express",
    "extended",
    "indoor",
    "outdoor",
    "refurbished",
    "remote",
    "local",
    "national",
    "international",
}


class AnswerType(str, Enum):
    """The kind of value a question is asking for."""

    COUNT = "count"
    AMOUNT = "amount"
    DURATION = "duration"
    FREQUENCY = "frequency"
    TIME = "time"
    PERSON = "person"
    PLACE = "place"
    YES_NO = "yes_no"
    ENTITY = "entity"
    DESCRIPTION = "description"


@dataclass(frozen=True)
class Answerability:
    """The outcome of an answerability check."""

    answerable: bool
    answer_type: AnswerType
    reason: str

    def __bool__(self) -> bool:
        return self.answerable


def focus_phrase(question: str) -> str:
    """Return the part of the question that names what is being asked about."""
    cleaned = question.strip().rstrip("?").strip()
    patterns = (
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
    for pattern in patterns:
        match = re.match(pattern, cleaned, re.I)
        if match:
            return match.group(1).strip()
    return cleaned


def focus_terms(question: str) -> Set[str]:
    """Return the stemmed content words of a question."""
    extra = {
        "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
        "does", "do", "did", "is", "are", "was", "were", "can", "could", "will",
        "would", "should", "must", "many", "much", "often", "long",
    }
    return {
        simple_stem(token)
        for token in tokenize(question)
        if token not in STOPWORDS and token not in extra and len(token) > 2
    }


def detect_answer_type(question: str) -> AnswerType:
    """Infer what kind of value the question is asking for.

    Args:
        question: The question text.

    Returns:
        The detected answer type.
    """
    lowered = question.strip().lower()
    if re.match(r"^how many\b", lowered):
        return AnswerType.COUNT
    if lowered.startswith("how much"):
        return AnswerType.AMOUNT
    if lowered.startswith("how long"):
        return AnswerType.DURATION
    if lowered.startswith("how often") or lowered.startswith("how frequently"):
        return AnswerType.FREQUENCY
    if lowered.startswith("when"):
        return AnswerType.TIME
    if lowered.startswith("who"):
        return AnswerType.PERSON
    if lowered.startswith("where"):
        return AnswerType.PLACE
    if re.match(
        r"^(do|does|did|is|are|was|were|can|could|will|would|should|shall|may|must|has|have|had)\b",
        lowered,
    ):
        return AnswerType.YES_NO
    if lowered.startswith("which"):
        return AnswerType.ENTITY
    return AnswerType.DESCRIPTION


def measure_family(question: str) -> Optional[Set[str]]:
    """Return the units that could answer the question, if it asks a quantity."""
    phrase = focus_phrase(question)
    words = [word.lower() for word in _WORD.findall(phrase)]
    for word in words:
        family = _UNIT_FAMILIES.get(simple_stem(word))
        if family:
            return family
    if words:
        family = _UNIT_FAMILIES.get(simple_stem(words[-1]))
        if family:
            return family
    return None


def specific_terms(question: str) -> List[str]:
    """Return the identifiers the evidence has to mention to answer the question.

    Product codes, model numbers and small capitalised codes identify a *thing*
    the question is about. If the evidence never mentions them, it is talking
    about something else, however similar the rest of the wording is.
    """
    terms: List[str] = []
    for fragment in split_claims(question):
        for token in _WORD.findall(fragment):
            if any(character.isdigit() for character in token):
                if token not in terms:
                    terms.append(token)
            elif token.isupper() and 2 <= len(token) <= 5 and token.lower() not in _TITLE_NOISE:
                if token not in terms:
                    terms.append(token)
    return terms


def _count_head(question: str) -> str:
    """Return the singular head noun of a 'how many X' question."""
    match = re.match(r"^how many\s+(.*)$", question.strip(), re.I)
    if not match:
        return ""
    words = re.findall(r"[A-Za-z][A-Za-z\-]*", match.group(1))
    stop_after = {"can", "could", "may", "might", "must", "are", "is", "were", "was", "will", "would"}
    head = ""
    for word in words:
        if word.lower() in stop_after:
            break
        head = word
    if not head and words:
        head = words[0]
    return simple_stem(head)


def _counted_in_evidence(text: str, head: str) -> bool:
    """Whether the text counts the thing the question asked about."""
    pattern = re.compile(
        r"\b\d[\d,]*(?:\.\d+)?\s+(?:[A-Za-z-]+\s+){0,1}" + re.escape(head) + r"s?\b",
        re.I,
    )
    return bool(pattern.search(text))


def _has_number_with_unit(text: str, family: Set[str]) -> bool:
    """Whether the text states a number followed by a unit from the family."""
    for match in _NUMBER.finditer(text):
        tail = text[match.end() : match.end() + 30].lower()
        words = re.findall(r"[a-z]+", tail)
        for index, word in enumerate(words):
            if word not in family:
                continue
            if index == 0 or words[index - 1] in _NON_COUNT_UNITS:
                return True
    return False


def _proper_nouns(text: str) -> List[str]:
    """Return capitalised tokens that are not sentence-initial and not noise."""
    found: List[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        words = re.findall(r"[A-Za-z][A-Za-z\-]*", sentence)
        for index, word in enumerate(words):
            if index == 0:
                continue
            if word[0].isupper() and word.lower() not in _TITLE_NOISE:
                found.append(word)
    return found


def _has_type_match(question: str, evidence: str, answer_type: AnswerType) -> Tuple[bool, str]:
    """Check that the evidence contains a value of the expected type."""
    lowered = evidence.lower()

    family = measure_family(question)
    if family is not None and answer_type in (AnswerType.DESCRIPTION, AnswerType.AMOUNT):
        if _has_number_with_unit(evidence, family):
            return True, f"a quantity in {sorted(family)[:3]} is stated in the evidence"
        return False, f"no quantity in {sorted(family)[:3]} is stated in the evidence"

    if answer_type is AnswerType.COUNT:
        head = _count_head(question)
        if not head:
            return False, "no head noun for the count question"
        if _counted_in_evidence(evidence, head):
            return True, f"the evidence counts {head}"
        return False, f"the evidence never counts {head}"

    if answer_type is AnswerType.DURATION:
        if re.search(r"\b\d[\d,.]*\s*-?\s*(?:" + "|".join(sorted(_TIME_UNITS)) + r")", lowered):
            return True, "a duration is stated in the evidence"
        return False, "no duration is stated in the evidence"

    if answer_type is AnswerType.AMOUNT:
        if _has_number_with_unit(evidence, _UNIT_FAMILIES["charge"] | {"percent"}):
            return True, "a quantity with a unit is stated in the evidence"
        return False, "no quantity with a unit is stated in the evidence"

    if answer_type is AnswerType.FREQUENCY:
        if any(word in lowered for word in _FREQUENCY_WORDS):
            return True, "a frequency is stated in the evidence"
        return False, "no frequency is stated in the evidence"

    if answer_type is AnswerType.TIME:
        if _TIME_EXPRESSIONS.search(evidence):
            return True, "a point in time is stated in the evidence"
        return False, "no point in time is stated in the evidence"

    if answer_type is AnswerType.PERSON:
        if any(word in lowered for word in _ROLE_WORDS):
            return True, "a role is named in the evidence"
        if _proper_nouns(evidence):
            return True, "a named person or entity is present in the evidence"
        return False, "no person is named in the evidence"

    if answer_type is AnswerType.PLACE:
        if any(word in lowered for word in _PLACE_WORDS):
            return True, "a place is named in the evidence"
        if _proper_nouns(evidence):
            return True, "a named place is present in the evidence"
        return False, "no place is named in the evidence"

    if answer_type is AnswerType.YES_NO:
        terms = focus_terms(question)
        evidence_stems = {simple_stem(token) for token in tokenize(evidence)}
        coverage = len(terms & evidence_stems) / max(len(terms), 1)
        if coverage >= 0.34:
            return True, f"the evidence discusses the subject of the question ({coverage:.0%})"
        return False, f"the evidence barely discusses the subject of the question ({coverage:.0%})"

    if answer_type is AnswerType.ENTITY:
        question_stems = {simple_stem(token) for token in tokenize(question)}
        candidates = [
            name for name in _proper_nouns(evidence) if simple_stem(name) not in question_stems
        ]
        if candidates:
            return True, f"the evidence names {candidates[0]}"
        return False, "the evidence names no entity that could answer the question"

    # DESCRIPTION: the passage has to add information beyond the question itself.
    question_stems = {simple_stem(token) for token in tokenize(question)}
    evidence_stems = {simple_stem(token) for token in tokenize(evidence)}
    extra = evidence_stems - question_stems
    if len(extra) >= 3:
        return True, "the evidence adds information beyond the question"
    return False, "the evidence adds no information beyond the question"


def check_answerability(question: str, evidence: str) -> Answerability:
    """Decide whether the evidence contains an answer to the question.

    Args:
        question: The question text.
        evidence: The retrieved text to inspect.

    Returns:
        The verdict, the detected answer type and a human-readable reason.
    """
    answer_type = detect_answer_type(question)
    lowered_evidence = evidence.lower()

    identifiers = specific_terms(question)
    if identifiers:
        missing = [term for term in identifiers if term.lower() not in lowered_evidence]
        if len(missing) == len(identifiers):
            reason = f"the evidence never mentions {missing[0]}"
            logger.debug("question %r is not answerable: %s", question, reason)
            return Answerability(answerable=False, answer_type=answer_type, reason=reason)

    matched, reason = _has_type_match(question, evidence, answer_type)
    if not matched:
        logger.debug("question %r is not answerable from the evidence: %s", question, reason)
    return Answerability(answerable=matched, answer_type=answer_type, reason=reason)
