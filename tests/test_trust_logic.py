"""Tests for the trust logic that does not need a model download."""

from __future__ import annotations

from trustlayer.answerability import (
    AnswerType,
    check_answerability,
    detect_answer_type,
    focus_phrase,
    simple_stem,
    specific_terms,
)
from trustlayer.statements import split_claims, to_claims

PRICING = "The base charge is 3.20 credits for the first item and 1.40 credits for each additional item in the same consignment."
CLAIMS = "Claims must be submitted within 14 calendar days of the delivery date."
LIVESTOCK = "Livestock, and any consignment containing live animals, is prohibited."
COURIER = "The warranty period starts on the delivery date recorded by the courier, not on the order date."
VT40W = "A wireless variant, the VT-40W, adds an embedded radio that transmits the same payload at 2 Hz."
RATING = "The unit is rated for operation between 0 and 45 degrees Celsius and at relative humidity up to 85 percent."


def test_simple_stem_handles_plurals() -> None:
    assert simple_stem("items") == "item"
    assert simple_stem("policies") == "policy"
    assert simple_stem("samples") == "sample"


def test_detect_answer_type() -> None:
    assert detect_answer_type("How many items fit?") is AnswerType.COUNT
    assert detect_answer_type("How long is the warranty?") is AnswerType.DURATION
    assert detect_answer_type("When is it delivered?") is AnswerType.TIME
    assert detect_answer_type("Who signs it off?") is AnswerType.PERSON
    assert detect_answer_type("Does it include a stipend?") is AnswerType.YES_NO
    assert detect_answer_type("Which scheme applies?") is AnswerType.ENTITY
    assert detect_answer_type("What is the sampling rate?") is AnswerType.DESCRIPTION


def test_count_question_needs_a_count_of_the_right_noun() -> None:
    question = "How many items can be included in a single Kestrel Freight consignment?"
    # The pricing paragraph mentions items but never counts them.
    assert not check_answerability(question, PRICING).answerable
    counted = "A consignment may contain up to 12 items in total."
    assert check_answerability(question, counted).answerable


def test_duration_question_needs_a_duration() -> None:
    question = "How long do I have to file a damage claim with Kestrel Freight?"
    assert check_answerability(question, CLAIMS).answerable
    assert not check_answerability(question, LIVESTOCK).answerable


def test_identifier_must_appear_in_the_evidence() -> None:
    question = "What is the maximum ambient temperature the VT-40W wireless variant is rated for?"
    # The VT-40W paragraph names the variant but states no temperature.
    assert not check_answerability(question, VT40W).answerable
    # The temperature paragraph states a temperature but never names the variant.
    assert not check_answerability(question, RATING).answerable


def test_entity_question_needs_a_named_entity() -> None:
    question = "Which courier service does Orrin Retail Group use for deliveries?"
    assert not check_answerability(question, COURIER).answerable


def test_specific_terms_only_catches_identifiers() -> None:
    assert specific_terms("What is the maximum ambient temperature the VT-40W variant is rated for?") == [
        "VT-40W"
    ]
    assert specific_terms("Which courier service does Orrin Freight use?") == []


def test_focus_phrase() -> None:
    assert focus_phrase("What is the sampling rate of the VT-40?") == "sampling rate of the VT-40"
    assert focus_phrase("How many items fit?") == "items fit"


def test_polar_questions_assert_something_and_wh_questions_do_not() -> None:
    polar = to_claims("Can a Kestrel Freight customer ship livestock?")[0]
    assert polar.asserted
    assert any("text states" in form.lower() for form in polar.forms)

    wh = to_claims("What is the sampling rate of the VT-40?")[0]
    assert not wh.asserted


def test_split_claims_strips_a_leading_conditional() -> None:
    fragments = split_claims("If I return it after 20 days, do I get a full refund?")
    assert fragments == ["do I get a full refund"]
