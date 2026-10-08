"""Matching a caller's question to an approved answer: keywords first, whole words, questions only."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

import faq as F  # noqa: E402

E = F.load_faq()


def ids(text, llm_topic=None):
    return [e.id for e in F.match_questions(text, E, llm_topic).entries]


def test_simple_questions_find_their_answer():
    assert ids("What are your opening hours?") == ["hours"]
    assert ids("Do you cover Overmere?") == ["area"]
    assert ids("Is there a charge for out of hours call outs?") == ["emergency_callout"]
    assert ids("Are you Gas Safe registered?") == ["qualifications"]
    assert ids("What's your cancellation policy?") == ["cancellation"]
    assert ids("How do I pay?") == ["payment"]


def test_statements_are_never_answered():
    assert ids("I'd like a quote to redo our bathroom.") == []
    assert ids("My boiler service is due and I pay by card.") == []
    assert ids("The area is flooded.") == []


def test_the_question_can_be_a_sentence_among_statements():
    assert ids("I've got a radiator that won't heat up. How soon could someone come out?") == ["booking_time"]


def test_whole_words_only():
    assert ids("Is Overmeres a nice place?") == []  # 'overmere' inside another word is not a match
    assert ids("What about the thours?") == []


def test_the_longest_phrase_wins_over_a_shorter_one_inside_it():
    assert ids("How soon can you get here?") == ["emergency_callout"]  # not also booking_time ('how soon')
    assert ids("How soon could someone come out?") == ["booking_time"]


def test_a_generic_price_question_is_about_the_sentence_before_it():
    assert ids("I'd like to book a boiler service. How much does that cost?") == ["boiler_service"]
    assert ids("I need a gas safety certificate for my rental property. How much is it?") == ["landlord_certificate"]


def test_a_self_contained_price_question_is_just_answered():
    assert ids("My neighbour recommended you, she had her bathroom done. How much do you charge an hour?") == ["prices"]


def test_two_questions_in_one_turn_give_two_answers_best_first():
    assert set(ids("Do you cover Overmere, and can I pay by card?")) == {"area", "payment"}
    assert len(ids("What are your hours, do you cover Overmere, can I pay by card and are you insured?")) <= F.MAX_ANSWERS


def test_punctuation_and_case_do_not_matter():
    assert set(ids("ARE YOU OPEN 24/7?")) == {"hours", "emergency_callout"}  # '24/7' = '24 7', and 'are you open' = hours
    assert ids("what's the cost per hour?") == ["prices"]


def test_unknown_question_is_passed_on_but_a_request_is_not_a_question():
    unknown = F.match_questions("Do you fit solar panels?", E)
    assert unknown.entries == [] and unknown.unknown == "Do you fit solar panels?"
    assert F.match_questions("Can you give me a call back when you get a sec.", E).unknown is None
    assert F.match_questions("Could you ask him to ring me back?", E).unknown is None
    assert F.match_questions("Can you come out today?", E).unknown is None


def test_small_talk_questions_are_not_unknown_questions():
    for text in ("Hello?", "Pardon?", "Hello? Are you there?", "Can you hear me?", "Sorry?", "What?"):
        r = F.match_questions(text, E)
        assert r.unknown is None and r.entries == [], text


def test_a_greeting_in_front_of_a_real_question_does_not_make_it_small_talk():
    assert F.match_questions("Hello, do you fit solar panels?", E).unknown == "Do you fit solar panels?"
    assert F.match_questions("Hi, sorry, hello? Hello?", E).unknown is None


def test_a_model_pick_is_used_only_when_no_keyword_matches():
    r = F.match_questions("Do you work at weekends at all?", E, llm_topic="hours")
    assert [e.id for e in r.entries] == ["hours"] and r.via == "llm"
    r = F.match_questions("What are your opening hours?", E, llm_topic="payment")  # keywords win: the facts come from the text
    assert [e.id for e in r.entries] == ["hours"] and r.via == "keywords" and r.llm_disagreed


def test_a_model_pick_that_is_not_an_entry_is_ignored():
    assert F.match_questions("Do you fit solar panels?", E, llm_topic="solar").unknown
    assert F.match_questions("Do you fit solar panels?", E, llm_topic="other").unknown


def test_every_keyword_finds_its_own_entry():
    """Each keyword, put in a question, brings back at least its own entry (or a longer, more specific one)."""
    for entry in E.values():
        for keyword in entry.keywords:
            result = F.match_questions(f"Quick question, {keyword}?", E)
            got = {e.id for e in result.entries}
            longer = {e.id for e in E.values() if any(keyword != k and f" {F.normalize_text(keyword)} " in f" {F.normalize_text(k)} " for k in e.keywords)}
            assert entry.id in got or got & longer, (entry.id, keyword, got)


def test_scoring_helpers():
    assert F.normalize_text("It's 24/7, isn't it?") == "its 24 7 isnt it"
    assert F.looks_like_question("Do you cover Kelmbridge") and not F.looks_like_question("I need a boiler.")
    assert F.sentences("One. Two? Three!") == ["One.", "Two?", "Three!"]
