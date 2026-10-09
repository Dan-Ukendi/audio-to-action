"""The conversation flow fixes after a real test call: business first, no needless name questions, a refusal is not a goodbye."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

from dialog import bare_question, next_reply, start_call  # noqa: E402
from faq import load_faq  # noqa: E402
from persona import load_persona  # noqa: E402
from tests.test_dialog import form  # noqa: E402
from turn import Understanding  # noqa: E402

P, FAQ = load_persona(), load_faq()


def say(state, text, **fields):
    return next_reply(state, text, P, FAQ, lambda t, c: Understanding(form(**fields)))


def new():
    return start_call(P)[1]


def test_spelling_is_off_by_default_so_a_full_name_goes_straight_to_the_number():
    assert P.ask_spelling is False
    s = new()
    _, s = say(s, "I need a quote for a bathroom", reason="a bathroom quote")
    reply, s = say(s, "My name is Daniel Kendi", name="Daniel Kendi")
    assert reply == P.say("ask_number")


def test_a_bare_question_is_not_taken_as_the_reason_and_holly_offers_help_or_a_message():
    s = new()
    reply, s = say(s, "Hey, what kind of service do you propose?", reason="what kind of service do you propose",
                   question_topic="services")
    assert s.value("reason") is None
    assert reply == f"{FAQ['services'].answer} {P.say('offer_help')}"
    # the caller then says what they need: the message is taken
    reply, s = say(s, "I'd like a boiler service", reason="a boiler service")
    assert reply == P.say("ask_name") and s.value("reason") == "a boiler service"


def test_a_question_plus_a_request_keeps_the_request_as_the_reason():
    assert not bare_question("My boiler is leaking, how much do you charge?")
    assert bare_question("Hey, what kind of service do you propose?")
    assert bare_question("Hello. Do you cover Overmere?")


def test_refusing_one_detail_does_not_end_the_call():
    s = new()
    _, s = say(s, "I need a boiler service", reason="a boiler service")
    _, s = say(s, "Daniel Kendi", name="Daniel Kendi")
    reply, s = say(s, "No, I don't want to give my number.", wants_to_end=True)  # a model may wrongly call this an end
    assert s.state != "ENDED" and "Goodbye" not in reply


def test_a_real_goodbye_still_ends_the_call():
    s = new()
    _, s = say(s, "I need a boiler service", reason="a boiler service")
    reply, s = say(s, "Never mind, goodbye", wants_to_end=True)
    assert s.state == "ENDED"


def test_a_spoken_number_the_model_missed_is_taken_by_plain_code():
    s = new()
    _, s = say(s, "I need a boiler service", reason="a boiler service")
    _, s = say(s, "Daniel Kendi", name="Daniel Kendi")
    reply, s = say(s, "Ok it is oh seven seven double oh, nine hundred, one two three.")  # the form has no number
    assert s.value("number") == "07700900123" and "oh seven seven" in reply
