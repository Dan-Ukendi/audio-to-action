"""The per-turn form: validators, grounding against what was said, and the model call (no real model: a fake chat)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

import asks  # noqa: E402
import faq as faq_module  # noqa: E402
import turn as turn_module  # noqa: E402
from shared.llm import LLMFormError, StructuredReply, structured_chat  # noqa: E402
from shared.schemas import Analysis, normalize_uk_number  # noqa: E402
from turn import CallerTurn, UnderstandContext, ground, understand  # noqa: E402

FAQ = faq_module.load_faq()
IDS = list(FAQ)


def form(**changes) -> CallerTurn:
    base = dict(heard_summary="x", name=None, number=None, reason=None, is_correction=False, correction_field=None,
                question_topic=None, emergency=False, wants_to_end=False, is_automated=False)
    return CallerTurn.model_validate({**base, **changes}, context={"faq_ids": IDS})


def ctx(**kw) -> UnderstandContext:
    return UnderstandContext(faq_topics={e.id: e.topic for e in FAQ.values()}, **kw)


# ---------------------------------------------------------------- the shared UK number rule

def test_the_uk_number_rule_is_one_function_for_part_1_and_part_3():
    assert normalize_uk_number("07700 900-123") == "07700900123"
    assert normalize_uk_number(None) is None and normalize_uk_number("no digits here") is None
    with pytest.raises(ValueError, match="not a UK phone number"):
        normalize_uk_number("1632960789")  # the leading 0 was lost
    with pytest.raises(ValueError):
        normalize_uk_number("0770090012")  # too short... 10 digits is valid, 9 is not
        normalize_uk_number("077009001")
    assert Analysis.check_uk_number("01632 960 456") == "01632960456"  # Part 1 still goes through the same rule


def test_form_validators():
    assert form(number="01632 960 501").number == "01632960501"
    with pytest.raises(ValidationError, match="UK phone number"):
        form(number="12345")
    assert form(name="unknown").name is None and form(name="  Dave ").name == "Dave" and form(reason="null").reason is None
    with pytest.raises(ValidationError, match="short phrase"):
        form(reason="one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen")
    assert form(question_topic="hours").question_topic == "hours" and form(question_topic="other").question_topic == "other"
    assert form(question_topic="none").question_topic is None
    with pytest.raises(ValidationError, match="not a topic"):
        form(question_topic="weather")
    assert form(is_correction=False, correction_field="name").correction_field is None  # a field without a correction is dropped


# ---------------------------------------------------------------- grounding: only what was said survives

def test_a_number_the_caller_never_said_is_dropped_not_kept():
    t, notes = ground(form(number="07700900123"), "Hi, it's Mark, I need someone today.")
    assert t.number is None and "not in what was said" in notes[0]


def test_a_number_that_was_said_survives_in_words_or_digits():
    assert ground(form(number="07700900123"), "my number is oh seven seven double oh, nine hundred, one two three")[0].number
    assert ground(form(number="07700900123"), "ring me on 07700 900 123 please")[0].number
    assert ground(form(number="01632960501"), "it's zero one six three two nine six oh five oh one")[0].number


def test_a_house_number_or_error_code_cannot_become_the_phone_number():
    t, _ = ground(form(number="07700900123"), "the heater at forty-two Mill Lane shows F twenty-eight")
    assert t.number is None


def test_a_name_must_be_made_of_words_that_were_said_or_spelled():
    assert ground(form(name="Mark Thompson"), "Hi, this is Mark Thompson")[0].name == "Mark Thompson"
    assert ground(form(name="Siobhan"), "that's S, I, O, B, H, A, N")[0].name == "Siobhan"
    assert ground(form(name="Mum"), "Hi love, it's Mum")[0].name == "Mum"
    t, notes = ground(form(name="Siobhan Gallagher"), "It's Shiv awn Gallagher")  # the model 'knew better' than the speech
    assert t.name is None and "not in what was said" in notes[0]
    assert ground(form(name="Sam Jones"), "I'd like a quote please")[0].name is None


def test_a_reason_paraphrase_is_kept_but_an_invented_one_is_replaced_by_the_callers_own_words():
    text = "Hi, my outside tap has been dripping for weeks. It's not urgent."
    kept, _ = ground(form(reason="a dripping outside tap"), text)
    assert kept.reason == "a dripping outside tap"
    replaced, notes = ground(form(reason="a gas boiler explosion"), text)
    assert replaced.reason == "my outside tap has been dripping for weeks" and "own words" in notes[0]


def test_an_ungrounded_reason_with_nothing_to_fall_back_on_is_dropped():
    t, notes = ground(form(reason="a bathroom quote"), "My name is Dave.")
    assert t.reason is None and "dropped" in notes[0]


# ---------------------------------------------------------------- the model call (a fake stands in for the chat)

def fake_chat(**answer):
    seen = {}

    def chat(schema, messages, llm=None, context=None, fix_hint=None, options=None):
        seen.update(schema=schema, messages=messages, context=context, fix_hint=fix_hint)
        return StructuredReply(form(**answer), answer.pop("_attempts", 1), None, None, 0.01)

    chat.seen = seen
    return chat


def test_the_prompt_tells_the_model_what_was_asked_what_is_known_and_the_topics():
    chat = fake_chat(name="Dave")
    understand("It's Dave.", ctx(asked=asks.NAME, reason="a leak", number=None), chat=chat)
    system, user = (m["content"] for m in chat.seen["messages"])
    assert "Never guess" in system and "hours:" in system and "safety_gas:" in system
    assert "the caller's name" in user and "reason='a leak'" in user and "It's Dave." in user
    assert set(chat.seen["context"]["faq_ids"]) == set(IDS)


def test_understand_grounds_the_models_answer_and_keeps_the_raw_form():
    u = understand("Hello, I'd like a quote for my bathroom.", ctx(), chat=fake_chat(name="Dave", number="07700900123", reason="a bathroom quote"))
    assert u.turn.name is None and u.turn.number is None and u.turn.reason == "a bathroom quote"
    assert u.raw["name"] == "Dave" and len(u.notes) == 2 and not u.fallback


def test_a_rejected_first_answer_is_noted():
    chat = lambda *a, **k: StructuredReply(form(name="Dave"), 2, "{bad}", "- number: not a UK number", 0.5)  # noqa: E731
    u = understand("It's Dave", ctx(), chat=chat)
    assert u.attempts == 2 and any("rejected" in n for n in u.notes)


def test_when_the_model_fails_the_rules_baseline_answers_and_the_call_goes_on():
    def broken(*a, **k):
        raise LLMFormError("invalid answer twice")

    u = understand("Hi, this is Mark Thompson. My number is 07700 900123.", ctx(), chat=broken)
    assert u.fallback and u.turn.name == "Mark Thompson" and u.turn.number == "07700900123"
    assert "baseline" in u.notes[0]


def test_a_network_error_also_falls_back_but_a_bug_does_not():
    def down(*a, **k):
        raise ConnectionError("Ollama is not running")

    assert understand("It's Dave", ctx(asked=asks.NAME), chat=down).fallback

    def bug(*a, **k):
        raise KeyError("my own mistake")

    with pytest.raises(KeyError):
        understand("It's Dave", ctx(), chat=bug)


def test_the_real_structured_chat_is_wired_to_the_form(monkeypatch):
    """structured_chat with a fake Ollama client: the schema is sent as the required format, the retry works."""
    replies = ['{"heard_summary": "x", "name": null, "number": "123", "reason": null, "is_correction": false, '
               '"correction_field": null, "question_topic": null, "emergency": false, "wants_to_end": false, "is_automated": false}',
               '{"heard_summary": "x", "name": "Dave", "number": null, "reason": null, "is_correction": false, '
               '"correction_field": null, "question_topic": null, "emergency": false, "wants_to_end": false, "is_automated": false}']
    calls = []

    class FakeClient:
        def __init__(self, host=None):
            pass

        def chat(self, model, messages, format, options):
            calls.append({"model": model, "messages": messages, "format": format, "options": options})
            return SimpleNamespace(message=SimpleNamespace(content=replies[len(calls) - 1]))

    import shared.llm as llm_module
    monkeypatch.setattr(llm_module.ollama, "Client", FakeClient)
    u = understand("It's Dave", ctx(asked=asks.NAME), chat=structured_chat, llm="test-model")
    assert u.turn.name == "Dave" and u.attempts == 2 and not u.fallback
    assert calls[0]["format"] == CallerTurn.model_json_schema() and calls[0]["options"]["temperature"] == 0
    assert "UK phone number" in calls[1]["messages"][-1]["content"]  # the retry showed the model what was wrong


def test_prompt_version_is_recorded_constant():
    assert turn_module.PROMPT_VERSION == "t1"
