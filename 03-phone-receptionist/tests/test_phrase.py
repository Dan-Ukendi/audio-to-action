"""phrase.py: the model may reword the approved sentences, the checks must stop anything invented (no model needed)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

import dialog  # noqa: E402
from dialog import Action, Part  # noqa: E402
from faq import load_faq  # noqa: E402
from persona import load_persona, speak_number  # noqa: E402
from phrase import Phrased, check_sentence, make_phraser, number_runs  # noqa: E402
from shared.llm import LLMFormError  # noqa: E402

PERSONA, FAQ = load_persona(), load_faq()
NUMBER = speak_number("01632960501")
READ_BACK = f"So that's Siobhan Gallagher, on {NUMBER}, about a bathroom quote. Is that right?"


def bad(new, draft="Could I take your name, please?", keep=(), ordered=False, said="", mention=("name", "who"), coverage=0.0):
    return check_sentence(new, Part(draft, rewrite=True, keep=keep, ordered_numbers=ordered, mention=mention, coverage=coverage), said)


def test_a_friendly_rewording_is_accepted():
    assert bad("Of course, and who am I speaking to?", said="I need a plumber") == []


def test_digits_and_symbols_are_refused():
    assert any("digit" in p for p in bad("Could I take your name, please? Ref 12"))


def test_the_question_must_stay_a_question():
    assert any("question" in p for p in bad("Tell me your name please."))
    assert any("question" in p for p in bad("Thanks, you are all set?", draft="Thank you, I've got that down."))


def test_names_and_reasons_must_survive_a_read_back():
    ok = f"Right, Siobhan Gallagher, on {NUMBER}, about a bathroom quote. Have I got that right?"
    assert bad(ok, READ_BACK, keep=("Siobhan Gallagher", "a bathroom quote"), ordered=True, mention=()) == []
    wrong_name = ok.replace("Siobhan", "Sian")
    assert any("Siobhan Gallagher" in p for p in bad(wrong_name, READ_BACK, keep=("Siobhan Gallagher",), ordered=True, mention=()))


def test_a_changed_or_added_number_is_refused():
    changed = READ_BACK.replace("oh one six", "oh one seven")
    assert any("numbers" in p for p in bad(changed, READ_BACK, ordered=True, mention=()))
    added = READ_BACK.replace("Is that right?", "Our office is on four five six. Is that right?")
    assert any("numbers" in p for p in bad(added, READ_BACK, ordered=True, mention=()))
    dropped = "So that's Siobhan Gallagher about a bathroom quote. Is that right?"
    assert any("numbers" in p for p in bad(dropped, READ_BACK, ordered=True, mention=()))


def test_digits_in_the_wrong_order_fail_a_read_back_but_a_pronoun_one_is_fine():
    swapped = READ_BACK.replace("oh one six three two, nine six oh", "nine six oh, oh one six three two")
    assert any("numbers" in p for p in bad(swapped, READ_BACK, ordered=True, mention=()))
    assert number_runs("Oh dear, which one?") == []


def test_invented_places_and_promises_are_refused():
    assert any("Kelmbridge" in p or "kelmbridge" in p for p in bad("Are you in Kelmbridge? Could I take your name?"))
    assert any("tomorrow" in p for p in bad("Someone will call you tomorrow. Could I take your name?"))
    assert any("refund" in p for p in bad("We can give a refund. Could I take your name?"))
    # words the caller said may come back
    assert bad("Oh dear, in Kelmbridge. Could I take your name?", said="I'm in Kelmbridge and it's leaking") == []


def test_a_question_cannot_turn_into_a_different_question():
    number = dict(draft="And what is the best number to call you back on?", mention=("number",))
    assert bad("Lovely. What's the best number to reach you on?", **number) == []
    assert any("same thing" in p for p in bad("Lovely. Could you spell your full name, please?", **number))


def test_information_lines_must_keep_their_facts():
    faq = dict(draft="Yes, our quotes are free.", mention=(), coverage=0.7)
    assert bad("Yes, of course, our quotes are free.", **faq) == []
    assert any("dropped information" in p for p in bad("Oh dear, a burst pipe, how awful.", **faq))
    goodbye = dict(draft="Thank you for calling Brightwater Plumbing and Heating. I've passed your message to the team. Goodbye.",
                   mention=(), coverage=0.6)
    assert bad("Thanks for calling Brightwater Plumbing and Heating. I've passed your message to the team. Goodbye.", **goodbye) == []
    assert any("dropped" in p for p in bad("Thank you for calling. Goodbye.", **goodbye))


def attack(new, draft, **kw):
    return check_sentence(new, Part(draft, rewrite=True, **kw), kw.pop("said", ""))


ASK = dict(mention=("name", "who"))
GOODBYE = "Thank you for calling Brightwater Plumbing and Heating. I've passed your message to the team. Goodbye."


def test_promises_instructions_and_negations_are_refused_even_if_the_caller_said_the_word():
    name = "Could I take your name, please?"
    for new, said in [("Oh dear, we'll get someone out today. Could I take your name?", "can someone come out today?"),
                      ("Yes, it's safe to keep using it. Could I take your name?", "is it safe to keep using the boiler?"),
                      ("Don't worry, someone will be round straight away. Could I take your name?", ""),
                      ("Please open the windows and leave the house. Could I take your name?", "I opened the windows"),
                      ("Sorry, I haven't got that down. Could I take your name?", "")]:
        assert check_sentence(new, Part(name, rewrite=True, **ASK), said), new
    goodbye = Part(GOODBYE, rewrite=True, coverage=0.75)
    assert check_sentence("Thank you for calling Brightwater Plumbing and Heating. I've passed your message to the team "
                          "and they will call you back. Goodbye.", goodbye, "")
    confirmed = Part("Thank you, I've got that down.", rewrite=True, mention=("thank", "thanks", "got", "noted", "down"))
    assert check_sentence("Thank you, I've got that down and marked it as urgent.", confirmed, "")


def test_information_lines_cannot_flip_or_grow():
    free = Part("Yes, our quotes are free.", rewrite=True, coverage=0.85)
    assert check_sentence("Yes, our quotes are not free.", free, "")
    assert check_sentence("Yes, our quotes are free and we also do roofing.", free, "")
    hours = Part("We're open Monday to Friday from eight in the morning until half past five, and on Saturdays from nine until one.",
                 rewrite=True, coverage=0.85)
    swapped = "We're open Monday to Friday from nine in the morning until half past five, and on Saturdays from eight until one."
    assert check_sentence(swapped, hours, "")
    lost_one = "We're open Monday to Friday from eight in the morning until half past five, and on Saturdays from nine until."
    assert check_sentence(lost_one, hours, "")


def test_invented_names_are_caught_even_as_the_first_word_of_a_sentence():
    name = Part("Could I take your name, please?", rewrite=True, **ASK)
    assert check_sentence("Thanks. Sam will ring you back. Could I take your name, please?", name, "")
    assert check_sentence("Overmere is lovely. Could I take your name?", name, "")
    assert check_sentence("Lovely, thank you. Could I take your name?", name, "") == []


def test_read_back_items_cannot_be_extended():
    keep = ("Siobhan Gallagher", "a leaking tap")
    line = Part(f"So that's Siobhan Gallagher, on {NUMBER}, about a leaking tap. Is that right?", rewrite=True, keep=keep,
                keep_end=("a leaking tap",), ordered_numbers=True)
    assert check_sentence(f"So that's Siobhan Gallagher, on {NUMBER}, about a leaking tap. Is that right?", line, "") == []
    assert check_sentence(f"So that's Siobhan Gallagher, on {NUMBER}, about a leaking tap and a gas smell. Is that right?", line, "")
    assert check_sentence(f"So that's Siobhan Gallagher-smith, on {NUMBER}, about a leaking tap. Is that right?", line, "")


def test_a_repeat_request_cannot_ask_for_a_detail():
    line = Part("Sorry, I didn't catch that. Could you say that again?", rewrite=True, kind="repeat_request")
    assert check_sentence("Sorry, I didn't catch that. Could I take your name?", line, "")
    assert check_sentence("Sorry, I missed that. Could you say it once more?", line, "") == []


def test_harmless_rewordings_are_not_rejected():
    number = Part("And what is the best number to call you back on?", rewrite=True, mention=("number",))
    assert check_sentence("And what number's best to ring you back on?", number, "") == []
    ask = Part("Could I take your name, please?", rewrite=True, **ASK)
    assert check_sentence("Could I take your name? Thank you.", ask, "") == []


def test_too_long_and_empty_are_refused():
    assert bad("")
    assert any("longer" in p for p in bad("Could I possibly take " + "your very kind name " * 20 + "please?"))


def test_phrased_form_needs_one_sentence_per_line():
    line = Part("Could I take your name, please?", rewrite=True)
    with pytest.raises(ValueError, match="exactly 1"):
        Phrased.model_validate({"sentences": ["a b c?", "d e f?"]}, context={"lines": [line]})
    assert Phrased.model_validate({"sentences": ["And who am I speaking to?"]}, context={"lines": [line]})


# ---------------------------------------------------------------- inside the dialog

def fake_chat(answers):
    """A stand-in for structured_chat: validates like the real one (with the checks), no retry loop needed here."""
    calls = []

    def chat(schema, messages, llm=None, context=None, options=None, fix_hint="", timeout=None):
        calls.append(messages)
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(value=schema.model_validate({"sentences": answer}, context=context), attempts=1,
                               rejected_because=None)

    chat.calls = calls
    return chat


def first_turn(text="Hi, I need a quote for a bathroom."):
    from rules_turn import rules_understand
    from turn import Understanding
    _, state = dialog.start_call(PERSONA)
    return state, (lambda t, ctx: Understanding(rules_understand(t, ctx)))


def test_the_reply_is_reworded_and_logged():
    state, understand = first_turn()
    chat = fake_chat([["Lovely, a bathroom quote. Could I take your name, please?"]])
    reply, state = dialog.next_reply(state, "Hi, I need a quote for a bathroom.", PERSONA, FAQ, understand,
                                     phrase_fn=make_phraser(chat=chat))
    assert reply.startswith("Lovely")
    assert state.log[-1]["phrasing"]["used"] == "model"
    assert "Hi, I need a quote for a bathroom." in chat.calls[0][1]["content"]


def test_a_rejected_wording_falls_back_to_the_fixed_sentence():
    state, understand = first_turn()
    chat = fake_chat([["Sure, I will book you in tomorrow. Could I take your name, please?"]])
    plain, _ = dialog.next_reply(first_turn()[0], "Hi, I need a quote for a bathroom.", PERSONA, FAQ, understand)
    reply, state = dialog.next_reply(state, "Hi, I need a quote for a bathroom.", PERSONA, FAQ, understand,
                                     phrase_fn=make_phraser(chat=chat))
    assert reply == plain
    assert state.log[-1]["phrasing"]["used"] == "fixed"


def test_a_model_that_is_down_never_breaks_the_call():
    state, understand = first_turn()
    chat = fake_chat([ConnectionError("ollama is not running")])
    plain, _ = dialog.next_reply(first_turn()[0], "Hi, I need a quote for a bathroom.", PERSONA, FAQ, understand)
    reply, state = dialog.next_reply(state, "Hi, I need a quote for a bathroom.", PERSONA, FAQ, understand,
                                     phrase_fn=make_phraser(chat=chat))
    assert reply == plain and "ConnectionError" in state.log[-1]["phrasing"]["problems"][0]


def test_safety_lines_never_reach_the_model_and_stay_word_for_word():
    state, _ = first_turn()
    actions = [Action(kind="urgent_ack"), Action(kind="goodbye", arg="urgent")]
    parts = dialog.render_parts(actions, state, PERSONA, FAQ)
    assert not any(p.rewrite for p in parts)
    chat = fake_chat([])
    state.silent_streak = PERSONA.max_silent_turns - 1  # the next silence ends the call: a fixed line
    reply, state = dialog.next_reply(state, "", PERSONA, FAQ, lambda t, c: None, phrase_fn=make_phraser(chat=chat))
    assert reply == PERSONA.say("silence_end") and chat.calls == []  # nothing reworded: no model call at all


def test_mixed_reply_keeps_the_safety_advice_exactly_and_rewords_only_the_question():
    state, _ = first_turn()
    actions = [Action(kind="urgent_ack"), Action(kind="ask", arg="name")]
    parts = dialog.render_parts(actions, state, PERSONA, FAQ)
    chat = fake_chat([["And who am I speaking to?"]])
    text, trace = make_phraser(chat=chat)(parts, state, "my pipe burst", PERSONA)
    assert text == f"{PERSONA.say('urgent_ack')} And who am I speaking to?"
    sent = " ".join(m["content"] for m in chat.calls[0])
    assert PERSONA.say("urgent_ack") not in sent and "flagging it for the team" not in sent  # safety text never reaches the model
    assert trace["used"] == "model"


def test_invalid_twice_uses_fixed(monkeypatch):
    state, _ = first_turn()
    parts = dialog.render_parts([Action(kind="ask", arg="name")], state, PERSONA, FAQ)
    chat = fake_chat([LLMFormError("invalid answer twice.\nline 1: bad")])
    text, trace = make_phraser(chat=chat)(parts, state, "hello", PERSONA)
    assert text == PERSONA.say("ask_name") and trace["used"] == "fixed" and trace["attempts"] == 2


def test_the_spam_goodbye_and_urgent_goodbye_are_not_rewritable_but_a_normal_one_is():
    state, _ = first_turn()
    for arg, rewritable in (("spam", False), ("urgent", False), (None, True), ("info", True)):
        (part,) = dialog.render_parts([Action(kind="goodbye", arg=arg)], state, PERSONA, FAQ)
        assert part.rewrite is rewritable, arg
