"""Every state transition, correction, limit, emergency and FAQ fallback of version A (no model: scripted understanding)."""

import json
import re
import dataclasses
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

import asks  # noqa: E402
import dialog  # noqa: E402
import faq as faq_module  # noqa: E402
from audio_io import Heard  # noqa: E402
from dialog import CallState, next_reply, start_call  # noqa: E402
from persona import load_persona, speak_number  # noqa: E402
from turn import CallerTurn, Understanding  # noqa: E402

P = dataclasses.replace(load_persona(), ask_spelling=True)  # the spelling tests; the default (off) is tested in test_dialog_flow.py
FAQ = faq_module.load_faq()
NUMBER = "01632960501"


def form(**changes) -> CallerTurn:
    base = dict(heard_summary="x", name=None, number=None, reason=None, is_correction=False, correction_field=None,
                question_topic=None, emergency=False, wants_to_end=False, is_automated=False)
    return CallerTurn.model_validate({**base, **changes}, context={"faq_ids": list(FAQ)})


def compose(fields) -> str:
    """Words that really contain the values: grounding runs inside apply_turn, so a value the text lacks would be dropped."""
    parts = []
    if fields.get("reason"):
        parts.append(f"It's about {fields['reason'].rstrip('.')}.")
    if fields.get("name"):
        parts.append(f"My name is {fields['name']}.")
    if fields.get("number"):
        parts.append(f"My number is {fields['number']}.")
    return " ".join(parts) or "Um, well."


def say(state, text, **fields):
    """One caller turn whose understanding is exactly `fields` (what a perfect model would report). Text 'x' = compose it."""
    text = compose(fields) if text == "x" else text
    return next_reply(state, text, P, FAQ, lambda t, c: Understanding(form(**fields)))


def new():
    greeting, state = start_call(P)
    return state


def line(key, **kw):
    return P.say(key, **kw)


def read_back(name, number, reason):
    return line("read_back", name=name, number=speak_number(number), reason=reason)


# ---------------------------------------------------------------- collecting

def test_the_call_starts_with_the_greeting_and_waits_for_the_reason():
    greeting, state = start_call(P)
    assert "automated" in greeting and "recorded" in greeting
    assert (state.state, state.waiting_for, state.asking) == ("GREETING", asks.GREETING, asks.GREETING)


def test_details_are_asked_one_at_a_time_in_the_order_reason_name_number():
    s = new()
    reply, s = say(s, "Hello?", )  # said something, but nothing usable
    assert reply == line("ask_reason") and s.waiting_for == asks.REASON
    reply, s = say(s, "My tap drips", reason="a dripping tap")
    assert reply == line("ask_name") and s.waiting_for == asks.NAME and s.state == "COLLECTING"
    reply, s = say(s, "Dave", name="Dave")
    assert reply == line("ask_number") and s.waiting_for == asks.NUMBER  # single name: no spelling
    reply, s = say(s, f"{NUMBER}", number=NUMBER)
    assert reply == line("read_back_no_name_no_number", reason="x") or "Dave" in reply
    assert s.state == "READ_BACK" and s.waiting_for == asks.CONFIRM


def test_what_the_caller_volunteers_is_not_asked_again():
    s = new()
    reply, s = say(s, "x", reason="a leak", name="Dave", number=NUMBER)
    assert reply == read_back("Dave", NUMBER, "a leak")  # nothing missing: straight to the read-back
    s2 = new()
    reply, s2 = say(s2, "x", reason="a leak", name="Dave")
    assert reply == line("ask_number")  # only the number is missing


def test_nothing_is_stored_that_the_turn_did_not_contain():
    s = new()
    _, s = say(s, "Hmm.")
    assert {k: v.value for k, v in s.slots.items()} == {"reason": None, "name": None, "number": None}
    _, s = say(s, "Dave", name="Dave")
    _, s = say(s, "Hmm.")  # a later turn without values cannot erase or invent anything
    assert s.value("name") == "Dave" and s.value("number") is None


def test_a_value_already_stored_is_not_changed_by_a_later_chatty_turn():
    s = new()
    _, s = say(s, "x", reason="a leak", name="Dave")
    _, s = say(s, "x", name="David", reason="a different thing")  # not a correction: ignored while collecting
    assert s.value("name") == "Dave" and s.value("reason") == "a leak"


# ---------------------------------------------------------------- spelling

def test_a_full_name_is_asked_to_be_spelled_once_right_after_the_name():
    s = new()
    _, s = say(s, "x", reason="a quote")
    reply, s = say(s, "Shiv awn Gallagher", name="Shiv awn Gallagher")
    assert reply == line("ask_name_spelling", first_name="Shiv") and s.waiting_for == asks.SPELLING
    reply, s = say(s, "S, I, O, B, H, A, N, G, A, double L, A, G, H, E, R")
    assert s.value("name") == "Siobhan Gallagher" and reply == line("ask_number")
    assert s.slots["name"].spelled


def test_spelling_is_not_asked_for_a_single_name_nor_twice_nor_when_already_spelled():
    s = new()
    _, s = say(s, "x", reason="r", name="Dave")
    assert s.waiting_for == asks.NUMBER  # Dave: no spelling
    s = new()
    _, s = say(s, "x", reason="r", name="Mark Thompson")
    assert s.waiting_for == asks.SPELLING
    reply, s = say(s, "It's Mark Thompson.")  # did not spell: asked once, never again
    assert s.waiting_for == asks.NUMBER and s.value("name") == "Mark Thompson"
    s = new()
    _, s = say(s, "M, A, R, K, T, H, O, M, P, S, O, N", reason="r", name="Mark Thompson")  # spelled up front
    assert s.waiting_for == asks.NUMBER and s.slots["name"].spelled


def test_spelling_that_fits_nothing_is_not_used():
    s = new()
    _, s = say(s, "x", reason="r", name="Mark Thompson")
    _, s = say(s, "X, Y, Z, Q, W, K, P, L, M, N, B, V, C, D, F, G, H")
    assert s.value("name") == "Mark Thompson"


# ---------------------------------------------------------------- refusals and re-asks

def test_a_refused_number_is_asked_once_more_with_the_apology_then_dropped():
    s = new()
    _, s = say(s, "x", reason="r", name="Dave")
    reply, s = say(s, "You've got my number.")
    assert reply == line("number_refused") and s.waiting_for == asks.NUMBER_AGAIN and s.number_refusals == 1
    reply, s = say(s, "I'd rather not give it.")
    assert s.slots["number"].given_up and s.value("number") is None
    assert reply == line("read_back_no_number", name="Dave", reason="r")


def test_a_detail_that_is_never_given_is_asked_three_times_then_dropped():
    s = new()
    _, s = say(s, "x", reason="r")  # asked name (1st)
    reply, s = say(s, "um")  # 2nd
    assert reply == line("ask_name")
    reply, s = say(s, "um")  # 3rd = initial + 2 re-asks
    assert reply == line("ask_name")
    reply, s = say(s, "um")  # now asks for the number instead
    assert reply == line("ask_number") and s.slots["name"].given_up


def test_a_refused_name_gets_one_more_try_not_two():
    s = new()
    _, s = say(s, "x", reason="r")
    reply, s = say(s, "I'd rather not say.")
    assert reply == line("ask_name")
    reply, s = say(s, "No, I'd rather not.")
    assert reply == line("ask_number") and s.slots["name"].given_up


def test_read_back_has_four_forms():
    number = speak_number(NUMBER)
    cases = ((("Dave", NUMBER), line("read_back", name="Dave", number=number, reason="a dripping tap")),
             (("Dave", None), line("read_back_no_number", name="Dave", reason="a dripping tap")),
             ((None, NUMBER), line("read_back_no_name", number=number, reason="a dripping tap")),
             ((None, None), line("read_back_no_name_no_number", reason="a dripping tap")))
    for (name, num), expected in cases:
        s = new()
        s.slots["reason"].value, s.slots["name"].value, s.slots["number"].value = "A dripping tap.", name, num
        assert dialog.render([dialog.Action(kind="read_back")], s, P, FAQ) == expected


def test_the_reason_reads_naturally_in_the_read_back():
    s = new()
    s.slots["reason"].value = "CO alarm going off"
    assert "about CO alarm going off," in dialog.render([dialog.Action(kind="read_back")], s, P, FAQ)
    s.slots["reason"].value = "I'm ringing about a tap"
    assert "about I'm ringing about a tap," in dialog.render([dialog.Action(kind="read_back")], s, P, FAQ)
    s.slots["reason"].value = "My outside tap is dripping!"
    assert "about my outside tap is dripping," in dialog.render([dialog.Action(kind="read_back")], s, P, FAQ)


def test_a_missing_reason_is_read_back_as_such():
    s = new()
    s.slots["reason"].given_up = True
    _, s = say(s, "x", name="Dave", number=NUMBER)
    text = dialog.render([dialog.Action(kind="read_back")], s, P, FAQ)
    assert line("reason_missing_phrase") in text


# ---------------------------------------------------------------- read-back, corrections, goodbye

def read_back_state(name="Dave", number=NUMBER, reason="a leak"):
    s = new()
    _, s = say(s, "x", reason=reason, name=name, number=number)
    assert s.state == "READ_BACK"
    return s


def test_yes_confirms_asks_anything_else_and_no_ends_the_call():
    s = read_back_state()
    reply, s = say(s, "Yes, that's right.")
    assert reply == f"{line('confirmed')} {line('anything_else')}" and s.state == "GOODBYE"
    assert all(slot.confirmed for slot in s.slots.values())
    reply, s = say(s, "No, that's all, thank you.")
    assert reply == line("goodbye") and s.state == "ENDED" and s.outcome == "completed"
    assert next_reply(s, "hello again", P, FAQ, lambda t, c: Understanding(form()))[0] == ""  # an ended call stays ended


def test_a_correction_with_the_new_value_replaces_only_that_field_and_reads_back_again():
    s = read_back_state()
    reply, s = say(s, "No, it's 349", number="01632960349", is_correction=True, correction_field="number")
    assert s.value("number") == "01632960349" and s.value("name") == "Dave" and s.value("reason") == "a leak"
    assert reply == read_back("Dave", "01632960349", "a leak") and s.state == "READ_BACK"
    assert not any(slot.confirmed for slot in s.slots.values())


def test_a_bare_no_asks_what_to_change_then_applies_the_answer():
    s = read_back_state()
    reply, s = say(s, "No.")
    assert reply == line("correction") and s.state == "CORRECTING" and s.asking == asks.CORRECTION
    reply, s = say(s, "My name is David", name="David")
    assert s.value("name") == "David" and reply == read_back("David", NUMBER, "a leak") and s.state == "READ_BACK"


def test_a_correction_round_that_never_gets_an_answer_ends_in_a_read_back_as_is():
    s = read_back_state()
    _, s = say(s, "No.")
    replies = [say(s, "hmm")[0] for _ in range(3)]
    assert replies[0] == line("correction") and replies[1] == line("correction")
    assert replies[2] == read_back("Dave", NUMBER, "a leak")


def test_yes_with_a_paraphrase_does_not_rewrite_the_message():
    s = read_back_state()
    _, s = say(s, "Yes, the leak", reason="a water leak", name="David")
    assert s.value("reason") == "a leak" and s.value("name") == "Dave"


def test_a_number_the_caller_never_gave_can_be_taken_off():
    s = read_back_state()
    reply, s = say(s, "No, I didn't give you a number.")
    assert s.value("number") is None and s.slots["number"].given_up
    assert reply == read_back_text_without_number()


def read_back_text_without_number():
    return line("read_back_no_number", name="Dave", reason="a leak")


def test_an_unclear_answer_to_the_read_back_repeats_it():
    s = read_back_state()
    reply, s = say(s, "Hmm, let me think")
    assert reply == read_back("Dave", NUMBER, "a leak") and s.state == "READ_BACK"


def test_anything_else_is_repeated_once_when_not_understood_then_the_call_ends():
    s = read_back_state()
    _, s = say(s, "Yes")
    reply, s = say(s, "Hmm")
    assert reply == line("anything_else")
    reply, s = say(s, "Hmm")
    assert reply == line("goodbye") and s.state == "ENDED"


def test_caller_who_hangs_up_early_gets_a_goodbye_and_the_outcome_says_so():
    s = new()
    _, s = say(s, "x", reason="a leak")
    reply, s = say(s, "Never mind, goodbye.", wants_to_end=True)
    assert reply == line("goodbye") and s.state == "ENDED" and s.outcome == "caller_ended"


# ---------------------------------------------------------------- emergencies

def test_a_gas_smell_gets_the_acknowledgement_and_the_gas_advice_once_then_collection_continues():
    s = new()
    reply, s = say(s, "There's a smell of gas in the hallway", reason="a smell of gas", emergency=True)
    assert reply.startswith(line("urgent_ack")) and FAQ["safety_gas"].answer in reply
    assert reply.endswith(line("ask_name")) and s.urgent and s.urgent_turn == 1
    assert s.safety_advised == ["gas"]
    reply, s = say(s, "Priya", name="Priya")
    assert FAQ["safety_gas"].answer not in reply and line("urgent_ack") not in reply  # said once


def test_the_emergency_goodbye_promises_a_callback_and_mentions_999():
    s = new()
    _, s = say(s, "x", reason="a gas smell", emergency=True, name="Priya", number=NUMBER)
    _, s = say(s, "Yes")
    reply, s = say(s, "No thanks", wants_to_end=True)
    assert reply == line("goodbye_urgent") and "nine nine nine" in reply and s.state == "ENDED"


def test_safety_words_flag_the_call_even_if_the_model_says_no():
    s = new()
    reply, s = say(s, "The water heater is leaking badly and it's sparking near the electrics", reason="a leak", emergency=False)
    assert s.urgent and reply.startswith(line("urgent_ack")) and FAQ["safety_water"].answer in reply


def test_the_model_alone_can_flag_an_emergency_without_specific_advice():
    s = new()
    reply, s = say(s, "It's got a lot worse since this morning", reason="a worsening leak", emergency=True)
    assert s.urgent and reply.startswith(line("urgent_ack")) and s.safety_advised == []
    assert "stopcock" not in reply


def test_no_heating_is_urgent_with_the_acknowledgement_only():
    s = new()
    reply, s = say(s, "I've got no heating and no hot water", reason="no heating")
    assert s.urgent and line("urgent_ack") in reply and "stopcock" not in reply and "gas" not in reply.lower().replace("gas safe", "")


def test_asking_what_to_do_about_gas_is_answered_by_the_advice_not_by_I_dont_know():
    s = new()
    reply, s = say(s, "There's a smell of gas. Should I turn it off?", reason="a smell of gas", emergency=True, question_topic="other")
    assert FAQ["safety_gas"].answer in reply and line("faq_unknown") not in reply
    assert s.unanswered_questions == []


# ---------------------------------------------------------------- spam, silence, limits

def test_a_robocall_gets_a_short_goodbye_even_if_it_says_urgent():
    s = new()
    reply, s = say(s, "Urgent. This is an urgent message about your listing. Press one.", is_automated=True, emergency=True)
    assert reply == line("goodbye_spam") and s.state == "ENDED" and s.outcome == "spam" and not s.urgent


def test_a_robocall_is_caught_by_code_even_if_the_model_misses_it():
    s = new()
    reply, s = say(s, "This is an automated message from the tax office. Press one now.")
    assert reply == line("goodbye_spam") and s.spam


def test_silence_is_asked_again_once_and_then_the_call_ends():
    s = new()
    reply, s = say(s, "")
    assert reply == line("repeat_request") and s.asking == asks.REPEAT and s.waiting_for == asks.GREETING
    reply, s = say(s, "   ")
    assert reply == line("silence_end") and s.state == "ENDED" and s.outcome == "silence"


def test_two_silent_turns_must_be_in_a_row():
    s = new()
    _, s = say(s, "")
    _, s = say(s, "Hi, it's Dave", name="Dave")
    assert s.silent_streak == 0
    reply, s = say(s, "")
    assert reply == line("repeat_request") and s.state != "ENDED"


def test_the_answer_after_a_repeat_request_still_answers_the_original_question():
    s = new()
    _, s = say(s, "x", reason="r")
    assert s.waiting_for == asks.NAME
    _, s = say(s, "")
    assert s.asking == asks.REPEAT and s.waiting_for == asks.NAME
    _, s = say(s, "Dave", name="Dave")
    assert s.value("name") == "Dave"


def test_the_turn_limit_wraps_the_call_up_with_what_we_have():
    s = new()
    reply = ""
    for i in range(P.max_turns):
        reply, s = say(s, "blah blah")
        if s.state == "ENDED":
            break
    assert s.turns == P.max_turns and reply.endswith(line("turn_limit")) and s.outcome == "turn_limit"


def test_the_turn_limit_after_a_complete_message_is_a_normal_goodbye():
    s = read_back_state()
    s.turns = P.max_turns - 1
    reply, s = say(s, "Yes, that's right.")
    assert reply.endswith(line("goodbye")) and s.outcome == "completed" and line("turn_limit") not in reply


# ---------------------------------------------------------------- FAQ

def test_a_question_is_answered_word_for_word_and_collection_goes_on():
    s = new()
    reply, s = say(s, "I've got a radiator problem. How soon could someone come out?", reason="a radiator problem")
    assert reply == f"{FAQ['booking_time'].answer} {line('ask_name')}" and s.faq_answered == ["booking_time"]


def test_an_unknown_question_is_passed_on_and_kept_in_the_message():
    s = new()
    reply, s = say(s, "Do you fit solar panels?", question_topic="other")
    assert reply == f"{line('faq_unknown')} {line('ask_reason')}"
    assert s.unanswered_questions == ["Do you fit solar panels?"]
    assert dialog.message_of(s)["unanswered_questions"] == ["Do you fit solar panels?"]


def test_someone_who_only_asks_a_question_is_not_pressed_for_a_message():
    s = new()
    reply, s = say(s, "What are your opening hours?")
    assert reply == f"{FAQ['hours'].answer} {line('offer_help')}" and s.state == "GOODBYE"
    reply, s = say(s, "No thanks, that's all.", wants_to_end=True)
    assert reply == line("goodbye_info") and s.outcome == "info_only" and all(v.value is None for v in s.slots.values())


def test_a_new_request_after_anything_else_goes_back_to_taking_a_message():
    s = new()
    _, s = say(s, "Do you cover Overmere?")
    reply, s = say(s, "Yes, I'd like a bathroom quote", reason="a bathroom quote")
    assert reply == line("ask_name") and s.state == "COLLECTING" and s.value("reason") == "a bathroom quote"


def test_a_second_question_in_the_closing_phase_is_answered_too():
    s = new()
    _, s = say(s, "Do you cover Overmere?")
    reply, s = say(s, "What payment methods do you take?")
    assert reply == f"{FAQ['payment'].answer} {line('anything_else')}"


def test_the_same_topic_is_not_answered_twice():
    s = new()
    _, s = say(s, "What are your opening hours?")
    reply, s = say(s, "What are your opening hours again?")
    assert FAQ["hours"].answer not in reply


def test_the_model_can_pick_the_entry_when_no_keyword_matches():
    s = new()
    reply, s = say(s, "Do you work at weekends at all?", question_topic="hours")
    assert FAQ["hours"].answer in reply


def test_two_answers_in_one_reply_when_two_questions_are_asked():
    s = new()
    reply, s = say(s, "Do you cover Overmere, and what payment methods do you take?")
    assert FAQ["area"].answer in reply and FAQ["payment"].answer in reply


# ---------------------------------------------------------------- the engine itself

def test_an_agent_can_replace_the_decision_step_and_keeps_the_same_safety_net():
    """decide_fn is the only difference between version A and B: a custom one is used, apply_turn still runs first."""
    def always_goodbye(state, facts, persona):
        return [dialog.Action(kind="goodbye", arg="normal")]

    s = new()
    reply, s = next_reply(s, "My name is Dave, number 01632 960 501", P, FAQ, lambda t, c: Understanding(form(name="Dave", number=NUMBER)), always_goodbye)
    assert reply == line("goodbye") and s.state == "ENDED" and s.value("name") == "Dave"


def test_the_log_records_every_turn_with_the_understanding_and_the_actions():
    s = new()
    _, s = say(s, "Hi, it's Dave", name="Dave", reason="a leak")
    entry = s.log[0]
    for key in ("turn", "caller_text", "waiting_for", "understanding", "facts", "actions", "reply", "state_after", "understand_s", "decide_s"):
        assert key in entry
    assert entry["understanding"]["turn"]["name"] == "Dave"


def test_heard_details_from_the_audio_step_are_logged():
    heard = Heard(text="Hi", ignored=False, why=None, raw_text="Hi", duration_s=1, speech_s=1, min_logprob=-0.2, seconds=0.1)
    s = new()
    _, s = next_reply(s, "Hi", P, FAQ, lambda t, c: Understanding(form()), heard=heard)
    assert s.log[0]["heard"]["min_logprob"] == -0.2


def test_the_state_is_plain_json():
    s = read_back_state()
    again = CallState.model_validate_json(s.model_dump_json())
    assert again.value("name") == "Dave" and again.state == "READ_BACK"
    json.dumps(dialog.message_of(s))


def test_unknown_actions_are_a_bug_not_silence():
    with pytest.raises(ValueError):
        dialog.render([dialog.Action(kind="sing")], new(), P, FAQ)


# ---------------------------------------------------------------- everything said is a fixed sentence

def allowed_sentence_pattern() -> re.Pattern:
    pieces = [re.escape(e.answer) for e in FAQ.values()]
    for key, text in P.lines.items():
        pattern = re.escape(re.sub(r"\{(\w+)\}", "\x00", text)).replace("\x00", ".+?")
        pieces.append(pattern)
    return re.compile(rf"(?:{'|'.join(pieces)})(?: (?:{'|'.join(pieces)}))*")


def test_every_reply_is_made_only_of_persona_lines_and_faq_answers():
    """The model never writes what is said: replies from whole calls must parse into approved pieces."""
    from cards import load_cards
    from rules_turn import rules_understand
    from call import run_call
    pattern = allowed_sentence_pattern()
    for card in load_cards():
        record = run_call(card, P, FAQ, lambda t, c: Understanding(rules_understand(t, c)), understand_label="rules")
        for entry in record.turns:
            assert pattern.fullmatch(entry["reply"]), (card.id, entry["reply"])


# ---------------------------------------------------------------- review round 1: the holes found by probing

def test_apply_turn_grounds_again_so_a_custom_understanding_step_cannot_store_inventions():
    """Grounding is inside apply_turn, not only inside turn.understand(): any understand_fn is safe."""
    s = new()
    reply, s = next_reply(s, "hello", P, FAQ, lambda t, c: Understanding(form(name="Invented Person", number="07700900999", reason="invented reason")))
    assert {k: v.value for k, v in s.slots.items()} == {"reason": None, "name": None, "number": None}
    assert reply == line("ask_reason") and any("not in what was said" in n for n in s.log[0]["understanding"]["notes"])


def test_a_correction_changes_only_the_field_it_names():
    s = read_back_state()
    # the (fake) model also fills a "reason" from the words "number is wrong": it must not overwrite the real reason
    reply, s = say(s, "No, the number is wrong, it's 01632 960 124", number="01632960124", reason="number is wrong",
                   is_correction=True, correction_field="number")
    assert s.value("number") == "01632960124" and s.value("reason") == "a leak" and s.value("name") == "Dave"
    reply, s = say(s, "No, my name is David", name="David", is_correction=True, correction_field="name", reason="my name")
    assert s.value("name") == "David" and s.value("reason") == "a leak"


def test_a_correction_without_a_named_field_changes_what_was_given_not_the_reason():
    s = read_back_state()
    _, s = say(s, "No, it's 01632 960 124, sorry about that", number="01632960124", reason="sorry about that", is_correction=True)
    assert s.value("number") == "01632960124" and s.value("reason") == "a leak"


def test_a_partial_number_correction_replaces_the_end_of_the_number():
    """The plan's own example: 'No, it's 349'. Every digit of the result was said (stored number + this turn)."""
    for phrase in ("No, it's 349", "No, it ends in three four nine", "No, the end is 349, not 501", "No, the last three digits are 349"):
        s = read_back_state(number="01632960501")
        reply, s = say(s, phrase, is_correction=True, correction_field="number")
        assert s.value("number") == "01632960349", phrase
        assert reply == read_back("Dave", "01632960349", "a leak"), phrase


def test_a_partial_correction_that_would_make_no_valid_number_is_not_applied():
    s = read_back_state()
    reply, s = say(s, "No, it's 3", is_correction=True, correction_field="number")
    assert s.value("number") == NUMBER  # unchanged: nothing was invented


def test_an_emergency_is_not_hung_up_on_as_spam():
    for model_says_automated in (False, True):
        s = new()
        reply, s = say(s, "I got a final notice from my landlord and now there's a smell of gas in the kitchen",
                       is_automated=model_says_automated, reason="a smell of gas")
        assert s.urgent and not s.spam and s.state != "ENDED" and FAQ["safety_gas"].answer in reply, model_says_automated


def test_a_robocall_that_says_urgent_is_still_spam_without_safety_words():
    s = new()
    reply, s = say(s, "Urgent. Your listing will be suspended. Press one now.", is_automated=False, emergency=True)
    assert s.spam and reply == line("goodbye_spam") and not s.urgent


def test_a_closing_remark_inside_a_request_does_not_hang_up():
    s = new()
    reply, s = say(s, "My tap is dripping, that's it really, can someone ring me?", reason="a dripping tap", wants_to_end=True)
    assert s.state == "COLLECTING" and reply == line("ask_name") and s.value("reason") == "a dripping tap"


def test_an_urgent_caller_who_also_says_goodbye_is_still_asked_for_name_and_number():
    s = new()
    reply, s = say(s, "My mum is eighty and has no heating, that's it really", reason="no heating", wants_to_end=True)
    assert s.urgent and s.state == "COLLECTING" and reply.endswith(line("ask_name"))


def test_a_bare_goodbye_ends_the_call_but_an_urgent_one_is_asked_for_a_number_first():
    s = new()
    reply, s = say(s, "Bye.", wants_to_end=True)
    assert reply == line("goodbye_info") and s.state == "ENDED"
    s = new()
    _, s = say(s, "x", reason="no heating", name="Dave", emergency=True)
    assert s.waiting_for == asks.NUMBER
    reply, s = say(s, "Bye.", wants_to_end=True)
    assert s.state != "ENDED" and reply == line("ask_number")


def test_a_second_request_after_anything_else_is_added_not_lost():
    s = read_back_state(reason="a radiator problem")
    _, s = say(s, "Yes, that's right")
    reply, s = say(s, "Yes, also my outside tap is dripping and needs looking at.", reason="my outside tap is dripping")
    assert s.extra_requests == ["my outside tap is dripping"] and s.value("reason") == "a radiator problem"
    assert reply == f"{line('confirmed')} {line('anything_else')}" and s.state == "GOODBYE"
    assert dialog.message_of(s)["extra_requests"] == ["my outside tap is dripping"]
    reply, s = say(s, "No, that's all.")
    assert s.state == "ENDED" and s.outcome == "completed"


def test_an_emergency_that_comes_up_after_the_message_is_flagged_and_added():
    s = read_back_state(reason="a radiator problem")
    _, s = say(s, "Yes, that's right")
    reply, s = say(s, "Actually, I can smell gas now", reason="a smell of gas", emergency=True)
    assert s.urgent and reply.startswith(line("urgent_ack")) and FAQ["safety_gas"].answer in reply
    assert s.extra_requests == ["a smell of gas"] and s.value("reason") == "a radiator problem"


def test_a_plain_yes_to_anything_else_asks_what_else_and_does_not_count_as_unclear():
    s = read_back_state()
    _, s = say(s, "Yes")
    reply, s = say(s, "Yes")
    assert reply == line("what_else") and s.unclear_closings == 0 and s.state == "GOODBYE"
    reply, s = say(s, "Yes, there is. Also my tap drips.", reason="my tap drips")
    assert s.extra_requests == ["my tap drips"] and s.state == "GOODBYE"


def test_a_question_about_the_danger_is_answered_by_the_advice_but_other_questions_are_passed_on():
    s = new()
    _, s = say(s, "There's a smell of gas. Should I turn it off?", reason="a smell of gas", emergency=True, question_topic="other")
    assert s.unanswered_questions == []  # the advice answers "what do I do?"
    s = new()
    _, s = say(s, "There's a smell of gas. Do you fit solar panels?", reason="a smell of gas", emergency=True, question_topic="other")
    assert s.unanswered_questions == ["Do you fit solar panels?"]  # not about the danger: still passed on in the message
    _, s = say(s, "x", name="Priya", number=NUMBER)
    _, s = say(s, "Yes")
    _, s = say(s, "One more thing: do you fit solar panels?", question_topic="other")
    assert s.unanswered_questions == ["Do you fit solar panels?", "One more thing: do you fit solar panels?"]


def test_asking_to_repeat_says_the_question_again_and_records_nothing():
    s = new()
    _, s = say(s, "x", reason="a leak")
    assert s.waiting_for == asks.NAME
    reply, s = say(s, "Could you repeat that?")
    assert reply == line("ask_name") and s.waiting_for == asks.NAME and s.unanswered_questions == [] and s.slots["name"].asks == 1
    s = read_back_state()
    reply, s = say(s, "Sorry, could you say that again?")
    assert reply == read_back("Dave", NUMBER, "a leak") and s.state == "READ_BACK" and s.unclear_readbacks == 0


def test_small_talk_and_questions_about_the_call_are_not_passed_on():
    for text in ("Can I leave a message for Sam?", "Is that Brightwater?", "Who am I speaking to?", "Are you a real person?",
                 "What's your name?", "Hello?"):
        s = new()
        _, s = say(s, text, question_topic="other")
        assert s.unanswered_questions == [], text


def test_a_model_that_sees_no_question_vetoes_the_keyword_guess_of_an_unknown_one():
    s = new()
    _, s = say(s, "Do you know where my parcel is?", question_topic=None)
    assert s.unanswered_questions == []
    s = new()
    _, s = say(s, "Do you know where my parcel is?", question_topic="other")
    assert s.unanswered_questions == ["Do you know where my parcel is?"]


def test_an_urgent_call_never_ends_without_the_urgent_promise():
    # turn limit
    s = new()
    _, s = say(s, "There's a smell of gas", reason="a smell of gas", emergency=True)
    s.turns = P.max_turns - 1  # the last turn the call may have
    reply, s = say(s, "blah blah")
    assert reply == line("turn_limit_urgent") and "nine nine nine" in reply and s.outcome == "turn_limit"
    # silence
    s = new()
    _, s = say(s, "There's a smell of gas", reason="a smell of gas", emergency=True)
    _, s = say(s, "")
    reply, s = say(s, "")
    assert reply == line("silence_end_urgent") and "nine nine nine" in reply and s.outcome == "silence"


def test_silence_after_a_confirmed_message_is_still_a_completed_call():
    s = read_back_state()
    _, s = say(s, "Yes")
    _, s = say(s, "")
    reply, s = say(s, "")
    assert s.state == "ENDED" and s.outcome == "completed"


def test_a_caller_who_leaves_at_the_read_back_gets_a_goodbye_and_the_message_stays_unconfirmed():
    s = read_back_state()
    reply, s = say(s, "Bye.", wants_to_end=True)
    assert reply == line("goodbye") and s.state == "ENDED" and s.outcome == "caller_ended"


def test_the_read_back_is_repeated_at_most_max_reasks_times_when_not_understood():
    s = read_back_state()
    replies = []
    for _ in range(4):
        reply, s = say(s, "Hmm, I don't know")
        replies.append(reply)
        if s.state == "ENDED":
            break
    assert all(r == read_back("Dave", NUMBER, "a leak") for r in replies[:P.max_reasks])
    assert s.state == "ENDED" and replies[-1] == line("goodbye")


def test_spelling_is_not_asked_on_an_urgent_call():
    s = new()
    _, s = say(s, "x", reason="a smell of gas", name="Priya Shah", emergency=True)
    assert s.waiting_for == asks.NUMBER and not s.spelling_asked


def test_info_only_calls_end_with_the_plain_goodbye_not_the_message_goodbye():
    s = new()
    _, s = say(s, "What are your opening hours?")
    reply, s = say(s, "No thanks", wants_to_end=True)
    assert reply == line("goodbye_info") and "passed your message" not in reply


# ---------------------------------------------------------------- review round 2

def rules_say(state, text):
    """One caller turn understood by the rules baseline (the model-free path)."""
    from rules_turn import rules_understand
    return next_reply(state, text, P, FAQ, lambda t, c: Understanding(rules_understand(t, c)))


def test_questions_and_fillers_are_never_stored_as_the_name_and_do_not_block_the_real_one():
    for junk in ("Pardon?", "Hmm", "Could you repeat that?", "Is that Brightwater?", "Who am I speaking to?", "Do you do underfloor heating?"):
        s = new()
        _, s = rules_say(s, "My boiler is broken.")
        assert s.waiting_for == asks.NAME
        _, s = rules_say(s, junk)
        assert s.value("name") is None, junk
        _, s = rules_say(s, "Ann Lee")
        assert s.value("name") == "Ann Lee", junk


def test_spelled_letters_alone_become_the_name_when_none_was_heard():
    for spoken, expected in (("S, M, I, T, H", "Smith"), ("S M I T H", "Smith")):
        s = new()
        _, s = rules_say(s, "My boiler is broken.")
        _, s = rules_say(s, spoken)
        assert s.value("name") == expected, spoken


def test_a_repeat_request_stores_nothing_even_from_a_model_that_fills_the_form():
    s = new()
    _, s = say(s, "x", reason="a leak")
    reply, s = say(s, "Could you repeat that?", name="Could you repeat")
    assert s.value("name") is None and reply == line("ask_name")


def test_naming_the_wrong_field_asks_for_it_directly_with_the_rules_path():
    for answer, ask_line, field in (("Number.", "ask_number", "number"), ("Your number.", "ask_number", "number"),
                                    ("The name.", "ask_name", "name"), ("What it's about.", "ask_reason", "reason")):
        s = read_back_state()
        _, s = rules_say(s, "No.")
        assert s.state == "CORRECTING"
        reply, s = rules_say(s, answer)
        assert reply == line(ask_line) and s.value("name") == "Dave", answer
        assert s.state == "CORRECTING" and s.waiting_for == {"number": asks.NUMBER, "name": asks.NAME, "reason": asks.REASON}[field]


def test_naming_the_wrong_field_works_straight_from_the_read_back_and_with_a_model_form():
    s = read_back_state()
    reply, s = say(s, "No, the number is wrong", is_correction=True, correction_field="number")
    assert reply == line("ask_number") and s.state == "CORRECTING"
    reply, s = say(s, "01632 960 124", number="01632960124")
    assert s.value("number") == "01632960124" and reply == read_back("Dave", "01632960124", "a leak") and s.state == "READ_BACK"
    s = read_back_state()
    reply, s = say(s, "No, the name is wrong")  # no field set by the model: the words decide
    assert reply == line("ask_name")
    reply, s = say(s, "David", name="David")
    assert s.value("name") == "David" and s.value("number") == NUMBER


def test_the_answer_to_an_asked_detail_changes_only_that_detail():
    s = read_back_state()
    _, s = say(s, "No, the number is wrong", is_correction=True, correction_field="number")
    _, s = say(s, "01632 960 124 and it's about a boiler", number="01632960124", reason="a boiler", name="David")
    assert s.value("number") == "01632960124" and s.value("name") == "Dave" and s.value("reason") == "a leak"


def test_a_number_is_not_repaired_by_digits_that_are_not_about_its_end():
    for phrase in ("No, I live at 42 Mill Lane, the number is fine but the name is wrong", "No the number is fine, it's flat 12",
                   "Wrong number, call me after 5 30 instead", "No, the number starts with 01632", "No, it's at number 349 Station Road"):
        s = read_back_state(number="07700900123")
        _, s = rules_say(s, phrase)
        assert s.value("number") == "07700900123", phrase


def test_the_plans_example_works_on_the_rules_path_without_any_field_hint():
    for phrase in ("No, it's 349", "No, it ends in three four nine", "No, the end is 349, not 123"):
        s = read_back_state(number="07700900123")
        reply, s = rules_say(s, phrase)
        assert s.value("number") == "07700900349", phrase


def test_a_yes_followed_by_a_no_is_a_no():
    for phrase in ("Yeah no, that's wrong.", "Right, no, the name's wrong.", "Yes, no, not quite."):
        s = read_back_state()
        _, s = rules_say(s, phrase)
        assert s.state == "CORRECTING" and not any(slot.confirmed for slot in s.slots.values()), phrase
    s = read_back_state()
    _, s = rules_say(s, "Yes, that's right, there's no rush.")  # "no rush" is not a no
    assert s.state == "GOODBYE" and all(slot.confirmed for slot in s.slots.values() if slot.value)


def test_a_caller_who_keeps_saying_no_is_not_read_back_to_for_ever():
    s = read_back_state()
    for _ in range(14):
        reply, s = say(s, "No.")
        if s.state == "ENDED":
            break
    assert s.state == "ENDED" and reply == line("goodbye") and s.outcome == "caller_ended"
    assert s.read_backs <= 1 + P.max_reasks and s.turns < P.max_turns


def test_a_correction_after_anything_else_is_not_lost():
    s = read_back_state()
    _, s = say(s, "Yes")
    reply, s = say(s, "Oh and my number is actually 07700 900222", number="07700900222")
    assert s.value("number") == "07700900222" and s.state == "READ_BACK"
    assert reply == read_back("Dave", "07700900222", "a leak") and not any(sl.confirmed for sl in s.slots.values())
    _, s = say(s, "Yes")
    assert s.state == "GOODBYE" and s.value("number") == "07700900222"


def test_the_same_late_correction_works_on_the_rules_path_and_a_late_name_is_read_back():
    s = read_back_state()
    _, s = rules_say(s, "Yes")
    reply, s = rules_say(s, "Actually my number is 07700 900 222")
    assert s.value("number") == "07700900222" and s.state == "READ_BACK"
    s = new()
    s.slots["name"].given_up = True
    _, s = say(s, "x", reason="a leak", number=NUMBER)
    _, s = say(s, "Yes")
    reply, s = say(s, "My name is Dave", name="Dave")
    assert s.value("name") == "Dave" and s.state == "READ_BACK"  # a name given late is stored AND read back


def test_a_real_question_with_is_that_or_is_this_is_not_small_talk():
    s = new()
    _, s = say(s, "My boiler is making a banging noise. Is that dangerous?", reason="a banging boiler", question_topic="other")
    assert s.unanswered_questions == ["Is that dangerous?"]
    s = new()
    _, s = say(s, "Is this something you can do?", question_topic="other")
    assert s.unanswered_questions == ["Is this something you can do?"]


def test_come_and_look_is_a_request_not_a_question():
    s = new()
    _, s = say(s, "Can someone come and look at it?", question_topic="other")
    assert s.unanswered_questions == []


def test_after_a_repeat_the_caller_is_asked_to_answer_the_same_question_as_before():
    s = new()
    _, s = say(s, "x", reason="a leak")
    _, s = say(s, "")
    assert s.asking == asks.REPEAT
    reply, s = say(s, "Pardon?")
    assert reply == line("ask_name") and s.asking == s.waiting_for == asks.NAME


def test_a_repeated_plain_yes_to_what_else_does_not_loop_to_the_turn_limit():
    s = read_back_state()
    _, s = say(s, "Yes")
    replies = []
    for _ in range(8):
        reply, s = say(s, "Yes")
        replies.append(reply)
        if s.state == "ENDED":
            break
    assert replies[0] == line("what_else") and s.state == "ENDED" and s.turns <= 6 and s.outcome == "completed"


# ---------------------------------------------------------------- review round 3

def test_a_person_who_mentions_an_automated_message_is_not_hung_up_on():
    for text in ("Hi, I got an automated message from you saying my appointment on Thursday was moved.",
                 "I left a recorded message yesterday about my boiler but nobody called back."):
        for step in (lambda s, t: rules_say(s, t), lambda s, t: say(s, t, reason="my appointment")):
            s = new()
            reply, s = step(s, text)
            assert not s.spam and s.state != "ENDED" and reply != line("goodbye_spam"), text


def test_an_urgent_call_can_never_end_as_spam():
    s = new()
    _, s = rules_say(s, "There's water everywhere, a pipe has burst!")
    _, s = rules_say(s, "Mark Thompson, 07700 900123")
    reply, s = rules_say(s, "Sorry, it says press one on my phone, hang on")
    assert s.urgent and not s.spam and s.outcome != "spam" and reply != line("goodbye_spam")
    s = new()  # not even a model that insists it is a robocall, once the call is urgent or has details
    _, s = say(s, "x", reason="a leak", name="Dave")
    reply, s = say(s, "Press one now", is_automated=True)
    assert not s.spam and s.state != "ENDED"


def test_a_robocall_is_still_hung_up_on_at_the_start_of_a_call():
    s = new()
    reply, s = rules_say(s, "This is an automated message from the tax office. Press one to speak to an officer.")
    assert s.spam and reply == line("goodbye_spam")


def test_the_phone_number_never_ends_up_in_the_reason_or_in_what_is_spoken():
    s = new()
    _, s = rules_say(s, "Hi, my kitchen tap is dripping and my number is 07700 900123.")
    assert s.value("reason") == "my kitchen tap is dripping" and s.value("number") == "07700900123"
    s = new()
    reply, s = rules_say(s, "My boiler is leaking a bit, you can ring me on oh seven seven double oh nine hundred one two three")
    assert s.value("reason") == "My boiler is leaking a bit" and s.value("number") == "07700900123"
    s = new()  # a model that paraphrases the reason on a sentence that also carries the number
    reply, s = say(s, "My faucet is leaking and my number is 07700 900123.", reason="a leaky faucet", number="07700900123")
    assert s.value("reason") and not any(ch.isdigit() for ch in s.value("reason"))
    assert not any(ch.isdigit() for ch in reply)


def test_digits_in_a_reason_are_spoken_as_words():
    s = new()
    s.slots["reason"].value = "the heater at 42 Mill Lane, error F 28"
    spoken = dialog.render([dialog.Action(kind="read_back")], s, P, FAQ)
    assert "forty-two Mill Lane" in spoken and not any(ch.isdigit() for ch in spoken)


def test_im_calling_about_and_its_about_give_a_reason_on_the_rules_path():
    for text in ("Hi, I'm calling about a leak under the sink.", "It's about a leak under the sink.", "It's a leak under the sink."):
        s = new()
        _, s = rules_say(s, text)
        assert s.value("reason") and "leak" in s.value("reason"), text
    s = read_back_state()
    _, s = rules_say(s, "No.")
    _, s = rules_say(s, "What it's about.")
    reply, s = rules_say(s, "It's about a radiator that won't heat up")
    assert "radiator" in s.value("reason") and s.value("name") == "Dave" and s.state == "READ_BACK"


def test_short_replies_to_the_name_question_are_not_names():
    for junk in ("Hold on", "Hang on a second", "Sure", "Cheers", "Great", "Speak up", "Can't hear you"):
        s = new()
        _, s = rules_say(s, "My boiler is broken.")
        _, s = rules_say(s, junk)
        assert s.value("name") is None, junk


def test_it_ends_in_digits_at_the_read_back_is_a_correction_without_the_word_no():
    s = read_back_state(number="07700900123")
    _, s = rules_say(s, "It ends in 349")
    assert s.value("number") == "07700900349"


# ---------------------------------------------------------------- a seeded fuzz: the engine's invariants over random calls

FUZZ_POOL = ["", "Hello?", "Yes", "No", "No.", "My name is Dave Smith", "It's Ann Lee", "My number is 07700 900123",
             "oh seven seven double oh nine hundred one two three", "There's a smell of gas in the hallway",
             "My tap is dripping and my number is 07700 900123.", "Do you cover Overmere?", "What are your opening hours?",
             "Could you repeat that?", "Bye.", "That's all, thanks", "Actually my number is 07700 900222", "No, it's 349",
             "Number.", "S, M, I, T, H", "Pardon?", "I'd rather not say", "You've got my number", "It's about a leak under the sink",
             "I'm calling about a leak under the sink", "Press one", "This is an automated message. Press one now.",
             "Do you fit solar panels?", "Is that dangerous?", "Yeah no, that's wrong", "42 Mill Lane", "Thank you", "Um, well.",
             "No heating and no hot water here", "The name.", "It ends in 349", "Can you give me a call back?"]


def test_random_calls_keep_every_invariant():
    import random
    from rules_turn import rules_understand
    rng = random.Random(20261008)
    for call in range(300):
        s = new()
        last = ""
        for _ in range(P.max_turns + 3):
            reply, s = next_reply(s, rng.choice(FUZZ_POOL), P, FAQ, lambda t, c: Understanding(rules_understand(t, c)))
            if s.state == "ENDED":
                last = reply
                break
        assert s.state == "ENDED" and s.turns <= P.max_turns, call
        assert s.outcome, call
        assert next_reply(s, "hello again", P, FAQ, lambda t, c: Understanding(form()))[0] == ""  # an ended call stays ended
        number = s.value("number")
        assert number is None or (number.startswith("0") and number.isdigit() and 10 <= len(number) <= 11), (call, number)
        if s.urgent:
            assert s.outcome != "spam" and "nine nine nine" in last, (call, s.outcome, last)
        for entry in s.log:
            assert not any(ch.isdigit() for ch in entry["reply"]), (call, entry["reply"])
