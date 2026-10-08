"""Every state transition, correction, limit, emergency and FAQ fallback of version A (no model: scripted understanding)."""

import json
import re
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

P = load_persona()
FAQ = faq_module.load_faq()
NUMBER = "01632960501"


def form(**changes) -> CallerTurn:
    base = dict(heard_summary="x", name=None, number=None, reason=None, is_correction=False, correction_field=None,
                question_topic=None, emergency=False, wants_to_end=False, is_automated=False)
    return CallerTurn.model_validate({**base, **changes}, context={"faq_ids": list(FAQ)})


def say(state, text, **fields):
    """One caller turn whose understanding is exactly `fields` (what a perfect model would report)."""
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
    reply, s = say(s, "Hi, Dave here, my number is ..., about a leak", reason="a leak", name="Dave", number=NUMBER)
    assert reply == read_back("Dave", NUMBER, "a leak")  # nothing missing: straight to the read-back
    s2 = new()
    reply, s2 = say(s2, "Dave, about a leak", reason="a leak", name="Dave")
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
    _, s = say(s, "gas", reason="gas", emergency=True, name="Priya", number=NUMBER)
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
    reply, s = say(s, "Do you fit solar panels?")
    assert reply == f"{line('faq_unknown')} {line('ask_reason')}"
    assert s.unanswered_questions == ["Do you fit solar panels?"]
    assert dialog.message_of(s)["unanswered_questions"] == ["Do you fit solar panels?"]


def test_someone_who_only_asks_a_question_is_not_pressed_for_a_message():
    s = new()
    reply, s = say(s, "What are your opening hours?")
    assert reply == f"{FAQ['hours'].answer} {line('anything_else')}" and s.state == "GOODBYE"
    reply, s = say(s, "No thanks, that's all.", wants_to_end=True)
    assert reply == line("goodbye") and s.outcome == "info_only" and all(v.value is None for v in s.slots.values())


def test_a_new_request_after_anything_else_goes_back_to_taking_a_message():
    s = new()
    _, s = say(s, "Do you cover Overmere?")
    reply, s = say(s, "Yes, I'd like a bathroom quote", reason="a bathroom quote")
    assert reply == line("ask_name") and s.state == "COLLECTING" and s.value("reason") == "a bathroom quote"


def test_a_second_question_in_the_closing_phase_is_answered_too():
    s = new()
    _, s = say(s, "Do you cover Overmere?")
    reply, s = say(s, "Can I pay by card?")
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
    reply, s = say(s, "Do you cover Overmere, and can I pay by card?")
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
