"""Version B, the tool-calling agent: guardrails, fallback, equivalence with version A, and the shared safety net (no model: scripted tool calls)."""

import json
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import ollama
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

import agent_dialog  # noqa: E402
import asks  # noqa: E402
import call as call_module  # noqa: E402
import dialog  # noqa: E402
import faq as faq_module  # noqa: E402
from agent_dialog import make_decide_b  # noqa: E402
from cards import load_cards  # noqa: E402
from dialog import next_reply, start_call  # noqa: E402
from persona import load_persona, speak_number  # noqa: E402
from rules_turn import rules_understand  # noqa: E402
from turn import CallerTurn, Understanding  # noqa: E402

P = load_persona()
FAQ = faq_module.load_faq()
NUMBER = "01632960501"
FULL = "My tap is dripping. I'm Dave. My number is 01632 960 501."  # three sentences the rules baseline reads completely
UNDERSTAND = lambda text, ctx: Understanding(rules_understand(text, ctx))  # noqa: E731


def tc(name, **args):
    return NS(function=NS(name=name, arguments=args))


def scripted(*steps):
    """A fake chat: each call returns the next list of tool calls. Records what it was shown."""
    seen = []
    queue = list(steps)

    def chat(messages, tools):
        seen.append({"messages": messages, "tools": tools})
        calls = queue.pop(0) if queue else []
        return NS(message=NS(tool_calls=calls, content="" if calls else "I think we should ask for the name"))

    chat.seen = seen
    return chat


def form(**changes) -> CallerTurn:
    base = dict(heard_summary="x", name=None, number=None, reason=None, is_correction=False, correction_field=None,
                question_topic=None, emergency=False, wants_to_end=False, is_automated=False)
    return CallerTurn.model_validate({**base, **changes}, context={"faq_ids": list(FAQ)})


def new():
    return start_call(P)[1]


def say(state, text, chat, **fields):
    understand = (lambda t, c: Understanding(form(**fields))) if fields else UNDERSTAND
    decide = make_decide_b(chat_fn=chat)
    reply, state = next_reply(state, text, P, FAQ, understand, decide_fn=decide)
    return reply, state, decide.last_trace


def line(key, **kw):
    return P.say(key, **kw)


# ---------------------------------------------------------------- the tools

def test_a_legal_choice_is_carried_out_and_the_reply_is_a_fixed_sentence():
    reply, s, trace = say(new(), "Hi, my tap is dripping", scripted([tc("ask", detail="name")]))
    assert reply == line("ask_name") and s.waiting_for == asks.NAME and not trace["fallback"] and trace["steps"] == 1


def test_asking_for_a_detail_we_already_have_is_refused_with_advice_then_corrected():
    chat = scripted([tc("ask", detail="reason")], [tc("ask", detail="name")])
    reply, s, trace = say(new(), "Hi, my tap is dripping", chat)
    assert reply == line("ask_name") and trace["steps"] == 2 and not trace["fallback"]
    assert "already known" in trace["trace"][0]["result"] and "never ask for a detail we have" in trace["trace"][0]["result"]
    assert "ERROR" in chat.seen[1]["messages"][-1]["content"]  # the model is shown its mistake in the next state message


def test_take_message_is_refused_until_the_caller_said_yes():
    s = new()
    _, s, _ = say(s, FULL, scripted([tc("read_back")]))
    chat = scripted([tc("take_message")], [tc("read_back")])
    reply, s, trace = say(s, "Hmm, let me think", chat)
    assert "not said yes" in trace["trace"][0]["result"] and s.state == "READ_BACK" and not any(x.confirmed for x in s.slots.values())
    s2 = new()
    _, s2, _ = say(s2, FULL, scripted([tc("read_back")]))
    reply, s2, trace = say(s2, "Yes, that's right", scripted([tc("take_message")]))
    assert reply == f"{line('confirmed')} {line('anything_else')}" and all(x.confirmed for x in s2.slots.values() if x.value)


def test_end_call_is_refused_while_the_caller_is_still_giving_a_message():
    chat = scripted([tc("end_call", why="nothing_more_to_do")], [tc("ask", detail="name")])
    reply, s, trace = say(new(), "Hi, my tap is dripping", chat)
    assert "not finished" in trace["trace"][0]["result"] and s.state != "ENDED" and reply == line("ask_name")


def test_read_back_is_refused_while_a_detail_is_missing():
    chat = scripted([tc("read_back")], [tc("ask", detail="name")])
    reply, s, trace = say(new(), "Hi, my tap is dripping", chat)
    assert "not legal" in trace["trace"][0]["result"] and reply == line("ask_name")


def test_a_caller_who_leaves_may_be_let_go():
    reply, s, trace = say(new(), "Bye.", scripted([tc("end_call", why="caller_left")]), wants_to_end=True)
    assert reply == line("goodbye_info") and s.state == "ENDED"


def test_every_question_must_be_answered_before_the_action_and_only_questions_that_were_asked():
    text = "My radiator is cold. How soon could someone come out? What are your opening hours?"
    chat = scripted([tc("ask", detail="name")],                                   # forgot the questions
                    [tc("answer_faq", topic="payment")],                          # a topic nobody asked about
                    [tc("answer_faq", topic="booking_time"), tc("answer_faq", topic="hours"), tc("ask", detail="name")])
    reply, s, trace = say(new(), text, chat)
    assert "call answer_faq for booking_time, hours" in trace["trace"][0]["result"]
    assert "did not ask about 'payment'" in trace["trace"][1]["result"]
    assert reply == f"{FAQ['booking_time'].answer} {FAQ['hours'].answer} {line('ask_name')}"


def test_flag_urgent_works_once_and_never_for_a_call_that_is_already_urgent():
    chat = scripted([tc("flag_urgent", why="elderly, no heating"), tc("flag_urgent", why="again"), tc("ask", detail="name")])
    reply, s, trace = say(new(), "It's very cold here and my mother is eighty-four", chat, reason="no heating")
    assert s.urgent and s.urgent_turn == 1 and reply.startswith(line("urgent_ack")) and reply.endswith(line("ask_name"))
    assert "already flagged" in trace["trace"][1]["result"] and reply.count(line("urgent_ack")) == 1


def test_unknown_tools_and_bad_arguments_are_answered_with_advice_not_crashes():
    chat = scripted([tc("delete_everything")], [tc("ask", detail="shoe size")], [tc("ask", detail="name")])
    reply, s, trace = say(new(), "Hi, my tap is dripping", chat)
    assert "unknown tool" in trace["trace"][0]["result"] and "bad arguments for ask" in trace["trace"][1]["result"]
    assert "Legal now: ask(name)" in trace["trace"][1]["result"] and reply == line("ask_name")


def test_no_tool_accepts_a_value_so_the_agent_cannot_invent_one():
    for name, (args, _) in agent_dialog.TOOLS.items():
        fields = set(args.model_fields)
        assert not fields & {"name", "number", "reason", "text", "sentence", "value", "phone"}, name


# ---------------------------------------------------------------- the safety net is code, not the agent's choice

def test_the_urgent_acknowledgement_and_the_safety_advice_are_said_even_if_the_agent_ignores_them():
    reply, s, trace = say(new(), "There's a smell of gas in the hallway", scripted([tc("ask", detail="name")]))
    assert reply.startswith(line("urgent_ack"))
    assert FAQ["safety_gas"].answer in reply and s.urgent and reply.endswith(line("ask_name"))


def test_an_unknown_question_is_passed_on_even_if_the_agent_ignores_it():
    reply, s, _ = say(new(), "Do you fit solar panels?", scripted([tc("ask", detail="reason")]), question_topic="other")
    assert reply == f"{line('faq_unknown')} {line('ask_reason')}" and s.unanswered_questions == ["Do you fit solar panels?"]


def test_robocalls_silence_and_the_turn_limit_never_reach_the_agent():
    def must_not_run(messages, tools):
        raise AssertionError("the agent was asked")

    decide = make_decide_b(chat_fn=must_not_run)
    s = new()
    reply, s = next_reply(s, "This is an automated message. Press one now.", P, FAQ, UNDERSTAND, decide_fn=decide)
    assert reply == line("goodbye_spam")
    s = new()
    reply, s = next_reply(s, "", P, FAQ, UNDERSTAND, decide_fn=decide)
    assert reply == line("repeat_request")
    s = new()
    s.turns = P.max_turns - 1
    _, s = next_reply(s, "blah blah", P, FAQ, UNDERSTAND, decide_fn=make_decide_b(chat_fn=scripted([tc("ask", detail="reason")])))
    assert s.outcome == "turn_limit"


def test_the_agent_cannot_store_anything_nothing_is_invented_whatever_it_does():
    s = new()
    _, s, _ = say(s, "hello", scripted([tc("ask", detail="reason")]), name="Invented", number="07700900999", reason="invented")
    assert {k: v.value for k, v in s.slots.items()} == {"reason": None, "name": None, "number": None}


# ---------------------------------------------------------------- where judgement may differ from the state machine

def test_after_an_unclear_answer_to_the_read_back_the_agent_may_ask_what_to_change_instead_of_repeating():
    s = new()
    _, s, _ = say(s, FULL, scripted([tc("read_back")]))
    reply, s, trace = say(s, "Hmm, I'm not sure", scripted([tc("ask", detail="correction")]))
    assert reply == line("correction") and s.state == "CORRECTING" and "ask:correction" in trace["legal"]


def test_after_a_refusal_the_agent_may_stop_asking_and_read_back_what_it_has():
    s = new()
    _, s, _ = say(s, "My tap is dripping. I'm Dave.", scripted([tc("ask", detail="number")]))
    reply, s, trace = say(s, "You've got my number", scripted([tc("read_back")]))
    assert reply == line("read_back_no_number", name="Dave", reason="my tap is dripping")
    assert s.state == "READ_BACK" and s.slots["number"].given_up and "read_back" in trace["legal"]


def test_an_unclear_answer_to_anything_else_may_end_the_call_or_ask_again():
    s = new()
    _, s, _ = say(s, FULL, scripted([tc("read_back")]))
    _, s, _ = say(s, "Yes", scripted([tc("take_message")]))
    reply, s, trace = say(s, "Hmm", scripted([tc("end_call", why="nothing_more_to_do")]))
    assert reply == line("goodbye") and s.state == "ENDED"


# ---------------------------------------------------------------- limits and fallback

def test_when_the_model_only_talks_the_state_machine_decides_and_the_turn_is_marked_a_fallback():
    chat = scripted([], [], [], [])
    reply, s, trace = say(new(), "Hi, my tap is dripping", chat)
    assert reply == line("ask_name") and trace["fallback"] and trace["steps"] <= agent_dialog.MAX_STEPS
    assert any(step.get("said") for step in trace["trace"])


def test_the_step_limit_ends_a_model_that_keeps_choosing_illegal_actions():
    chat = scripted(*[[tc("take_message")]] * 10)
    reply, s, trace = say(new(), "Hi, my tap is dripping", chat)
    assert trace["fallback"] and trace["steps"] <= agent_dialog.MAX_STEPS and reply == line("ask_name")


def test_the_same_input_twice_stops_the_loop_instead_of_repeating_the_same_calls():
    calls = []

    def stubborn(messages, tools):
        calls.append(1)
        return NS(message=NS(tool_calls=[tc("take_message")], content=""))

    decide = make_decide_b(chat_fn=stubborn, max_steps=6)
    reply, s = next_reply(new(), "Hi, my tap is dripping", P, FAQ, UNDERSTAND, decide_fn=decide)
    assert len(calls) < 6 and "no progress" in decide.last_trace["fallback_why"] and reply == line("ask_name")


def test_ollama_errors_fall_back_but_a_bug_in_the_chat_function_does_not():
    def runner_died(messages, tools):
        raise ollama.ResponseError("llama runner process has terminated", 500)

    reply, s, trace = say(new(), "Hi, my tap is dripping", runner_died)
    assert trace["fallback"] and "model error" in trace["fallback_why"] and reply == line("ask_name")

    def bug(messages, tools):
        raise KeyError("my own mistake")

    with pytest.raises(KeyError):
        say(new(), "Hi, my tap is dripping", bug)


def test_the_trace_is_in_the_call_log_and_the_record_says_which_decider_ran():
    card = next(c for c in load_cards() if c.id.startswith("c14"))
    chat = scripted(*[[tc("ask", detail="name")]] * 30)
    record = call_module.run_call(card, P, FAQ, UNDERSTAND, decide_fn=make_decide_b(chat_fn=chat), understand_label="rules", decide_label="b")
    assert record.decide == "b" and record.turns[0]["decide_trace"]["steps"] >= 1 and "legal" in record.turns[0]["decide_trace"]


def test_the_state_message_shows_what_the_model_needs_and_nothing_it_could_copy_as_an_answer():
    chat = scripted([tc("ask", detail="name")])
    say(new(), "Hi, my tap is dripping", chat)
    system, user = (m["content"] for m in chat.seen[0]["messages"])
    flat = " ".join(system.split())
    assert "You never write what she says" in flat and "take_message() is only for after the caller said yes" in flat
    assert 'The caller said: "Hi, my tap is dripping"' in user and "LEGAL NEXT ACTIONS" in user and "ask(name)" in user
    assert {t["function"]["name"] for t in chat.seen[0]["tools"]} == set(agent_dialog.TOOLS)


# ---------------------------------------------------------------- same interface, same behaviour

def first_legal_agent():
    """A fake agent that always takes the first legal option listed (= the state machine's own decision)."""
    def chat(messages, tools):
        user = messages[-1]["content"]
        questions = re.search(r"Questions to answer: (.*)\.", user).group(1)
        legal = re.search(r"LEGAL NEXT ACTIONS .*?: (.*)$", user, re.M).group(1).split(" | ")
        calls = [tc("answer_faq", topic=t) for t in questions.split(", ")] if questions != "none" else []
        first = legal[0]
        if first.startswith("ask("):
            calls.append(tc("ask", detail=first[4:-1]))
        elif first == "end_call()":
            calls.append(tc("end_call", why="caller_finished"))
        else:
            calls.append(tc(first[:-2]))
        return NS(message=NS(tool_calls=calls, content=""))
    return chat


def test_an_agent_that_takes_the_state_machines_choice_reproduces_version_a_on_all_24_callers():
    for card in load_cards():
        a = call_module.run_call(card, P, FAQ, UNDERSTAND, understand_label="rules")
        b = call_module.run_call(card, P, FAQ, UNDERSTAND, decide_fn=make_decide_b(chat_fn=first_legal_agent()), understand_label="rules",
                                 decide_label="b")
        assert [t["reply"] for t in a.turns] == [t["reply"] for t in b.turns], card.id
        assert a.message["name"] == b.message["name"] and a.message["number"] == b.message["number"] and a.outcome == b.outcome
        assert not any(t.get("decide_trace", {}).get("fallback") for t in b.turns), card.id


def test_the_command_line_can_run_the_agent(monkeypatch, capsys):
    monkeypatch.setattr(agent_dialog, "make_decide_b", lambda *a, **k: make_decide_b(chat_fn=first_legal_agent()))
    assert call_module.main(["--card", "c14", "--understand", "rules", "--decide", "b"]) == 0
    assert "--- outcome=completed" in capsys.readouterr().out


# ---------------------------------------------------------------- a badly behaved agent cannot break the engine

def test_a_random_agent_cannot_break_any_invariant():
    """Random tool calls, legal or not, for 200 calls: the call still ends within the limits, urgent calls keep their promise,
    nothing contains a digit, and every reply is made of approved pieces only."""
    rng = random.Random(20261008)
    pool = ["", "Hello?", "Yes", "No", "My name is Dave Smith", "My number is 07700 900123", "There's a smell of gas in the hallway",
            "It's about a leak under the sink", "Do you cover Overmere?", "What are your opening hours?", "That's all, thanks", "Bye.",
            "Do you fit solar panels?", "Actually my number is 07700 900222", "No, it's 349", "Press one"]
    tools = [("ask", lambda: {"detail": rng.choice(list(agent_dialog.DETAILS) + ["bogus"])}), ("answer_faq", lambda: {"topic": rng.choice(list(FAQ) + ["x"])}),
             ("read_back", lambda: {}), ("flag_urgent", lambda: {"why": "x"}), ("take_message", lambda: {}),
             ("end_call", lambda: {"why": rng.choice(["caller_finished", "bogus"])}), ("nonsense", lambda: {})]

    def random_chat(messages, tools_):
        if rng.random() < 0.15:
            return NS(message=NS(tool_calls=[], content="hmm"))
        calls = []
        for _ in range(rng.randint(1, 3)):
            name, make = rng.choice(tools)
            calls.append(tc(name, **make()))
        return NS(message=NS(tool_calls=calls, content=""))

    piece = re.compile("(?:" + "|".join([re.escape(e.answer) for e in FAQ.values()] +
                                       [re.escape(re.sub(r"\{(\w+)\}", "\x00", t)).replace("\x00", ".+?") for t in P.lines.values()]) + ")")
    for n in range(200):
        s = new()
        decide = make_decide_b(chat_fn=random_chat)
        last = ""
        for _ in range(P.max_turns + 2):
            reply, s = next_reply(s, rng.choice(pool), P, FAQ, UNDERSTAND, decide_fn=decide)
            if s.state == "ENDED":
                last = reply
                break
        assert s.state == "ENDED" and s.outcome and s.turns <= P.max_turns, n
        if s.urgent:
            assert s.outcome != "spam" and "nine nine nine" in last, (n, last)
        for entry in s.log:
            assert not any(ch.isdigit() for ch in entry["reply"]), (n, entry["reply"])
            assert re.fullmatch(f"(?:{piece.pattern})(?: (?:{piece.pattern}))*", entry["reply"]) or entry["reply"] == "", (n, entry["reply"])


def test_an_agent_that_flags_urgent_and_then_ends_the_call_still_gives_the_urgent_promise():
    s = new()
    chat = scripted([tc("flag_urgent", why="gas"), tc("end_call", why="caller_left")])
    reply, s, trace = say(s, "Bye.", chat, wants_to_end=True)
    assert s.urgent and reply == f"{line('urgent_ack')} {line('goodbye_urgent')}" and "nine nine nine" in reply


def test_an_agent_that_flags_urgent_and_then_fails_still_gets_the_acknowledgement():
    chat = scripted([tc("flag_urgent", why="gas")], [tc("take_message")], [tc("take_message")], [tc("take_message")])
    reply, s, trace = say(new(), "Hi, my tap is dripping", chat)
    assert trace["fallback"] and s.urgent and reply.startswith(line("urgent_ack")) and reply.endswith(line("ask_name"))
