"""Version B: the same receptionist, but a boxed tool-calling agent chooses the next action instead of the state machine.

    decide_b = make_decide_b()                                                  # llm = OLLAMA_MODEL (qwen2.5:7b)
    reply, state = next_reply(state, text, persona, faq, understand_fn, decide_fn=decide_b)

The interface is the one of version A: only `decide` is replaced. Everything before it (understanding, grounding, apply_turn:
nothing invented, emergencies, spam, silence, limits) and everything after it (render: fixed sentences only) is shared, so the
agent can change WHICH legal action comes next, never WHAT is said and never WHAT is stored. The guardrails are Part 2's, adapted:

  - TOOLS: ask, answer_faq, read_back, flag_urgent, take_message, end_call. Nothing else exists for it. No tool takes a value (a name,
    a number, a sentence): values come from apply_turn, which grounded them in the caller's words. The agent cannot invent one.
  - EVERY call is validated (Pydantic) and answered with advice, never a crash ("ERROR: the name is already known ...").
  - LEGALITY comes from the state. The state machine's own decision is always one legal option; a few more are allowed where
    judgement can matter (stop asking after a refusal, ask what to change instead of repeating a read-back, end a call that is
    going nowhere). take_message only after a yes, end_call only when it is legal, ask only for what is legal. An illegal choice is
    refused with what to do instead.
  - STATE, not history: every step the model gets a fresh message (what the caller said, what is known, what is legal, feedback on
    its last calls), as in Part 2, where a growing chat made the agent replay its failing calls.
  - A STEP LIMIT, a NO-PROGRESS stop (the same input twice at temperature 0 gives the same answer) and a FALLBACK: when the agent
    cannot produce a legal action, the state machine's own decision is used and the turn is marked fallback=True. A call never
    depends on the model behaving, and the evaluation counts fallbacks.
  - A TRACE of every step, kept in the call record (entry["decide_trace"]).
  - A SAFETY FLOOR in code: the urgent acknowledgement, the safety advice and "I'll pass your question on" are said whatever the
    agent does. The agent is only asked about the part where judgement can matter.
"""

import copy
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import ollama
from pydantic import BaseModel, Field, ValidationError

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import asks  # noqa: E402
import dialog  # noqa: E402
from dialog import Action, CallState, Facts  # noqa: E402
from persona import Persona  # noqa: E402
from shared.llm import settings  # noqa: E402
from shared.retry import is_transient  # noqa: E402

MAX_STEPS = 4  # LLM calls per caller turn: enough for a few answer_faq calls and the final action
DETAILS = ("reason", "name", "number", "spelling", "correction", "anything_else", "what_else", "repeat")

SYSTEM_PROMPT = """\
You choose the NEXT ACTION of the receptionist of Brightwater Plumbing & Heating, a small UK plumbing and heating business.
You never write what she says: every sentence is fixed. You only call tools. Each turn you get the current state: what the caller
just said, the details we have, and the list of LEGAL NEXT ACTIONS.
1. For every question listed under "Questions to answer", call answer_faq(topic) once.
2. Then call exactly ONE of the legal actions: ask(detail), read_back(), take_message(), end_call(why).
3. Call flag_urgent(why) only if the caller describes danger or damage happening now, or a vulnerable person without heating or hot
   water, and the state does not already say "Urgent: yes".
Rules: ask one thing at a time, for the detail the state says is next. Never ask for a detail we already have. take_message() is only for
after the caller said yes to the read-back, or when the state says the caller added a further request (it is added to the
message). end_call() only when the caller is finished, wants to leave, or the call is going nowhere.
If a tool answers ERROR, read it and choose a legal action.
"""


# ---------------------------------------------------------------- tool arguments (validated)

class AskArgs(BaseModel):
    detail: Literal["reason", "name", "number", "spelling", "correction", "anything_else", "what_else", "repeat"] = Field(
        description="What to ask: a missing detail, 'spelling' of the full name, 'correction' (what should be changed?), "
                    "'anything_else', 'what_else' (after a plain yes), or 'repeat' (say the last question again).")


class AnswerFaqArgs(BaseModel):
    topic: str = Field(description="The id of a question the caller asked, from 'Questions to answer'.")


class ReadBackArgs(BaseModel):
    pass


class FlagUrgentArgs(BaseModel):
    why: str = Field(description="One short phrase: what the danger is.")


class TakeMessageArgs(BaseModel):
    pass


class EndCallArgs(BaseModel):
    why: Literal["caller_finished", "caller_left", "nothing_more_to_do"] = "caller_finished"


TOOLS: dict[str, tuple[type[BaseModel], str]] = {
    "ask": (AskArgs, "Ask the caller for ONE thing (the receptionist's fixed question for it is said)."),
    "answer_faq": (AnswerFaqArgs, "Give the approved answer to a question the caller asked."),
    "read_back": (ReadBackArgs, "Read the message back (name, number, reason) and ask if it is right."),
    "flag_urgent": (FlagUrgentArgs, "Flag the call as urgent (only for danger now; never if already flagged)."),
    "take_message": (TakeMessageArgs, "The caller said yes to the read-back: confirm and ask if there is anything else."),
    "end_call": (EndCallArgs, "Say goodbye and end the call."),
}


def tool_schemas() -> list[dict]:
    return [{"type": "function", "function": {"name": name, "description": text, "parameters": args.model_json_schema()}}
            for name, (args, text) in TOOLS.items()]


# ---------------------------------------------------------------- what is legal now

PREFIX_KINDS = ("urgent_ack", "advice", "faq", "faq_unknown")


def terminal_key(actions: list[Action]) -> str:
    """The tool call a list of terminal actions corresponds to: 'ask:name', 'read_back', 'take_message', 'end_call', ..."""
    first = actions[0]
    if first.kind == "ask":
        return "ask:number" if first.arg == "number_again" else f"ask:{first.arg}"
    return {"read_back": "read_back", "confirmed": "take_message", "goodbye": "end_call", "anything_else": "ask:anything_else",
            "what_else": "ask:what_else", "ask_correction": "ask:correction", "repeat_last": "ask:repeat",
            "silence_end": "end_call", "turn_limit": "end_call", "repeat_request": "ask:repeat"}[first.kind]


@dataclass
class Turn:
    """Everything the agent may do this turn, and what it has done so far."""
    state: CallState
    facts: Facts
    persona: Persona
    candidate: list[Action]                      # the state machine's own decision
    floor: list[Action]                          # urgent_ack / advice / faq_unknown: said whatever the agent does
    pending_faq: list[str]                       # questions the agent must still answer_faq
    legal: dict[str, list[Action]]               # tool-call key -> the actions it stands for
    answered: list[Action] = field(default_factory=list)
    flagged: list[Action] = field(default_factory=list)
    terminal: list[Action] | None = None
    give_up: list[str] = field(default_factory=list)   # refused details that a read_back chosen now gives up (like the state machine)


def build_turn(state: CallState, facts: Facts, persona: Persona) -> Turn:
    # The state machine decides first (this also does its small bookkeeping once). Its answer is always legal.
    candidate = dialog.decide_a(state, facts, persona)
    floor = [a for a in candidate if a.kind in ("urgent_ack", "advice", "faq_unknown")]
    terminal = [a for a in candidate if a.kind not in PREFIX_KINDS]
    legal = {terminal_key(terminal): terminal}
    kind = terminal[0].kind
    # A few more options where judgement can matter. Each one is safe: it only changes WHICH fixed sentence comes next.
    if state.state == dialog.READ_BACK and kind == "read_back" and not facts.changed and not facts.yes and not facts.no:
        legal["ask:correction"] = [Action(kind="ask_correction")]        # not understood: ask what to change instead of repeating
    if state.state == dialog.GOODBYE and kind == "anything_else":
        legal["end_call"] = [Action(kind="goodbye", arg=dialog.goodbye_kind(state))]  # an unclear answer: let them go
    give_up = [d for d, refused in (("name", facts.refused_name), ("number", facts.refused_number)) if refused and not state.value(d)]
    if (state.state in (dialog.GREETING, dialog.COLLECTING) and kind == "ask" and give_up
            and any(s.value for s in state.slots.values()) and nothing_else_missing(state, persona, give_up)):
        legal["read_back"] = [Action(kind="read_back")]                  # they refused: respect it, read back what we have
    pending = [a.arg for a in candidate if a.kind == "faq"]
    return Turn(state, facts, persona, candidate, floor, pending, legal, give_up=give_up)


def nothing_else_missing(state: CallState, persona: Persona, given_up: list[str]) -> bool:
    """Would the state machine have nothing left to ask once the refused details are given up? (tried on a copy)"""
    trial = copy.deepcopy(state)
    for detail in given_up:
        trial.slots[detail].given_up = True
    return dialog.next_missing(trial, persona) is None


def keep_urgent_legal(turn: Turn) -> None:
    """After flag_urgent: an urgent call asks for the number before anything else and skips the spelling, like version A."""
    st = turn.state
    if st.value("number") is not None or dialog.exhausted(st.slots["number"], turn.persona):
        return
    dropped = ("end_call", "read_back", "ask:spelling")
    nxt = dialog.next_missing(st, turn.persona)
    fixed = [Action(kind="ask", arg=nxt)] if nxt else None
    for key in dropped:
        turn.legal.pop(key, None)
    if fixed:
        turn.legal.setdefault(f"ask:{nxt}", fixed)
    prefix = [a for a in turn.candidate if a.kind in PREFIX_KINDS]
    if fixed and terminal_key([a for a in turn.candidate if a.kind not in PREFIX_KINDS]) in dropped:
        turn.candidate = prefix + fixed


def refresh_goodbye(actions: list[Action], state: CallState) -> list[Action]:
    """A goodbye prepared before the agent flagged the call urgent must still carry the urgent promise."""
    return [Action(kind="goodbye", arg=dialog.goodbye_kind(state)) if a.kind == "goodbye" and a.arg in ("normal", "info", "urgent") else a
            for a in actions]


def legal_text(turn: Turn) -> str:
    return " | ".join(f"{k.split(':')[0]}({k.split(':')[1]})" if ":" in k else f"{k}()" for k in turn.legal)


# ---------------------------------------------------------------- the tools

def run_tool(turn: Turn, name: str, raw_args: dict) -> str:
    """Execute one tool call. Every problem becomes an 'ERROR: ...' message for the model, never a crash."""
    if name not in TOOLS:
        return f"ERROR: unknown tool '{name}'. Available: {', '.join(TOOLS)}."
    try:
        args = TOOLS[name][0].model_validate(raw_args)
    except ValidationError as error:
        first = error.errors()[0]
        return f"ERROR: bad arguments for {name}: {first['msg']} ({'.'.join(str(p) for p in first['loc'])}). Legal now: {legal_text(turn)}."
    if turn.terminal is not None:
        return "ERROR: the turn is already decided. Nothing more to do."

    if name == "answer_faq":
        if args.topic not in turn.pending_faq:
            pending = ", ".join(turn.pending_faq) or "none"
            return f"ERROR: the caller did not ask about '{args.topic}' (or it is already answered). Questions to answer: {pending}."
        turn.pending_faq.remove(args.topic)
        turn.answered.append(Action(kind="faq", arg=args.topic))
        return f"OK, the answer to '{args.topic}' will be said."

    if name == "flag_urgent":
        if turn.state.urgent:
            return "ERROR: the call is already flagged urgent. Do not flag it again."
        turn.state.urgent, turn.state.urgent_turn = True, turn.state.turns
        turn.flagged.append(Action(kind="urgent_ack"))
        keep_urgent_legal(turn)
        return "OK, the call is flagged urgent; the acknowledgement will be said."

    # everything else ends the turn: the agent must have answered the questions first, and the action must be legal
    if turn.pending_faq:
        return (f"ERROR: the caller asked a question first: call answer_faq for {', '.join(turn.pending_faq)} before "
                f"{name}. Legal after that: {legal_text(turn)}.")
    key = {"read_back": "read_back", "take_message": "take_message", "end_call": "end_call"}.get(name) or f"ask:{args.detail}"
    if key not in turn.legal:
        return f"ERROR: {key.replace(':', '(')}{')' if ':' in key else '()'} is not legal now. {advice(turn, key)} Legal now: {legal_text(turn)}."
    turn.terminal = refresh_goodbye(turn.legal[key], turn.state)
    if key == "read_back" and turn.give_up and terminal_key([a for a in turn.candidate if a.kind not in PREFIX_KINDS]) != "read_back":
        # reading back although a refused detail is missing: only the refused detail is given up, like the state machine does
        for detail in turn.give_up:
            turn.state.slots[detail].given_up = True
    return f"OK, {key.replace(':', '(')}{')' if ':' in key else '()'} will be done."


def advice(turn: Turn, key: str) -> str:
    """What to do instead of an illegal choice (a bare 'not legal' just gets repeated by a model)."""
    st, f = turn.state, turn.facts
    if key == "take_message":
        return "The caller has not said yes to a read-back or added a request. Use read_back() when nothing is missing, otherwise ask(the next missing detail)."
    if key == "end_call":
        return "The caller has not finished and the message is not complete. Keep going with ask(...) or read_back()."
    if key == "read_back":
        return "Something is still missing: ask for it first."
    if key.startswith("ask:") and key[4:] in ("reason", "name", "number") and st.value(key[4:]):
        return f"The {key[4:]} is already known ({st.value(key[4:])}); never ask for a detail we have."
    if key.startswith("ask:"):
        return "Ask for the detail the state says is next."
    return ""


# ---------------------------------------------------------------- the state message

def state_message(turn: Turn, feedback: list[str], max_turns: int) -> str:
    st, f = turn.state, turn.facts
    known = ", ".join(f"{d}={st.value(d)!r}" for d in dialog.DETAILS if st.value(d)) or "nothing yet"
    given_up = [d for d in dialog.DETAILS if st.slots[d].given_up and not st.value(d)]
    lines = [f"Turn {st.turns} of {max_turns}. State: {st.state}. The receptionist is waiting for: {st.waiting_for}.",
             f"The caller said: {json.dumps(f.caller_text)}",
             f"Details we have: {known}." + (f" Given up on: {', '.join(given_up)}." if given_up else ""),
             f"Urgent: {'yes' if st.urgent else 'no'}.",
             "This turn the caller: " + (", ".join(x for x in (
                 "said yes" if f.yes else "", "said no" if f.no else "", "wants to end the call" if f.wants_to_end else "",
                 "asked to repeat" if f.asks_repeat else "", "refused to give their number" if f.refused_number else "",
                 "refused to give their name" if f.refused_name else "", f"gave/changed {', '.join(f.changed)}" if f.changed else "",
                 f"added a further request: {json.dumps(f.extra_request)}" if f.extra_request else "") if x)
                 or "said nothing that changes the details.")]
    lines.append("Questions to answer: " + (", ".join(turn.pending_faq) if turn.pending_faq else "none") + ".")
    lines.append(f"LEGAL NEXT ACTIONS (choose exactly one after the questions): {legal_text(turn)}")
    if feedback:
        lines += ["", "Results of your last tool calls:"] + feedback
    return "\n".join(lines)


# ---------------------------------------------------------------- the decide function

def ollama_chat(llm: str | None = None):
    client = ollama.Client(host=settings()["host"])

    def chat(messages: list[dict], tools: list[dict]):
        # num_predict caps the reply: a model stuck repeating itself stops after a few hundred tokens, not minutes.
        return client.chat(model=llm or settings()["model"], messages=messages, tools=tools,
                           options={"temperature": 0, "num_predict": 256})

    return chat


def make_decide_b(llm: str | None = None, chat_fn=None, max_steps: int = MAX_STEPS):
    """The agent's decide function (same signature as dialog.decide_a). `chat_fn(messages, tools)` is injectable for tests."""
    chat = chat_fn or ollama_chat(llm)

    def decide_b(state: CallState, facts: Facts, persona: Persona) -> list[Action]:
        turn = build_turn(state, facts, persona)
        trace: list[dict] = []
        feedback: list[str] = []
        last_message = None
        llm_calls, seconds, fallback_why = 0, 0.0, None
        for step in range(1, max_steps + 1):
            message = state_message(turn, feedback, persona.max_turns)
            if message == last_message:  # the same input gives the same answer at temperature 0: stop, do not loop
                fallback_why = "no progress: the same input as the last step"
                trace.append({"step": step, "stopped": fallback_why})
                break
            last_message = message
            started = time.perf_counter()
            try:
                response = chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": message}], tool_schemas())
            except Exception as error:  # noqa: BLE001
                if not (isinstance(error, (ollama.ResponseError,)) or is_transient(error)):
                    raise
                fallback_why = f"model error: {type(error).__name__}"
                trace.append({"step": step, "error": str(error)[:200]})
                break
            finally:
                llm_calls += 1
                seconds += time.perf_counter() - started
            calls = response.message.tool_calls or []
            feedback = []
            if not calls:  # answered in prose: steer it back to the tools
                trace.append({"step": step, "tool": None, "said": (response.message.content or "")[:200]})
                feedback = ["You answered in text. Use the tools."]
                continue
            for call in calls:
                args = dict(call.function.arguments)
                result = run_tool(turn, call.function.name, args)
                trace.append({"step": step, "tool": call.function.name, "args": args, "result": result})
                feedback.append(f"{call.function.name}({json.dumps(args)}) -> {result}")
                if turn.terminal is not None:
                    break
            if turn.terminal is not None:
                break
        else:
            fallback_why = fallback_why or "step limit reached"

        decide_b.last_trace = {"steps": llm_calls, "llm_s": round(seconds, 3), "fallback": turn.terminal is None,
                               "fallback_why": fallback_why if turn.terminal is None else None, "trace": trace,
                               "legal": list(turn.legal)}
        if turn.terminal is None:
            # the state machine's own decision: a complete list, safe by construction (plus the acknowledgement if the agent
            # flagged the call urgent before it failed)
            return turn.flagged + refresh_goodbye(turn.candidate, state) if turn.flagged else turn.candidate
        faq_unknown = [a for a in turn.floor if a.kind == "faq_unknown"]
        floor = [a for a in turn.floor if a.kind != "faq_unknown"]
        return floor + turn.flagged + turn.answered + faq_unknown + turn.terminal

    decide_b.last_trace = None
    return decide_b
