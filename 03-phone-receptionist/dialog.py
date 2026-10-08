"""Version A: the dialog as a state machine in plain code.

    greeting, state = start_call(persona)
    reply, state = next_reply(state, "Hi, this is Mark Thompson, my number is ...", persona, faq, understand_fn)

Every turn goes through the same four steps, and only the third one differs between version A and the agent of Phase 5:

  1. UNDERSTAND  turn.py / rules_turn.py fill a CallerTurn from the caller's words (the only place a model is used).
  2. APPLY       apply_turn(): plain code updates the call state: the details that were really said, a correction, the
                 emergency flag, FAQ questions, yes / no. Both versions share this step, so neither can store an invented value.
  3. DECIDE      decide_a(): the state machine turns the new state into a list of ACTIONS (ask, answer, read back, say goodbye).
  4. RENDER      render(): every action becomes a fixed sentence from persona.json or an approved answer from faq.json.

Emergencies, robocalls, silence and the turn limit are handled by code before and after DECIDE in both versions: they are
safety behaviour, not something a model gets to choose.

States: GREETING (waiting for the first words), COLLECTING (asking for the missing details one at a time, in the order
reason, name, number), READ_BACK (waiting for yes / no), CORRECTING (waiting for what to change), GOODBYE ("anything else?"
was asked), ENDED. The pure-Python signature is  next_reply(state, caller_text, ...) -> (reply_text, new_state), so the
simulated-caller tests run with no screen, microphone or model.
"""

import re
import sys
import time
from pathlib import Path

from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import asks  # noqa: E402
from faq import FaqEntry, match_questions  # noqa: E402
from persona import Persona, first_name, speak_number  # noqa: E402
from rules_turn import AUTOMATED, ENDING, NO, YES  # noqa: E402
from safety import emergency_from_text  # noqa: E402
from spoken import apply_spelling, letter_runs  # noqa: E402
from turn import UnderstandContext, Understanding  # noqa: E402

GREETING, COLLECTING, READ_BACK, CORRECTING, GOODBYE, ENDED = (
    "GREETING", "COLLECTING", "READ_BACK", "CORRECTING", "GOODBYE", "ENDED")
DETAILS = ("reason", "name", "number")
ASK_FOR = {"reason": asks.REASON, "name": asks.NAME, "number": asks.NUMBER, "spelling": asks.SPELLING,
           "number_again": asks.NUMBER_AGAIN}
ASK_LINE = {"reason": "ask_reason", "name": "ask_name", "number": "ask_number", "number_again": "number_refused"}

REFUSED_NUMBER = re.compile(r"\b(you'?ve got my number|you have my number|you already have|already have (it|my number)|"
                            r"(he|she|sam)(?:'s| has| have)? got (it|my number)|(he|she|sam) (knows|has) (it|my number)|on your system|not going to (read|give)|"
                            r"rather not|don'?t want to give|won'?t give|no number)\b", re.I)
REMOVE_NUMBER = re.compile(r"(didn'?t give you a number|take (that|the) number off|there is no number|no number to give)", re.I)
REFUSED_NAME = re.compile(r"\b(rather not|don'?t want to (say|give)|not going to (say|give)|no name|he'?ll know|he knows who|she knows who|"
                          r"knows who i am|just say it'?s me|prefer not)\b", re.I)
MAX_CORRECTION_ASKS = 2   # how often "what should I change?" is asked before reading back what we have
MAX_UNCLEAR_CLOSINGS = 2  # how often "anything else?" is repeated when the answer is not understood


# ---------------------------------------------------------------- the state of one call

class Slot(BaseModel):
    value: str | None = None
    asks: int = 0               # how many times we asked for it (the first ask + re-asks)
    given_up: bool = False      # asked enough, or refused: we record nothing and move on
    confirmed: bool = False     # the caller said "yes" to a read-back that contained it
    spelled: bool = False       # (name) spelling was asked for or given


class CallState(BaseModel):
    state: str = GREETING
    waiting_for: str = asks.GREETING   # the answer the receptionist is waiting for
    asking: str = asks.GREETING        # what a simulated caller should answer (waiting_for, or REPEAT after silence)
    slots: dict[str, Slot] = Field(default_factory=lambda: {d: Slot() for d in DETAILS})
    turns: int = 0
    silent_streak: int = 0
    urgent: bool = False
    urgent_turn: int | None = None
    safety_advised: list[str] = []
    faq_answered: list[str] = []
    unanswered_questions: list[str] = []
    number_refusals: int = 0
    correction_asks: int = 0
    unclear_closings: int = 0
    read_backs: int = 0
    spelling_asked: bool = False
    spam: bool = False
    outcome: str | None = None
    log: list[dict] = []

    def value(self, detail: str) -> str | None:
        return self.slots[detail].value


class Action(BaseModel):
    kind: str          # urgent_ack | advice | faq | faq_unknown | ask | repeat_request | read_back | ask_correction |
    arg: str | None = None  # confirmed | anything_else | goodbye | silence_end | turn_limit


class Facts(BaseModel):
    """What this turn changed, found by plain code (apply_turn). Decisions are made from this, never from the model."""
    waiting_before: str
    spam: bool = False
    new_urgent: bool = False
    advice_kinds: list[str] = []
    faq_ids: list[str] = []
    unknown_question: str | None = None
    yes: bool = False
    no: bool = False
    wants_to_end: bool = False
    changed: list[str] = []
    refused_number: bool = False
    removed_number: bool = False
    refused_name: bool = False
    silent: bool = False


def start_call(persona: Persona) -> tuple[str, CallState]:
    return persona.say("greeting"), CallState()


# ---------------------------------------------------------------- step 2: apply

def accept(state: CallState, detail: str, value: str | None, correcting: bool, facts: Facts) -> bool:
    """Store a detail only if the caller said it. An existing value changes only on an explicit correction."""
    if not value:
        return False
    slot = state.slots[detail]
    if slot.value is None:
        slot.value, slot.confirmed, slot.given_up = value, False, False
        facts.changed.append(detail)
        return True
    if correcting and slot.value.lower() != value.lower():
        slot.value, slot.confirmed = value, False
        facts.changed.append(detail)
        return True
    return False


def apply_turn(state: CallState, text: str, understanding: Understanding, persona: Persona, faq: dict[str, FaqEntry]) -> Facts:
    """Update the call state from one understood turn. Pure code: this is where 'never invent' is enforced."""
    turn = understanding.turn
    facts = Facts(waiting_before=state.waiting_for)

    if turn.is_automated or AUTOMATED.search(text):
        facts.spam = True
        state.spam = True
        return facts

    # --- emergency (before anything else; the model's flag OR Part 1's safety words)
    danger = emergency_from_text(text)
    if turn.emergency or danger:
        facts.new_urgent = not state.urgent
        if facts.new_urgent:
            state.urgent, state.urgent_turn = True, state.turns
        facts.advice_kinds = [k for k in danger.kinds if k not in state.safety_advised]
        state.safety_advised += facts.advice_kinds

    waiting = state.waiting_for
    facts.no = bool(NO.match(text)) or turn.is_correction
    facts.yes = bool(YES.match(text)) and not facts.no
    facts.wants_to_end = turn.wants_to_end or bool(ENDING.search(text))
    # In CORRECTING any value the caller gives IS the correction; in READ_BACK it must be marked as one ("No, it's ...").
    correcting = waiting == asks.CORRECTION or (waiting == asks.CONFIRM and (facts.no or turn.is_correction))

    # --- details: a value counts only when the caller gave it in THIS turn (turn.py grounded it against the words)
    closing = waiting == asks.ANYTHING_ELSE and (facts.no or facts.wants_to_end)  # "No, that's all" is not a new request
    accept(state, "reason", None if closing else turn.reason, correcting, facts)
    if (waiting == asks.CONFIRM and facts.yes) or closing:
        pass  # "yes, that's right" must not re-write anything with a paraphrase
    else:
        accept(state, "name", turn.name, correcting, facts)
        accept(state, "number", turn.number, correcting, facts)

    # --- spelling: letters the caller spelled correct the name we heard
    letters = letter_runs(text)
    if letters and state.value("name") and (waiting in (asks.SPELLING, asks.NAME, asks.CORRECTION, asks.CONFIRM) or "name" in facts.changed):
        merged = apply_spelling(state.value("name"), letters[0])
        if merged != state.value("name"):
            state.slots["name"].value = merged
            if "name" not in facts.changed:
                facts.changed.append("name")
        state.slots["name"].spelled = True
    if waiting == asks.SPELLING:
        state.slots["name"].spelled = True  # asked once; whatever the answer, do not ask again

    # --- a refused name or number: one more try, then we stop asking (asks jumps to the re-ask limit)
    if waiting in (asks.NUMBER, asks.NUMBER_AGAIN) and "number" not in facts.changed and REFUSED_NUMBER.search(text):
        facts.refused_number = True
        state.number_refusals += 1
        state.slots["number"].asks = max(state.slots["number"].asks, persona.max_reasks)
    if waiting == asks.NAME and "name" not in facts.changed and REFUSED_NAME.search(text):
        facts.refused_name = True
        state.slots["name"].asks = max(state.slots["name"].asks, persona.max_reasks)
    if waiting in (asks.CONFIRM, asks.CORRECTION) and REMOVE_NUMBER.search(text) and state.value("number"):
        state.slots["number"].value, state.slots["number"].given_up = None, True
        facts.removed_number = True
        facts.changed.append("number")

    # --- questions: answered from faq.json only. Safety advice already covers "should I turn it off?"
    result = match_questions(text, faq, llm_topic=turn.question_topic)
    # (a topic already answered in this call is not answered again: the caller is probably repeating themselves)
    entries = [e for e in result.entries if not (e.safety and e.safety in state.safety_advised) and e.id not in state.faq_answered]
    for e in entries:
        if e.safety and e.safety not in state.safety_advised:
            state.safety_advised.append(e.safety)
            facts.advice_kinds.append(e.safety)
    facts.faq_ids = [e.id for e in entries if not e.safety]
    state.faq_answered += facts.faq_ids
    if result.unknown and not facts.advice_kinds and not state.safety_advised:
        facts.unknown_question = result.unknown
        state.unanswered_questions.append(result.unknown)
    return facts


# ---------------------------------------------------------------- step 3: decide (version A)

def exhausted(slot: Slot, persona: Persona) -> bool:
    return slot.given_up or slot.asks >= 1 + persona.max_reasks


def next_missing(state: CallState, persona: Persona) -> str | None:
    """The next thing to ask for, or None when nothing is missing. Order: persona.detail_order, with the spelling of
    a full name right after the name."""
    for detail in persona.detail_order:
        slot = state.slots[detail]
        if slot.value is None:
            if exhausted(slot, persona):
                slot.given_up = True
                continue
            return detail
        if detail == "name" and not slot.spelled and not state.spelling_asked and len(slot.value.split()) >= 2:
            return "spelling"
    return None


def decide_a(state: CallState, facts: Facts, persona: Persona) -> list[Action]:
    """The state machine. Pure: reads the state and the facts, returns what to say, changes nothing."""
    actions: list[Action] = []
    if facts.new_urgent:
        actions.append(Action(kind="urgent_ack"))
    actions += [Action(kind="advice", arg=k) for k in facts.advice_kinds]
    actions += [Action(kind="faq", arg=i) for i in facts.faq_ids]
    if facts.unknown_question:
        actions.append(Action(kind="faq_unknown"))
    answered = bool(facts.faq_ids or facts.unknown_question or facts.advice_kinds)
    here = state.state

    def say_goodbye() -> list[Action]:
        return actions + [Action(kind="goodbye", arg="urgent" if state.urgent else "normal")]

    if here == GOODBYE:  # "anything else?" was asked
        if "reason" in facts.changed and state.slots["reason"].value:
            return actions + collect(state, facts, persona)  # a new request: back to taking a message
        if answered:
            return actions + [Action(kind="anything_else")]
        if facts.no or facts.wants_to_end:
            return say_goodbye()
        state.unclear_closings += 1
        if state.unclear_closings >= MAX_UNCLEAR_CLOSINGS:
            return say_goodbye()
        return actions + [Action(kind="anything_else")]

    if here == READ_BACK:
        if facts.changed:
            return actions + [Action(kind="read_back")]
        if facts.yes:
            return actions + [Action(kind="confirmed"), Action(kind="anything_else")]
        if facts.no:
            return actions + [Action(kind="ask_correction")]
        return actions + [Action(kind="read_back")]  # not understood (or only a question): read it back again

    if here == CORRECTING:
        if facts.changed:
            return actions + [Action(kind="read_back")]
        state.correction_asks += 1
        if state.correction_asks > MAX_CORRECTION_ASKS:
            return actions + [Action(kind="read_back")]
        return actions + [Action(kind="ask_correction")]

    # GREETING / COLLECTING
    if facts.wants_to_end and not answered:
        return say_goodbye()
    only_questions = (here == GREETING and facts.faq_ids and not facts.unknown_question
                      and state.value("reason") is None and not facts.changed)
    if only_questions:  # decision 16: someone who only wants information is not pressed for a message
        return actions + [Action(kind="anything_else")]
    return actions + collect(state, facts, persona)


def collect(state: CallState, facts: Facts, persona: Persona) -> list[Action]:
    """Ask for the next missing detail, or read everything back when nothing is missing."""
    if facts.waiting_before in (asks.NUMBER, asks.NUMBER_AGAIN) and facts.refused_number and state.number_refusals >= 2:
        state.slots["number"].given_up = True  # asked, refused twice: record "no number given"
    missing = next_missing(state, persona)
    if missing is None:
        return [Action(kind="read_back")]
    if missing == "number" and facts.refused_number:
        return [Action(kind="ask", arg="number_again")]
    return [Action(kind="ask", arg=missing)]


# ---------------------------------------------------------------- step 4: render + commit

def reason_phrase(state: CallState, persona: Persona) -> str:
    reason = state.value("reason")
    if not reason:
        return persona.say("reason_missing_phrase")
    reason = reason.strip().rstrip(".!?")
    keeps_capital = len(reason) > 1 and ((reason[0].isupper() and reason[1].isupper())  # "CO alarm"
                                         or re.match(r"I(['’ ]|$)", reason))             # "I'm ringing about ..."
    return reason if keeps_capital else reason[0].lower() + reason[1:]


def read_back_text(state: CallState, persona: Persona) -> str:
    name, number = state.value("name"), state.value("number")
    reason = reason_phrase(state, persona)
    if name and number:
        return persona.say("read_back", name=name, number=speak_number(number), reason=reason)
    if name:
        return persona.say("read_back_no_number", name=name, reason=reason)
    if number:
        return persona.say("read_back_no_name", number=speak_number(number), reason=reason)
    return persona.say("read_back_no_name_no_number", reason=reason)


def render(actions: list[Action], state: CallState, persona: Persona, faq: dict[str, FaqEntry]) -> str:
    """Every action becomes a fixed sentence: nothing here is written by a model."""
    parts = []
    for a in actions:
        if a.kind == "urgent_ack":
            parts.append(persona.say("urgent_ack"))
        elif a.kind == "advice":
            parts.append(faq[f"safety_{a.arg}"].answer)
        elif a.kind == "faq":
            parts.append(faq[a.arg].answer)
        elif a.kind == "faq_unknown":
            parts.append(persona.say("faq_unknown"))
        elif a.kind == "ask" and a.arg == "spelling":
            parts.append(persona.say("ask_name_spelling", first_name=first_name(state.value("name") or "")))
        elif a.kind == "ask":
            parts.append(persona.say(ASK_LINE[a.arg]))
        elif a.kind == "repeat_request":
            parts.append(persona.say("repeat_request"))
        elif a.kind == "read_back":
            parts.append(read_back_text(state, persona))
        elif a.kind == "ask_correction":
            parts.append(persona.say("correction"))
        elif a.kind == "confirmed":
            parts.append(persona.say("confirmed"))
        elif a.kind == "anything_else":
            parts.append(persona.say("anything_else"))
        elif a.kind == "goodbye":
            parts.append(persona.say({"urgent": "goodbye_urgent", "spam": "goodbye_spam"}.get(a.arg, "goodbye")))
        elif a.kind == "silence_end":
            parts.append(persona.say("silence_end"))
        elif a.kind == "turn_limit":
            parts.append(persona.say("turn_limit"))
        else:
            raise ValueError(f"unknown action {a.kind!r}")
    return " ".join(parts)


def commit(state: CallState, actions: list[Action], facts: Facts) -> None:
    """Move the state forward to match what was just said (the LAST question-like action decides what we wait for)."""
    for a in actions:
        if a.kind == "ask":
            slot_name = "name" if a.arg == "spelling" else ("number" if a.arg == "number_again" else a.arg)
            state.slots[slot_name].asks += 1 if a.arg != "spelling" else 0
            if a.arg == "spelling":
                state.spelling_asked = True
                state.slots["name"].spelled = True
            state.state = COLLECTING
            state.waiting_for = state.asking = ASK_FOR[a.arg]
        elif a.kind == "repeat_request":
            state.asking = asks.REPEAT
        elif a.kind == "read_back":
            state.state, state.waiting_for = READ_BACK, asks.CONFIRM
            state.asking = asks.CONFIRM
            state.read_backs += 1
        elif a.kind == "ask_correction":
            state.state, state.waiting_for, state.asking = CORRECTING, asks.CORRECTION, asks.CORRECTION
        elif a.kind == "confirmed":
            for slot in state.slots.values():
                slot.confirmed = slot.value is not None
        elif a.kind == "anything_else":
            state.state, state.waiting_for, state.asking = GOODBYE, asks.ANYTHING_ELSE, asks.ANYTHING_ELSE
        elif a.kind in ("goodbye", "silence_end", "turn_limit"):
            state.state, state.waiting_for, state.asking = ENDED, asks.NOTHING, asks.NOTHING
    if state.state == ENDED and state.outcome is None:
        state.outcome = outcome_of(state, actions)


def outcome_of(state: CallState, actions: list[Action]) -> str:
    kinds = {a.kind for a in actions}
    if "silence_end" in kinds:
        return "silence"
    if "turn_limit" in kinds:
        return "turn_limit"
    if any(a.kind == "goodbye" and a.arg == "spam" for a in actions):
        return "spam"
    has_message = any(s.value for s in state.slots.values())
    if not has_message and state.faq_answered and not state.unanswered_questions:
        return "info_only"
    if any(s.confirmed for s in state.slots.values()):
        return "completed"
    return "caller_ended" if has_message or state.unanswered_questions else "no_message"


# ---------------------------------------------------------------- the turn

def default_understand(text: str, ctx: UnderstandContext) -> Understanding:
    from turn import understand
    return understand(text, ctx)


def message_of(state: CallState) -> dict:
    """What the call produced, for the hand-off and the evaluation. Only what the caller said."""
    return {"reason": state.value("reason"), "name": state.value("name"), "number": state.value("number"),
            "urgent": state.urgent, "safety_advised": list(state.safety_advised), "faq_answered": list(state.faq_answered),
            "unanswered_questions": list(state.unanswered_questions), "confirmed": any(s.confirmed for s in state.slots.values()),
            "outcome": state.outcome}


def next_reply(state: CallState, text: str, persona: Persona, faq: dict[str, FaqEntry], understand_fn=None,
               decide_fn=decide_a, heard=None) -> tuple[str, CallState]:
    """One caller turn in, one reply out. `heard` (audio_io.Heard) is optional and only recorded in the log."""
    if state.state == ENDED:
        return "", state
    understand_fn = understand_fn or default_understand
    state.turns += 1
    text = (text or "").strip()
    entry: dict = {"turn": state.turns, "caller_text": text, "waiting_for": state.waiting_for}
    if heard is not None:
        entry["heard"] = {"ignored": heard.ignored, "why": heard.why, "raw_text": heard.raw_text, "min_logprob": heard.min_logprob}

    if not text:  # nobody spoke (or the turn was ignored as silence)
        state.silent_streak += 1
        facts = Facts(waiting_before=state.waiting_for, silent=True)
        actions = ([Action(kind="silence_end")] if state.silent_streak >= persona.max_silent_turns
                   else [Action(kind="repeat_request")])
        entry["understanding"] = None
    else:
        state.silent_streak = 0
        ctx = UnderstandContext(asked=state.waiting_for, name=state.value("name"), number=state.value("number"),
                                reason=state.value("reason"), faq_topics={e.id: e.topic for e in faq.values()})
        started = time.perf_counter()
        understanding = understand_fn(text, ctx)
        entry["understand_s"] = round(time.perf_counter() - started, 3)
        entry["understanding"] = {"turn": understanding.turn.model_dump(), "notes": understanding.notes,
                                  "fallback": understanding.fallback, "attempts": understanding.attempts}
        facts = apply_turn(state, text, understanding, persona, faq)
        if facts.spam:
            actions = [Action(kind="goodbye", arg="spam")]
        else:
            started = time.perf_counter()
            actions = decide_fn(state, facts, persona)
            entry["decide_s"] = round(time.perf_counter() - started, 3)

    actions = apply_limits(state, actions, persona)
    reply = render(actions, state, persona, faq)
    commit(state, actions, facts)
    entry.update(facts=facts.model_dump(), actions=[a.model_dump() for a in actions], reply=reply,
                 state_after=state.state, waiting_for_after=state.waiting_for)
    state.log.append(entry)
    return reply, state


def apply_limits(state: CallState, actions: list[Action], persona: Persona) -> list[Action]:
    """The turn limit: at the last allowed turn the call is wrapped up with what we have."""
    kinds = [a.kind for a in actions]
    if state.turns < persona.max_turns or any(k in ("goodbye", "silence_end", "turn_limit") for k in kinds):
        return actions
    keep = [a for a in actions if a.kind in ("urgent_ack", "advice", "faq", "faq_unknown")]
    if "anything_else" in kinds and state.state != COLLECTING:  # the message was complete: a normal goodbye
        return keep + [a for a in actions if a.kind == "confirmed"] + [Action(kind="goodbye", arg="urgent" if state.urgent else "normal")]
    return keep + [Action(kind="turn_limit")]
