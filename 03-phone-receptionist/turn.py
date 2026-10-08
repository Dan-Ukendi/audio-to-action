"""Understanding ONE caller turn: the LLM fills a small form, plain code checks every value against what was said.

    u = understand("It's Siobhan Gallagher, my number is 01632 960 501", context)
    u.turn.name, u.turn.number, u.notes      # validated, grounded values + what the checks changed

The model never decides what happens next (that is dialog.py) and never writes what Holly says. It only reads the
caller's words and reports: the details given, a correction, a question, an emergency, the wish to end the call.

Two layers keep it honest (the Part 1/2 lessons: models satisfy a rule the cheapest way, and a validator needs a
check that the "fix" is real):
  1. Validators on the form (CallerTurn): the UK number rule shared with Part 1, a known question topic, short reason.
     A failure is shown to the model once (shared/llm.structured_chat), exactly as in Parts 1-2.
  2. GROUNDING (ground()): a value is kept only if the caller really said it in THIS turn. A number must appear as digits
     in the speech, a name must be made of words (or spelled letters) that were said, a reason must share its content
     words with the speech. Anything else is dropped, never "repaired" by guessing.

If the model cannot produce a valid form (twice invalid, or Ollama unreachable) the call does not crash: the plain-code
baseline in rules_turn.py answers instead and the result says so (fallback=True).
"""

import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import ollama
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import asks  # noqa: E402
from shared.llm import LLMFormError, structured_chat  # noqa: E402
from shared.retry import is_transient  # noqa: E402
from shared.schemas import normalize_uk_number  # noqa: E402
from spoken import digit_runs, letter_runs  # noqa: E402

PROMPT_VERSION = "t1"  # bump when the prompt changes (never edit a version in place: saved calls record it)
MAX_REASON_WORDS = 14

NOT_A_VALUE = {"", "unknown", "none", "n/a", "null", "not said", "not given", "nobody"}


class CallerTurn(BaseModel):
    """What the model must fill in for one caller turn. Field ORDER matters: it summarizes first, then commits."""

    heard_summary: str = Field(description="One short sentence: what the caller just said.")
    name: str | None = Field(description="The caller's own name as they said or spelled it; null if not said in this message.")
    number: str | None = Field(description="UK phone number to call them back on, digits only; null if not said in this message.")
    reason: str | None = Field(description="Why they call: a short phrase in their own words, max 12 words; null if not said.")
    is_correction: bool = Field(description="True if the caller is correcting something said earlier, e.g. answering 'No, it's ...'.")
    correction_field: Literal["name", "number", "reason"] | None = Field(description="Which detail they are correcting; null if none.")
    question_topic: str | None = Field(description="If they ask a question about the business: the id of the topic that answers it, "
                                                   "or 'other' if no topic fits; null if they ask no question.")
    emergency: bool = Field(description="True only for danger or damage happening now, a vulnerable person without heating or "
                                        "hot water, or a hard deadline today or tomorrow.")
    wants_to_end: bool = Field(description="True if the caller says they are finished: that's all, no thanks, goodbye.")
    is_automated: bool = Field(description="True for a robocall, a recorded or automated message, a scam or 'press one'.")

    @field_validator("name", "reason")
    @classmethod
    def empty_is_none(cls, value: str | None) -> str | None:
        # Models write "" or "unknown" instead of null; treat them the same (as Part 1's Analysis does).
        if value is None or value.strip().lower() in NOT_A_VALUE:
            return None
        return value.strip()

    @field_validator("reason")
    @classmethod
    def reason_is_short(cls, value: str | None) -> str | None:
        if value and len(value.split()) > MAX_REASON_WORDS:
            raise ValueError(f"the reason must be a short phrase of at most {MAX_REASON_WORDS} words, not a copy of the message.")
        return value

    @field_validator("number")
    @classmethod
    def uk_number(cls, value: str | None) -> str | None:
        return normalize_uk_number(value)  # the same rule as Part 1's callback_number

    @field_validator("question_topic")
    @classmethod
    def known_topic(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None or value.strip().lower() in NOT_A_VALUE:
            return None
        ids = (info.context or {}).get("faq_ids")
        value = value.strip()
        if ids is not None and value != "other" and value not in ids:
            raise ValueError(f"'{value}' is not a topic. Use one of: {', '.join(ids)}, or 'other' if none fits, or null.")
        return value

    @model_validator(mode="after")
    def correction_field_needs_correction(self) -> "CallerTurn":
        if not self.is_correction:
            self.correction_field = None  # a field without a correction means nothing; drop it rather than argue
        return self


@dataclass
class UnderstandContext:
    """What the model is told about the call so far. Known values are shown so it does not repeat them as news."""
    asked: str = asks.GREETING
    name: str | None = None
    number: str | None = None
    reason: str | None = None
    faq_topics: dict[str, str] = field(default_factory=dict)  # id -> topic title


@dataclass
class Understanding:
    turn: CallerTurn
    notes: list[str] = field(default_factory=list)  # what grounding dropped or replaced, for the call record
    fallback: bool = False       # True = the model failed and the plain-code baseline answered
    attempts: int = 1
    seconds: float = 0.0
    raw: dict | None = None      # the model's form before grounding (kept for inspection)


ASKED_TEXT = {
    asks.GREETING: "how she can help (the call just started)",
    asks.REASON: "what the call is about",
    asks.NAME: "the caller's name",
    asks.SPELLING: "the caller's full name spelled letter by letter",
    asks.NUMBER: "the best number to call back on",
    asks.NUMBER_AGAIN: "the number to call back on (asked a second time)",
    asks.REPEAT: "the caller to say that again",
    asks.CONFIRM: "whether the name, number and reason she read back are right",
    asks.CORRECTION: "what should be changed: the name, the number or what it is about",
    asks.ANYTHING_ELSE: "whether there is anything else she can help with",
    asks.NOTHING: "nothing",
}

SYSTEM_PROMPT = """\
You help the receptionist of Brightwater Plumbing & Heating, a small UK plumbing and heating business.
You read ONE message from a caller on the phone and fill in a form. The message was made by speech recognition,
so it may contain mistakes. Answer with JSON only, in the requested format.

Fill in ONLY what the caller said in THIS message. Never guess. Never copy a value from "Already known":
it is shown only so you can tell news from repetition.

- name: the caller's own name, not the business and not the person they are calling. If they spell it letter by letter,
  use the spelling. A relationship word like "Mum" counts if that is all they give. null if they do not say it.
- number: the number to call them back on, digits only. Convert spoken forms: "oh" = 0, "double 7" = 77, "nine hundred" = 900.
  Ignore house numbers, times, prices and error codes. If they correct themselves, use the corrected number.
  null if no number is spoken (for example "you've got my number"). Never guess.
- reason: why they are calling, as a short phrase in their own words (max 12 words), e.g. "a dripping outside tap".
- is_correction / correction_field: true if they are fixing something said earlier (usually after a read-back: "No, it's ...").
- question_topic: if they ask a question about the business, the id of the topic below that answers it, or "other" if no topic fits.
  Asking someone to call back or come out is a request, not a question: null.
- emergency: true only for danger or damage happening now (gas smell, carbon monoxide, flooding, a burst pipe, sparks near
  water), a vulnerable person without heating or hot water, or a hard deadline today or tomorrow. A message that says "urgent"
  but is automated, a scam or a sales pitch is NOT an emergency.
- wants_to_end: true if they say they are finished (that's all, no thanks, goodbye).
- is_automated: true for a robocall, a recorded message, a scam, or "press one".
"""

FIX_HINT = ("Fix the values using the caller's words only; do not replace a value the caller actually said with null, "
            "and do not invent one. Return the corrected JSON.")


def build_messages(text: str, ctx: UnderstandContext) -> list[dict]:
    topics = "\n".join(f"  {topic_id}: {title}" for topic_id, title in ctx.faq_topics.items())
    known = ", ".join(f"{k}={v!r}" for k, v in (("name", ctx.name), ("number", ctx.number), ("reason", ctx.reason)) if v) or "nothing yet"
    return [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\nTopics for question_topic:\n{topics}\n"},
        {"role": "user", "content": f"The receptionist just asked for: {ASKED_TEXT[ctx.asked]}.\nAlready known: {known}.\n"
                                    f"Caller said:\n<<<\n{text}\n>>>"},
    ]


# ---------------------------------------------------------------- grounding

def stem(word: str) -> str:
    """Crude on purpose: 'dripping' / 'drips' / 'dripped' -> 'drip'. Only used to compare words with each other."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def words_of(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower().replace("’", "'"))


STOPWORDS = {"a", "an", "the", "my", "our", "your", "of", "to", "for", "with", "and", "about", "some", "in", "on", "at", "it", "is",
             "are", "be", "from", "that", "this", "i", "we", "me"}  # ("no" and "not" are NOT stopwords: "no heating" != "heating")


def name_is_grounded(name: str, text: str) -> bool:
    """The name is made of words the caller said, or of letters they spelled (the WHOLE spelled run, not a slice of it)."""
    parts = words_of(name)
    if not parts:
        return False  # "12345" or "-" is not a name
    said = set(words_of(text))
    runs = {run.lower() for run in letter_runs(text)}
    if all(part in said or part in runs for part in parts):
        return True
    return "".join(parts) in runs  # the whole name spelled in one run: S-I-O-B-H-A-N-G-A-L-L-A-G-H-E-R


def reason_is_grounded(reason: str, text: str) -> bool:
    """EVERY content word of the reason (stopwords aside, short words included) appears, stemmed, in what was said.

    Strict on purpose: a paraphrase that adds a word the caller never said ("gas boiler" for "boiler") is replaced by the
    caller's own words (see ground()). Nothing in a message may be the model's invention."""
    content = [stem(w) for w in words_of(reason) if w not in STOPWORDS]
    if not content:
        return False
    said = {stem(w) for w in words_of(text)}
    return all(w in said for w in content)


GREETING_PREFIX = re.compile(r"^(?:(?:hi|hello|hey|oh|um|yeah|yes|alright|good (?:morning|afternoon|evening)|there|mate|love|sorry about the noise)[,.!\s]+)+", re.I)


TRAILING_FILLER = {"and", "we", "the", "a", "an", "to", "of", "my", "i", "it", "that", "but", "or", "so", "in", "on", "at", "with", "for", "is", "are"}


INTRODUCTION = re.compile(r"^(?:my name is|my name's|this is|it's|it is|i'm|i am|name's)\s+\S+(?:\s+\S+)?$", re.I)


def fallback_reason(text: str) -> str | None:
    """The caller's own first substantial sentence, trimmed: used when the model's phrase is not in the speech.

    Grounded by construction (it IS the speech), so nothing is invented; it just reads less neatly in the read-back.
    """
    for sentence in re.split(r"(?<=[.?!])\s+", text.strip()):
        sentence = GREETING_PREFIX.sub("", sentence).strip(" ,.!?")
        has_number = bool(digit_runs(sentence, min_len=6))
        if len(words_of(sentence)) >= 3 and not (has_number and len(sentence.split()) <= 10) and not INTRODUCTION.match(sentence):
            words = sentence.split()[:MAX_REASON_WORDS]
            while len(words) > 3 and words[-1].lower().strip(",") in TRAILING_FILLER:
                words.pop()  # do not end on "and we" / "the" when the sentence was cut
            return " ".join(words).rstrip(",")
    return None


def ground(turn: CallerTurn, text: str) -> tuple[CallerTurn, list[str]]:
    """Keep only what the caller really said in this turn. Returns the checked turn and a note for every change."""
    notes: list[str] = []
    out = turn.model_copy()
    if out.number is not None:
        if not any(out.number in run for run in digit_runs(text, min_len=8)):
            notes.append(f"number {out.number} is not in what was said: dropped")
            out.number = None
    if out.name is not None and not name_is_grounded(out.name, text):
        notes.append(f"name {out.name!r} is not in what was said: dropped")
        out.name = None
    if out.reason is not None and not reason_is_grounded(out.reason, text):
        replacement = fallback_reason(text)
        notes.append(f"reason {out.reason!r} is not in what was said: " + (f"replaced by the caller's own words" if replacement else "dropped"))
        out.reason = replacement
    return out, notes


# ---------------------------------------------------------------- the step

def understand(text: str, ctx: UnderstandContext, chat=structured_chat, llm: str | None = None,
               fallback=None) -> Understanding:
    """One caller turn -> a validated, grounded CallerTurn. `chat` is injectable so tests need no model."""
    messages = build_messages(text, ctx)
    started = time.perf_counter()
    try:
        reply = chat(CallerTurn, messages, llm=llm, context={"faq_ids": list(ctx.faq_topics)}, fix_hint=FIX_HINT)
    except Exception as error:  # an invalid answer twice, or Ollama not reachable: never end the call because of it
        if not (isinstance(error, (LLMFormError, ollama.ResponseError)) or is_transient(error)):  # e.g. 500 "runner terminated", 404 no model
            raise
        if fallback is None:
            from rules_turn import rules_understand as fallback  # lazy: only needed on failure
        turn = fallback(text, ctx)
        return Understanding(turn, [f"model failed ({type(error).__name__}); plain-code baseline used"], True, 0,
                             round(time.perf_counter() - started, 3))
    raw = reply.value.model_dump()
    turn, notes = ground(reply.value, text)
    if reply.attempts > 1:
        notes.append(f"the model's first answer was rejected: {reply.rejected_because}")
    return Understanding(turn, notes, False, reply.attempts, reply.seconds, raw)
