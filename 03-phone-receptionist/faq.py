"""The approved answers (faq.json): loading, checking, and matching a caller's question to an entry.

    entries = load_faq()
    result = match_questions("Hi, I'd like a boiler service. How much does that cost?", entries)
    result.entries   # [the boiler_service entry]: its answer is said word for word

The receptionist may only state facts from this file, word for word. Matching is keywords first; a model may only
PICK among the entries (result.via == "llm"), never write an answer: that keeps invented prices and opening hours out
of the call.

How a question is matched, in order:
  1. Only sentences that LOOK like questions are searched ("What are your hours?", "Do you cover Overmere?"), so a
     statement like "I'd like a quote for the bathroom" never triggers an answer.
  2. Keywords are matched on whole words. A keyword that is just part of a longer matched keyword is dropped
     ("how soon" inside "how soon can you get here"); the longest, most specific phrases win.
  3. A generic entry (fallback: prices) only answers on its own when nothing specific matched. A question that points
     back ("How much does THAT cost?", "...is IT?") asks about whatever the sentence BEFORE it was about ("I'd like a
     boiler service."); a self-contained one ("How much do you charge an hour?") is just answered.
  4. No keyword hit at all: the model's pick (a valid entry id) is used; "other" or nothing = an unknown question.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_FAQ = HERE / "faq.json"


class FaqError(Exception):
    """faq.json is inconsistent."""


@dataclass(frozen=True)
class FaqEntry:
    id: str
    topic: str
    keywords: tuple[str, ...]
    answer: str
    safety: str | None  # "gas" | "co" | "water" for the emergency advice entries, else None
    fallback: bool = False  # a generic entry (prices): answers on its own only when nothing specific matched


def load_faq(path: str | Path = DEFAULT_FAQ) -> dict[str, FaqEntry]:
    """id -> entry, in file order. Fails loudly on duplicates, empty fields, upper case or digits in answers."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))["entries"]
    entries: dict[str, FaqEntry] = {}
    for item in raw:
        entry = FaqEntry(id=item["id"], topic=item["topic"], keywords=tuple(item["keywords"]),
                         answer=item["answer"], safety=item.get("safety"), fallback=bool(item.get("fallback", False)))
        if entry.id in entries:
            raise FaqError(f"duplicate FAQ id '{entry.id}'")
        if not entry.keywords or not entry.answer.strip():
            raise FaqError(f"FAQ entry '{entry.id}' needs keywords and an answer")
        if any(k != k.lower() or k != k.strip() for k in entry.keywords):
            raise FaqError(f"FAQ entry '{entry.id}': keywords must be lowercase and trimmed")
        if re.search(r"[\d£$€%&/@#*+=<>]", entry.answer):
            raise FaqError(f"FAQ entry '{entry.id}': write numbers and symbols as words in the spoken answer")
        if entry.safety not in (None, "gas", "co", "water"):
            raise FaqError(f"FAQ entry '{entry.id}': safety must be gas, co or water")
        entries[entry.id] = entry
    return entries


# ---------------------------------------------------------------- matching a question

MAX_ANSWERS = 2  # at most this many answers to one turn: more would be a speech, not a reply

QUESTION_WORDS = re.compile(
    r"\b(do you|does your|can you|could you|will you|would you|are you|is there|are there|have you|what (are|is|time|do)|"
    r"when (are|do|is|can|could|will)|where (are|is)|which|how (much|long|soon|far|many|do|can|quickly)|"
    r"any chance)\b")
REQUEST_WORDS = re.compile(
    r"\b(give me a (call|ring)|call me|ring me|phone me|text me|get back to me|call back|ring back|call him|call her|"
    r"come out|come round|come and (look|see|take)|look at (it|that)|send (someone|a|an)|book (me|us|in)|arrange|let (him|her|sam) know|tell (him|her|sam)|"
    r"pass (it|that|this) on|ask (him|her|sam))\b")


POINTS_BACK = re.compile(r"\b(that|this|it|those|these|them)\b")


def normalize_text(text: str) -> str:
    """Lower case, everything that is not a letter or digit becomes a space: '24/7' matches the keyword '24 7'."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower().replace("'", "")).split())


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+", text.strip()) if s.strip()]


def looks_like_question(sentence: str) -> bool:
    return "?" in sentence or bool(QUESTION_WORDS.search(normalize_text(sentence)))


# Not questions about the business: small talk, the line, the call itself, the receptionist.
CONVERSATIONAL = re.compile(
    r"^(hello|hi|hey|pardon|sorry|what|really|are you there|can you hear me|is (anyone|anybody|someone) there|hello are you there|"
    r"could you (repeat|say that again)|can you (repeat|say that again)|can i (leave|speak|talk|have a word)|may i (speak|leave)|"
    r"can you take|could you take|who am i|who is this|who are you|are you (a )?(real|robot|human|machine|person|automated|recording)|"
    r"whats your name|what is your name|what are you called|is (that|this) (brightwater|right|correct|the (right|brightwater|plumber|company|business|office)|them)|can you add|could you add|can you hear|you there)\b")


def is_small_talk(sentence: str) -> bool:
    """'Hello?', 'Pardon?', 'Could you repeat that?', 'Is that Brightwater?': about the call, not the business."""
    plain = re.sub(r"^((hello|hi|hey|oh|um|yes|yeah|sorry)\s+)+", "", normalize_text(sentence))
    return bool(CONVERSATIONAL.match(plain)) or len(plain.split()) < 3


def is_information_question(sentence: str) -> bool:
    """A real question about the business, not 'Hello?', 'Pardon?' or 'Can you hear me?' and not a request for an action."""
    plain = re.sub(r"^((hello|hi|hey|oh|um|yes|yeah|sorry)\s+)+", "", normalize_text(sentence))  # "Hello, do you fit ...?"
    return len(plain.split()) >= 3 and not CONVERSATIONAL.match(plain) and not is_request(sentence)


GREETING_WORDS = re.compile(r"^((hello|hi|hey|oh|um|yes|yeah|sorry)[,.!\s]+)+", re.I)


def strip_greeting(sentence: str) -> str:
    """'Hello, do you fit solar panels?' -> 'Do you fit solar panels?' (what is passed on to the team)."""
    stripped = GREETING_WORDS.sub("", sentence.strip())
    return stripped[:1].upper() + stripped[1:] if stripped else sentence.strip()


def is_request(sentence: str) -> bool:
    """'Can you give me a call back?' asks for an action, not for information: it is never an unknown question."""
    return bool(REQUEST_WORDS.search(normalize_text(sentence)))


def light_stem(word: str) -> str:
    """'bathrooms' -> 'bathroom', 'callouts' -> 'callout'. Applied to the caller's words AND the keywords, so a plural
    question finds a singular keyword (and the other way round). Words of three letters or fewer ('gas', 'bus') are left alone."""
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def match_form(text: str) -> str:
    return " ".join(light_stem(w) for w in normalize_text(text).split())


def keyword_matches(text: str, entries: dict[str, FaqEntry]) -> list[tuple[str, str]]:
    """(entry id, keyword) for every keyword found as whole words, minus keywords inside a longer matched keyword."""
    padded = f" {match_form(text)} "
    found = [(e.id, match_form(k)) for e in entries.values() for k in e.keywords if f" {match_form(k)} " in padded]
    return [(eid, k) for eid, k in found
            if not any(k != other and f" {k} " in f" {other} " for _, other in found)]


def scores(text: str, entries: dict[str, FaqEntry]) -> dict[str, int]:
    """entry id -> score (words matched); an entry with no match is absent."""
    out: dict[str, int] = {}
    for eid, keyword in set(keyword_matches(text, entries)):
        out[eid] = out.get(eid, 0) + len(keyword.split())
    return out


def best(found: dict[str, int]) -> list[str]:
    """The strongest entries: within half of the best score, at most MAX_ANSWERS, best first (ties in file order)."""
    if not found:
        return []
    top = max(found.values())
    ranked = sorted((eid for eid, sc in found.items() if sc * 2 >= top), key=lambda e: (-found[e], list(found).index(e)))
    return ranked[:MAX_ANSWERS]


@dataclass
class QuestionResult:
    entries: list[FaqEntry] = field(default_factory=list)
    unknown: str | None = None   # the question text when faq.json has no answer
    via: str | None = None       # "keywords" | "llm" | None
    llm_disagreed: bool = False  # the model picked a different entry than the keywords (keywords won)


def match_questions(text: str, entries: dict[str, FaqEntry], llm_topic: str | None = None) -> QuestionResult:
    """Which FAQ entries answer the questions in this turn? See the module docstring for the rules."""
    parts = sentences(text)
    asked = [i for i, s in enumerate(parts) if looks_like_question(s)]
    llm_valid = llm_topic in entries
    if not asked and not llm_valid:
        return QuestionResult()

    found: dict[str, int] = {}
    for i in asked:
        for eid, sc in scores(parts[i], entries).items():
            found[eid] = found.get(eid, 0) + sc
    specific = {e: sc for e, sc in found.items() if not entries[e].fallback}
    if specific:
        chosen = best(specific)
    elif found:  # only generic hits ("how much ..."): a question that points back is about the sentence before it
        context = {e: sc for i in asked if i > 0 and POINTS_BACK.search(normalize_text(parts[i]))
                   for e, sc in scores(parts[i - 1], entries).items() if not entries[e].fallback and not entries[e].safety}
        chosen = best(context) if context else best(found)
    else:
        chosen = []

    if chosen:
        return QuestionResult([entries[e] for e in chosen], via="keywords", llm_disagreed=llm_valid and llm_topic not in chosen)
    if llm_valid:
        return QuestionResult([entries[llm_topic]], via="llm")
    # Nothing matched. An information question that is not a request is one we cannot answer: pass it on.
    for i in asked:
        if is_information_question(parts[i]):
            return QuestionResult(unknown=strip_greeting(parts[i]))
    return QuestionResult()
