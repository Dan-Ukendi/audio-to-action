"""Understanding a turn WITHOUT a model: plain-code rules that fill the same CallerTurn form.

    turn = rules_understand("Hi, this is Mark Thompson. My number is 07700 900123.", ctx)

Two jobs:
  1. The safety net of turn.understand(): when the model fails twice or Ollama is down, the call goes on with these rules.
  2. The simple baseline ("build the simple baseline first"): it lets the whole dialog run headless in the tests, with no
     model, and gives the laptop a zero-latency reference to compare the model against.

It is deliberately modest. It finds numbers (the same digit parser the grounding uses), names after "my name is / it's /
this is", the caller's own sentences as the reason, FAQ questions by keyword, and safety words with Part 1's patterns.
It cannot judge urgency from context ("got a lot worse") or paraphrase a reason: that is what the model is for.
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import asks  # noqa: E402
import faq as faq_module  # noqa: E402
from safety import emergency_from_text  # noqa: E402
from shared.schemas import normalize_uk_number  # noqa: E402
from spoken import digit_runs  # noqa: E402
from turn import MAX_REASON_WORDS, CallerTurn, UnderstandContext, fallback_reason  # noqa: E402

NAME_INTRO = re.compile(r"\b(?:my name is|my name's|this is|it's|it is|i'm|i am|name's|call me)\s+([^.,!?;]+)", re.I)
NAME_STOP = {"from", "here", "and", "calling", "at", "on", "in", "with", "about", "the", "just", "again", "speaking", "of", "for",
             "an", "a", "me", "my", "he", "she", "we", "it", "its", "oh", "um", "yes", "no", "please", "sorry", "well", "yeah", "ok",
             "okay", "hi", "hello", "thanks", "thank", "i", "i'd", "i'm", "i've", "good", "just", "not", "that", "so", "dear", "love"}
ENDING = re.compile(r"\b(that'?s all|that is all|that'?s everything|that'?s it|no thanks|no,? thank you|nothing else|goodbye|bye bye|bye)\b", re.I)
AUTOMATED = re.compile(r"\b(automated message|recorded message|press (one|1|two|2)|final notice|legal proceedings|this is an automated)\b", re.I)
YES = re.compile(r"^\W*(yes|yeah|yep|yup|correct|right|that'?s (right|correct)|okay|ok|sure|exactly|perfect|that is right)\b", re.I)
NO = re.compile(r"^\W*(no|nope|not quite|wrong|that'?s (wrong|not right|incorrect)|incorrect)\b", re.I)


def find_name(text: str, asked: str) -> str | None:
    """A name after an introduction ('this is Mark Thompson'), or the capitalised words of a short answer to 'your name?'."""
    candidates = [m.group(1) for m in NAME_INTRO.finditer(text)]
    if asked in (asks.NAME, asks.CORRECTION) and len(text.split()) <= 6:
        candidates.append(re.split(r"[.,!?;]", text.strip())[0])
    for chunk in candidates:
        taken = []
        for word in chunk.split():
            bare = word.strip("'’")
            if bare.lower() in NAME_STOP or re.search(r"['’](ll|s|d|ve|re)$", bare, re.I) or not re.match(r"^[A-Z][\w'’-]*$", bare) and not (taken and re.match(r"^[a-z]{2,6}$", bare) and asked == asks.NAME):
                break
            taken.append(bare)
            if len(taken) == 3:
                break
        if taken and taken[0].lower() not in NAME_STOP and re.match(r"^[A-Z]", taken[0]):
            return " ".join(taken)
    return None


def find_number(text: str) -> str | None:
    """The last run of digits that is a valid UK number (so a self-correction's restated number wins)."""
    for run in reversed(digit_runs(text, min_len=8)):
        try:
            number = normalize_uk_number(run)
        except ValueError:
            continue
        if number:
            return number
    return None


def find_reason(text: str, asked: str) -> str | None:
    if asked not in (asks.GREETING, asks.REASON, asks.ANYTHING_ELSE) or (asked == asks.ANYTHING_ELSE and (NO.match(text) or ENDING.search(text))):
        return None
    kept = []
    for sentence in re.split(r"(?<=[.?!])\s+", text.strip()):
        if (digit_runs(sentence, min_len=6) or NAME_INTRO.search(sentence)) and len(sentence.split()) <= 10:
            continue  # a short "my number is ..." or "this is X" sentence is not the reason
        if faq_module.looks_like_question(sentence) and faq_module.match_questions(sentence, FAQ).entries:
            continue  # a question about the business is answered, not recorded as the reason
        kept.append(sentence)
    reason = fallback_reason(" ".join(kept))
    return reason


FAQ = faq_module.load_faq()


def rules_understand(text: str, ctx: UnderstandContext) -> CallerTurn:
    asked = ctx.asked
    number = find_number(text)
    name = find_name(text, asked)
    emergency = bool(emergency_from_text(text))
    automated = bool(AUTOMATED.search(text))
    result = faq_module.match_questions(text, FAQ)
    topic = result.entries[0].id if result.entries else ("other" if result.unknown else None)
    correcting = asked in (asks.CONFIRM, asks.CORRECTION) and (bool(NO.match(text)) or bool(name or number))
    field = "number" if correcting and number else "name" if correcting and name else None
    return CallerTurn(
        heard_summary=text.strip()[:120] or "(nothing)", name=name, number=number, reason=find_reason(text, asked),
        is_correction=correcting, correction_field=field, question_topic=topic,
        emergency=emergency and not automated, wants_to_end=bool(ENDING.search(text)) and not automated,
        is_automated=automated)
