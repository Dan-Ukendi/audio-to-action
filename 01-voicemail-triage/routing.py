"""Routing step: plain-code rules that decide what happens to an analyzed voicemail.

    route(transcript, result) -> Decision      (pure: no sending, no saving)

The LLM's output is only an INPUT here. Every decision is an if-statement you can read,
test and change, and the same input always gives the same decision.

Rules, in order:
  1. category -> route  (urgent: notify_now, other: inbox, personal: personal, spam/sales: archive)
  2. safety net         safety words in the transcript but LLM didn't say urgent -> never archive,
                        push anyway, flag for review (catches LLM misses with dumb, reliable code)
  3. review flags       things a human should double-check; they never change the route
"""

import re
from typing import Literal

from pydantic import BaseModel

from shared.schemas import Result, Transcript

Route = Literal["notify_now", "inbox", "personal", "archive"]

CATEGORY_ROUTE: dict[str, Route] = {
    "urgent": "notify_now",
    "other": "inbox",
    "personal": "personal",
    "spam": "archive",
    "sales": "archive",  # user's choice: trade offers stay findable but out of the way
}

# Phrases that mean danger or damage in progress. Phrases, not single words: "gas" alone
# would fire on every "gas boiler service" request. \b = word boundary, so "flood" doesn't
# match inside "floodlight".
SAFETY_PATTERNS = [
    r"\bsmell(s|ing)? (of )?gas\b", r"\bgas (leak|smell)\b", r"\bcarbon monoxide\b",
    r"\bspark(s|ing)\b", r"\belectrics\b", r"\bflood(ed|ing)?\b", r"\bburst\b",
    r"\bwater (is )?everywhere\b", r"\bthrough the ceiling\b",
    r"\bno heating\b", r"\bno hot water\b",
]

# Whisper's own "probably garbage" threshold. Our test set never gets close (lowest -0.38),
# even where Whisper was wrong: confidence catches unintelligible audio, not misheard names.
LOW_CONFIDENCE_LOGPROB = -1.0


class Decision(BaseModel):
    route: Route
    notify: bool  # send a push now?
    review: bool  # should a human double-check this one?
    reasons: list[str]  # plain-language trail of every rule that fired


def safety_hits(text: str) -> list[str]:
    """Which safety phrases appear in the transcript (the matched words, for the reasons list)."""
    lowered = text.lower()
    return [m.group(0) for p in SAFETY_PATTERNS if (m := re.search(p, lowered))]


def route(transcript: Transcript, result: Result) -> Decision:
    a = result.analysis
    reasons = [f"category '{a.category}' -> {CATEGORY_ROUTE[a.category]}"]
    chosen: Route = CATEGORY_ROUTE[a.category]
    notify = chosen == "notify_now"
    review = False

    # Rule 2: safety net. Only needed when the LLM did NOT say urgent.
    hits = safety_hits(transcript.text)
    if hits and a.category != "urgent":
        if chosen == "archive":
            chosen = "inbox"  # never archive something that mentions danger
        notify, review = True, True
        reasons.append(f"safety words {hits} but category '{a.category}': push + review")

    # Rule 3: review flags.
    if a.category in ("urgent", "other") and a.callback_number is None:
        review = True
        reasons.append("customer call without a callback number")
    if a.category in ("urgent", "other") and a.caller_name is None:
        review = True
        reasons.append("customer call without a caller name")
    if result.attempts > 1:
        review = True
        reasons.append(f"analysis needed a retry ({result.rejected_because or 'see result JSON'})")
    worst = min((s.avg_logprob for s in transcript.segments), default=0.0)
    if worst < LOW_CONFIDENCE_LOGPROB:
        review = True
        reasons.append(f"unclear audio (Whisper confidence {worst:.2f})")
    if not transcript.segments:
        review = True
        reasons.append("no speech found")

    return Decision(route=chosen, notify=notify, review=review, reasons=reasons)


def push_text(decision: Decision, received: str) -> tuple[str, str, str]:
    """(title, message, priority) for the push. Deliberately contains NO caller data (privacy)."""
    if decision.route == "notify_now":
        return "Urgent voicemail", f"Received {received}. Open the triage list on the laptop.", "urgent"
    return "Voicemail needs a look", f"Received {received}. Possible emergency, please check.", "high"
