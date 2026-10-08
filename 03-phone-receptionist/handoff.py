"""Hand-off: a finished call becomes a row in Part 1's voicemails.db, routed by Part 1's rules.

    live = LivePush()                                         # pushes the moment a call is flagged urgent
    record = run_call(card, ..., on_urgent=live)              # the call itself
    outcome = hand_off(record, conn, live=live)               # analyze the caller's side -> route -> push if urgent -> save

What happens, in order (the order of Part 1's deliver(): decide, push, THEN save, so a crash can repeat a push but never lose one):
  1. call_transcript()    the CALLER'S side only, one answer per line, as a Part 1 Transcript. The receptionist's own words are
                          fixed sentences and would only add noise to the analysis.
  2. analysis_for_call()  Part 1's analyze() with the "call-v1" prompt (a call, not a voicemail) for calls that carry a message.
                          Robocalls, info-only calls and silent calls need no model: plain code builds their analysis. The dialog's
                          own checked name and number ALWAYS replace whatever the model says for those two fields, and a call
                          flagged urgent during the call stays urgent. If the model fails, plain code builds the analysis and the
                          row is marked for review (attempts = 2).
  3. route()              Part 1's routing, unchanged: category -> route, Part 1's safety-word net, review flags.
  4. push                 minimal text, no caller data (Part 1's rule); a dry run unless NTFY is configured. A call flagged urgent
                          during the call was pushed THEN (LivePush) and is not pushed again here.
  5. save()               Part 1's store.save(): one row keyed by a hash of the call, so handing the same call off twice is a no-op.

A call is recognisable in the table by source_file = "call-<id>.json" (Part 1's table has no source column and is not changed).

    python 03-phone-receptionist/handoff.py 03-phone-receptionist/calls/*.json [--db PATH] [--analysis model|rules]
"""

import argparse
import hashlib
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import ollama

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "01-voicemail-triage"))

import call as call_module  # noqa: E402
import shared.analyze as analyze_module  # noqa: E402
from routing import Decision, push_text, route  # noqa: E402  (Part 1)
from shared.notify import send_push  # noqa: E402
from shared.retry import is_transient, with_retries  # noqa: E402
from shared.schemas import Analysis, Result, Segment, Transcript  # noqa: E402
from store import DEFAULT_DB, connect, save  # noqa: E402  (Part 1)

SOURCE_PREFIX = "call-"
CALL_PROMPT = "call-v1"
SUMMARY_WORDS = 25            # Analysis.summary is "one sentence, max 25 words"
NO_MODEL_OUTCOMES = {"spam", "info_only", "silence", "no_message"}  # nothing to judge: plain code is exact and instant


def is_call(source_file: str) -> bool:
    return source_file.startswith(SOURCE_PREFIX)


def caller_lines(record) -> list[str]:
    return [t["caller_text"] for t in record.turns if t["caller_text"]]


def call_transcript(record) -> Transcript:
    """The caller's side of the call as a Part 1 Transcript (one segment per answer; start/end are the answer's position)."""
    text = "\n".join(caller_lines(record))
    segments = []
    for i, turn in enumerate(t for t in record.turns if t["caller_text"]):
        logprob = (turn.get("heard") or {}).get("min_logprob")  # only calls through the audio loop have it
        segments.append(Segment(start=float(i), end=float(i + 1), text=turn["caller_text"],
                                avg_logprob=0.0 if logprob is None else float(logprob), no_speech_prob=0.0))
    return Transcript(
        source_file=f"{SOURCE_PREFIX}{record.call_id}.json",
        audio_sha256=hashlib.sha256(f"{record.call_id}\n{text}".encode("utf-8")).hexdigest(),  # same call = same row
        model="call", hint=None, language="en", language_probability=1.0, duration_s=0.0, transcribe_s=0.0,
        text=text, segments=segments, created_at=datetime.fromisoformat(record.started))


# ---------------------------------------------------------------- the analysis

def short(text: str, words: int = SUMMARY_WORDS) -> str:
    return " ".join(text.split()[:words])


def rule_analysis(record) -> Analysis:
    """The analysis built by plain code from the dialog's own message (no model): exact, instant, never invented."""
    m = record.message
    if m["urgent"]:
        category, urgency = "urgent", 3
    elif record.outcome == "spam":
        category, urgency = "spam", 1
    else:
        category, urgency = "other", 1
    if record.outcome == "spam":
        summary = "Automated or scam call; the receptionist ended it after the first message."
    elif record.outcome == "silence":
        summary = "The caller said nothing; no message was left."
    elif record.outcome == "info_only" or not (m["reason"] or m["name"] or m["number"] or m["extra_requests"]):
        asked = ", ".join(m["faq_answered"]) or "nothing the receptionist could use"
        summary = f"Caller only asked questions ({asked}) and left no message."
    else:
        who = m["name"] or "An unnamed caller"
        about = m["reason"] or "something the receptionist could not take down"
        summary = f"{who} called about {about}."
        if m["extra_requests"]:
            summary += " Also: " + "; ".join(m["extra_requests"]) + "."
        if m["unanswered_questions"]:
            summary += " Asked: " + " ".join(m["unanswered_questions"])
    return Analysis(summary=short(summary), reason="Built from the phone call by plain code, without a model.",
                    category=category, urgency=urgency, caller_name=m["name"], callback_number=m["number"], language="en")


def merge_model_analysis(record, model: Analysis) -> Analysis:
    """The model's judgement (category, urgency, summary) with the dialog's own verified name and number.

    The model only SAW the caller's words; the dialog CHECKED them (every value was grounded in the speech and read back to the
    caller), so for name and number the dialog wins even when the model disagrees or invents. A call the dialog flagged urgent
    stays urgent: it saw a safety word or the model's live flag; the end-of-call analysis may only add urgency, never remove it."""
    m = record.message
    category, urgency, reason = model.category, model.urgency, model.reason
    if m["urgent"] and category != "urgent":
        category, urgency, reason = "urgent", 3, short(f"{model.reason} Flagged urgent during the call.", 40)
    return Analysis(summary=model.summary, reason=reason, category=category, urgency=urgency, caller_name=m["name"],
                    callback_number=m["number"], language=model.language)


def default_analyze(transcript: Transcript) -> Result:
    return analyze_module.analyze(transcript, cache_dir=None, prompt_version=CALL_PROMPT)  # no cache: its key has no hint fingerprint


def analysis_for_call(record, transcript: Transcript, analyze_fn=None, use_model: bool = True) -> Result:
    now = datetime.now(timezone.utc)

    def result(analysis, llm, prompt, attempts=1, because=None, seconds=0.0) -> Result:
        return Result(source_file=transcript.source_file, audio_sha256=transcript.audio_sha256, transcript_model=transcript.model,
                      llm_model=llm, prompt_version=prompt, attempts=attempts, rejected_reply=None, rejected_because=because,
                      analyze_s=seconds, analysis=analysis, created_at=now)

    if not use_model or record.outcome in NO_MODEL_OUTCOMES:
        return result(rule_analysis(record), "none (rules)", "call-rules")
    try:
        model_result = (analyze_fn or default_analyze)(transcript)
    except Exception as error:
        if not (isinstance(error, (analyze_module.AnalysisError, ollama.ResponseError)) or is_transient(error)):
            raise
        # The model is down or confused: the message is still complete in the dialog's own words. attempts=2 makes Part 1's
        # routing flag the row for review ("analysis needed a retry"), so a human looks at it.
        return result(rule_analysis(record), "none (model failed)", "call-rules", attempts=2,
                      because=f"model analysis failed ({type(error).__name__}); built by plain code from the dialog")
    return result(merge_model_analysis(record, model_result.analysis), model_result.llm_model, model_result.prompt_version,
                  model_result.attempts, model_result.rejected_because, model_result.analyze_s)


# ---------------------------------------------------------------- the push and the row

class LivePush:
    """The push sent the moment a call is flagged urgent, while the caller is still on the line (run_call's on_urgent hook).

    Minimal text, no caller data (Part 1's rule). It never raises: a push that fails must not end the call; hand_off() then
    sends the push at the end instead. `status` is "sent" or "dry_run" (nothing is sent until NTFY is configured)."""

    def __init__(self, sender=send_push):
        self.sender = sender
        self.status: str | None = None
        self.sent_at: str | None = None
        self.error: str | None = None

    def __call__(self, state=None) -> None:
        now = datetime.now(timezone.utc)
        try:
            self.status = with_retries(self.sender, "Urgent call in progress",
                                       f"A caller was flagged urgent at {now.astimezone():%H:%M}. Open the triage list on the laptop.",
                                       attempts=2, base_delay=1.0, priority="urgent", tags="rotating_light")
            self.sent_at = now.isoformat(timespec="seconds") if self.status == "sent" else None
        except Exception as error:  # noqa: BLE001  (never end a call because of a push)
            self.error = f"{type(error).__name__}: {error}"


@dataclass
class HandOff:
    transcript: Transcript
    result: Result
    decision: Decision
    pushed: str                 # "live" (during the call) | "end" (at hand-off) | "already" (row existed) | "none"
    notified_at: str | None


def hand_off(record, conn: sqlite3.Connection, analyze_fn=None, live: LivePush | None = None, sender=send_push,
             use_model: bool = True) -> HandOff:
    """Analyze, route, push if needed and save one finished call. Handing the same call off again changes nothing."""
    transcript = call_transcript(record)
    result = analysis_for_call(record, transcript, analyze_fn, use_model)
    decision = route(transcript, result)
    if record.outcome == "info_only":
        # Part 1 flags "customer call without a number/name"; a caller who only asked the opening hours has nothing to leave.
        decision.review = False
        decision.reasons.append("info-only call: the caller left no message, so the missing name and number are not a problem")
    decision.reasons.insert(1, f"phone call, outcome '{record.outcome}', {len(record.turns)} turns"
                               + (f", flagged urgent in turn {record.urgent_flagged_at_turn}" if record.urgent_flagged_at_turn else ""))

    existing = conn.execute("SELECT notified_at FROM voicemails WHERE audio_sha256 = ?", (transcript.audio_sha256,)).fetchone()
    pushed, notified_at = "none", None
    if decision.notify:
        if live is not None and live.status is not None:
            pushed, notified_at = "live", live.sent_at
        elif existing is not None:
            pushed, notified_at = "already", existing["notified_at"]  # a row exists: the push went out when it was first saved
        else:
            title, message, priority = push_text(decision, received=datetime.now().strftime("%H:%M"))
            status = with_retries(sender, title, message, priority=priority, tags="rotating_light")  # raises after the retries: not saved
            pushed = "end"
            notified_at = datetime.now(timezone.utc).isoformat(timespec="seconds") if status == "sent" else None
    save(conn, transcript, result, decision, notified_at)  # push BEFORE save: at-least-once, like Part 1
    record.handoff = {"source_file": transcript.source_file, "audio_sha256": transcript.audio_sha256, "route": decision.route,
                      "notify": decision.notify, "review": decision.review, "reasons": decision.reasons, "pushed": pushed,
                      "category": result.analysis.category, "urgency": result.analysis.urgency, "llm_model": result.llm_model,
                      "prompt_version": result.prompt_version, "attempts": result.attempts}
    return HandOff(transcript, result, decision, pushed, notified_at)


def run_and_hand_off(card, persona, faq, understand_fn, conn, channel=None, analyze_fn=None, sender=send_push,
                     save_dir: Path | None = None, use_model: bool = True, understand_label: str = "model",
                     prompt_version: str | None = None):
    """A whole simulated call with the live urgent push, then the hand-off. Returns (record, hand-off)."""
    live = LivePush(sender)
    record = call_module.run_call(card, persona, faq, understand_fn, channel=channel, on_urgent=live,
                                  understand_label=understand_label, prompt_version=prompt_version)
    outcome = hand_off(record, conn, analyze_fn, live, sender, use_model)
    if save_dir:
        call_module.save_record(record, save_dir)
    return record, outcome


# ---------------------------------------------------------------- command line

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hand saved calls (calls/*.json) to the Part 1 pipeline.")
    parser.add_argument("calls", nargs="+", help="call record files")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Part 1's SQLite database (default: 01-voicemail-triage/voicemails.db)")
    parser.add_argument("--analysis", choices=["model", "rules"], default="model",
                        help="model = Part 1's analyze() with the call prompt (needs Ollama); rules = plain code, no model")
    args = parser.parse_args(argv)
    conn = connect(args.db)
    for path in args.calls:
        record = call_module.CallRecord.model_validate_json(Path(path).read_text(encoding="utf-8"))
        outcome = hand_off(record, conn, use_model=args.analysis == "model", live=None)
        call_module.save_record(record, Path(path).parent)
        print(f"{record.call_id}: route={outcome.decision.route} notify={outcome.decision.notify} review={outcome.decision.review} "
              f"pushed={outcome.pushed}\n    " + "\n    ".join(outcome.decision.reasons))
    return 0


if __name__ == "__main__":
    sys.exit(main())
