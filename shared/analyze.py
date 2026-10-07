"""Analysis step: Transcript -> Result (category, urgency, name, number, summary).

    analyze(transcript, cache_dir=...) -> Result

Steps, in order:
    1. cache lookup   <cache_dir>/<hash>_<whisper>_<llm>_<prompt version>.json exists? return it
    2-4. shared/llm.structured_chat(): Ollama with format=<Analysis JSON schema>, Pydantic validation
         (types + our plain-code rules: UK number, urgent <-> urgency 3), one retry with the errors
    5. save           write the Result JSON into the cache

The LLM only fills in fields. It never decides what happens next: that's routing, plain code.

Try it:  python -m shared.analyze path\\to\\voicemail.wav
"""

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from shared.llm import LLMFormError, settings, structured_chat
from shared.schemas import Analysis, Result, Transcript

load_dotenv()

# Prompt versions. The version is part of the cache key, so a changed prompt never reuses
# old answers. Never edit a version in place: add a new one.
# v1 (not kept): first attempt; put customer requests in 'sales', retry could delete values.
# v2: 'sales' = selling TO the business; retry says fix, don't delete; greeting name = callee.
# v3: Phase 6 experiment, ONE change: the company-caller example keeps the full name.
PROMPT_V2 = """\
You triage voicemails for Brightwater Plumbing & Heating, a small UK plumbing and heating business.
The owner is Sam. The transcript was made by speech recognition, so it may contain mistakes.
Answer with JSON only, in the requested format.

CATEGORY (choose exactly one):
- urgent: needs action TODAY. Damage happening now (leak, flooding), a safety risk (gas smell,
  sparking, carbon monoxide), a vulnerable person without heating or hot water, or a hard deadline
  today or first thing tomorrow. Judge the situation, not the words: a message that says "urgent"
  but is automated, a scam or a sales pitch is NOT urgent.
- spam: robocalls, scams, automated messages, "press 1", threats, fake renewals or listings.
- sales: a real business or person trying to SELL something TO Brightwater (supplies, marketing,
  software), with a real callback. A customer who wants to buy from Brightwater is NOT sales.
- personal: friends or family, not about work.
- other: normal customer business that is not urgent (quotes, bookings, repairs, confirmations),
  or a message too unclear to tell.

URGENCY: 3 = act today (only and always for urgent), 2 = reply within a day or two
(e.g. the caller mentions a date this week, or the message is unclear), 1 = no time pressure.

CALLER NAME:
- The caller's own name, not the company and not the person they are calling. A name right
  after a greeting ("Hi Sam, ...") is usually the person being called, not the caller.
- A person who calls from a company still has a name: "this is Rachel from Acme" -> "Rachel".
- If they spell it letter by letter, use the spelling.
- First name only, or a relationship word like "Mum", if that is all they give.
- null if they do not say a name. Never invent one.

CALLBACK NUMBER:
- The number the caller wants to be called back on, as digits only.
- Convert spoken forms: "oh" or "o" = 0, "double 7" = 77, "nine hundred" = 900.
- Speech recognition often writes a leading "oh" as "a": "a 1632 960 456" means 01632960456.
- If the caller corrects a number, use the corrected one.
- Do not confuse it with error codes, house numbers, prices or times.
- null if no number is spoken (for example "you've got my number"). Never guess.
"""

V2_EXAMPLE = '"this is Rachel from Acme" -> "Rachel".'
V3_EXAMPLE = '"this is Rachel Moore from Acme" -> "Rachel Moore". Keep the full name when one is given.'
assert V2_EXAMPLE in PROMPT_V2  # the experiment must change exactly this one line
PROMPTS = {"v2": PROMPT_V2, "v3": PROMPT_V2.replace(V2_EXAMPLE, V3_EXAMPLE)}

# Which prompt the pipeline uses by default (Phase 6 decides: see docs/eval-results.md).
DEFAULT_PROMPT_VERSION = os.getenv("ANALYSIS_PROMPT", "v2")


class AnalysisError(Exception):
    """The model gave an invalid answer twice. Carries both raw replies for inspection."""


# The retry message for voicemails (kept word for word from before the shared helper existed).
FIX_HINT = ("Fix the values using the transcript; do not replace a value "
            "the caller actually said with null. Return the corrected JSON.")


def cache_path(cache_dir: Path, transcript: Transcript, llm: str, prompt_version: str) -> Path:
    # ':' is not allowed in Windows file names ("qwen2.5:7b"), so swap it.
    safe_llm = llm.replace(":", "-")
    return cache_dir / f"{transcript.audio_sha256[:16]}_{transcript.model}_{safe_llm}_{prompt_version}.json"


def analyze(transcript: Transcript, cache_dir: str | Path | None = None, llm: str | None = None,
            prompt_version: str | None = None) -> Result:
    """Analyze one transcript. With cache_dir, a transcript already analyzed is returned from disk."""
    llm = llm or settings()["model"]
    prompt_version = prompt_version or DEFAULT_PROMPT_VERSION

    cached = cache_path(Path(cache_dir), transcript, llm, prompt_version) if cache_dir else None
    if cached and cached.exists():
        return Result.model_validate_json(cached.read_text(encoding="utf-8"))

    messages = [
        {"role": "system", "content": PROMPTS[prompt_version]},
        {"role": "user", "content": f"Voicemail transcript:\n<<<\n{transcript.text}\n>>>"},
    ]
    try:
        reply = structured_chat(Analysis, messages, llm=llm, fix_hint=FIX_HINT)
    except LLMFormError as error:
        raise AnalysisError(f"{transcript.source_file}: {error}") from error

    result = Result(
        source_file=transcript.source_file,
        audio_sha256=transcript.audio_sha256,
        transcript_model=transcript.model,
        llm_model=llm,
        prompt_version=prompt_version,
        attempts=reply.attempts,
        rejected_reply=reply.rejected_reply,
        rejected_because=reply.rejected_because,
        analyze_s=reply.seconds,
        analysis=reply.value,
        created_at=datetime.now(timezone.utc),
    )

    if cached:
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = cached.with_suffix(".tmp")
        tmp_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        tmp_file.replace(cached)

    return result


def main() -> None:
    from shared.transcribe import transcribe  # only needed for this command-line demo

    parser = argparse.ArgumentParser(description="Transcribe + analyze one audio file and print the result.")
    parser.add_argument("path")
    parser.add_argument("--transcripts", default="01-voicemail-triage/transcripts")
    parser.add_argument("--results", default="01-voicemail-triage/results")
    args = parser.parse_args()

    transcript = transcribe(args.path, cache_dir=args.transcripts)
    print(f"TRANSCRIPT ({transcript.model}): {transcript.text}\n")
    result = analyze(transcript, cache_dir=args.results)
    print(f"RESULT ({result.llm_model}, attempts={result.attempts}, {result.analyze_s:.1f}s):")
    print(result.analysis.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
