"""Analysis step: Transcript -> Result (category, urgency, name, number, summary).

    analyze(transcript, cache_dir=...) -> Result

Steps, in order:
    1. cache lookup   <cache_dir>/<hash>_<whisper>_<llm>_<prompt version>.json exists? return it
    2. ask the LLM    Ollama chat with format=<Analysis JSON schema>: the reply MUST have that shape
    3. validate       Pydantic checks types + our plain-code rules (UK number, urgent <-> urgency 3)
    4. retry once     if validation fails, send the error back and ask for a corrected answer
    5. save           write the Result JSON into the cache

The LLM only fills in fields. It never decides what happens next: that's routing, plain code.

Try it:  python -m shared.analyze path\\to\\voicemail.wav
"""

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import ollama
from dotenv import load_dotenv
from pydantic import ValidationError

from shared.schemas import Analysis, Result, Transcript

load_dotenv()

# Change this whenever SYSTEM_PROMPT or the retry message changes.
# v1 -> v2: 'sales' means selling TO the business (v1 put customer requests in sales);
# retry says fix, don't delete; a name right after "Hi" is usually who is being called.
PROMPT_VERSION = "v2"

SYSTEM_PROMPT = """\
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


class AnalysisError(Exception):
    """The model gave an invalid answer twice. Carries both raw replies for inspection."""


def settings() -> dict:
    return {
        "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        "model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
    }


def cache_path(cache_dir: Path, transcript: Transcript, llm: str) -> Path:
    # ':' is not allowed in Windows file names ("qwen2.5:7b"), so swap it.
    safe_llm = llm.replace(":", "-")
    return cache_dir / f"{transcript.audio_sha256[:16]}_{transcript.model}_{safe_llm}_{PROMPT_VERSION}.json"


def short_errors(error: ValidationError) -> str:
    """Turn Pydantic's error list into a few readable lines for the model (and for us)."""
    lines = []
    for e in error.errors():
        where = ".".join(str(part) for part in e["loc"]) or "answer"
        lines.append(f"- {where}: {e['msg']}")
    return "\n".join(lines)


def ask_llm(client: ollama.Client, llm: str, messages: list[dict]) -> str:
    """One chat call. format=<schema> makes Ollama constrain the output to that JSON shape."""
    response = client.chat(
        model=llm,
        messages=messages,
        format=Analysis.model_json_schema(),
        options={"temperature": 0},  # always the most likely answer: repeatable runs
    )
    return response.message.content


def analyze(transcript: Transcript, cache_dir: str | Path | None = None, llm: str | None = None) -> Result:
    """Analyze one transcript. With cache_dir, a transcript already analyzed is returned from disk."""
    cfg = settings()
    llm = llm or cfg["model"]

    cached = cache_path(Path(cache_dir), transcript, llm) if cache_dir else None
    if cached and cached.exists():
        return Result.model_validate_json(cached.read_text(encoding="utf-8"))

    client = ollama.Client(host=cfg["host"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Voicemail transcript:\n<<<\n{transcript.text}\n>>>"},
    ]

    started = time.perf_counter()
    replies = []
    analysis = None
    rejected_because = None
    for attempt in (1, 2):  # first try + exactly one retry
        raw = ask_llm(client, llm, messages)
        replies.append(raw)
        try:
            analysis = Analysis.model_validate_json(raw)
            break
        except ValidationError as error:
            if attempt == 2:
                raise AnalysisError(
                    f"{transcript.source_file}: invalid answer twice.\n{short_errors(error)}\nReplies: {replies}"
                ) from error
            rejected_because = short_errors(error)
            # Show the model its own answer and what was wrong with it. Without this feedback a
            # temperature-0 model would just repeat the same answer.
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Your answer broke these rules:\n"
                                            f"{rejected_because}\n"
                                            "Fix the values using the transcript; do not replace a value "
                                            "the caller actually said with null. Return the corrected JSON."},
            ]
    elapsed = time.perf_counter() - started

    result = Result(
        source_file=transcript.source_file,
        audio_sha256=transcript.audio_sha256,
        transcript_model=transcript.model,
        llm_model=llm,
        prompt_version=PROMPT_VERSION,
        attempts=attempt,
        rejected_reply=replies[0] if attempt == 2 else None,
        rejected_because=rejected_because,
        analyze_s=round(elapsed, 2),
        analysis=analysis,
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
