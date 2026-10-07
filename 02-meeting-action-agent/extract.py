"""Extraction step (workflow): meeting Transcript -> MeetingResult (action items with resolved due dates).

    extract(transcript, meeting_date, cache_dir=...) -> MeetingResult

Steps, in order:
    1. cache lookup      <cache_dir>/<hash>_<whisper>[_h<hint>]_<llm>_<prompt version>.json exists? return it
    2. chunk             split the segments into pieces of <= MAX_WORDS words (only long meetings split)
    3. ask the LLM       per chunk: shared/llm.structured_chat(MeetingItems) with the team + transcript as
                         validation context (owners must be team members, quotes must be in the transcript)
    4. merge + ground    repeated items about the same task become one (later one wins);
                         an owner whose name isn't in the nearby words is dropped (ground_owner)
    5. resolve dates     dates.resolve_due(due_text, meeting_date): plain code, not the LLM
                         (4 + 5 = finish_items(): the model's raw items are kept as llm_items)
    6. save              write the MeetingResult JSON into the cache

Stateless on purpose: it only reads THIS meeting. Relating items to earlier tasks is the agent's job (Phase 4).

Try it:  python 02-meeting-action-agent/extract.py 02-meeting-action-agent/testset/audio/m2_2026-09-14.m4a 2026-09-14
"""

import argparse
import hashlib
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import context  # noqa: E402
from dates import resolve_due  # noqa: E402
from shared.llm import settings, structured_chat  # noqa: E402
from shared.schemas import ActionItem, ExtractedItem, MeetingItems, MeetingResult, Segment, Transcript  # noqa: E402

# Change whenever SYSTEM_PROMPT or the post-processing changes (part of the cache key).
# x1: rules only -> recall 71 %: missed most "done" news about older tasks, merged done + follow-up.
# x2: + one worked example (invented jobs, not from the test set) + owner grounding in code -> recall 75 %:
#     the misses were mostly two jobs merged into one item ("ordered it + I'll fit it Friday").
# x3: + "jobs_mentioned" list before the items (think first, one entry per job) -> recall 75 %, owners 83 %
#     where named; but items didn't follow the job list (listed 2 jobs, wrote 1 merged item).
# x4: + done/cancelled items get no due date (code) -> precision 84 %, recall 75 %, owners 78 % where named.
#     Tried and removed: a code check that every listed job has its own item (one retry naming the
#     missing jobs). It fired on 2 meetings, but the model "fixed" it by rewording its job list instead
#     of adding items: same answers, 60 % more LLM time. (Same lesson as Part 1: a rule + retry invites
#     the model to satisfy the rule by deleting.)  [cached as x5]
# x6: same prompt; code fixes only. merge_items had been joining SIMILAR items of the same chunk:
#     "Order the suite" (done) + "Fit the suite" (open) became one item, which looked like the model
#     merging jobs (m4, m5). Now only exact repeats merge within a chunk. Also: evidence check ignores
#     [mm:ss]; "Tom's" names Tom; the raw model items are saved as llm_items.
#     Note: so part of the "merged jobs" blamed on the model in x1-x4 was this code bug.
# x7: the misheard-word example in the prompt was taken from a test meeting ("the Gallagher court" =
#     quote): a test-set leak, replaced by an invented one. Every number from x7 on is leak-free.
PROMPT_VERSION = "x7"
MAX_WORDS = 1200  # per chunk: prompt + chunk + answer stay well inside Ollama's 4096-token context

SYSTEM_PROMPT = f"""\
You extract action items from a weekly team meeting of {context.BUSINESS}, a small UK plumbing business.
Team: Sam (the owner, chairs the meeting), Priya (office), Tom and Jamie (plumbers).
The transcript comes from speech recognition: one line per segment with its time, NO speaker names,
and it may contain misheard words (guess the obvious meaning, e.g. "the Patel in voice" = the Patel invoice).
Answer with JSON only, in the requested format.

AN ITEM is a piece of work someone must do, or news about a piece of work (it is done, or cancelled).
- List every item mentioned in this meeting, ONCE each, with its latest state in this meeting.
- Work reported as finished -> status "done". Work called off -> status "cancelled".
- Work still to do -> status "open", also when it was mentioned before and nothing changed.
- If finished work leads to new work ("ordered it, I'll fit it Friday"), give TWO items: done + open.
- NOT items: ideas that are postponed or rejected ("let's leave it for now"), facts about customers
  ("she'd like to start in October"), greetings and thanks.

task: the work itself, not its status, with customer and place names, e.g.
  "Order a pressure valve for the Mill Lane boiler" (also when it's done).
owner: Sam, Priya, Tom or Jamie, whoever will do (or did) the work.
  - Someone asked by name ("Tom, can you...?") who agrees is the owner.
  - "I'll do it" from someone whose name you can't tell: null. Never guess. Never use a customer.
due_text: the words for WHEN it will be done, copied exactly ("by Wednesday", "this afternoon", "on the 30th").
  If both a deadline and a promised time are said, copy the promised time. null if not said.
  For done or cancelled items: null.
evidence: copy the transcript words that show this item.
status: open, done or cancelled.

Go through the transcript line by line: every question about a job ("did X go out?", "how's Y?") and
its answer is an item, usually "done". A name said just before ("Priya, did...?") is that item's owner.

EXAMPLE (a different team and different jobs, only to show the format):
[00:00] Morning. Alex, did the Patel invoice go out?
[00:03] Yes, sent it Friday.
[00:05] Chris, how's the radiator for Elm Street?
[00:08] Delivered yesterday. I'll fit it on Thursday.
[00:11] Someone needs to unblock the drain at the depot.
[00:14] I'll do that.
[00:16] Let's leave the new logo for now.
Answer:
{{"jobs_mentioned": ["Patel invoice (sent)", "Elm Street radiator delivery (done)", "fitting the Elm Street radiator",
  "unblocking the depot drain", "new logo (postponed)"],
 "items": [
 {{"evidence": "Alex, did the Patel invoice go out? Yes, sent it Friday.", "task": "Send Mrs Patel the invoice",
   "owner": "Alex", "due_text": null, "status": "done"}},
 {{"evidence": "Chris, how's the radiator for Elm Street? Delivered yesterday.", "task": "Get the radiator for Elm Street delivered",
   "owner": "Chris", "due_text": null, "status": "done"}},
 {{"evidence": "I'll fit it on Thursday.", "task": "Fit the radiator at Elm Street",
   "owner": "Chris", "due_text": "on Thursday", "status": "open"}},
 {{"evidence": "Someone needs to unblock the drain at the depot. I'll do that.", "task": "Unblock the drain at the depot",
   "owner": null, "due_text": null, "status": "open"}}
]}}
(No item for the logo: it was postponed. The drain's owner is null: nobody was named.)
"""


def format_transcript(segments: list[Segment]) -> str:
    """One line per segment, '[mm:ss] text': keeps the turn breaks that a flat text would lose."""
    return "\n".join(f"[{int(s.start) // 60:02}:{int(s.start) % 60:02}] {s.text}" for s in segments)


def chunk_segments(segments: list[Segment], max_words: int = MAX_WORDS) -> list[list[Segment]]:
    """Split into pieces of <= max_words, on segment boundaries, overlapping by one segment.

    The overlap keeps a question and its answer together when the split falls between them.
    """
    chunks, current, words = [], [], 0
    for segment in segments:
        n = len(segment.text.split())
        if current and words + n > max_words:
            chunks.append(current)
            current, words = [current[-1]], len(current[-1].text.split())  # overlap
        current.append(segment)
        words += n
    if current:
        chunks.append(current)
    return chunks


def task_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def same_task(a: str, b: str) -> bool:
    """Two task descriptions are the same task if most of their words overlap (Jaccard >= 0.5)."""
    wa, wb = task_words(a), task_words(b)
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= 0.5


def merge_items(per_chunk: list[list[ActionItem]]) -> list[ActionItem]:
    """Items about the same task become one: the later one's status wins, missing owner/due are
    filled from the earlier one. Two kinds of twin:

    - an exact repeat (same task words), anywhere: the model sometimes writes one item per line
      where a job is discussed ("Tom, can you order...?" / "Yes, I'll order it...").
    - a similar name (same_task) only from an EARLIER chunk: the overlap segment makes each chunk
      see a little of the previous one. Within one chunk similar names are different jobs on purpose:
      "Order the Gallagher suite" (done) + "Fit the Gallagher suite" (open) share most of their words.
    """
    merged: list[ActionItem] = []
    for items in per_chunk:
        earlier = list(merged)  # items from previous chunks
        for item in items:
            twin = next((m for m in merged if task_words(m.task) == task_words(item.task)), None)
            if twin is None:
                twin = next((m for m in earlier if same_task(m.task, item.task)), None)
            if twin is None:
                merged.append(item)
                continue
            # Each earlier item absorbs one item per chunk: a repeated "order" item in the overlap
            # must not also swallow the new "fit" item next to it.
            earlier = [m for m in earlier if m is not twin]
            twin.status = item.status
            twin.owner = item.owner or twin.owner
            twin.due_text = item.due_text or twin.due_text
            twin.evidence = item.evidence
    return merged


def words_of(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def ground_owner(item: ActionItem, segments: list[Segment], window: int = 2) -> str | None:
    """Keep the owner only if their name is in the evidence or in the `window` segments before it.

    We have no speaker labels, so an owner is knowable only when the words name them
    ("Jamie, can you take that one?" / "Yes, tomorrow."). Anything else is a guess, and a wrong
    owner is worse than an honest "unknown" that a human fills in.
    """
    if item.owner is None or not segments:
        return item.owner
    evidence = words_of(item.evidence)
    # The segment that best matches the evidence quote is where the item was said.
    at = max(range(len(segments)), key=lambda i: len(evidence & words_of(segments[i].text)))
    nearby = " ".join([item.evidence] + [s.text for s in segments[max(0, at - window): at + 1]]).lower()
    # \b...\b rather than a word set, so "Tom's" also counts as naming Tom.
    return item.owner if re.search(rf"\b{re.escape(item.owner.lower())}\b", nearby) else None


def finish_items(llm_items: list[list[ActionItem]], segments: list[Segment], meeting_date: date) -> list[ExtractedItem]:
    """The plain-code steps after the LLM: merge, ground owners, resolve dates.

    Works on copies, so llm_items stays exactly what the model said (saved in the result for comparison).
    """
    copies = [[item.model_copy() for item in chunk_items] for chunk_items in llm_items]
    items = []
    for item in merge_items(copies):
        llm_owner = item.owner
        item.owner = ground_owner(item, segments)  # plain-code rule: no name in the words, no owner
        # Finished or called-off work has no deadline any more (same convention as the labels).
        due = resolve_due(item.due_text, meeting_date) if item.status == "open" else None
        items.append(ExtractedItem(**item.model_dump(), due=due, owner_from_llm=llm_owner))
    return items


def cache_path(cache_dir: Path, transcript: Transcript, llm: str, prompt_version: str = PROMPT_VERSION) -> Path:
    hint = f"_h{hashlib.sha256(transcript.hint.encode('utf-8')).hexdigest()[:8]}" if transcript.hint else ""
    safe_llm = llm.replace(":", "-")  # ':' is not allowed in Windows file names
    return cache_dir / f"{transcript.audio_sha256[:16]}_{transcript.model}{hint}_{safe_llm}_{prompt_version}.json"


def extract(transcript: Transcript, meeting_date: date, cache_dir: str | Path | None = None,
            llm: str | None = None, max_words: int = MAX_WORDS, prompt_version: str = PROMPT_VERSION) -> MeetingResult:
    """prompt_version other than the current one: re-read an old cached result (its prompt is gone)."""
    llm = llm or settings()["model"]
    cached = cache_path(Path(cache_dir), transcript, llm, prompt_version) if cache_dir else None
    if cached and cached.exists():
        return MeetingResult.model_validate_json(cached.read_text(encoding="utf-8"))
    if prompt_version != PROMPT_VERSION:
        raise ValueError(f"no cached result for old prompt version {prompt_version}; only {PROMPT_VERSION} can run")

    validation = {"team": context.TEAM, "transcript": transcript.text}
    llm_items, jobs, attempts, rejected, seconds = [], [], 0, [], 0.0
    chunks = chunk_segments(transcript.segments, max_words)
    for chunk in chunks:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Meeting date: {meeting_date:%A %Y-%m-%d}\n"
                                        f"Transcript:\n<<<\n{format_transcript(chunk)}\n>>>"},
        ]
        reply = structured_chat(MeetingItems, messages, llm=llm, context=validation)
        llm_items.append(reply.value.items)
        jobs += reply.value.jobs_mentioned
        attempts += reply.attempts
        seconds += reply.seconds
        if reply.rejected_because:
            rejected.append(reply.rejected_because)

    items = finish_items(llm_items, transcript.segments, meeting_date)
    result = MeetingResult(
        source_file=transcript.source_file, audio_sha256=transcript.audio_sha256, meeting_date=meeting_date,
        transcript_model=transcript.model, transcript_hint=bool(transcript.hint), llm_model=llm,
        prompt_version=PROMPT_VERSION, chunks=len(chunks), attempts=attempts, rejected=rejected, jobs_mentioned=jobs,
        extract_s=round(seconds, 2), items=items, llm_items=llm_items, created_at=datetime.now(timezone.utc),
    )
    if cached:
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = cached.with_suffix(".tmp")  # write then rename: never a half-written cache file
        tmp_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        tmp_file.replace(cached)
    return result


def main() -> None:
    from shared.transcribe import transcribe

    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Transcribe (with name hint) + extract one meeting.")
    parser.add_argument("path")
    parser.add_argument("date", type=date.fromisoformat, help="meeting date, YYYY-MM-DD")
    parser.add_argument("--transcripts", default=str(HERE / "testset" / "transcripts"))
    parser.add_argument("--results", default=str(HERE / "testset" / "results"))
    args = parser.parse_args()

    transcript = transcribe(args.path, cache_dir=args.transcripts, hint=context.whisper_hint())
    print(format_transcript(transcript.segments), "\n")
    result = extract(transcript, args.date, cache_dir=args.results)
    print(f"{len(result.items)} items ({result.llm_model}, {result.attempts} LLM call(s), {result.extract_s:.0f} s):")
    for item in result.items:
        print(f"  [{item.status:9}] {item.task}  | owner={item.owner}  due={item.due} ({item.due_text!r})")


if __name__ == "__main__":
    main()
