"""The meeting pipeline: inbox/ -> transcribe -> extract -> sync tracker -> processed/.

    python 02-meeting-action-agent/run.py                  # process what's in inbox/ once, then stop
    python 02-meeting-action-agent/run.py --watch          # keep checking inbox/ (Ctrl+C stops)
    python 02-meeting-action-agent/run.py --retry-failed   # move failed/ meetings back to inbox/ first

Per meeting (process_file), OLDEST MEETING FIRST (meeting 3 talks about tasks from meetings 1-2):
    0. hash        already in the tracker's meetings table? -> move to processed/, nothing else
    1. transcribe  Whisper + the known-names hint (cached in transcripts/)
    2. extract     action items (cached in results/), Ollama calls retried with backoff
    3. sync        the agent updates tracker.db; if it fails halfway, the meeting's changes are rolled back
                   (tracker.undo_meeting) so a retry starts clean. Changes left by a run that was killed
                   before step 4 are rolled back first too, so a meeting is never applied twice.
    4. record      meetings table (items, sync mode, items left for review) + move to processed/
Any error -> failed/<file> + <file>.error.txt. Logs: logs/run.log. Agent traces: traces/.

Meeting date: the first YYYY-MM-DD in the file name (e.g. "team_2026-09-21.m4a"), else the file's modified date.
Meeting id: the file name without extension; if a recorded meeting already uses it ("standup.m4a" every
week), the start of the audio hash is added ("standup_3f9a12bc") so the two meetings' changes stay apart.
Only the files in inbox/ at the same time are sorted by date. A meeting OLDER than one already in the
tracker is still synced (with a warning): its news is older, so it may undo newer news. Check it by hand.
Run one copy at a time: two copies would interleave their tracker changes.
"""

import argparse
import logging
import re
import sqlite3
import sys
import time
from datetime import date, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import context  # noqa: E402
import tracker  # noqa: E402
from agent import sync_meeting  # noqa: E402
from extract import extract  # noqa: E402
from shared.pipeline import find_ready_files, move_to, retry_failed, setup_logging, write_error_note  # noqa: E402
from shared.retry import with_retries  # noqa: E402
from shared.schemas import ExtractedItem  # noqa: E402
from shared.transcribe import file_sha256, transcribe  # noqa: E402

log = logging.getLogger("meetings")


class Folders:
    """All working locations, relative to one base folder (so tests can use a scratch copy)."""

    def __init__(self, base: Path):
        self.base = base
        self.inbox, self.processed, self.failed = base / "inbox", base / "processed", base / "failed"
        self.transcripts, self.results, self.traces = base / "transcripts", base / "results", base / "traces"
        self.log_file = base / "logs" / "run.log"
        self.db = base / "tracker.db"
        for folder in (self.inbox, self.processed, self.failed, self.transcripts, self.results, self.traces):
            folder.mkdir(parents=True, exist_ok=True)


def meeting_date(path: Path) -> date:
    """The (first) date in the file name if there is one, else the day the file was last modified.
    An impossible date like 2026-13-40 raises ValueError: guessing would resolve "next Friday" wrongly."""
    found = re.search(r"(\d{4})-(\d{2})-(\d{2})", path.name)
    if found:
        try:
            return date(*map(int, found.groups()))
        except ValueError:
            raise ValueError(f"'{found.group()}' in the file name is not a real date; rename the file") from None
    return datetime.fromtimestamp(path.stat().st_mtime).date()


def date_for_sorting(path: Path) -> date:
    """meeting_date() that never raises: a bad name sorts first and then fails on its own in process_file,
    instead of crashing the whole run (and every later --watch round)."""
    try:
        return meeting_date(path)
    except (ValueError, OSError):
        return date.min


def meeting_id(conn: sqlite3.Connection, path: Path, audio_hash: str) -> str:
    """The id the tracker files this meeting's changes under. Must differ from every recorded meeting:
    rolling back a failed sync undoes changes by id, and must never touch another meeting's changes."""
    if tracker.meeting_id_used(conn, path.stem):
        return f"{path.stem}_{audio_hash[:8]}"
    return path.stem


def agent_sync(conn: sqlite3.Connection, meeting: str, day: date, items: list[ExtractedItem], folders: Folders) -> list[str]:
    """Run the agent; returns the items it left for a human (step limit / no progress)."""
    run = sync_meeting(conn, meeting, day, items, trace_dir=folders.traces)
    log.info("[%s] agent: %d LLM calls, %.0f s, finished=%s", meeting, run.llm_calls, run.seconds, run.finished)
    return [f"{i}: {run.items[i].task}" for i in run.unhandled()]


SYNC_MODES = {"agent": agent_sync}  # Phase 6 adds a plain-code "rules" mode to compare against


def sync_with_rollback(sync, conn: sqlite3.Connection, meeting: str, day: date, items: list[ExtractedItem],
                       folders: Folders) -> list[str]:
    """All of a meeting's tracker changes, or none: on any error, undo what this meeting changed, then re-raise."""
    start_clean(conn, meeting)
    try:
        return sync(conn, meeting, day, items, folders)
    except BaseException:  # BaseException so Ctrl+C (KeyboardInterrupt) in the middle of a sync is rolled back too
        undone = tracker.undo_meeting(conn, meeting)
        log.warning("[%s] sync failed: rolled back %d tracker change(s)", meeting, undone)
        raise


def start_clean(conn: sqlite3.Connection, meeting: str) -> None:
    """A meeting that isn't recorded yet should have no tracker changes. If it has some, an earlier run was
    killed after the sync saved changes but before record_meeting (power cut, window closed): undo them,
    or this run would apply the meeting a second time."""
    if not tracker.changes(conn, meeting):
        return
    undone = tracker.undo_meeting(conn, meeting)
    log.warning("[%s] undid %d tracker change(s) left by an interrupted run", meeting, undone)
    if tracker.changes(conn, meeting):
        # Another meeting changed the tracker after them, so undoing them could clobber its changes.
        raise RuntimeError(f"{meeting} has tracker changes from an interrupted run that later changes depend "
                           "on; fix them by hand (tracker.changes) before retrying")


def timed(step: str, name: str, fn, *args, **kwargs):
    """Run one pipeline step and log how long it took."""
    started = time.perf_counter()
    value = fn(*args, **kwargs)
    log.info("[%s] %-10s done in %.1f s", name, step, time.perf_counter() - started)
    return value


def process_file(path: Path, folders: Folders, conn: sqlite3.Connection, sync_mode: str) -> str:
    """One meeting through the pipeline. Returns 'processed', 'skipped' or 'failed'. Never raises."""
    name, step = path.name, "hash"
    try:
        audio_hash = file_sha256(path)
        if tracker.meeting_done(conn, audio_hash):
            log.info("[%s] already processed (same audio, hash %s): nothing to do", name, audio_hash[:12])
            move_to(path, folders.processed)
            return "skipped"
        meeting = meeting_id(conn, path, audio_hash)
        step = "date"
        day = meeting_date(path)
        newest = tracker.latest_meeting_date(conn)
        if newest and day.isoformat() < newest:
            log.warning("[%s] is older than a meeting already synced (%s): its news may overwrite newer news",
                        name, newest)

        step = "transcribe"
        transcript = timed(step, name, transcribe, path, cache_dir=folders.transcripts, hint=context.whisper_hint())
        step = "extract"
        result = timed(step, name, with_retries, extract, transcript, day, cache_dir=folders.results)
        step = "sync"
        review = timed(step, name, with_retries, sync_with_rollback, SYNC_MODES[sync_mode], conn, meeting, day,
                       result.items, folders)

        tracker.record_meeting(conn, audio_hash, meeting, day.isoformat(), name, len(result.items), sync_mode, review)
        log.info("[%s] %s: %d items, %d tracker changes%s", name, day, len(result.items),
                 len(tracker.changes(conn, meeting)), f", {len(review)} LEFT FOR REVIEW" if review else "")
        for line in review:
            log.warning("[%s] review: %s", name, line)
        step = "move"
        move_to(path, folders.processed)
        return "processed"

    except Exception as exc:  # one bad meeting must never stop the others
        log.error("[%s] FAILED at %s: %s: %s", name, step, type(exc).__name__, exc)
        try:
            moved = move_to(path, folders.failed)
        except OSError as move_error:  # Windows: another program (e.g. a media player) has the file open
            log.error("[%s] can't move it to failed/ (%s): left in inbox/ for the next run", name, move_error)
            return "failed"
        write_error_note(moved, name, step)
        return "failed"


def run_once(folders: Folders, conn: sqlite3.Connection, sync_mode: str) -> dict[str, int]:
    counts = {"processed": 0, "skipped": 0, "failed": 0}
    # Oldest meeting first: later meetings refer to tasks the earlier ones created.
    for path in sorted(find_ready_files(folders.inbox), key=lambda p: (date_for_sorting(p), p.name)):
        counts[process_file(path, folders, conn, sync_mode)] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--watch", action="store_true", help="keep polling inbox/ until Ctrl+C")
    parser.add_argument("--interval", type=float, default=10, help="seconds between checks in --watch mode")
    parser.add_argument("--retry-failed", action="store_true", help="move failed/ meetings back to inbox/ first")
    parser.add_argument("--sync", choices=sorted(SYNC_MODES), default="agent", help="how to update the tracker")
    parser.add_argument("--base", type=Path, default=HERE, help="folder containing inbox/, processed/, ...")
    args = parser.parse_args()

    folders = Folders(args.base.resolve())
    setup_logging(folders.log_file)
    conn = tracker.connect(folders.db)
    log.info("start: base=%s sync=%s watch=%s", folders.base, args.sync, args.watch)
    if args.retry_failed:
        for moved in retry_failed(folders.failed, folders.inbox):
            log.info("[%s] moved back to inbox for another try", moved)
    try:
        while True:
            counts = run_once(folders, conn, args.sync)
            if sum(counts.values()) or not args.watch:
                log.info("run: %d processed, %d skipped (already done), %d failed",
                         counts["processed"], counts["skipped"], counts["failed"])
            if not args.watch:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log.info("stopped by user")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
