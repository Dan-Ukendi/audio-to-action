"""The voicemail triage pipeline: inbox/ -> transcribe -> analyze -> deliver -> processed/.

    python 01-voicemail-triage/run.py                  # process what's in inbox/ once, then stop
    python 01-voicemail-triage/run.py --watch          # keep checking inbox/ every 10 s (Ctrl+C stops)
    python 01-voicemail-triage/run.py --retry-failed   # move failed/ files back to inbox/ first

Per file (process_file):
    0. hash        already in the database? -> move to processed/, do nothing else (idempotent)
    1. transcribe  cached by hash in transcripts/
    2. analyze     cached in results/; Ollama calls retried with backoff
    3. deliver     route -> push (retried) -> save row in SQLite
    4. move        to processed/  (or failed/ + <name>.error.txt if any step raised)
Every step is logged to the console and to logs/run.log.

A fixed sequence of plain function calls: the LLM fills in one step's form, it never chooses
the next step. That's what makes this a workflow, not an agent.
"""

import argparse
import logging
import shutil
import sqlite3
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from deliver import deliver  # noqa: E402
from shared.analyze import analyze  # noqa: E402
from shared.retry import with_retries  # noqa: E402
from shared.transcribe import file_sha256, transcribe  # noqa: E402
from store import connect  # noqa: E402

log = logging.getLogger("run")

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".oga", ".opus", ".flac", ".aac", ".amr", ".wma", ".webm", ".3gp"}
MIN_FILE_AGE_S = 3  # a file changed more recently than this may still be copying in


class Folders:
    """All working locations, relative to one base folder (so tests can use a scratch copy)."""

    def __init__(self, base: Path):
        self.base = base
        self.inbox = base / "inbox"
        self.processed = base / "processed"
        self.failed = base / "failed"
        self.transcripts = base / "transcripts"
        self.results = base / "results"
        self.log_file = base / "logs" / "run.log"
        self.db = base / "voicemails.db"
        for folder in (self.inbox, self.processed, self.failed, self.transcripts, self.results, self.log_file.parent):
            folder.mkdir(parents=True, exist_ok=True)


def setup_logging(log_file: Path) -> None:
    """Console + file. Log file names, steps, timings and routes; never transcripts or numbers."""
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")):
        handler.setFormatter(fmt)
        root.addHandler(handler)
    for noisy in ("httpx", "faster_whisper"):  # their per-request chatter isn't useful here
        logging.getLogger(noisy).setLevel(logging.WARNING)


def find_ready_files(inbox: Path) -> list[Path]:
    """Audio files in inbox/, oldest first, skipping ones that may still be copying in."""
    now = time.time()
    files = [
        p for p in inbox.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        and now - p.stat().st_mtime >= MIN_FILE_AGE_S
    ]
    return sorted(files, key=lambda p: p.stat().st_mtime)


def already_processed(conn: sqlite3.Connection, audio_hash: str) -> bool:
    return conn.execute("SELECT 1 FROM voicemails WHERE audio_sha256 = ?", (audio_hash,)).fetchone() is not None


def move_to(path: Path, folder: Path) -> Path:
    """Move a file into folder; if the name is taken, prefix a timestamp instead of overwriting."""
    target = folder / path.name
    if target.exists():
        target = folder / f"{datetime.now():%Y%m%d-%H%M%S}_{path.name}"
    return Path(shutil.move(str(path), str(target)))


def timed(step: str, name: str, fn, *args, **kwargs):
    """Run one pipeline step and log how long it took."""
    started = time.perf_counter()
    value = fn(*args, **kwargs)
    log.info("[%s] %-10s done in %.1f s", name, step, time.perf_counter() - started)
    return value


def process_file(path: Path, folders: Folders, conn: sqlite3.Connection) -> str:
    """Run one voicemail through the pipeline. Returns 'processed', 'skipped' or 'failed'. Never raises."""
    name = path.name
    step = "hash"
    try:
        audio_hash = file_sha256(path)
        if already_processed(conn, audio_hash):
            log.info("[%s] already processed (same audio, hash %s): nothing to do", name, audio_hash[:12])
            move_to(path, folders.processed)
            return "skipped"

        step = "transcribe"
        transcript = timed(step, name, transcribe, path, cache_dir=folders.transcripts)
        step = "analyze"
        result = timed(step, name, with_retries, analyze, transcript, cache_dir=folders.results)
        step = "deliver"
        decision = timed(step, name, deliver, conn, transcript, result)

        log.info("[%s] -> %s%s%s", name, decision.route, " +push" if decision.notify else "",
                 " +REVIEW" if decision.review else "")
        move_to(path, folders.processed)
        return "processed"

    except Exception as exc:  # one bad file must never stop the others
        log.error("[%s] FAILED at %s: %s: %s", name, step, type(exc).__name__, exc)
        moved = move_to(path, folders.failed)
        error_note = moved.with_name(moved.name + ".error.txt")
        error_note.write_text(
            f"file: {name}\nstep: {step}\ntime: {datetime.now().isoformat(timespec='seconds')}\n\n"
            f"{traceback.format_exc()}",
            encoding="utf-8",
        )
        return "failed"


def retry_failed(folders: Folders) -> None:
    """Put failed files back in the inbox. Caches make the retry cheap: finished steps are reused."""
    for audio in [p for p in folders.failed.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS]:
        Path(str(audio) + ".error.txt").unlink(missing_ok=True)
        move_to(audio, folders.inbox)
        log.info("[%s] moved back to inbox for another try", audio.name)


def run_once(folders: Folders, conn: sqlite3.Connection) -> dict[str, int]:
    counts = {"processed": 0, "skipped": 0, "failed": 0}
    for path in find_ready_files(folders.inbox):
        counts[process_file(path, folders, conn)] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--watch", action="store_true", help="keep polling inbox/ until Ctrl+C")
    parser.add_argument("--interval", type=float, default=10, help="seconds between checks in --watch mode")
    parser.add_argument("--retry-failed", action="store_true", help="move failed/ files back to inbox/ first")
    parser.add_argument("--base", type=Path, default=HERE, help="folder containing inbox/, processed/, ...")
    args = parser.parse_args()

    folders = Folders(args.base.resolve())
    setup_logging(folders.log_file)
    conn = connect(folders.db)
    log.info("start: base=%s watch=%s", folders.base, args.watch)

    if args.retry_failed:
        retry_failed(folders)  # moving keeps the old modified time, so they're "ready" at once

    try:
        while True:
            counts = run_once(folders, conn)
            if sum(counts.values()) or not args.watch:
                log.info("run: %d processed, %d skipped (already done), %d failed",
                         counts["processed"], counts["skipped"], counts["failed"])
            if not args.watch:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:  # Ctrl+C in --watch mode: stop cleanly
        log.info("stopped by user")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
