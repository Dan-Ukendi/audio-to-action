"""Small helpers every inbox-style pipeline needs (Part 2 uses them; Part 1 has its own copies in run.py).

    files = find_ready_files(inbox)            # audio files done copying, oldest first
    move_to(path, processed_dir)                # never overwrites
    write_error_note(moved_path, name, step)    # <file>.error.txt with the traceback
"""

import logging
import time
import traceback
from datetime import datetime
from pathlib import Path

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".oga", ".opus", ".flac", ".aac", ".amr", ".wma", ".webm", ".3gp"}
MIN_FILE_AGE_S = 3  # a file changed more recently than this may still be copying in


def setup_logging(log_file: Path) -> None:
    """Console + file. Callers log file names, steps, timings and counts: never transcript content."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # force=True replaces handlers set up earlier, so calling this twice doesn't print every line twice.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", force=True,
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")])
    for noisy in ("httpx", "faster_whisper"):  # their per-request chatter isn't useful here
        logging.getLogger(noisy).setLevel(logging.WARNING)


def find_ready_files(inbox: Path) -> list[Path]:
    """Audio files in inbox/, oldest first, skipping ones that may still be copying in."""
    now = time.time()
    files = [p for p in inbox.iterdir()
             if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS and now - p.stat().st_mtime >= MIN_FILE_AGE_S]
    return sorted(files, key=lambda p: p.stat().st_mtime)


def move_to(path: Path, folder: Path) -> Path:
    """Move a file into folder; if the name is taken, prefix a timestamp instead of overwriting.

    A plain rename (inbox/, processed/ and failed/ share one base folder, so one drive): it either happens
    or fails. shutil.move would fall back to copy + delete, and on Windows, when another program has the
    file open, the delete fails and the file ends up in BOTH folders."""
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / path.name
    if target.exists():
        target = folder / f"{datetime.now():%Y%m%d-%H%M%S}_{path.name}"
    return path.rename(target)


def write_error_note(moved: Path, original_name: str, step: str) -> Path:
    """<file>.error.txt next to a failed file: which step failed, when, and the full traceback.
    Call it inside the `except` block, so traceback.format_exc() has the error."""
    note = moved.with_name(moved.name + ".error.txt")
    note.write_text(f"file: {original_name}\nstep: {step}\ntime: {datetime.now().isoformat(timespec='seconds')}\n\n"
                    f"{traceback.format_exc()}", encoding="utf-8")
    return note


def retry_failed(failed: Path, inbox: Path) -> list[str]:
    """Move failed audio back to the inbox (error notes are deleted: the log keeps the history)."""
    moved = []
    for audio in [p for p in failed.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS]:
        Path(str(audio) + ".error.txt").unlink(missing_ok=True)
        move_to(audio, inbox)
        moved.append(audio.name)
    return moved
