"""Plain-Python pieces of the local app (no Streamlit here, so they can be unit-tested).

The app never re-implements the pipelines: uploads go into the same inbox/ folders, runs start the same
run.py scripts in a separate process, and results are read from the same SQLite databases and JSON files.
"""

import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable  # the app runs in the project's .venv, so its Python runs the pipelines too
AUDIO_TYPES = ["wav", "mp3", "m4a", "ogg", "oga", "opus", "flac", "aac", "amr", "wma", "webm", "3gp"]
RUNS_DIR = ROOT / "app" / "runs"  # one log file per pipeline run (git-ignored: may contain file names)


@dataclass(frozen=True)
class Part:
    """One pipeline the app can drive."""

    key: str
    folder: Path
    db: Path

    @property
    def inbox(self) -> Path:
        return self.folder / "inbox"

    @property
    def processed(self) -> Path:
        return self.folder / "processed"

    @property
    def failed(self) -> Path:
        return self.folder / "failed"


def part(key: str, default_folder: Path, db_name: str, env_var: str) -> Part:
    """The real pipeline folder, or another one set in an environment variable (e.g. a scratch test copy).
    run.py is always taken from the real folder; --base points it at the chosen data folder."""
    folder = Path(os.getenv(env_var) or default_folder)
    return Part(key, folder, folder / db_name)


VOICEMAILS = part("voicemails", ROOT / "01-voicemail-triage", "voicemails.db", "APP_VOICEMAIL_BASE")
MEETINGS = part("meetings", ROOT / "02-meeting-action-agent", "tracker.db", "APP_MEETING_BASE")
SCRIPTS = {"voicemails": ROOT / "01-voicemail-triage" / "run.py", "meetings": ROOT / "02-meeting-action-agent" / "run.py"}


# ---------------------------------------------------------------- adding audio

def safe_name(name: str) -> str:
    """Keep letters, digits, '.', '-', '_' (a file name typed by a person may contain anything)."""
    stem, dot, ext = name.rpartition(".")
    stem = re.sub(r"[^\w\-]+", "_", stem if dot else name).strip("_") or "audio"
    return f"{stem}.{ext.lower()}" if dot else stem


def with_meeting_date(name: str, day: date) -> str:
    """Meetings need a date in the file name (run.py reads YYYY-MM-DD from it); add it if missing."""
    if re.search(r"\d{4}-\d{2}-\d{2}", name):
        return name
    stem, _, ext = name.rpartition(".")
    return f"{stem}_{day.isoformat()}.{ext}"


def save_upload(data: bytes, name: str, inbox: Path) -> Path:
    """Write an uploaded file into inbox/ without ever overwriting an existing one."""
    inbox.mkdir(parents=True, exist_ok=True)
    target = inbox / safe_name(name)
    if target.exists():
        target = inbox / f"{datetime.now():%Y%m%d-%H%M%S}_{target.name}"
    target.write_bytes(data)
    return target


def files_in(folder: Path) -> list[Path]:
    """Audio files in a folder, newest first."""
    if not folder.exists():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower().lstrip(".") in AUDIO_TYPES]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def failed_notes(part: Part) -> list[dict]:
    """Files waiting in failed/ with the step that failed and the first line of the error."""
    notes = []
    for audio in files_in(part.failed):
        note = Path(str(audio) + ".error.txt")
        text = note.read_text(encoding="utf-8") if note.exists() else ""
        step = next((l.split(":", 1)[1].strip() for l in text.splitlines() if l.startswith("step:")), "?")
        error = next((l for l in reversed(text.splitlines()) if l.strip()), "")
        notes.append({"file": audio.name, "step": step, "error": error, "detail": text})
    return notes


# ---------------------------------------------------------------- running the pipelines

def environment_problems() -> list[str]:
    """Things that would make a run fail straight away, in words a person can act on."""
    problems = []
    if not shutil.which("ffmpeg"):
        problems.append("ffmpeg is not on PATH: open a new terminal (after installing ffmpeg) and start the app from it.")
    try:
        import urllib.request
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        urllib.request.urlopen(f"{host}/api/tags", timeout=3).read()
    except Exception:
        problems.append("Ollama is not answering: start the Ollama app (tray icon) and try again.")
    return problems


def free_ram_gb() -> float | None:
    """Free physical memory in GB (Windows only; None elsewhere). Whisper needs ~1 GB, the 7B LLM ~5 GB."""
    if sys.platform != "win32":
        return None
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.ullAvailPhys / 1024 ** 3


def start_run(part: Part, extra_args: list[str] | None = None) -> Path:
    """Start part's run.py in its own process (the app stays responsive); returns the log file path."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    log = RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{part.key}.log"
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"}
    args = [PYTHON, str(SCRIPTS[part.key]), "--base", str(part.folder), *(extra_args or [])]
    with open(log, "w", encoding="utf-8") as out:
        process = subprocess.Popen(args, cwd=ROOT, stdout=out,
                                   stderr=subprocess.STDOUT, env=env,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    (log.with_suffix(".pid")).write_text(str(process.pid), encoding="utf-8")
    return log


def run_status(log: Path) -> str:
    """'running' while run.py is still busy, else 'finished'. run.py always ends with a 'run: ...' summary line."""
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    if re.search(r"\brun: \d+ processed", text) or "Traceback" in text:
        return "finished"
    pid_file = log.with_suffix(".pid")
    if pid_file.exists() and not pid_alive(int(pid_file.read_text(encoding="utf-8"))):
        return "finished"
    return "running"


def pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def latest_run(part: Part) -> Path | None:
    logs = sorted(RUNS_DIR.glob(f"*_{part.key}.log")) if RUNS_DIR.exists() else []
    return logs[-1] if logs else None


def read_log(log: Path, last_lines: int = 200) -> str:
    """The end of a run log, with the long run paths shortened so it reads easily."""
    text = log.read_text(encoding="utf-8", errors="replace")
    text = text.replace(str(ROOT) + os.sep, "")
    return "\n".join(text.splitlines()[-last_lines:])


# ---------------------------------------------------------------- reading results

def query(db: Path, sql: str, args: tuple = ()) -> list[dict]:
    """Rows as dicts; an empty list if the database doesn't exist yet (nothing processed so far)."""
    if not db.exists():
        return []
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)  # read-only: the app never changes results
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, args)]
    except sqlite3.OperationalError:
        return []  # table not created yet
    finally:
        conn.close()


def voicemails() -> list[dict]:
    rows = query(VOICEMAILS.db, "SELECT * FROM voicemails ORDER BY processed_at DESC")
    for r in rows:
        r["reasons"] = json.loads(r["reasons"])
    return rows


def find_audio(part: Part, source_file: str) -> Path | None:
    """The processed audio file for a result (it may have a timestamp prefix if the name was taken)."""
    candidates = [p for p in files_in(part.processed) if p.name == source_file or p.name.endswith("_" + source_file)]
    return candidates[0] if candidates else None


def tracker_tasks() -> list[dict]:
    return query(MEETINGS.db, "SELECT * FROM tasks ORDER BY CASE status WHEN 'open' THEN 0 WHEN 'done' THEN 1 ELSE 2 END, "
                              "due IS NULL, due, id")


def meetings() -> list[dict]:
    rows = query(MEETINGS.db, "SELECT * FROM meetings ORDER BY meeting_date DESC")
    for r in rows:
        r["review"] = json.loads(r["review"])
    return rows


def changes(meeting: str | None = None) -> list[dict]:
    sql, args = "SELECT * FROM changes", ()
    if meeting:
        sql, args = sql + " WHERE meeting = ?", (meeting,)
    rows = query(MEETINGS.db, sql + " ORDER BY id", args)
    for r in rows:
        before = json.loads(r["before"]) if r["before"] else None
        after = json.loads(r["after"])
        r["what"] = describe_change(r["action"], before, after)
    return rows


def describe_change(action: str, before: dict | None, after: dict) -> str:
    """'added' or 'status open → done, owner None → Jamie' in plain words."""
    if action == "add" or before is None:
        return f"added [{after['status']}]"
    diffs = [f"{field} {before.get(field)} → {after.get(field)}"
             for field in ("status", "owner", "due") if before.get(field) != after.get(field)]
    return ", ".join(diffs) or "mentioned again, no change"


def meeting_items(audio_sha256: str) -> list[dict]:
    """The extracted action items of one meeting (newest cached extraction for that audio)."""
    results = sorted((MEETINGS.folder / "results").glob(f"{audio_sha256[:16]}_*.json"), key=lambda p: p.stat().st_mtime)
    if not results:
        return []
    return json.loads(results[-1].read_text(encoding="utf-8"))["items"]


def agent_trace(meeting: str) -> dict | None:
    trace = MEETINGS.folder / "traces" / f"{meeting}.trace.json"
    return json.loads(trace.read_text(encoding="utf-8")) if trace.exists() else None
