"""Storage: one SQLite table, one row per voicemail, keyed by the audio hash.

SQLite is a whole database in one local file: no server, built into Python (import sqlite3).
The JSON files on disk stay the detailed record; this table is the list you query
("what's in my inbox?", "what needs review?").

    python 01-voicemail-triage/store.py [path/to/voicemails.db]   # print a summary
"""

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # so 'shared' is importable when run as a script

from shared.schemas import Result, Transcript  # noqa: E402

DEFAULT_DB = HERE / "voicemails.db"  # git-ignored (*.db): real callers' data

SCHEMA = """
CREATE TABLE IF NOT EXISTS voicemails (
    audio_sha256    TEXT PRIMARY KEY,   -- same audio = same row, never a duplicate
    source_file     TEXT NOT NULL,
    processed_at    TEXT NOT NULL,      -- ISO time, UTC
    route           TEXT NOT NULL,      -- notify_now | inbox | personal | archive
    review          INTEGER NOT NULL,   -- 0/1 (SQLite has no real boolean)
    reasons         TEXT NOT NULL,      -- JSON list of the rules that fired
    notified_at     TEXT,               -- when the push went out; NULL = no push (yet)
    category        TEXT NOT NULL,
    urgency         INTEGER NOT NULL,
    caller_name     TEXT,
    callback_number TEXT,
    summary         TEXT NOT NULL,
    transcript      TEXT NOT NULL,
    whisper_model   TEXT NOT NULL,
    llm_model       TEXT NOT NULL,
    prompt_version  TEXT NOT NULL
)
"""


def connect(db_path: str | Path = DEFAULT_DB) -> sqlite3.Connection:
    """Open (or create) the database and make sure the table exists."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # rows behave like dicts: row["route"]
    conn.execute(SCHEMA)
    return conn


def save(conn: sqlite3.Connection, transcript: Transcript, result: Result, decision, notified_at: str | None) -> None:
    """Insert the voicemail, or update it if this audio was stored before.

    The ? placeholders let sqlite3 insert the values safely. Never build SQL with f-strings
    from transcript text: a caller's words would become part of the query.
    """
    a = result.analysis
    conn.execute(
        """
        INSERT INTO voicemails VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(audio_sha256) DO UPDATE SET
            processed_at = excluded.processed_at, route = excluded.route, review = excluded.review,
            reasons = excluded.reasons, category = excluded.category, urgency = excluded.urgency,
            caller_name = excluded.caller_name, callback_number = excluded.callback_number,
            summary = excluded.summary, transcript = excluded.transcript,
            whisper_model = excluded.whisper_model, llm_model = excluded.llm_model,
            prompt_version = excluded.prompt_version,
            -- keep the first push time: re-processing must not pretend we notified again
            notified_at = COALESCE(voicemails.notified_at, excluded.notified_at)
        """,
        (
            result.audio_sha256, result.source_file, datetime.now(timezone.utc).isoformat(timespec="seconds"),
            decision.route, int(decision.review), json.dumps(decision.reasons), notified_at,
            a.category, a.urgency, a.caller_name, a.callback_number, a.summary,
            transcript.text, transcript.model, result.llm_model, result.prompt_version,
        ),
    )
    conn.commit()


def summary(conn: sqlite3.Connection) -> None:
    """Print counts per route and the review list."""
    print("Per route:")
    for row in conn.execute("SELECT route, COUNT(*) AS n FROM voicemails GROUP BY route ORDER BY route"):
        print(f"  {row['route']:11} {row['n']}")
    print("\nNeeds review:")
    for row in conn.execute("SELECT source_file, route, reasons FROM voicemails WHERE review = 1 ORDER BY source_file"):
        print(f"  {row['source_file']:34} {row['route']:11} {'; '.join(json.loads(row['reasons'])[1:])}")


if __name__ == "__main__":
    summary(connect(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB))
