"""The task tracker: one SQLite file with the tasks and a log of every change (with its reason).

    conn = connect("tracker.db")
    add_task(conn, "Send the quote", owner="Priya", due="2026-09-09", status="open", meeting="m1", reason="...")

Every change records the task's state before and after, so any agent decision can be inspected and undone
(`undo_last()`). Plain functions, no LLM: the agent calls these through its tools.
"""

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task        TEXT NOT NULL,
    owner       TEXT,                 -- NULL = nobody yet
    due         TEXT,                 -- ISO date or NULL
    status      TEXT NOT NULL,        -- open | done | cancelled
    created_in  TEXT NOT NULL,        -- meeting id
    updated_in  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS changes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     INTEGER NOT NULL,
    meeting     TEXT NOT NULL,
    action      TEXT NOT NULL,        -- add | update
    before      TEXT,                 -- JSON of the task before (NULL for add)
    after       TEXT NOT NULL,        -- JSON of the task after
    reason      TEXT NOT NULL,        -- why (the agent must always say)
    at          TEXT NOT NULL
);
"""
FIELDS = ("id", "task", "owner", "due", "status", "updated_in")  # updated_in too, so undo can restore it


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def as_dict(row: sqlite3.Row | None) -> dict | None:
    return {k: row[k] for k in FIELDS} if row else None


def get_task(conn: sqlite3.Connection, task_id: int) -> dict | None:
    return as_dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())


def all_tasks(conn: sqlite3.Connection, status: str | None = None) -> list[dict]:
    sql, args = "SELECT * FROM tasks", ()
    if status:
        sql, args = sql + " WHERE status = ?", (status,)
    return [as_dict(r) for r in conn.execute(sql + " ORDER BY id", args)]


# Filler words that two unrelated tasks can share ("Send THE quote" / "Order THE valve").
STOPWORDS = {"the", "and", "for", "with", "from", "this", "that", "new", "get", "make", "out", "about", "into"}


def words(text: str) -> set[str]:
    """The meaningful words of a task description: for fuzzy search and the agent's relevance check."""
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2 and w not in STOPWORDS}


def similarity(a: str, b: str) -> float:
    """Share of meaningful words two task descriptions have in common (Jaccard, 0..1)."""
    wa, wb = words(a), words(b)
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


def search_tasks(conn: sqlite3.Connection, query: str, limit: int = 5) -> list[dict]:
    """Tasks (any status) ranked by shared words with the query. Fuzzy on purpose: the meeting may say
    "the Ellis booking" for "Book Margaret Ellis's annual gas safety check"."""
    q = words(query)
    scored = [(len(q & words(t["task"])), t) for t in all_tasks(conn)]
    return [t for score, t in sorted(scored, key=lambda s: -s[0]) if score > 0][:limit]


def _log(conn, task_id: int, meeting: str, action: str, before: dict | None, reason: str) -> None:
    conn.execute(
        "INSERT INTO changes (task_id, meeting, action, before, after, reason, at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (task_id, meeting, action, json.dumps(before) if before else None, json.dumps(get_task(conn, task_id)),
         reason, datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )


def add_task(conn: sqlite3.Connection, task: str, owner: str | None, due: str | None, status: str,
             meeting: str, reason: str) -> dict:
    cur = conn.execute(
        "INSERT INTO tasks (task, owner, due, status, created_in, updated_in) VALUES (?, ?, ?, ?, ?, ?)",
        (task, owner, due, status, meeting, meeting),
    )
    _log(conn, cur.lastrowid, meeting, "add", None, reason)
    conn.commit()
    return get_task(conn, cur.lastrowid)


def update_task(conn: sqlite3.Connection, task_id: int, owner: str | None, due: str | None, status: str,
                meeting: str, reason: str) -> dict:
    """Apply what a meeting said about a task. A null owner/due never wipes a known one: "done" news
    usually doesn't repeat who or when, and that mustn't erase it."""
    before = get_task(conn, task_id)
    conn.execute(
        "UPDATE tasks SET owner = COALESCE(?, owner), due = COALESCE(?, due), status = ?, updated_in = ? WHERE id = ?",
        (owner, due, status, meeting, task_id),
    )
    _log(conn, task_id, meeting, "update", before, reason)
    conn.commit()
    return get_task(conn, task_id)


def changes(conn: sqlite3.Connection, meeting: str | None = None) -> list[dict]:
    sql, args = "SELECT * FROM changes", ()
    if meeting:
        sql, args = sql + " WHERE meeting = ?", (meeting,)
    return [dict(r) for r in conn.execute(sql + " ORDER BY id", args)]


def undo_last(conn: sqlite3.Connection) -> dict | None:
    """Reverse the most recent change (an add is deleted, an update restored). Returns the undone change."""
    last = conn.execute("SELECT * FROM changes ORDER BY id DESC LIMIT 1").fetchone()
    if last is None:
        return None
    if last["action"] == "add":
        conn.execute("DELETE FROM tasks WHERE id = ?", (last["task_id"],))
    else:
        b = json.loads(last["before"])
        conn.execute("UPDATE tasks SET task = ?, owner = ?, due = ?, status = ?, updated_in = ? WHERE id = ?",
                     (b["task"], b["owner"], b["due"], b["status"], b["updated_in"], b["id"]))
    conn.execute("DELETE FROM changes WHERE id = ?", (last["id"],))
    conn.commit()
    return dict(last)
