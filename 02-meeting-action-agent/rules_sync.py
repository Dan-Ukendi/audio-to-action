"""The NON-agent sync: plain-code rules relate a meeting's items to the tracker. No LLM, instant, deterministic.

    review = rules_sync(conn, meeting, day, items)

Built as the baseline the agent has to beat (Phase 6). For each item, in order:
  1. Repeat:    an earlier item of today with the same status and mostly the same words -> skip it.
  2. Done/cancelled news: the most similar open task (at least LINK similar) -> update it (close it).
  3. Open item: an open task that is SAME similar -> update it (re-dated, reassigned, mentioned again).
  4. Otherwise: a cancelled item with no task -> skip (a postponed idea); anything else -> add a new task.
A task already changed by an earlier item of this meeting is not considered again.
"""

import sqlite3
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import tracker  # noqa: E402
from shared.schemas import ExtractedItem  # noqa: E402

LINK = 0.2    # done/cancelled news needs only a weak match: there is usually one open task it can be about
SAME = 0.6    # an open item must match an open task strongly, or it's new work ("fit" after "order the valve")
REPEAT = 0.5  # two items of one meeting with the same status and this similar are one mention said twice


def most_similar(conn: sqlite3.Connection, text: str, exclude: set[int]) -> tuple[float, dict | None]:
    """The open task most like `text` (and its similarity), skipping tasks already changed today."""
    candidates = [(tracker.similarity(t["task"], text), t) for t in tracker.all_tasks(conn, status="open")
                  if t["id"] not in exclude]
    return max(candidates, key=lambda c: c[0], default=(0.0, None))


def rules_sync(conn: sqlite3.Connection, meeting: str, day: date, items: list[ExtractedItem], folders=None) -> list[str]:
    """Same signature as run.py's agent_sync. Returns items left for review (always none: rules always decide)."""
    changed: set[int] = set()
    seen: list[ExtractedItem] = []
    for item in items:
        due = item.due.isoformat() if item.due else None
        if any(o.status == item.status and tracker.similarity(o.task, item.task) >= REPEAT for o in seen):
            continue  # rule 1: said twice in one meeting
        seen.append(item)
        score, task = most_similar(conn, item.task, changed)
        threshold = SAME if item.status == "open" else LINK
        if task is not None and score >= threshold:  # rules 2 and 3
            tracker.update_task(conn, task["id"], item.owner, due, item.status, meeting,
                                f"rule: {item.status} item matches #{task['id']} (similarity {score:.2f})")
            changed.add(task["id"])
        elif item.status == "cancelled":
            continue  # rule 4a: cancelling something that never became a task ("let's leave the website")
        else:  # rule 4b
            new = tracker.add_task(conn, item.task, item.owner, due, item.status, meeting,
                                   f"rule: no open task is similar enough (best {score:.2f})")
            changed.add(new["id"])
    return []
