"""Act on a routing decision: the only place in the routing step with side effects.

    deliver(conn, transcript, result) -> Decision
        1. route()      decide (pure, in routing.py)
        2. send_push()  if the decision says notify (minimal text, no caller data)
        3. save()       one row in SQLite with the decision and its reasons
"""

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))  # sibling modules (routing, store) when imported from elsewhere

from routing import Decision, push_text, route  # noqa: E402
from shared.notify import send_push  # noqa: E402
from shared.schemas import Result, Transcript  # noqa: E402
from store import save  # noqa: E402


def deliver(conn: sqlite3.Connection, transcript: Transcript, result: Result) -> Decision:
    decision = route(transcript, result)

    notified_at = None
    if decision.notify:
        title, message, priority = push_text(decision, received=datetime.now().strftime("%H:%M"))
        status = send_push(title, message, priority=priority, tags="rotating_light")
        if status == "sent":
            notified_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        # Known gap (fixed in Phase 5): if save() below crashes after a real push,
        # re-running would push again because the row doesn't record it yet.

    save(conn, transcript, result, decision, notified_at)
    return decision
