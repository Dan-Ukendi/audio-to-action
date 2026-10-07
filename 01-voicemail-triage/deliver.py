"""Act on a routing decision: the only place in the routing step with side effects.

    deliver(conn, transcript, result) -> Decision
        1. route()      decide (pure, in routing.py)
        2. send_push()  if the decision says notify (minimal text, no caller data), with retries
        3. save()       one row in SQLite with the decision and its reasons

Order matters. Push BEFORE save means "at least once": if we crash between the two, the row
is missing, the next run processes the file again and pushes a second time. The opposite order
(save, then push) would be "at most once": a crash could leave an urgent voicemail marked
done with no push ever sent. For emergencies a duplicate push is the safer mistake.
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
from shared.retry import with_retries  # noqa: E402
from shared.schemas import Result, Transcript  # noqa: E402
from store import save  # noqa: E402


def deliver(conn: sqlite3.Connection, transcript: Transcript, result: Result) -> Decision:
    decision = route(transcript, result)

    notified_at = None
    if decision.notify:
        title, message, priority = push_text(decision, received=datetime.now().strftime("%H:%M"))
        # If ntfy is still unreachable after the retries, this raises: run.py then moves the file
        # to failed/ (visible, retryable) instead of storing an urgent voicemail nobody was told about.
        status = with_retries(send_push, title, message, priority=priority, tags="rotating_light")
        if status == "sent":
            notified_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    save(conn, transcript, result, decision, notified_at)
    return decision
