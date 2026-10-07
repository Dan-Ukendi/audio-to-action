"""Daily digest: everything from the last 24 h on one page, plus anything stuck in failed/.

    python 01-voicemail-triage/digest.py                 # print + save digests/<date>.md
    python 01-voicemail-triage/digest.py --push          # also send a counts-only push
    python 01-voicemail-triage/digest.py --hours 168     # last week

The page has names, numbers and summaries, so it stays local (digests/ is git-ignored).
The optional push only carries counts ("2 to call back, 1 to review"), like the urgent push.
"""

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from run import AUDIO_EXTENSIONS  # noqa: E402
from shared.notify import send_push  # noqa: E402
from store import connect  # noqa: E402

ROUTE_ORDER = {"notify_now": 0, "inbox": 1, "personal": 2, "archive": 3}


def recent_rows(conn: sqlite3.Connection, hours: float) -> list[sqlite3.Row]:
    # processed_at is stored as ISO text in UTC, so plain string comparison sorts correctly.
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(timespec="seconds")
    rows = conn.execute("SELECT * FROM voicemails WHERE processed_at >= ? ORDER BY processed_at", (since,)).fetchall()
    return sorted(rows, key=lambda r: ROUTE_ORDER[r["route"]])


def failed_files(failed_dir: Path) -> list[tuple[str, str]]:
    """(file name, first line of the error) for each audio file waiting in failed/."""
    items = []
    for audio in sorted(p for p in failed_dir.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS):
        note = Path(str(audio) + ".error.txt")
        step = note.read_text(encoding="utf-8").splitlines()[1] if note.exists() else "step: unknown"
        items.append((audio.name, step))
    return items


def line(row: sqlite3.Row) -> str:
    who = row["caller_name"] or "unknown caller"
    number = row["callback_number"] or "no number"
    flag = "  **REVIEW: " + "; ".join(json.loads(row["reasons"])[1:]) + "**" if row["review"] else ""
    return f"- {who}, {number}: {row['summary']}{flag}"


def build(rows: list[sqlite3.Row], failed: list[tuple[str, str]], hours: float) -> str:
    out = [f"# Voicemail digest {datetime.now():%Y-%m-%d %H:%M} (last {hours:g} h)", ""]
    sections = [("Urgent (push was sent)", "notify_now"), ("Call back", "inbox"),
                ("Personal", "personal"), ("Archived (sales / spam)", "archive")]
    for title, route_name in sections:
        selected = [r for r in rows if r["route"] == route_name]
        out += [f"## {title}: {len(selected)}", ""] + [line(r) for r in selected] + [""]
    out += [f"## Failed, needs attention: {len(failed)}", ""]
    out += [f"- {name} ({step}); fix, then `python 01-voicemail-triage/run.py --retry-failed`" for name, step in failed]
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=HERE)
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--push", action="store_true", help="send a counts-only push")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    conn = connect(args.base / "voicemails.db")
    rows = recent_rows(conn, args.hours)
    failed = failed_files(args.base / "failed")
    page = build(rows, failed, args.hours)

    out_dir = args.base / "digests"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{datetime.now():%Y-%m-%d}.md"
    out_file.write_text(page, encoding="utf-8")
    print(page)
    print(f"Saved: {out_file}")

    if args.push:
        callbacks = sum(r["route"] in ("notify_now", "inbox") for r in rows)
        review = sum(r["review"] for r in rows)
        send_push("Voicemail digest", f"{callbacks} to call back, {review} to review, {len(failed)} failed.",
                  priority="default", tags="memo")


if __name__ == "__main__":
    main()
