"""Load the Part 2 answer key and derive the expected tracker state after each meeting.

Used by generate.py (to check the labels) and by the evaluation (Phase 6), so both read
the labels the same way.
"""

import json
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATUSES = {"open", "done", "cancelled"}
MENTION_KEYS = {"task", "status", "owner", "due", "owner_from_text"}  # owner_from_text is optional (default true)


def load() -> dict:
    return json.loads((HERE / "labels.json").read_text(encoding="utf-8"))


def matches(rule: dict, text: str) -> bool:
    """Does a task description fit a task's match rule? Plain substrings, so 'book' also hits 'booked'."""
    text = text.lower()
    has_all = all(any(word in text for word in group) for group in rule["all"])
    no_forbidden = not any(word in text for word in rule.get("none", []))
    return has_all and no_forbidden


def tracker_states(labels: dict) -> dict[str, dict[str, dict]]:
    """{meeting id: {task key: {"owner", "due", "status"}}} after each meeting.

    Rule (same as labels.json _conventions): apply mentions in order; non-null owner/due
    overwrite, status always overwrites. A task first seen in a meeting starts from its mention.
    """
    tracker: dict[str, dict] = {}
    states = {}
    for meeting in labels["meetings"]:
        for m in meeting["mentions"]:
            task = tracker.setdefault(m["task"], {"owner": None, "due": None, "status": "open"})
            if m["owner"] is not None:
                task["owner"] = m["owner"]
            if m["due"] is not None:
                task["due"] = m["due"]
            task["status"] = m["status"]
        states[meeting["id"]] = {key: dict(value) for key, value in tracker.items()}
    return states


def problems(labels: dict, script_ids: set[str]) -> list[str]:
    """Structural mistakes in the labels (not wrong answers: only a human can spot those)."""
    found = []
    team = set(labels["team"])
    for meeting in labels["meetings"]:
        if meeting["id"] not in script_ids:
            found.append(f"{meeting['id']}: no script")
        meeting_date = date.fromisoformat(meeting["date"])
        seen = set()
        for m in meeting["mentions"]:
            where = f"{meeting['id']} {m['task']}"
            if m.keys() - MENTION_KEYS:  # a typo like 'owner_from_txt' would silently mean "true"
                found.append(f"{where}: unknown field(s) {sorted(m.keys() - MENTION_KEYS)}")
            if m["task"] not in labels["tasks"]:
                found.append(f"{where}: unknown task key")
            if m["task"] in seen:
                found.append(f"{where}: mentioned twice in one meeting")
            seen.add(m["task"])
            if m["status"] not in STATUSES:
                found.append(f"{where}: bad status {m['status']!r}")
            if m["owner"] is not None and m["owner"] not in team:
                found.append(f"{where}: owner {m['owner']!r} not in team")
            if m["due"] is not None and date.fromisoformat(m["due"]) < meeting_date:
                found.append(f"{where}: due date before the meeting")
    used = {m["task"] for meeting in labels["meetings"] for m in meeting["mentions"]}
    found += [f"{key}: defined but never mentioned" for key in sorted(labels["tasks"].keys() - used)]
    found += match_problems(labels["tasks"])
    return found


def match_problems(tasks: dict) -> list[str]:
    """Each task's own description must fit its own match rule and no other (e.g. T1 vs T8, T7 vs T11)."""
    found = []
    for key, task in tasks.items():
        hits = [other for other, spec in tasks.items() if matches(spec["match"], task["task"])]
        if hits != [key]:
            found.append(f"{key}: its own text matches the rules of {hits}, expected only [{key!r}]")
    return found
