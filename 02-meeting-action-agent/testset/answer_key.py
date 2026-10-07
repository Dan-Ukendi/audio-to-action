"""Load the Part 2 answer key and derive the expected tracker state after each meeting.

Used by generate.py (to check the labels) and by the evaluation (Phase 6), so both read
the labels the same way.
"""

import json
import re
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


def overlap(a: str, b: str) -> float:
    """Share of words two texts have in common (Jaccard): to pick the closest of several candidates."""
    wa, wb = set(re.findall(r"[a-z]+", a.lower())), set(re.findall(r"[a-z]+", b.lower()))
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


def match_items(item_tasks: list[str], mentions: list[dict], tasks: dict) -> tuple[list[tuple[int, dict]], list[int], list[dict]]:
    """Pair extracted items (by their task text) with this meeting's mentions, one-to-one.

    Returns (pairs [(item index, mention)], unmatched item indexes, missed mentions).
    Candidate pairs are items whose text fits a mentioned task's match rule; the closest pairs
    (word overlap with the task description) are taken first, so "Get three quotes for the website"
    beats "Review the website" for T14. Items fitting no mentioned task are false positives.
    """
    candidates = [(overlap(text, tasks[m["task"]]["task"]), index, m)
                  for index, text in enumerate(item_tasks)
                  for m in mentions if matches(tasks[m["task"]]["match"], text)]
    pairs, used_items, used_tasks = [], set(), set()
    for _, index, m in sorted(candidates, key=lambda c: -c[0]):
        if index not in used_items and m["task"] not in used_tasks:
            pairs.append((index, m))
            used_items.add(index)
            used_tasks.add(m["task"])
    extra = [i for i in range(len(item_tasks)) if i not in used_items]
    missed = [m for m in mentions if m["task"] not in used_tasks]
    return sorted(pairs, key=lambda p: p[0]), extra, missed
