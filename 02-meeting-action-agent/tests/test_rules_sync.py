"""The plain-code sync rules (no LLM, instant)."""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 02-meeting-action-agent

import tracker  # noqa: E402
from rules_sync import rules_sync  # noqa: E402
from shared.schemas import ExtractedItem  # noqa: E402

DAY = date(2026, 9, 14)


def item(task: str, status: str = "open", owner=None, due=None) -> ExtractedItem:
    return ExtractedItem(evidence="e", task=task, owner=owner, due_text=None, status=status, due=due)


def tasks(conn):
    return [(t["task"], t["owner"], t["status"]) for t in tracker.all_tasks(conn)]


def test_done_news_closes_the_matching_task_and_follow_up_is_new():
    conn = tracker.connect(":memory:")
    tracker.add_task(conn, "Order a pressure valve for the Mill Lane boiler", "Tom", None, "open", "m1", "earlier")
    rules_sync(conn, "m2", DAY, [item("Order the Mill Lane valve", "done"),
                                 item("Fit the new pressure valve at Mill Lane", owner="Tom")])
    assert tasks(conn) == [("Order a pressure valve for the Mill Lane boiler", "Tom", "done"),
                           ("Fit the new pressure valve at Mill Lane", "Tom", "open")]


def test_reassignment_updates_instead_of_duplicating():
    conn = tracker.connect(":memory:")
    tracker.add_task(conn, "Fix the van's rear brake light", None, None, "open", "m1", "earlier")
    rules_sync(conn, "m2", DAY, [item("Fix the van's rear brake light", owner="Jamie")])
    assert tasks(conn) == [("Fix the van's rear brake light", "Jamie", "open")]


def test_repeat_in_one_meeting_and_postponed_idea_are_skipped():
    conn = tracker.connect(":memory:")
    rules_sync(conn, "m1", DAY, [item("Get three quotes from web designers"),
                                 item("Get three quotes from web designers for the website"),
                                 item("Discuss the new website", "cancelled")])
    assert tasks(conn) == [("Get three quotes from web designers", None, "open")]
