"""The agent's guardrails, tested by calling its tools directly (no LLM, instant).

Tasks that already exist are created with earlier() (as if a previous meeting added them), because the
agent may change a task only once per meeting.
"""

import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 02-meeting-action-agent

import agent  # noqa: E402
import tracker  # noqa: E402
from agent import SyncRun, run_tool, state_message  # noqa: E402
from shared.schemas import ExtractedItem  # noqa: E402


def item(task: str, status: str = "open", owner=None, due=None) -> ExtractedItem:
    return ExtractedItem(evidence="e", task=task, owner=owner, due_text=None, status=status, due=due)


def setup(*items: ExtractedItem):
    """A fresh tracker and a run for meeting m9 with items A, B, C, ..."""
    conn = tracker.connect(":memory:")
    return conn, SyncRun("m9", {chr(65 + n): it for n, it in enumerate(items)})


def earlier(conn, task: str, owner=None, due=None, status="open") -> dict:
    """A task added by a previous meeting."""
    return tracker.add_task(conn, task, owner, due, status, "m8", "earlier meeting")


def test_unknown_tool_and_bad_arguments_are_errors_not_crashes():
    conn, run = setup(item("Fix the van light"))
    assert run_tool(run, conn, "delete_everything", {}).startswith("ERROR: unknown tool")
    assert "needs task_id" in run_tool(run, conn, "update_task", {"task_id": "seven", "item_id": "A", "reason": "r"})
    assert run_tool(run, conn, "add_task", {"item_id": "Z", "reason": "r"}).startswith("ERROR: there is no item")
    assert tracker.all_tasks(conn) == []


def test_each_item_gets_exactly_one_action():
    conn, run = setup(item("Fix the van light"))
    assert run_tool(run, conn, "add_task", {"item_id": "A", "reason": "new"}).startswith("OK")
    assert "already handled" in run_tool(run, conn, "add_task", {"item_id": "A", "reason": "again"})
    assert len(tracker.all_tasks(conn)) == 1


def test_finish_refused_until_every_item_is_handled():
    conn, run = setup(item("Fix the van light"), item("Fix the van light again"))
    run_tool(run, conn, "add_task", {"item_id": "A", "reason": "new"})
    assert "B" in run_tool(run, conn, "finish", {"summary": "done"}) and not run.finished
    assert "needs other_item" in run_tool(run, conn, "skip_item", {"item_id": "B", "kind": "repeat_of_item", "reason": "?"})
    run_tool(run, conn, "skip_item", {"item_id": "B", "kind": "repeat_of_item", "other_item": "A", "reason": "repeat"})
    assert run_tool(run, conn, "finish", {"summary": "done"}) == "OK, finished." and run.finished


def test_values_come_from_the_item_and_update_never_wipes_known_values():
    conn, run = setup(item("Book the Ellis gas check", "done"))
    earlier(conn, "Book the Ellis gas check", owner="Tom", due="2026-09-30")
    run_tool(run, conn, "update_task", {"task_id": 1, "item_id": "A", "reason": "booked"})
    task = tracker.get_task(conn, 1)
    assert (task["owner"], task["due"], task["status"]) == ("Tom", "2026-09-30", "done")
    assert [c["reason"] for c in tracker.changes(conn)] == ["earlier meeting", "booked"]  # every change has a reason


def test_undo_restores_the_previous_state():
    conn, run = setup(item("Fix the van light", "done"))
    earlier(conn, "Fix the van light", owner="Jamie")
    run_tool(run, conn, "update_task", {"task_id": 1, "item_id": "A", "reason": "fixed"})
    tracker.undo_last(conn)
    assert tracker.get_task(conn, 1)["status"] == "open"
    tracker.undo_last(conn)
    assert tracker.all_tasks(conn) == []


def test_update_needs_a_real_related_task():
    conn, run = setup(item("Send Siobhan Gallagher the quote", "done"))
    earlier(conn, "Order a pressure valve for the Mill Lane boiler")
    assert "needs task_id" in run_tool(run, conn, "update_task", {"item_id": "A", "reason": "r"})
    assert "use add_task" in run_tool(run, conn, "update_task", {"task_id": 9, "item_id": "A", "reason": "r"})
    assert "too little in common" in run_tool(run, conn, "update_task", {"task_id": 1, "item_id": "A", "reason": "guess"})
    assert tracker.get_task(conn, 1)["status"] == "open"  # the guessed update was blocked


def test_one_shared_word_is_not_enough_to_link():
    conn, run = setup(item("Fix the van's rear brake light", owner="Jamie"))
    earlier(conn, "Renew the van insurance")
    assert "too little in common" in run_tool(run, conn, "update_task", {"task_id": 1, "item_id": "A", "reason": "?"})


def test_duplicate_add_needs_confirmation():
    conn, run = setup(item("Fix the van's rear brake light", owner="Jamie"))
    earlier(conn, "Fix the van's rear brake light")
    assert "update_task(task_id=1" in run_tool(run, conn, "add_task", {"item_id": "A", "reason": "new?"})
    assert run_tool(run, conn, "add_task", {"item_id": "A", "reason": "really new", "confirm_new": True}).startswith("OK")


def test_one_item_per_task_per_meeting():
    conn, run = setup(item("Fit the new pressure valve at Mill Lane", "done"),
                      item("Follow-up visit to Mill Lane to check the pressure", owner="Tom"))
    earlier(conn, "Fit the new pressure valve at Mill Lane", owner="Tom")
    run_tool(run, conn, "update_task", {"task_id": 1, "item_id": "A", "reason": "fitted"})
    assert "already changed by item A" in run_tool(run, conn, "update_task", {"task_id": 1, "item_id": "B", "reason": "?"})
    assert tracker.get_task(conn, 1)["status"] == "done"  # not reopened


def test_a_repeat_must_be_the_same_work():
    conn, run = setup(item("Order the Gallagher bathroom suite", "done"), item("Lead the Gallagher bathroom fit", owner="Tom"))
    assert "different work" in run_tool(run, conn, "skip_item",
                                        {"item_id": "B", "kind": "repeat_of_item", "other_item": "A", "reason": "?"})


def test_done_news_is_never_not_work():
    conn, run = setup(item("Fix the van's rear brake light", "done"))
    assert "news about real work" in run_tool(run, conn, "skip_item", {"item_id": "A", "kind": "not_work", "reason": "?"})


def test_state_message_shows_progress_and_empty_tracker():
    conn, run = setup(item("Fix the van light"), item("Renew the van tax"))
    first = state_message(run, conn, date(2026, 9, 7), [])
    assert "tracker is empty" in first and "Still to do: A, B" in first
    run_tool(run, conn, "add_task", {"item_id": "A", "reason": "new"})
    later = state_message(run, conn, date(2026, 9, 7), ["add_task(...) -> OK"])
    assert "HANDLED (add #1)" in later and "Still to do: B" in later and "Results of your last tool calls" in later


def test_every_change_needs_a_real_reason():
    conn, run = setup(item("Fix the van light"))
    assert "reason must not be empty" in run_tool(run, conn, "add_task", {"item_id": "A", "reason": "  "})
    assert tracker.all_tasks(conn) == [] and run.unhandled() == ["A"]


def test_undo_also_restores_which_meeting_last_changed_the_task():
    conn, run = setup(item("Fix the van light", "done"))
    earlier(conn, "Fix the van light")
    run_tool(run, conn, "update_task", {"task_id": 1, "item_id": "A", "reason": "fixed"})
    assert tracker.get_task(conn, 1)["updated_in"] == "m9"
    tracker.undo_last(conn)
    assert tracker.get_task(conn, 1)["updated_in"] == "m8"


def test_state_message_does_not_call_a_tracker_of_closed_tasks_empty():
    conn, run = setup(item("Fix the van light", "done"))
    earlier(conn, "Fix the van light", status="done")
    message = state_message(run, conn, date(2026, 9, 7), [])
    assert "tracker is empty" not in message and "search_tasks" in message


class StuckModel:
    """A fake Ollama client that always makes the same refused call, like the real model in m3 (no LLM)."""
    calls = 0

    def __init__(self, host=None):
        pass

    def chat(self, **kwargs):
        StuckModel.calls += 1
        call = SimpleNamespace(function=SimpleNamespace(name="add_task", arguments={"item_id": "A", "reason": "new"}))
        return SimpleNamespace(message=SimpleNamespace(tool_calls=[call], content=""))


def test_a_run_that_makes_no_progress_stops_instead_of_repeating(monkeypatch):
    conn = tracker.connect(":memory:")
    earlier(conn, "Fix the van's rear brake light")  # so add_task is always refused as a duplicate
    monkeypatch.setattr(agent.ollama, "Client", StuckModel)
    run = agent.sync_meeting(conn, "m9", date(2026, 9, 7), [item("Fix the van's rear brake light", "done")])
    # step 1 is refused, step 2 is refused with the same feedback, step 3 would be an identical input: stop.
    assert StuckModel.calls == 2 and not run.finished and run.unhandled() == ["A"]
    assert run.trace[-1]["stopped"].startswith("no progress")
