"""run.py pieces that don't need audio or the LLM: meeting dates, ordering, rollback, idempotency."""

import os
import sys
import time
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 02-meeting-action-agent

import tracker  # noqa: E402
from shared.pipeline import move_to  # noqa: E402
from run import Folders, meeting_date, meeting_id, run_once, sync_with_rollback  # noqa: E402


def test_meeting_date_from_name_or_file_time(tmp_path):
    named = tmp_path / "team_2026-09-21.m4a"
    named.write_bytes(b"x")
    assert meeting_date(named) == date(2026, 9, 21)
    unnamed = tmp_path / "standup.m4a"
    unnamed.write_bytes(b"x")
    os.utime(unnamed, (1789000000, 1789000000))  # 2026-09-09 (UTC)
    assert meeting_date(unnamed).isoformat().startswith("2026-09")


def test_failed_sync_is_rolled_back_completely():
    conn = tracker.connect(":memory:")
    tracker.add_task(conn, "Order the valve", "Tom", None, "open", "m1", "earlier meeting")

    def half_then_crash(conn, meeting, day, items, folders):
        tracker.update_task(conn, 1, None, None, "done", meeting, "ordered")
        tracker.add_task(conn, "Fit the valve", "Tom", None, "open", meeting, "follow-up")
        raise ConnectionError("Ollama went away")

    with pytest.raises(ConnectionError):
        sync_with_rollback(half_then_crash, conn, "m2", date(2026, 9, 14), [], None)
    assert [(t["task"], t["status"]) for t in tracker.all_tasks(conn)] == [("Order the valve", "open")]
    assert len(tracker.changes(conn)) == 1  # only meeting 1's change is left


def test_meetings_are_recorded_once():
    conn = tracker.connect(":memory:")
    assert not tracker.meeting_done(conn, "abc")
    tracker.record_meeting(conn, "abc", "m1", "2026-09-07", "m1.wav", 5, "agent", [])
    assert tracker.meeting_done(conn, "abc")


def add_one_task(conn, meeting, day, items, folders):
    """A fake sync (no LLM): the meeting adds one task."""
    tracker.add_task(conn, f"Task from {meeting}", "Tom", None, "open", meeting, "fake sync")
    return []


def test_interrupted_sync_is_not_applied_twice():
    # A run was killed after the sync saved its changes but before record_meeting (power cut, closed window).
    conn = tracker.connect(":memory:")
    add_one_task(conn, "m2", None, [], None)
    assert not tracker.meeting_done(conn, "hash-m2")
    # The file is still in inbox/, so the next run syncs the same meeting again: it must start clean.
    sync_with_rollback(add_one_task, conn, "m2", date(2026, 9, 14), [], None)
    assert [t["task"] for t in tracker.all_tasks(conn)] == ["Task from m2"]


def test_ctrl_c_during_sync_is_rolled_back():
    conn = tracker.connect(":memory:")

    def add_then_ctrl_c(conn, meeting, day, items, folders):
        add_one_task(conn, meeting, day, items, folders)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        sync_with_rollback(add_then_ctrl_c, conn, "m2", date(2026, 9, 14), [], None)
    assert tracker.all_tasks(conn) == []


def test_same_file_name_new_recording_gets_its_own_meeting_id():
    # "standup.m4a" every week: a failed sync of week 2 must never roll back week 1's recorded changes.
    conn = tracker.connect(":memory:")
    week1 = meeting_id(conn, Path("standup.m4a"), "aaaa1111bbbb")
    add_one_task(conn, week1, None, [], None)
    tracker.record_meeting(conn, "aaaa1111bbbb", week1, "2026-09-07", "standup.m4a", 1, "agent", [])
    week2 = meeting_id(conn, Path("standup.m4a"), "cccc2222dddd")
    assert (week1, week2) == ("standup", "standup_cccc2222")

    def crash(conn, meeting, day, items, folders):
        raise ConnectionError("Ollama went away")

    with pytest.raises(ConnectionError):
        sync_with_rollback(crash, conn, week2, date(2026, 9, 14), [], None)
    assert [t["task"] for t in tracker.all_tasks(conn)] == ["Task from standup"]


def test_leftovers_that_cannot_be_undone_stop_the_sync():
    # m2 was interrupted, then m1 (an older meeting copied in later) was synced on top of it.
    conn = tracker.connect(":memory:")
    add_one_task(conn, "m2", None, [], None)
    add_one_task(conn, "m1", None, [], None)
    with pytest.raises(RuntimeError, match="interrupted"):
        sync_with_rollback(add_one_task, conn, "m2", date(2026, 9, 14), [], None)
    assert len(tracker.all_tasks(conn)) == 2  # nothing applied twice, nothing of m1 undone


def test_invalid_date_in_name_fails_that_file_only(tmp_path):
    folders = Folders(tmp_path)
    bad = folders.inbox / "team_2026-13-40.wav"
    bad.write_bytes(b"not audio")
    os.utime(bad, (time.time() - 60, time.time() - 60))  # old enough to count as fully copied in
    conn = tracker.connect(":memory:")
    assert run_once(folders, conn, "agent") == {"processed": 0, "skipped": 0, "failed": 1}
    assert (folders.failed / "team_2026-13-40.wav").exists()
    assert "step: date" in (folders.failed / "team_2026-13-40.wav.error.txt").read_text(encoding="utf-8")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file locking")
def test_move_of_an_open_file_fails_without_leaving_a_copy(tmp_path):
    audio = tmp_path / "m1.wav"
    audio.write_bytes(b"x")
    with open(audio, "rb"):  # e.g. a media player has the recording open
        with pytest.raises(PermissionError):
            move_to(audio, tmp_path / "processed")
    assert audio.exists() and not (tmp_path / "processed" / "m1.wav").exists()
