"""The app's helpers and a headless render of every tab (no browser, no LLM)."""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/

import helpers as h  # noqa: E402


def test_safe_name_keeps_it_simple():
    assert h.safe_name("My Voicemail (2).MP3") == "My_Voicemail_2.mp3"
    assert h.safe_name("../../evil.wav") == "evil.wav"


def test_meeting_date_added_only_when_missing():
    assert h.with_meeting_date("standup.m4a", date(2026, 10, 12)) == "standup_2026-10-12.m4a"
    assert h.with_meeting_date("team_2026-09-21.mp3", date(2026, 10, 12)) == "team_2026-09-21.mp3"


def test_save_upload_never_overwrites(tmp_path):
    first = h.save_upload(b"one", "call.wav", tmp_path)
    second = h.save_upload(b"two", "call.wav", tmp_path)
    assert first != second and first.read_bytes() == b"one" and second.read_bytes() == b"two"


def test_describe_change_in_plain_words():
    assert h.describe_change("add", None, {"status": "open"}) == "added [open]"
    before = {"status": "open", "owner": None, "due": None}
    after = {"status": "done", "owner": "Jamie", "due": None}
    assert h.describe_change("update", before, after) == "status open → done, owner None → Jamie"


def test_missing_database_reads_as_empty(tmp_path):
    assert h.query(tmp_path / "nope.db", "SELECT 1") == []


def test_app_renders_without_errors():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=60)
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs][:3] == ["📞 Voicemails", "🗓️ Meetings", "❓ How it works"]
