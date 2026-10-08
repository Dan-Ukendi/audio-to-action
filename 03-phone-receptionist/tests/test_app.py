"""The push-to-talk page renders and starts a call without a browser, microphone or models (Streamlit's AppTest)."""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(HERE.parent / "app.py")


def test_page_renders_and_explains_the_missing_voice(monkeypatch):
    monkeypatch.delenv("RECEPTIONIST_SPEAKER", raising=False)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value.startswith("Receptionist")
    assert "No receptionist voice yet" in at.warning[0].value and "choose_voice" in at.warning[0].value
    assert [b.label for b in at.button] == ["Start a call"]


def test_starting_a_call_shows_the_greeting(monkeypatch):
    monkeypatch.delenv("RECEPTIONIST_SPEAKER", raising=False)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.button[0].click().run()
    assert not at.exception
    assert [m.name for m in at.chat_message] == ["assistant"]
    text = " ".join(e.value for e in at.markdown)
    assert "This is Holly, the automated assistant" in text and "This call is recorded" in text
