"""Tests for the routing rules, using hand-made voicemails (no audio, no LLM, instant).

    python -m pytest 01-voicemail-triage/tests -v

The test set never made the safety net fire (the LLM caught every urgent call), so these
cases force each rule to run, including the LLM getting it wrong.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # repo root, for 'shared'
sys.path.insert(0, str(HERE.parent))  # 01-voicemail-triage, for 'routing'

from routing import push_text, route  # noqa: E402
from shared.schemas import Analysis, Result, Segment, Transcript  # noqa: E402

NOW = datetime.now(timezone.utc)


def make(text: str, category: str, urgency: int = 1, name: str | None = "Jo Bloggs",
         number: str | None = "07700900123", attempts: int = 1, logprob: float = -0.2):
    """Build a (Transcript, Result) pair as if Whisper and the LLM had produced them."""
    transcript = Transcript(
        source_file="fake.wav", audio_sha256="0" * 64, model="small", language="en",
        language_probability=1.0, duration_s=10, transcribe_s=1, text=text,
        segments=[Segment(start=0, end=10, text=text, avg_logprob=logprob, no_speech_prob=0.0)],
        created_at=NOW,
    )
    analysis = Analysis(summary="s", reason="r", category=category, urgency=urgency,
                        caller_name=name, callback_number=number, language="en")
    result = Result(source_file="fake.wav", audio_sha256="0" * 64, transcript_model="small",
                    llm_model="test", prompt_version="test", attempts=attempts, analyze_s=1,
                    analysis=analysis, created_at=NOW)
    return transcript, result


def test_each_category_goes_to_its_route():
    expected = {"urgent": "notify_now", "other": "inbox", "personal": "personal",
                "spam": "archive", "sales": "archive"}
    for category, wanted in expected.items():
        d = route(*make("hello", category, urgency=3 if category == "urgent" else 1))
        assert d.route == wanted, category
        assert d.notify == (category == "urgent")


def test_safety_net_catches_llm_miss():
    # The LLM said 'other', but the caller smells gas: push anyway and flag for review.
    d = route(*make("There's a smell of gas in the kitchen, call me back", "other"))
    assert d.route == "inbox" and d.notify and d.review
    assert any("safety words" in r for r in d.reasons)


def test_safety_net_never_lets_danger_reach_the_archive():
    d = route(*make("Our pipe has burst and water is everywhere", "spam"))
    assert d.route != "archive"
    assert d.notify


def test_ordinary_gas_boiler_request_is_not_an_emergency():
    # 'gas' alone must not trigger: plumbers hear "gas boiler" all day.
    d = route(*make("Could you service my gas boiler next week?", "other"))
    assert not d.notify and not d.review


def test_urgent_without_number_still_pushes_but_needs_review():
    d = route(*make("It's Dave, the leak is worse, you've got my number", "urgent", urgency=3, number=None))
    assert d.route == "notify_now" and d.notify and d.review


def test_retry_and_unclear_audio_flag_review():
    assert route(*make("hi", "other", attempts=2)).review
    assert route(*make("hi", "other", logprob=-1.5)).review


def test_push_text_contains_no_caller_data():
    transcript, result = make("Smell of gas, it's Jo Bloggs on 07700900123", "urgent", urgency=3)
    title, message, _ = push_text(route(transcript, result), received="14:32")
    for private in ("Jo", "Bloggs", "07700900123", "gas"):
        assert private not in title + message
