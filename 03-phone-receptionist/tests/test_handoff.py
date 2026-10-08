"""A finished call becomes a row in Part 1's voicemails.db: caller-side transcript, analysis, routing, push, storage."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "01-voicemail-triage"))

import call as call_module  # noqa: E402
import faq as faq_module  # noqa: E402
import handoff  # noqa: E402
import shared.analyze as analyze_module  # noqa: E402
import store  # noqa: E402
from cards import load_cards  # noqa: E402
from persona import load_persona  # noqa: E402
from rules_turn import rules_understand  # noqa: E402
from shared.llm import LLMFormError, StructuredReply  # noqa: E402
from shared.schemas import Analysis, Result  # noqa: E402
from turn import Understanding  # noqa: E402

P = load_persona()
FAQ = faq_module.load_faq()
CARDS = {c.id.split("_")[0]: c for c in load_cards()}
UNDERSTAND = lambda text, ctx: Understanding(rules_understand(text, ctx))  # noqa: E731


class Sender:
    """A recording stand-in for ntfy: what would have been pushed."""

    def __init__(self, status="dry_run"):
        self.calls, self.status = [], status

    def __call__(self, title, message, priority="high", tags=""):
        self.calls.append({"title": title, "message": message, "priority": priority})
        return self.status


def model_result(**changes) -> Result:
    """What Part 1's analyze() would return for a call (fake model output)."""
    analysis = dict(summary="Customer asks for a bathroom quote.", reason="A quote request, no time pressure.", category="other",
                    urgency=1, caller_name="Siobhan Gallagher", callback_number="01632960501", language="en")
    analysis.update(changes)
    return Result(source_file="x", audio_sha256="0" * 64, transcript_model="call", llm_model="qwen-test", prompt_version="call-v1",
                  attempts=1, rejected_reply=None, rejected_because=None, analyze_s=1.5, analysis=Analysis(**analysis),
                  created_at=datetime.now(timezone.utc))


def call(prefix, conn=None, **kw):
    conn = conn or store.connect(":memory:")
    sender = kw.pop("sender", Sender())
    record, outcome = handoff.run_and_hand_off(CARDS[prefix], P, FAQ, UNDERSTAND, conn, sender=sender, understand_label="rules", **kw)
    return record, outcome, conn, sender


# ---------------------------------------------------------------- the transcript

def test_only_the_callers_side_goes_to_analysis_and_it_is_marked_as_a_call():
    record, outcome, conn, _ = call("c14", use_model=False)
    t = outcome.transcript
    assert t.source_file == f"call-{record.call_id}.json" and handoff.is_call(t.source_file) and not handoff.is_call("01_urgent.wav")
    assert "Holly" not in t.text and "Could I take your name" not in t.text
    assert t.text.splitlines()[0].startswith("Hello, I'm looking for a quote") and "S, I, O, B, H, A, N" in t.text
    assert len(t.segments) == len(t.text.splitlines()) and t.model == "call"


def test_the_same_call_always_has_the_same_hash_and_different_calls_differ():
    r1, _, _, _ = call("c14", use_model=False)
    assert handoff.call_transcript(r1).audio_sha256 == handoff.call_transcript(r1).audio_sha256
    r2, _, _, _ = call("c14", use_model=False)
    assert handoff.call_transcript(r1).audio_sha256 != handoff.call_transcript(r2).audio_sha256


def test_audio_confidence_reaches_part_1s_review_flag_for_unclear_audio():
    record, _, _, _ = call("c14", use_model=False)
    record.turns[0]["heard"] = {"ignored": False, "why": None, "raw_text": "x", "min_logprob": -1.4}
    t = handoff.call_transcript(record)
    assert t.segments[0].avg_logprob == -1.4
    from routing import route
    decision = route(t, handoff.analysis_for_call(record, t, use_model=False))
    assert decision.review and any("unclear audio" in r for r in decision.reasons)


# ---------------------------------------------------------------- the analysis

def test_the_model_judges_category_and_summary_but_the_dialogs_name_and_number_win():
    record, _, conn, _ = call("c14", use_model=False)
    fake = lambda transcript: model_result(caller_name="Sean Gallagher", callback_number="01632960999")  # noqa: E731  (invented)
    result = handoff.analysis_for_call(record, handoff.call_transcript(record), fake)
    assert result.analysis.caller_name == "Siobhan Gallagher" and result.analysis.callback_number == "01632960501"
    assert result.analysis.summary == "Customer asks for a bathroom quote." and result.llm_model == "qwen-test"
    assert result.prompt_version == "call-v1"


def test_a_number_the_dialog_did_not_record_is_not_taken_from_the_model():
    record, _, _, _ = call("c04", use_model=False)  # the caller refused the number
    assert record.message["number"] is None
    fake = lambda transcript: model_result(caller_name="Dave", callback_number="07700900555")  # noqa: E731
    result = handoff.analysis_for_call(record, handoff.call_transcript(record), fake)
    assert result.analysis.callback_number is None


def test_a_call_flagged_urgent_during_the_call_stays_urgent_even_if_the_model_says_otherwise():
    record, _, _, _ = call("c03", use_model=False)
    fake = lambda transcript: model_result(category="other", urgency=1)  # noqa: E731
    result = handoff.analysis_for_call(record, handoff.call_transcript(record), fake)
    assert result.analysis.category == "urgent" and result.analysis.urgency == 3
    assert "Flagged urgent during the call" in result.analysis.reason


def test_the_model_can_add_urgency_the_dialog_did_not_see():
    record, _, _, _ = call("c14", use_model=False)
    result = handoff.analysis_for_call(record, handoff.call_transcript(record),
                                       lambda transcript: model_result(category="urgent", urgency=3))
    assert result.analysis.category == "urgent"


def test_calls_without_a_message_need_no_model_at_all():
    def must_not_run(transcript):
        raise AssertionError("the model was called")

    for prefix, category, words in (("c11", "spam", "Automated or scam"), ("f03", "other", "only asked questions")):
        record, _, _, _ = call(prefix, use_model=False)
        result = handoff.analysis_for_call(record, handoff.call_transcript(record), must_not_run)
        assert result.analysis.category == category and words in result.analysis.summary and result.llm_model == "none (rules)"


def test_when_the_model_fails_plain_code_builds_the_analysis_and_the_row_is_flagged_for_review():
    record, _, _, _ = call("c14", use_model=False)

    def broken(transcript):
        raise analyze_module.AnalysisError("invalid answer twice")

    result = handoff.analysis_for_call(record, handoff.call_transcript(record), broken)
    assert result.attempts == 2 and "failed" in result.rejected_because and result.llm_model == "none (model failed)"
    assert result.analysis.caller_name == "Siobhan Gallagher" and "called about" in result.analysis.summary
    from routing import route
    assert route(handoff.call_transcript(record), result).review  # Part 1's "analysis needed a retry" rule


def test_a_bug_in_the_analysis_is_not_swallowed():
    record, _, _, _ = call("c14", use_model=False)
    with pytest.raises(KeyError):
        handoff.analysis_for_call(record, handoff.call_transcript(record), lambda t: (_ for _ in ()).throw(KeyError("my bug")))


def test_summaries_respect_part_1s_25_word_limit():
    record, _, _, _ = call("c16", use_model=False)
    assert len(handoff.rule_analysis(record).summary.split()) <= handoff.SUMMARY_WORDS


def test_part_1s_analyze_uses_the_call_prompt_and_label_for_calls(monkeypatch):
    seen = {}

    def fake_chat(schema, messages, llm=None, context=None, fix_hint=None, options=None):
        seen["messages"] = messages
        return StructuredReply(model_result().analysis, 1, None, None, 0.1)

    monkeypatch.setattr(analyze_module, "structured_chat", fake_chat)
    record, _, _, _ = call("c14", use_model=False)
    analyze_module.analyze(handoff.call_transcript(record), cache_dir=None, prompt_version="call-v1")
    system, user = (m["content"] for m in seen["messages"])
    assert "THIS IS A PHONE CALL, NOT A VOICEMAIL" in system and user.startswith("Phone call (what the caller said")
    analyze_module.analyze(handoff.call_transcript(record), cache_dir=None, prompt_version="v2")  # Part 1 is unchanged
    system, user = (m["content"] for m in seen["messages"])
    assert "PHONE CALL" not in system and user.startswith("Voicemail transcript:")


# ---------------------------------------------------------------- routing, push, storage

def test_every_call_lands_in_the_database_with_a_route_and_reasons():
    conn = store.connect(":memory:")
    for prefix in CARDS:
        call(prefix, conn=conn, use_model=False)
    rows = conn.execute("SELECT * FROM voicemails").fetchall()
    assert len(rows) == 24
    for row in rows:
        assert handoff.is_call(row["source_file"]) and row["route"] in ("notify_now", "inbox", "personal", "archive")
        reasons = json.loads(row["reasons"])
        assert len(reasons) >= 2 and "phone call" in reasons[1]
    routes = {r["source_file"].split("-")[1].split("_")[0]: r["route"] for r in rows}
    assert routes["c11"] == routes["c12"] == "archive"
    assert routes["c03"] == "notify_now" and routes["c14"] == "inbox"


def test_the_row_holds_the_checked_name_number_and_the_callers_words_only():
    record, outcome, conn, _ = call("c14", use_model=False)
    row = conn.execute("SELECT * FROM voicemails").fetchone()
    assert (row["caller_name"], row["callback_number"]) == ("Siobhan Gallagher", "01632960501")
    assert row["whisper_model"] == "call" and row["prompt_version"] == "call-rules" and "Holly" not in row["transcript"]


def test_an_urgent_call_is_pushed_once_while_the_caller_is_still_on_the_line():
    record, outcome, conn, sender = call("c03", use_model=False)
    assert outcome.pushed == "live" and len(sender.calls) == 1
    assert sender.calls[0]["title"] == "Urgent call in progress" and sender.calls[0]["priority"] == "urgent"
    text = json.dumps(sender.calls)
    for secret in ("Priya", "gas", "07700", "hallway"):
        assert secret not in text  # Part 1's rule: a push says that something needs attention, never what
    row = conn.execute("SELECT route, notified_at FROM voicemails").fetchone()
    assert row["route"] == "notify_now"
    assert record.handoff["pushed"] == "live" and record.handoff["route"] == "notify_now"


def test_a_failed_live_push_does_not_end_the_call_and_the_push_is_sent_at_the_end_instead():
    def down(title, message, priority="high", tags=""):
        raise ConnectionError("ntfy unreachable")

    live = handoff.LivePush(down)
    record = call_module.run_call(CARDS["c03"], P, FAQ, UNDERSTAND, on_urgent=live, understand_label="rules")
    live.wait()  # the push runs in the background: wait for it before looking at the result
    assert record.outcome == "completed" and live.status is None and "ntfy unreachable" in live.error
    conn, sender = store.connect(":memory:"), Sender()
    outcome = handoff.hand_off(record, conn, use_model=False, live=live, sender=sender)
    assert outcome.pushed == "end" and len(sender.calls) == 1 and sender.calls[0]["title"] == "Urgent voicemail"


def test_an_urgent_call_without_a_live_push_is_pushed_at_hand_off_with_part_1s_minimal_text():
    record = call_module.run_call(CARDS["c03"], P, FAQ, UNDERSTAND, understand_label="rules")
    conn, sender = store.connect(":memory:"), Sender()
    outcome = handoff.hand_off(record, conn, use_model=False, sender=sender)
    assert outcome.pushed == "end" and len(sender.calls) == 1 and "Priya" not in sender.calls[0]["message"]


def test_a_push_that_cannot_be_sent_at_all_leaves_no_row_so_the_next_run_tries_again():
    """Part 1's at-least-once order: push first, save second."""
    record = call_module.run_call(CARDS["c03"], P, FAQ, UNDERSTAND, understand_label="rules")
    conn = store.connect(":memory:")

    def down(title, message, priority="high", tags=""):
        raise ValueError("a bug, not a network error: not retried")

    with pytest.raises(ValueError):
        handoff.hand_off(record, conn, use_model=False, sender=down)
    assert conn.execute("SELECT COUNT(*) FROM voicemails").fetchone()[0] == 0


def test_handing_the_same_call_off_twice_changes_nothing_and_pushes_nothing_more():
    record, first, conn, sender = call("c03", use_model=False)
    again = handoff.hand_off(record, conn, use_model=False, sender=sender)  # no live push this time
    assert conn.execute("SELECT COUNT(*) FROM voicemails").fetchone()[0] == 1
    assert again.pushed == "already" and len(sender.calls) == 1


def test_non_urgent_calls_are_never_pushed():
    for prefix in ("c14", "c11", "f03", "c07"):
        _, outcome, _, sender = call(prefix, use_model=False)
        assert outcome.pushed == "none" and sender.calls == [], prefix


def test_missing_details_are_flagged_for_review_by_part_1s_rules():
    _, outcome, _, _ = call("c07", use_model=False)  # a caller who left no number
    assert outcome.decision.review and any("without a callback number" in r for r in outcome.decision.reasons)


def test_an_info_only_call_is_stored_but_not_flagged_for_a_missing_name_or_number():
    _, outcome, conn, _ = call("f03", use_model=False)
    assert not outcome.decision.review and any("info-only call" in r for r in outcome.decision.reasons)
    assert conn.execute("SELECT COUNT(*) FROM voicemails").fetchone()[0] == 1


def test_part_1s_summary_report_works_on_a_database_full_of_calls(capsys):
    conn = store.connect(":memory:")
    for prefix in ("c14", "c03", "c11"):
        call(prefix, conn=conn, use_model=False)
    store.summary(conn)
    assert "notify_now" in capsys.readouterr().out


def test_calls_through_the_model_path_are_stored_with_the_models_judgement():
    conn = store.connect(":memory:")
    record, outcome, _, _ = call("c14", conn=conn, analyze_fn=lambda t: model_result(summary="Bathroom quote request."))
    row = conn.execute("SELECT summary, llm_model, prompt_version FROM voicemails").fetchone()
    assert (row["summary"], row["llm_model"], row["prompt_version"]) == ("Bathroom quote request.", "qwen-test", "call-v1")


# ---------------------------------------------------------------- the files

def test_the_record_remembers_its_hand_off_and_the_command_line_hands_saved_calls_off(tmp_path, capsys):
    conn = store.connect(":memory:")
    record, _, _, _ = call("c14", conn=conn, use_model=False, save_dir=tmp_path)
    saved = json.loads(next(tmp_path.glob("c14_*.json")).read_text(encoding="utf-8"))
    assert saved["handoff"]["route"] == "inbox" and saved["handoff"]["source_file"].startswith("call-c14_")
    db = tmp_path / "t.db"
    assert handoff.main([str(next(tmp_path.glob("c14_*.json"))), "--db", str(db), "--analysis", "rules"]) == 0
    assert "route=inbox" in capsys.readouterr().out
    assert store.connect(db).execute("SELECT COUNT(*) FROM voicemails").fetchone()[0] == 1


def test_the_default_database_is_git_ignored():
    ignore = (HERE.parents[1] / ".gitignore").read_text(encoding="utf-8")
    assert "*.db" in ignore


# ---------------------------------------------------------------- review round 1

def test_a_call_that_becomes_urgent_on_a_second_hand_off_is_pushed_then():
    """--analysis rules first, --analysis model later: the later run may find urgency the first one did not."""
    record, first, conn, sender = call("c14", use_model=False)
    assert first.pushed == "none" and sender.calls == []
    again = handoff.hand_off(record, conn, analyze_fn=lambda t: model_result(category="urgent", urgency=3), sender=sender)
    assert again.decision.route == "notify_now" and again.pushed == "end" and len(sender.calls) == 1
    # and once it HAS been pushed, a third hand-off does not push again
    third = handoff.hand_off(record, conn, analyze_fn=lambda t: model_result(category="urgent", urgency=3), sender=sender)
    assert third.pushed == "already" and len(sender.calls) == 1


def test_a_message_that_ends_in_silence_is_still_a_message():
    record, _, conn, _ = call("c14", use_model=False)
    record.outcome = "silence"  # the caller left a full message, then went quiet
    seen = []
    result = handoff.analysis_for_call(record, handoff.call_transcript(record), lambda t: seen.append(t) or model_result())
    assert seen and result.llm_model == "qwen-test"  # the model WAS asked: there is something to judge
    rules = handoff.rule_analysis(record)
    assert "said nothing" not in rules.summary and "Siobhan Gallagher" in rules.summary
    empty, _, _, _ = call("c11", use_model=False)
    empty.outcome = "silence"
    empty.message.update(reason=None, name=None, number=None)
    assert "said nothing" in handoff.rule_analysis(empty).summary


def test_an_info_only_call_keeps_every_review_flag_except_the_two_about_a_missing_name_and_number():
    record, _, _, _ = call("f03", use_model=False)
    record.turns[0]["heard"] = {"ignored": False, "why": None, "raw_text": "x", "min_logprob": -2.0}
    outcome = handoff.hand_off(record, store.connect(":memory:"), use_model=False)
    assert outcome.decision.review and any("unclear audio" in r for r in outcome.decision.reasons)
    assert not any(r.startswith("customer call without") for r in outcome.decision.reasons)
    # a safety word in an info-only call keeps its push AND its review flag
    record, _, _, _ = call("f03", use_model=False)
    record.turns[0]["caller_text"] = "The pipe has burst but what are your opening hours?"
    outcome = handoff.hand_off(record, store.connect(":memory:"), use_model=False, sender=Sender())
    assert outcome.decision.notify and outcome.decision.review and any("push + review" in r for r in outcome.decision.reasons)


def test_the_call_record_survives_a_failed_end_of_call_push(tmp_path):
    def down(title, message, priority="high", tags=""):
        raise ValueError("push failed for a reason that is not a network error")

    conn = store.connect(":memory:")
    with pytest.raises(ValueError):
        handoff.run_and_hand_off(CARDS["c03"], P, FAQ, UNDERSTAND, conn, sender=down, save_dir=tmp_path, use_model=False,
                                 understand_label="rules")
    saved = list(tmp_path.glob("c03_*.json"))
    assert len(saved) == 1 and json.loads(saved[0].read_text(encoding="utf-8"))["handoff"] is None  # saved, not yet handed off
    assert conn.execute("SELECT COUNT(*) FROM voicemails").fetchone()[0] == 0
    assert handoff.main([str(saved[0]), "--db", str(tmp_path / "t.db"), "--analysis", "rules"]) == 0  # the next run completes it
    assert store.connect(tmp_path / "t.db").execute("SELECT COUNT(*) FROM voicemails").fetchone()[0] == 1


def test_the_command_line_expands_patterns_itself_and_survives_one_bad_file(tmp_path, capsys):
    for prefix in ("c14", "c11"):
        record = call_module.run_call(CARDS[prefix], P, FAQ, UNDERSTAND, understand_label="rules")
        call_module.save_record(record, tmp_path)
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "half.tmp").write_text("x", encoding="utf-8")
    db = tmp_path / "new" / "folder" / "t.db"  # a folder that does not exist yet
    status = handoff.main([str(tmp_path / "*.json"), "--db", str(db), "--analysis", "rules"])
    out = capsys.readouterr().out
    assert status == 1 and "broken.json: FAILED" in out and out.count("route=") == 2
    assert store.connect(db).execute("SELECT COUNT(*) FROM voicemails").fetchone()[0] == 2
    with pytest.raises(SystemExit, match="no call files match"):
        handoff.main([str(tmp_path / "nothing*.json"), "--db", str(db)])


def test_the_live_push_never_makes_the_caller_wait():
    import time
    released = []

    def slow(title, message, priority="high", tags=""):
        time.sleep(0.4)
        released.append(1)
        return "dry_run"

    live = handoff.LivePush(slow)
    started = time.perf_counter()
    live(None)
    assert time.perf_counter() - started < 0.2 and live.status is None  # returned at once: the call goes on
    live.wait()
    assert live.status == "dry_run" and released == [1]


def test_a_live_push_that_is_still_running_when_waited_on_counts_as_failed():
    import threading
    block = threading.Event()

    def stuck(title, message, priority="high", tags=""):
        block.wait(2)
        return "sent"

    live = handoff.LivePush(stuck)
    live(None)
    live.wait(timeout=0.05)
    assert live.status is None and "still sending" in live.error
    block.set()


def test_a_hand_off_remembers_whether_the_live_push_worked():
    record, outcome, _, _ = call("c03", use_model=False)
    assert record.handoff["live_push"] == {"status": "dry_run", "error": None}
    record, outcome, _, _ = call("c14", use_model=False)
    assert record.handoff["live_push"]["status"] is None  # no urgent flag: no live push


def test_expand_keeps_files_only_and_each_file_once(tmp_path):
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    found = handoff.expand([str(tmp_path / "*"), str(tmp_path / "a.json"), str(tmp_path / "*.json")])
    assert found == [tmp_path / "a.json"]


def test_every_review_reason_part_1_can_give_is_one_the_handoff_knows_about():
    """handoff.REVIEW_REASONS must stay in step with routing.py: feed route() inputs for each review rule."""
    from datetime import datetime, timezone as tz
    from routing import route
    from shared.schemas import Segment, Transcript
    cases = {
        "customer call without a caller name": dict(text="please ring me", name=None, number="07700900123", category="other", urgency=1),
        "customer call without a callback number": dict(text="please ring me", name="Dave", number=None, category="other", urgency=1),
        "safety words": dict(text="there is a smell of gas", name="Dave", number="07700900123", category="other", urgency=1),
        "unclear audio": dict(text="hello there", name="Dave", number="07700900123", category="other", urgency=1, logprob=-2.0),
        "analysis needed a retry": dict(text="hello there", name="Dave", number="07700900123", category="other", urgency=1, attempts=2),
        "no speech found": dict(text="", name="Dave", number="07700900123", category="other", urgency=1, segments=False),
    }
    now = datetime.now(tz.utc)
    for expected, c in cases.items():
        segments = [Segment(start=0, end=1, text=c["text"], avg_logprob=c.get("logprob", 0.0), no_speech_prob=0.0)] if c.get("segments", True) else []
        t = Transcript(source_file="call-x.json", audio_sha256="0" * 64, model="call", language="en", language_probability=1.0,
                       duration_s=0, transcribe_s=0, text=c["text"], segments=segments, created_at=now)
        analysis = Analysis(summary="s", reason="r", category=c["category"], urgency=c["urgency"], caller_name=c["name"],
                            callback_number=c["number"], language="en")
        result = Result(source_file="x", audio_sha256="0" * 64, transcript_model="call", llm_model="m", prompt_version="p",
                        attempts=c.get("attempts", 1), rejected_reply=None, rejected_because="x" if c.get("attempts", 1) > 1 else None,
                        analyze_s=0, analysis=analysis, created_at=now)
        decision = route(t, result)
        flagged = [r for r in decision.reasons[1:] if r.startswith(handoff.REVIEW_REASONS)]
        assert decision.review and flagged and any(r.startswith(expected) for r in flagged), (expected, decision.reasons)
