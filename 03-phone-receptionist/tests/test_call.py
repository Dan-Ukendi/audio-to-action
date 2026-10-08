"""Whole simulated calls: every caller card talks to the dialog until the call ends (no model: the rules baseline understands).

These check the INVARIANTS of the design (a call always ends, nothing is invented, the emergency path fires, everything is
saved) on all 24 cards. They deliberately do not score the baseline's accuracy: that is Phase 6, on the model.
Exact slot assertions are made on dev cards only.
"""

import json
import re
import sys
import wave
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

import asks  # noqa: E402
import call as call_module  # noqa: E402
import dialog  # noqa: E402
import faq as faq_module  # noqa: E402
from audio_io import Heard  # noqa: E402
from cards import load_cards  # noqa: E402
from persona import load_persona  # noqa: E402
from rules_turn import rules_understand  # noqa: E402
from spoken import digit_runs, letter_runs  # noqa: E402
from turn import Understanding  # noqa: E402

P = load_persona()
FAQ = faq_module.load_faq()
CARDS = {c.id.split("_")[0]: c for c in load_cards()}
UNDERSTAND = lambda text, ctx: Understanding(rules_understand(text, ctx))  # noqa: E731


def run(prefix, **kw):
    return call_module.run_call(CARDS[prefix], P, FAQ, UNDERSTAND, understand_label="rules", **kw)


@pytest.fixture(scope="module")
def records():
    return {prefix: run(prefix) for prefix in CARDS}


def test_every_call_ends_within_the_turn_limit_and_is_not_cut_off(records):
    for prefix, r in records.items():
        assert not r.cut_off and r.outcome and len(r.turns) <= P.max_turns, (prefix, r.outcome, len(r.turns))


def test_no_number_is_ever_invented(records):
    """The recorded number must have been said by the caller in this call, digit for digit."""
    for prefix, r in records.items():
        said = [run_ for t in r.turns for run_ in digit_runs(t["caller_said"], min_len=8)]
        number = r.message["number"]
        assert number is None or any(number in run_ for run_ in said), (prefix, number, said)


def test_no_name_is_ever_invented(records):
    for prefix, r in records.items():
        name = r.message["name"]
        if name:
            said = " ".join(t["caller_said"] for t in r.turns).lower()
            spelled = [x.lower() for t in r.turns for x in letter_runs(t["caller_said"])]
            words = set(re.findall(r"[a-z']+", said))
            assert all(part in words or any(part in s for s in spelled) for part in re.findall(r"[a-z']+", name.lower())), (prefix, name)


def test_dev_card_c11_the_robocall_gets_a_short_goodbye_and_is_not_urgent(records):
    r = records["c11"]
    assert r.outcome == "spam" and len(r.turns) == 1 and not r.message["urgent"] and r.turns[0]["reply"] == P.say("goodbye_spam")


def test_every_spam_ending_is_one_short_turn_and_never_urgent(records):
    for prefix, r in records.items():
        if r.outcome == "spam":
            assert len(r.turns) == 1 and not r.message["urgent"] and r.turns[0]["reply"] == P.say("goodbye_spam"), prefix


def test_dev_card_c03_the_gas_caller_is_flagged_in_the_first_turn_and_gets_the_gas_advice(records):
    r = records["c03"]
    assert r.urgent_flagged_at_turn == 1 and r.message["urgent"] and r.message["safety_advised"] == ["gas"]
    assert FAQ["safety_gas"].answer in r.turns[0]["reply"] and r.turns[0]["reply"].startswith(P.say("urgent_ack"))


def test_a_call_is_flagged_urgent_in_the_turn_Part_1s_safety_words_are_first_said(records):
    """Invariant over ALL cards (no card names): the first turn whose words contain a safety word flags the call, with that advice."""
    from safety import emergency_from_text
    for prefix, r in records.items():
        first = next((t for t in r.turns if emergency_from_text(t["caller_said"])), None)
        if first is None:
            continue
        assert r.urgent_flagged_at_turn == first["turn"], prefix
        for kind in emergency_from_text(first["caller_said"]).kinds:
            assert kind in r.message["safety_advised"] and FAQ[f"safety_{kind}"].answer in first["reply"], (prefix, kind)


def test_a_call_is_never_flagged_urgent_without_a_safety_word_or_the_models_flag(records):
    """The baseline has no model flag, so with it a call is urgent exactly when a safety word was said."""
    from safety import emergency_from_text
    for prefix, r in records.items():
        said_danger = any(emergency_from_text(t["caller_said"]) for t in r.turns)
        assert r.message["urgent"] == said_danger, prefix


def test_the_info_only_caller_is_answered_and_not_pressed_for_a_message(records):
    r = records["f03"]
    assert r.outcome == "info_only" and len(r.turns) == 2
    assert r.message["faq_answered"] == ["hours"] and r.message["name"] is None and r.message["number"] is None


def test_the_unanswerable_question_is_passed_on_and_still_a_message_is_taken(records):
    r = records["f04"]
    assert r.message["unanswered_questions"] == ["Do you fit solar panels?"]
    assert r.message["name"] == "Gareth Lloyd" and r.message["number"] == "01632960955" and r.outcome == "completed"


def test_dev_faq_callers_get_their_expected_answers(records):
    assert records["f01"].message["faq_answered"] == ["booking_time"]
    assert records["f03"].message["faq_answered"] == ["hours"]


def test_every_faq_answer_given_is_a_real_entry_said_word_for_word(records):
    """Invariant over all cards: whatever topic was answered, that entry's approved text is in a reply."""
    for prefix, r in records.items():
        for topic in r.message["faq_answered"]:
            assert topic in FAQ and any(FAQ[topic].answer in t["reply"] for t in r.turns), (prefix, topic)


def test_dev_card_c14_the_hard_name_is_fixed_by_the_spelling(records):
    r = records["c14"]
    assert r.message["name"] == "Siobhan Gallagher" and r.message["number"] == "01632960501"
    asked = [t["asked"] for t in r.turns]
    assert asks.SPELLING in asked


def test_dev_card_c18_the_self_corrected_number_is_the_right_one(records):
    assert records["c18"].message["number"] == "07700900349"


def test_dev_card_c13_the_wrong_number_is_corrected_at_the_read_back(records):
    r = records["c13"]
    assert r.message["number"] == "07700900204"
    assert [t["asked"] for t in r.turns].count(asks.CONFIRM) >= 2  # read back, "no, the number is ...", read back again
    first_number = next(t for t in r.turns if t["asked"] == asks.NUMBER)["understanding"]["turn"]["number"]
    assert first_number and first_number != "07700900204"


def test_dev_card_c07_a_caller_who_will_not_give_a_number_leaves_none(records):
    r = records["c07"]
    assert r.message["number"] is None and r.message["name"] == "Mum"
    assert asks.NUMBER_AGAIN in [t["asked"] for t in r.turns]  # asked once more, then recorded as "no number given"


def test_records_are_complete_and_json(records, tmp_path):
    r = records["c14"]
    path = call_module.save_record(r, tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["card_id"] == "c14_quote_hard_name" and data["channel"] == "text" and data["understand"] == "rules"
    assert data["greeting"] == P.say("greeting") and data["message"]["outcome"] == "completed"
    assert data["final_state"]["state"] == "ENDED" and "log" not in data["final_state"]
    turn = data["turns"][0]
    for key in ("turn", "caller_said", "caller_text", "asked", "understanding", "actions", "reply", "state_after", "understand_s"):
        assert key in turn
    assert not list(tmp_path.glob("*.tmp"))  # written through a temp name, then renamed


def test_call_ids_are_unique(records):
    again = run("c14")
    assert again.call_id != records["c14"].call_id and again.call_id.startswith("c14_quote_hard_name-")


def test_the_call_folder_is_git_ignored():
    ignore = (HERE.parents[1] / ".gitignore").read_text(encoding="utf-8")
    assert "03-phone-receptionist/calls/*" in ignore and "!03-phone-receptionist/**/.gitkeep" in ignore


def test_on_urgent_is_called_once_in_the_turn_the_call_becomes_urgent():
    seen = []
    r = run("c03", on_urgent=lambda state: seen.append(state.turns))
    assert seen == [1] and r.urgent_flagged_at_turn == 1
    seen.clear()
    run("c14", on_urgent=lambda state: seen.append(state.turns))
    assert seen == []


def test_a_dialog_that_never_ends_is_cut_off_not_looped_forever(monkeypatch):
    monkeypatch.setattr(dialog, "apply_limits", lambda state, actions, persona: actions)  # the engine's own limit disabled
    never_ends = lambda state, facts, persona: [dialog.Action(kind="repeat_request")]  # noqa: E731
    r = call_module.run_call(CARDS["c14"], P, FAQ, UNDERSTAND, decide_fn=never_ends)
    assert r.cut_off and r.outcome == "cut_off" and len(r.turns) == P.max_turns + call_module.SAFETY_MARGIN_TURNS


# ---------------------------------------------------------------- the audio channel (fake voice, fake ears)

def fake_audio_channel(tmp_path, speaker=20):
    def render_fn(card, text, out):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")  # the "audio" file carries its words so the fake ears can read them
        return out

    def listen_fn(wav, hint=None, model=None):
        text = wav.read_text(encoding="utf-8")
        return Heard(text=text, ignored=not text, why=None if text else "silence", raw_text=text, duration_s=2, speech_s=2,
                     min_logprob=-0.3, seconds=0.25)

    def speak_fn(text, out, spk):
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(22050)
            w.writeframes(b"\x00\x00" * 2205)
        return 0.05

    return call_module.AudioChannel(tmp_path, P.hint, speaker, render_fn, listen_fn, speak_fn, model="base")


def test_an_audio_call_goes_through_listen_and_speak_and_records_the_files(tmp_path):
    r = run("c14", channel=fake_audio_channel(tmp_path))  # a dev card
    assert r.channel == "audio" and r.outcome == "completed" and r.message["name"] == "Siobhan Gallagher"
    first = r.turns[0]
    assert Path(first["caller_audio"]).exists() and Path(first["reply_audio"]).exists()
    assert first["listen_s"] == 0.25 and first["speak_s"] == 0.05 and first["heard"]["raw_text"]
    assert (tmp_path / "turn_00_receptionist.wav").exists()  # the greeting was spoken too


def test_an_audio_call_without_a_chosen_voice_still_runs_as_text_replies(tmp_path):
    r = run("c14", channel=fake_audio_channel(tmp_path, speaker=None))
    assert r.outcome == "completed" and "reply_audio" not in r.turns[0]


def test_an_ignored_turn_reaches_the_dialog_as_silence_and_ends_a_silent_call(tmp_path):
    channel = fake_audio_channel(tmp_path)
    channel.listen_fn = lambda wav, hint=None, model=None: Heard(text="", ignored=True, why="silence", raw_text="Thank you.", duration_s=1,
                                                              speech_s=0, min_logprob=None, seconds=0.1)
    r = run("c14", channel=channel)
    assert r.outcome == "silence" and len(r.turns) == 2
    assert [t["reply"] for t in r.turns] == [P.say("repeat_request"), P.say("silence_end")]


def test_the_command_line_runs_cards_without_a_model_and_saves_only_on_request(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(call_module, "HERE", tmp_path)
    assert call_module.main(["--card", "c14", "--understand", "rules"]) == 0
    out = capsys.readouterr().out
    assert "=== c14_quote_hard_name" in out and "--- outcome=completed" in out and "Siobhan Gallagher" in out
    assert not (tmp_path / "calls").exists()
    assert call_module.main(["--card", "c11", "--understand", "rules", "--save"]) == 0
    assert len(list((tmp_path / "calls").glob("c11_*.json"))) == 1


def test_the_command_line_selects_dev_or_score_cards_and_rejects_unknown_ones(capsys):
    call_module.main(["--card", "dev", "--understand", "rules"])
    printed = capsys.readouterr().out
    assert printed.count("=== ") == 9
    with pytest.raises(SystemExit):
        call_module.main(["--card", "zzz", "--understand", "rules"])


def test_the_call_starts_when_it_starts_not_when_the_record_is_built():
    from datetime import datetime, timezone
    import time as _time
    before = datetime.now(timezone.utc)

    def slow(text, ctx):
        _time.sleep(0.15)
        return UNDERSTAND(text, ctx)

    r = call_module.run_call(CARDS["c14"], P, FAQ, slow, understand_label="rules")
    started = datetime.fromisoformat(r.started)
    assert (started - before).total_seconds() < 0.5 and r.total_s > 0.5  # the call lasted a while; it began right away


def test_the_urgent_hook_runs_before_the_reply_is_spoken():
    events = []

    class Channel(call_module.TextChannel):
        def speak(self, reply, turn):
            events.append(("speak", turn))
            return {}

    run("c03", channel=Channel(), on_urgent=lambda state: events.append(("push", state.turns)))
    assert events.index(("push", 1)) < events.index(("speak", 1))
