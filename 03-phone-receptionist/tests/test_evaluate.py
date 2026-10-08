"""Phase 6: the evaluation harness. Everything runs without a model: these tests check the counting and the A-vs-B rule, not the receptionist."""

import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import dialog  # noqa: E402
import evaluate  # noqa: E402
from cards import load_cards, split_cards  # noqa: E402
from faq import load_faq  # noqa: E402
from persona import load_persona  # noqa: E402
from turn import Understanding  # noqa: E402
from rules_turn import rules_understand  # noqa: E402

P, FAQ = load_persona(), load_faq()
UNDERSTAND = lambda text, ctx: Understanding(rules_understand(text, ctx))  # noqa: E731


def summary(**changes) -> dict:
    base = dict(calls=15, audio=False, name_correct=12, number_correct=12, details_correct=24, invented_numbers=0, missed_urgent=0,
                false_urgent=0, faq_cards=5, faq_ok=5, faq_extra_answers=0, non_approved_replies=0, latency_p50=2.0, latency_p95=4.0,
                agent_turns=50, fallback_turns=2, fallback_share=0.04, handoff_calls=15, missed_push=0, false_push=0, category_ok=14,
                completed_calls=12, outcome_ok=15, cut_off=0, turns=60, name_classes={}, number_classes={})
    return {**base, **changes}


# ---------------------------------------------------------------- the pre-registered rule

def test_b_replaces_a_only_when_every_rule_holds():
    a = [summary(details_correct=20, latency_p50=1.0), summary(details_correct=21, latency_p50=1.0)]
    assert evaluate.decide_ab(a, summary(details_correct=23, latency_p50=3.5))["verdict"] == "B replaces A"


@pytest.mark.parametrize("changes", [
    dict(invented_numbers=1), dict(missed_urgent=1), dict(non_approved_replies=1),     # rule 1
    dict(details_correct=21),                                                          # rule 2: only 1 more
    dict(latency_p50=4.5), dict(fallback_share=0.11),                                  # rule 3
])
def test_a_single_broken_rule_keeps_a(changes):
    a = [summary(details_correct=20, latency_p50=1.0), summary(details_correct=20, latency_p50=1.0)]
    b = summary(details_correct=24, latency_p50=2.0) | changes
    assert evaluate.decide_ab(a, b)["verdict"] in ("A stays", "inconclusive, A stays")


def test_when_a_varies_by_as_much_as_b_gained_the_result_is_inconclusive_and_a_stays():
    a = [summary(details_correct=20), summary(details_correct=23)]
    verdict = evaluate.decide_ab(a, summary(details_correct=23))
    assert verdict["verdict"] == "inconclusive, A stays"


def test_without_a_second_a_run_b_cannot_win():
    verdict = evaluate.decide_ab([summary(details_correct=20)], summary(details_correct=25))
    assert verdict["verdict"] != "B replaces A" and "no noise estimate" in verdict["rules"][-1]["evidence"]


def test_the_rule_constants_match_the_readme():
    text = (Path(evaluate.HERE) / "README.md").read_text(encoding="utf-8")
    assert "at least **2 more**" in text and "at most A's + 3 s" in text and "at most 10 %" in text
    assert evaluate.MIN_EXTRA_DETAILS == 2 and evaluate.MAX_EXTRA_SECONDS == 3.0
    assert evaluate.MAX_FALLBACK_SHARE == 0.10


# ---------------------------------------------------------------- counting

def test_only_approved_pieces_pass_the_sentence_check():
    pattern = evaluate.approved_pattern(P, FAQ)
    _, state = dialog.start_call(P)
    assert pattern.fullmatch(P.say("ask_number"))
    assert pattern.fullmatch(P.say("ask_number") + " " + P.say("ask_number"))
    assert not pattern.fullmatch("We will fix it for fifty pounds.")


def record(numbers_said: list[str], recorded: str | None):
    return NS(message={"number": recorded}, turns=[{"caller_text": t} for t in numbers_said])


def test_a_number_the_caller_never_said_counts_as_invented():
    assert not evaluate.invented_number(record(["my number is 01632 960 501"], "01632960501"))
    assert evaluate.invented_number(record(["my number is 01632 960 501"], "01632960502"))
    assert not evaluate.invented_number(record(["hello"], None))
    assert not evaluate.invented_number(record(["zero seven seven zero zero nine zero zero one two three"], "07700900123"))


def test_a_number_corrected_over_two_turns_is_not_called_invented():
    assert not evaluate.invented_number(record(["07700 900", "123"], "07700900123"))


def test_numbers_are_compared_as_digits():
    assert evaluate.digits_of("01632 960 501") == "01632960501" and evaluate.digits_of(None) is None and evaluate.digits_of("") is None


# ---------------------------------------------------------------- running cards

def test_a_rules_run_on_the_dev_cards_scores_every_call_and_hands_each_off():
    cards = split_cards(load_cards(), "dev")
    scores = evaluate.run_set(cards, P, FAQ, UNDERSTAND, "a", lambda: dialog.decide_a)
    s = evaluate.summarize(scores)
    assert s["calls"] == 9 and s["handoff_calls"] == 9 and s["invented_numbers"] == 0 and s["non_approved_replies"] == 0
    assert s["missed_push"] == 0 and s["false_push"] == 0
    assert all(sc["turn_seconds"] and len(sc["turn_seconds"]) == sc["turns"] for sc in scores)


def test_a_wrong_recorded_name_is_scored_wrong_not_correct():
    card = next(c for c in load_cards() if c.id.startswith("c14"))
    from call import run_call
    rec = run_call(card, P, FAQ, UNDERSTAND)
    rec.message["name"] = "Somebody Else"
    rec.message["number"] = "01632960999"
    sc = evaluate.score_call(card, rec, evaluate.approved_pattern(P, FAQ))
    assert sc["name"] == "wrong" and sc["number"] == "wrong" and sc["details_correct"] == 0 and sc["invented_number"]


def test_the_definition_of_done_table_says_not_measured_for_speed_without_an_audio_run():
    rows = {c: r for c, r, _ in evaluate.dod_table(summary())}
    assert rows["Median reply <= 5 s"] == "NOT MEASURED"
    assert all(r in ("PASS", "FAIL", "NOT MEASURED", "OWNER") for r in rows.values())
    slow = {c: r for c, r, _ in evaluate.dod_table(summary(audio=True, latency_p50=6.0))}
    assert slow["Median reply <= 5 s"] == "FAIL"


# ---------------------------------------------------------------- the command and the results file

def test_the_committed_results_file_is_the_honest_placeholder_until_a_laptop_run_replaces_it():
    text = evaluate.RESULTS_FILE.read_text(encoding="utf-8")
    assert text == evaluate.placeholder_markdown() or "Run 20" in text   # either never measured, or a real run's report
    if "Run 20" not in text:
        assert "NOT MEASURED YET" in text and "evaluate.py" in text


def test_a_run_without_a_model_never_writes_the_results_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(evaluate, "RESULTS_FILE", tmp_path / "results.md")
    monkeypatch.setattr(evaluate, "LOGS", tmp_path / "logs")
    assert evaluate.main(["--split", "f03", "--understand", "rules", "--decide", "a", "--repeat-a", "1", "--write-docs"]) == 0
    out = capsys.readouterr().out
    assert not (tmp_path / "results.md").exists() and "Harness check only" in out and "NOT written" in out
    assert list((tmp_path / "logs").glob("eval-*.json"))


def test_both_versions_give_a_verdict_and_the_model_flag_decides_whether_docs_are_written(tmp_path, monkeypatch, capsys):
    import agent_dialog

    def fake_b():
        def decide(state, facts, persona):
            return dialog.decide_a(state, facts, persona)
        decide.last_trace = None
        return decide

    monkeypatch.setattr(agent_dialog, "make_decide_b", fake_b)
    monkeypatch.setattr(evaluate, "LOGS", tmp_path / "logs")
    assert evaluate.main(["--split", "f0", "--understand", "rules", "--decide", "both", "--repeat-a", "2"]) == 0
    out = capsys.readouterr().out
    assert "A or B?" in out and "A stays" in out


# ---------------------------------------------------------------- review fixes

def test_noise_plus_a_failed_cost_rule_is_a_plain_a_stays_not_inconclusive():
    a = [summary(details_correct=20), summary(details_correct=23)]
    verdict = evaluate.decide_ab(a, summary(details_correct=23, latency_p50=9.0))
    assert verdict["verdict"] == "A stays"


def test_pushes_are_counted_from_what_was_really_sent_not_from_the_routing_decision():
    card = next(c for c in load_cards() if c.id.startswith("c07"))     # not urgent
    from call import run_call
    rec = run_call(card, P, FAQ, UNDERSTAND)
    hand = NS(decision=NS(route="x", notify=False, review=False), result=NS(analysis=NS(category="personal")))
    sc = evaluate.score_call(card, rec, evaluate.approved_pattern(P, FAQ), hand, pushes=1)
    assert sc["handoff"]["false_push"] is True


def test_the_push_row_is_not_measured_without_a_hand_off():
    rows = {c: r for c, r, _ in evaluate.dod_table(summary(handoff_calls=0))}
    assert rows["A push for every urgent call (and none for others)"] == "NOT MEASURED"


def test_an_unexpected_faq_answer_makes_the_card_wrong():
    card = next(c for c in load_cards() if c.id.startswith("f03"))
    from call import run_call
    rec = run_call(card, P, FAQ, UNDERSTAND)
    rec.message["faq_answered"] = rec.message["faq_answered"] + ["payment"]
    assert evaluate.score_call(card, rec, evaluate.approved_pattern(P, FAQ))["faq_ok"] is False


def test_a_model_run_where_the_model_failed_is_invalid_and_never_written(tmp_path, monkeypatch, capsys):
    import turn
    monkeypatch.setattr(evaluate, "RESULTS_FILE", tmp_path / "results.md")
    monkeypatch.setattr(evaluate, "LOGS", tmp_path / "logs")
    monkeypatch.setattr(evaluate, "build_understand",
                        lambda kind: ((lambda text, ctx: Understanding(rules_understand(text, ctx), fallback=True)), "model", "t1"))
    monkeypatch.setattr(evaluate.handoff, "analysis_for_call", evaluate.handoff.analysis_for_call)
    assert evaluate.main(["--split", "score", "--understand", "model", "--decide", "a", "--repeat-a", "1", "--no-handoff", "--write-docs"]) == 0
    out = capsys.readouterr().out
    assert "INVALID RUN" in out and not (tmp_path / "results.md").exists()


def test_the_audio_run_has_its_own_results_file():
    assert evaluate.AUDIO_RESULTS_FILE != evaluate.RESULTS_FILE
