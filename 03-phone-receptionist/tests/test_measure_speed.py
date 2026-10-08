"""The speed script's arithmetic and its rule. All numbers below are SYNTHETIC test inputs, not measurements."""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

import choose_voice  # noqa: E402
import measure_speed as ms  # noqa: E402
from audio_io import Heard  # noqa: E402
from persona import load_persona  # noqa: E402
from shared.evaluation import median, percentile  # noqa: E402


def test_percentile_and_median():
    assert median([3, 1, 2]) == 2 and median([1, 2, 3, 4]) == 2.5 and median([]) is None
    assert percentile([1, 2, 3, 4, 5], 0) == 1 and percentile([1, 2, 3, 4, 5], 100) == 5
    assert percentile([0, 10], 95) == 9.5
    assert percentile([7], 95) == 7


def test_whisper_measurement_keeps_the_warm_up_apart_and_counts_name_hits():
    calls = []
    seconds = iter([9.0, 1.0, 2.0, 3.0])  # the first call is the model load

    def listen_fn(path, model=None):
        calls.append((path, model))
        return Heard(text="x", ignored=False, why=None, raw_text=f"it's {path}", duration_s=2, speech_s=2, min_logprob=-0.2,
                     seconds=next(seconds))

    clips = [{"path": "warm", "name_tokens": None}, {"path": "mark thompson", "name_tokens": ["mark", "thompson"]},
             {"path": "somebody", "name_tokens": ["siobhan", "gallagher"]}, {"path": "no name here", "name_tokens": None}]
    out = ms.measure_whisper(clips, "base", listen_fn)
    assert out["load_and_first_s"] == 9.0 and out["median_s"] == 2.0 and out["n"] == 3
    assert (out["name_hits"], out["name_clips"]) == (1, 2)
    assert all(model == "base" for _, model in calls)


def test_llm_and_piper_measurements():
    times = iter([30.0, 2.0, 4.0, 3.0])
    llm = ms.measure_llm("m", 3, lambda model: next(times))
    assert llm == {"load_and_first_s": 30.0, "n": 3, "median_s": 3.0, "p95_s": 3.9}
    spoken = []
    piper = ms.measure_piper({"a": "x", "b": "y"}, 20, 2, lambda text, spk: spoken.append(text) or 0.5)
    assert piper["median_s"] == 0.5 and set(piper["per_line"]) == {"a", "b"} and spoken[0] == "x"  # warm-up first


def results(**whisper_hits):
    """Synthetic merged results. whisper_hits: 'base/cpu'=(median_s, name_hits)."""
    out = {"whisper": {}, "llm": {}, "piper": {"median_s": 0.5, "per_line": {}}}
    for key, (median_s, hits) in whisper_hits.items():
        out["whisper"][key] = {"load_and_first_s": 5, "n": 6, "median_s": median_s, "p95_s": median_s + 1, "name_hits": hits, "name_clips": 7}
    return out


def test_rule_prefers_base_when_it_is_within_one_name_hit_of_small():
    r = results(**{"base/cuda": (0.4, 5), "small/cuda": (0.9, 6)})
    r["llm"]["qwen2.5:7b"] = {"median_s": 2.0}
    rec = ms.recommend(r)
    assert rec["whisper"] == "base/cuda" and rec["llm"] == "qwen2.5:7b"
    assert rec["estimated_turn_s"] == 2.9 and rec["meets_target"] is True


def test_rule_takes_small_when_base_misses_two_or_more_names():
    r = results(**{"base/cpu": (0.8, 3), "small/cpu": (2.5, 6)})
    r["llm"]["qwen2.5:7b"] = {"median_s": 2.5}
    assert ms.recommend(r)["whisper"] == "small/cpu"


def test_slow_7b_falls_back_to_3b_and_target_can_be_missed():
    r = results(**{"small/cpu": (4.0, 6)})
    r["llm"] = {"qwen2.5:7b": {"median_s": 40.0}, "qwen2.5:3b": {"median_s": 12.0}}
    rec = ms.recommend(r)
    assert rec["llm"] == "qwen2.5:3b" and rec["meets_target"] is False and rec["estimated_turn_s"] == 16.5


def test_slow_7b_without_a_measured_3b_says_so_instead_of_inventing_one():
    r = results(**{"small/cpu": (4.0, 6)})
    r["llm"] = {"qwen2.5:7b": {"median_s": 40.0}}
    rec = ms.recommend(r)
    assert rec["llm"] == "qwen2.5:7b" and any("3b was not measured" in n for n in rec["notes"])


def test_nothing_measured_gives_no_decision_and_the_placeholder_report():
    rec = ms.recommend({})
    assert rec["whisper"] is None and rec["llm"] is None and rec["estimated_turn_s"] is None
    assert ms.render_markdown({}, rec) == ms.PLACEHOLDER


def test_report_shows_every_measured_row_and_the_decision():
    r = results(**{"base/cuda": (0.4, 5), "small/cuda": (0.9, 6)})
    r["llm"]["qwen2.5:7b"] = {"load_and_first_s": 20, "n": 5, "median_s": 2.0, "p95_s": 2.5, "processor": "100% GPU"}
    r["machine"], r["date"], r["gpu"] = "test-machine", "2000-01-01", "nvidia-smi: test"
    text = ms.render_markdown(r, ms.recommend(r))
    for needle in ("base/cuda", "small/cuda", "qwen2.5:7b", "100% GPU", "target met", "nvidia-smi: test", "test-machine"):
        assert needle in text


def test_the_committed_report_is_the_honest_placeholder_until_the_laptop_has_measured():
    committed = (HERE.parents[1] / "docs" / "part3-speed.md").read_text(encoding="utf-8")
    if (HERE.parents[1] / "docs" / "part3-speed-results.json").exists():
        assert "NOT MEASURED YET" not in committed  # real results were committed: the report must have been regenerated
    else:
        assert committed == ms.PLACEHOLDER and "NOT MEASURED YET" in committed


def test_sample_texts_come_from_dev_cards_only():
    from cards import load_cards
    cards = load_cards()
    texts = ms.sample_texts(cards)
    dev_openings = {c.script.opening.split("{")[0][:30] for c in cards if c.split == "dev"}
    assert texts and all(any(t.startswith(o) for o in dev_openings) for t in texts)
    assert not any("{" in t for t in texts)


def test_voice_candidates_are_free_spread_out_and_enough():
    p = load_persona()
    ids = choose_voice.candidate_ids(p, 12)
    assert len(ids) == 12 and len(set(ids)) == 12 and not set(ids) & p.taken_voice_ids
    assert ids == sorted(ids) and ids[-1] - ids[0] > 60  # spread over the range, not twelve neighbours
    assert choose_voice.candidate_ids(p, 1000) == [i for i in range(109) if i not in p.taken_voice_ids]


def test_probe_form_is_valid_json_schema_for_ollama():
    schema = ms.ProbeTurn.model_json_schema()
    assert set(schema["properties"]) == {"heard_summary", "name", "number", "reason", "emergency"}
    json.dumps(schema)


# ---------------------------------------------------------------- the whole run, with fakes

def fake_run(tmp_path, llm_call, speaker=20, whisper=("base", "small"), llm=("qwen2.5:7b", "qwen2.5:3b"), skip_piper=False):
    persona = load_persona()
    args = argparse.Namespace(device="cpu", whisper=list(whisper), llm=list(llm), repeats=2, skip_piper=skip_piper)
    clips = [{"path": "warm", "name_tokens": None}, {"path": "a", "name_tokens": ["mark"]}, {"path": "b", "name_tokens": None}]

    def listen_fn(path, model):
        return Heard(text="x", ignored=False, why=None, raw_text="mark", duration_s=1, speech_s=1, min_logprob=-0.2,
                     seconds=1.0 if model == "base" else 2.0)

    return ms.run_measurements(args, persona, clips, ["hello", "there"], listen_fn=listen_fn, llm_call=llm_call,
                               speak_fn=lambda text, spk: 0.4, speaker=speaker, processor_fn=lambda: "100% GPU",
                               gpu_text="nvidia-smi: test", results_path=tmp_path / "r.json", report_path=tmp_path / "r.md")


def test_a_full_run_writes_results_and_a_report_with_a_decision(tmp_path):
    out = fake_run(tmp_path, lambda model, text: 2.0)
    saved = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert set(saved["whisper"]) == {"base/cpu", "small/cpu"} and set(saved["llm"]) == {"qwen2.5:7b", "qwen2.5:3b"}
    assert saved["piper"]["median_s"] == 0.4 and saved["llm"]["qwen2.5:7b"]["processor"] == "100% GPU"
    assert out["recommendation"]["whisper"] == "base/cpu" and out["recommendation"]["llm"] == "qwen2.5:7b"
    assert out["recommendation"]["estimated_turn_s"] == 3.4  # 1.0 + 2.0 + 0.4
    report = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "base/cpu" in report and "target met" in report
    import platform
    assert saved["machine"] == platform.platform() and platform.node() not in saved["machine"].split()  # no host name in a committed file


def test_a_model_that_is_not_pulled_is_recorded_and_everything_else_is_kept(tmp_path):
    def llm_call(model, text):
        if model == "qwen2.5:3b":
            raise RuntimeError("model 'qwen2.5:3b' not found, try pulling it first")
        return 2.0

    out = fake_run(tmp_path, llm_call)
    saved = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert "qwen2.5:7b" in saved["llm"] and "qwen2.5:3b" not in saved["llm"]
    assert "not found" in saved["llm_errors"]["qwen2.5:3b"]
    assert "could not be measured" in (tmp_path / "r.md").read_text(encoding="utf-8")
    assert out["recommendation"]["llm"] == "qwen2.5:7b"


def test_a_crash_late_in_the_run_keeps_what_was_already_measured(tmp_path):
    def exploding_speak(text, spk):
        raise RuntimeError("Piper voice missing")

    persona = load_persona()
    args = argparse.Namespace(device="cuda", whisper=["base"], llm=["qwen2.5:7b"], repeats=2, skip_piper=False)
    clips = [{"path": "warm", "name_tokens": None}, {"path": "a", "name_tokens": None}]
    heard = Heard(text="x", ignored=False, why=None, raw_text="x", duration_s=1, speech_s=1, min_logprob=-0.2, seconds=0.5)
    import pytest
    with pytest.raises(RuntimeError, match="Piper voice missing"):
        ms.run_measurements(args, persona, clips, ["hi"], listen_fn=lambda p, model: heard, llm_call=lambda m, t: 1.5,
                            speak_fn=exploding_speak, speaker=20, processor_fn=lambda: "?", gpu_text="",
                            results_path=tmp_path / "r.json", report_path=tmp_path / "r.md")
    saved = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert "base/cuda" in saved["whisper"] and "qwen2.5:7b" in saved["llm"]  # saved before Piper failed


def test_runs_on_different_devices_are_merged(tmp_path):
    fake_run(tmp_path, lambda model, text: 2.0, whisper=("base",), llm=("qwen2.5:7b",), skip_piper=True)
    persona = load_persona()
    args = argparse.Namespace(device="cuda", whisper=["base"], llm=[], repeats=2, skip_piper=True)
    heard = Heard(text="x", ignored=False, why=None, raw_text="x", duration_s=1, speech_s=1, min_logprob=-0.2, seconds=0.3)
    ms.run_measurements(args, persona, [{"path": "w", "name_tokens": None}, {"path": "a", "name_tokens": None}], ["hi"],
                        listen_fn=lambda p, model: heard, llm_call=lambda m, t: 1.0, speak_fn=lambda t, s: 0.1, speaker=20,
                        processor_fn=lambda: "?", gpu_text="", results_path=tmp_path / "r.json", report_path=tmp_path / "r.md")
    saved = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert {"base/cpu", "base/cuda"} <= set(saved["whisper"]) and "qwen2.5:7b" in saved["llm"]


def test_percentile_outside_0_100_is_an_error_not_a_wrong_number():
    import pytest
    with pytest.raises(ValueError):
        percentile([1, 2, 3], 101)
    with pytest.raises(ValueError):
        percentile([1, 2, 3], -1)


def test_voice_chooser_edge_cases():
    import pytest
    p = load_persona()
    with pytest.raises(ValueError):
        choose_voice.candidate_ids(p, 0)
    assert choose_voice.candidate_ids(p, 1) == [1]


def test_no_small_measurement_is_noted():
    r = results(**{"base/cpu": (0.8, 5)})
    r["llm"]["qwen2.5:7b"] = {"median_s": 2.0}
    rec = ms.recommend(r)
    assert rec["whisper"] == "base/cpu" and any("small was not measured" in n for n in rec["notes"])


def test_a_whisper_model_that_cannot_load_is_recorded_and_the_rest_continues(tmp_path):
    persona = load_persona()
    args = argparse.Namespace(device="cuda", whisper=["base", "small"], llm=["qwen2.5:7b"], repeats=2, skip_piper=True)
    ok = Heard(text="x", ignored=False, why=None, raw_text="x", duration_s=1, speech_s=1, min_logprob=-0.2, seconds=0.5)

    def listen_fn(path, model):
        if model == "base":
            raise RuntimeError("CUDA failed to initialize")
        return ok

    clips = [{"path": "w", "name_tokens": None}, {"path": "a", "name_tokens": None}]
    out = ms.run_measurements(args, persona, clips, ["hi"], listen_fn=listen_fn, llm_call=lambda m, t: 1.0,
                              speak_fn=lambda t, s: 0.1, speaker=20, processor_fn=lambda: "?", gpu_text="",
                              results_path=tmp_path / "r.json", report_path=tmp_path / "r.md")
    saved = out["results"]
    assert "small/cuda" in saved["whisper"] and "base/cuda" not in saved["whisper"]
    assert "CUDA failed" in saved["whisper_errors"]["base/cuda"]
    assert "Whisper runs that could not be measured" in (tmp_path / "r.md").read_text(encoding="utf-8")


def test_an_old_row_is_dropped_when_the_same_model_fails_in_a_later_run(tmp_path):
    fake_run(tmp_path, lambda model, text: 2.0, whisper=("base",), llm=("qwen2.5:7b",), skip_piper=True)

    def failing(model, text):
        raise RuntimeError("out of memory")

    fake_run(tmp_path, failing, whisper=("base",), llm=("qwen2.5:7b",), skip_piper=True)
    saved = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert "qwen2.5:7b" not in saved["llm"] and "out of memory" in saved["llm_errors"]["qwen2.5:7b"]
