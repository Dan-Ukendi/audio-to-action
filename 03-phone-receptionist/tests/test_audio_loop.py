"""process_turn times listen / respond / speak and never speaks an empty reply."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

from audio_io import Heard  # noqa: E402
from audio_loop import process_turn  # noqa: E402
from test_audio_io import fake_synth  # noqa: E402


def heard(text="hello", ignored=False):
    return Heard(text="" if ignored else text, ignored=ignored, why="silence" if ignored else None, raw_text=text,
                 duration_s=2.0, speech_s=1.5, min_logprob=-0.3, seconds=0.2)


def speak_fn(text, out_wav, speaker):
    fake_synth(text, speaker, out_wav)


def test_stages_run_in_order_and_are_timed(tmp_path):
    order = []

    def listen_fn(audio, hint=None, model=None):
        order.append(("listen", hint, model))
        return heard("my name is Sam")

    def respond(text, h):
        order.append(("respond", text))
        return "Thank you, Sam."

    def spk(text, out_wav, speaker):
        order.append(("speak", text, speaker))
        fake_synth(text, speaker, out_wav)

    result = process_turn(tmp_path / "in.wav", respond, tmp_path / "out.wav", hint="Sam", speaker=20, model="base",
                          listen_fn=listen_fn, speak_fn=spk)
    assert order == [("listen", "Sam", "base"), ("respond", "my name is Sam"), ("speak", "Thank you, Sam.", 20)]
    assert result.reply_text == "Thank you, Sam." and result.reply_wav == tmp_path / "out.wav" and result.reply_seconds == 1.0
    t = result.timing
    assert min(t.listen_s, t.respond_s, t.speak_s) >= 0 and abs(t.total_s - (t.listen_s + t.respond_s + t.speak_s)) < 0.01


def test_an_ignored_turn_reaches_the_dialog_as_empty_text(tmp_path):
    got = []
    process_turn(tmp_path / "in.wav", lambda text, h: got.append((text, h.ignored)) or "Sorry?", tmp_path / "o.wav", None, 20,
                 listen_fn=lambda *a, **k: heard("Thank you.", ignored=True), speak_fn=speak_fn)
    assert got == [("", True)]


def test_empty_reply_means_the_call_is_over_and_no_audio(tmp_path):
    result = process_turn(tmp_path / "in.wav", lambda t, h: "", tmp_path / "o.wav", None, 20,
                          listen_fn=lambda *a, **k: heard(), speak_fn=speak_fn)
    assert result.reply_wav is None and result.reply_seconds == 0.0 and not (tmp_path / "o.wav").exists()


def test_no_voice_chosen_yet_still_returns_the_reply_as_text(tmp_path):
    result = process_turn(tmp_path / "in.wav", lambda t, h: "Hello", tmp_path / "o.wav", None, None,
                          listen_fn=lambda *a, **k: heard(), speak_fn=speak_fn)
    assert result.reply_text == "Hello" and result.reply_wav is None


def test_the_result_is_json_serialisable_for_the_call_record(tmp_path):
    result = process_turn(tmp_path / "in.wav", lambda t, h: "Hi", tmp_path / "o.wav", None, 20,
                          listen_fn=lambda *a, **k: heard(), speak_fn=speak_fn)
    data = json.loads(json.dumps(result.as_dict()))
    assert data["reply_text"] == "Hi" and data["timing"]["total_s"] >= 0 and data["heard"]["text"] == "hello"


def test_errors_in_a_stage_are_not_swallowed(tmp_path):
    def broken(audio, hint=None, model=None):
        raise RuntimeError("ffmpeg could not convert in.wav")

    try:
        process_turn(tmp_path / "in.wav", lambda t, h: "x", tmp_path / "o.wav", None, 20, listen_fn=broken, speak_fn=speak_fn)
    except RuntimeError as error:
        assert "ffmpeg" in str(error)
    else:
        raise AssertionError("the error disappeared")
