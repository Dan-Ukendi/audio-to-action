"""Listening decides whether anyone spoke; speaking times the voice. Fakes stand in for Whisper and Piper."""

import sys
import wave
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

from audio_io import MIN_SPEECH_S, listen, speak, wav_seconds  # noqa: E402
from shared.schemas import Segment, Transcript  # noqa: E402


def transcript(text: str, segments: list[tuple[float, float, float, float]]) -> Transcript:
    """segments = (start, end, avg_logprob, no_speech_prob)"""
    return Transcript(source_file="t.wav", audio_sha256="0" * 64, model="base", language="en", language_probability=1.0,
                      duration_s=5.0, transcribe_s=0.1, text=text, created_at=datetime.now(timezone.utc),
                      segments=[Segment(start=a, end=b, text=text, avg_logprob=lp, no_speech_prob=ns) for a, b, lp, ns in segments])


def fake(t: Transcript):
    seen = {}

    def transcribe_fn(path, cache_dir=None, model=None, hint=None):
        seen.update(path=path, cache_dir=cache_dir, model=model, hint=hint)
        return t

    transcribe_fn.seen = seen
    return transcribe_fn


def test_a_normal_turn_is_kept_and_nothing_is_cached():
    fn = fake(transcript("My name is Mark Thompson.", [(0.2, 3.0, -0.3, 0.01)]))
    heard = listen(Path("t.wav"), hint="Sam, Priya", model="base", transcribe_fn=fn)
    assert (heard.text, heard.ignored, heard.why) == ("My name is Mark Thompson.", False, None)
    assert heard.speech_s == 2.8 and heard.min_logprob == -0.3 and heard.seconds >= 0
    assert fn.seen["cache_dir"] is None and fn.seen["hint"] == "Sam, Priya" and fn.seen["model"] == "base"  # every turn is new audio


def test_no_segments_or_blank_text_is_silence():
    for t in (transcript("", []), transcript("   ", [(0, 1, -0.5, 0.1)])):
        heard = listen(Path("t.wav"), transcribe_fn=fake(t))
        assert heard.ignored and heard.text == "" and heard.why == "no speech found"


def test_a_blip_of_speech_is_ignored_even_if_whisper_wrote_words():
    heard = listen(Path("t.wav"), transcribe_fn=fake(transcript("Hello.", [(0.0, MIN_SPEECH_S - 0.1, -0.4, 0.05)])))
    assert heard.ignored and heard.text == "" and heard.raw_text == "Hello." and "under" in heard.why


def test_segments_whisper_calls_silence_are_ignored():
    heard = listen(Path("t.wav"), transcribe_fn=fake(transcript("Yes please.", [(0, 2, -0.8, 0.9), (2, 4, -0.8, 0.8)])))
    assert heard.ignored and "silence" in heard.why


def test_phrases_whisper_invents_on_silence_are_ignored_in_short_shaky_clips():
    for phrase in ("Thank you.", "Thanks for watching!", "you", "Bye."):
        heard = listen(Path("t.wav"), transcribe_fn=fake(transcript(phrase, [(0, 1.0, -0.95, 0.05)])))
        assert heard.ignored, phrase
        assert "invents" in heard.why


def test_a_confident_short_thank_you_is_a_real_answer():
    """After 'anything else?' a clear 'Thank you.' / 'Bye.' is one of the likeliest real replies."""
    for phrase in ("Thank you.", "Thanks.", "Bye."):
        heard = listen(Path("t.wav"), transcribe_fn=fake(transcript(phrase, [(0, 0.9, -0.2, 0.02)])))
        assert not heard.ignored and heard.text == phrase


def test_a_bare_yes_survives_the_speech_threshold():
    heard = listen(Path("t.wav"), transcribe_fn=fake(transcript("Yes.", [(0.1, 0.45, -0.3, 0.02)])))
    assert not heard.ignored


def test_the_same_phrase_is_kept_when_it_is_clearly_speech():
    """A caller who really says 'thank you' for three confident seconds is not silence."""
    heard = listen(Path("t.wav"), transcribe_fn=fake(transcript("Thank you.", [(0, 3.0, -0.2, 0.02)])))
    assert not heard.ignored and heard.text == "Thank you."


def test_the_same_phrase_is_ignored_if_whisper_doubts_there_was_speech():
    heard = listen(Path("t.wav"), transcribe_fn=fake(transcript("Thank you.", [(0, 3.0, -0.9, 0.45)])))
    assert heard.ignored


def test_real_sentences_are_never_mistaken_for_invented_phrases():
    heard = listen(Path("t.wav"), transcribe_fn=fake(transcript("Thank you, my number is oh one six three two.", [(0, 1.0, -0.5, 0.1)])))
    assert not heard.ignored


def fake_synth(text, speaker, out_wav):
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(b"\x00\x00" * 22050)  # one second


def test_speak_writes_the_file_and_reports_how_long_synthesis_took(tmp_path):
    out = tmp_path / "sub" / "reply.wav"
    seconds = speak("Hello", out, speaker=20, synth_fn=fake_synth)
    assert out.exists() and wav_seconds(out) == 1.0 and 0 <= seconds < 5


def test_a_bare_okay_is_a_real_answer_to_a_read_back():
    heard = listen(Path("t.wav"), transcribe_fn=fake(transcript("Okay.", [(0, 0.8, -0.4, 0.05)])))
    assert not heard.ignored and heard.text == "Okay."
