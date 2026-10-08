"""The call as the page sees it: turns, files, timings. No browser, no models."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

from audio_io import Heard  # noqa: E402
from persona import load_persona  # noqa: E402
from session import AudioSession, dialog_responder, scripted_responder  # noqa: E402
from test_audio_io import fake_synth  # noqa: E402


def make_heard(text="", ignored=False):
    return Heard(text="" if ignored else text, ignored=ignored, why="silence" if ignored else None, raw_text=text,
                 duration_s=2.0, speech_s=1.5, min_logprob=-0.3, seconds=0.2)


def session(tmp_path, speaker=20, texts=None):
    queue = list(texts or ["hello there", "my name is Sam"])
    return AudioSession(scripted_responder(load_persona()), tmp_path / "call", hint="Sam", speaker=speaker,
                        listen_fn=lambda audio, hint=None, model=None: queue.pop(0) if isinstance(queue[0], Heard) else make_heard(queue.pop(0)),
                        speak_fn=lambda text, out, spk: fake_synth(text, spk, out))


def test_the_receptionist_greets_first_and_the_greeting_is_turn_zero(tmp_path):
    s = session(tmp_path)
    turn = s.greet("Hello, this is Holly.")
    assert turn.n == 0 and turn.caller_text == "" and turn.timing is None
    assert turn.reply_wav == tmp_path / "call" / "reply_00.wav" and turn.reply_wav.exists()


def test_hearing_a_recording_saves_it_and_answers(tmp_path):
    s = session(tmp_path)
    s.greet("Hello.")
    turn = s.hear(b"RIFF-not-really-audio")
    assert turn.n == 1 and turn.caller_text == "hello there" and turn.reply_wav.exists()
    assert (tmp_path / "call" / "turn_01.wav").read_bytes() == b"RIFF-not-really-audio"
    assert turn.timing.total_s >= 0 and len(s.turns) == 2
    assert s.median_total_s() == turn.timing.total_s


def test_without_a_voice_the_session_works_in_text(tmp_path):
    s = session(tmp_path, speaker=None)
    greeting = s.greet("Hello.")
    turn = s.hear(b"x")
    assert greeting.reply_wav is None and turn.reply_wav is None and turn.reply_text


def test_median_ignores_the_greeting_and_is_none_before_any_turn(tmp_path):
    s = session(tmp_path)
    s.greet("Hello.")
    assert s.median_total_s() is None


def test_scripted_responder_cycles_persona_lines_and_asks_again_after_silence():
    p = load_persona()
    respond = scripted_responder(p)
    say = lambda text: respond(text, make_heard(text))  # noqa: E731
    assert say("a") == p.say("ask_reason") and say("b") == p.say("ask_name")
    assert respond("", make_heard("Thank you.", ignored=True)) == p.say("repeat_request")  # silence does not advance the script
    assert say("c") == p.say("ask_number") and say("d") == p.say("anything_else") and say("e") == p.say("ask_reason")


def test_an_ignored_turn_is_recorded_with_its_reason(tmp_path):
    s = session(tmp_path, texts=[make_heard("Thank you.", ignored=True)])
    s.greet("Hello.")
    turn = s.hear(b"x")
    assert turn.ignored and turn.ignored_why == "silence" and turn.caller_text == "" and turn.raw_text == "Thank you."


def test_the_real_dialog_answers_through_the_audio_session_and_ignored_turns_are_silence(tmp_path):
    from faq import load_faq
    from rules_turn import rules_understand
    from turn import Understanding
    persona = load_persona()
    respond = dialog_responder(persona, load_faq(), lambda text, ctx: Understanding(rules_understand(text, ctx)))
    queue = [make_heard("My tap is dripping. I'm Dave. My number is 01632 960 501."), make_heard("", ignored=True)]
    s = AudioSession(respond, tmp_path / "call", hint=None, speaker=None, listen_fn=lambda a, hint=None, model=None: queue.pop(0))
    s.greet(respond.greeting)
    first = s.hear(b"x")
    assert first.reply_text.startswith("Let me read that back") or "dripping" in first.reply_text
    assert respond.state.value("name") == "Dave" and respond.state.value("number") == "01632960501"
    second = s.hear(b"x")
    assert second.reply_text == persona.say("repeat_request") and respond.state.silent_streak == 1
