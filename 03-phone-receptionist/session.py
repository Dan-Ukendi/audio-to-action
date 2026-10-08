"""A call as the browser page sees it: a list of turns, with the audio files and timings behind each.

    session = AudioSession(respond, workdir, hint=persona.hint, speaker=None)
    session.greet(persona.say("greeting"))       # the receptionist speaks first
    turn = session.hear(wav_bytes)               # one push-to-talk recording from the caller

All logic lives here, none in the Streamlit page (app.py), so the page stays a thin shell and this file is tested
without a browser. `respond(text, heard)` is where the dialog engine plugs in: dialog_responder() is the real
receptionist; scripted_responder() (fixed persona lines whatever the caller said) is kept for measuring the audio loop alone.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import dialog  # noqa: E402
from audio_io import Heard, listen, speak, wav_seconds  # noqa: E402
from audio_loop import Timing, process_turn  # noqa: E402
from persona import Persona  # noqa: E402
from shared.evaluation import median  # noqa: E402


@dataclass
class Turn:
    n: int                      # 0 = the greeting
    caller_text: str            # what Whisper heard ("" for the greeting or an ignored turn)
    ignored: bool
    ignored_why: str | None
    reply_text: str
    reply_wav: Path | None
    timing: Timing | None       # None for the greeting (nobody was waiting for it)
    raw_text: str = ""


@dataclass
class AudioSession:
    respond: object             # (text, heard) -> reply text
    workdir: Path
    hint: str | None
    speaker: int | None         # None = no voice chosen yet: replies are text only
    model: str | None = None
    listen_fn: object = listen
    speak_fn: object = speak
    turns: list[Turn] = field(default_factory=list)

    def greet(self, text: str) -> Turn:
        """The receptionist speaks first."""
        self.workdir.mkdir(parents=True, exist_ok=True)
        wav = None
        if self.speaker is not None:
            wav = self.workdir / "reply_00.wav"
            self.speak_fn(text, wav, self.speaker)
        turn = Turn(0, "", False, None, text, wav, None)
        self.turns.append(turn)
        return turn

    def hear(self, audio_bytes: bytes) -> Turn:
        """One recording from the caller -> the receptionist's reply."""
        self.workdir.mkdir(parents=True, exist_ok=True)
        n = len(self.turns)
        clip = self.workdir / f"turn_{n:02d}.wav"
        clip.write_bytes(audio_bytes)
        result = process_turn(clip, self.respond, self.workdir / f"reply_{n:02d}.wav", self.hint, self.speaker,
                              model=self.model, listen_fn=self.listen_fn, speak_fn=self.speak_fn)
        turn = Turn(n, result.heard.text, result.heard.ignored, result.heard.why, result.reply_text,
                    result.reply_wav, result.timing, raw_text=result.heard.raw_text)
        self.turns.append(turn)
        return turn

    def median_total_s(self) -> float | None:
        """Median seconds the caller waited for an answer (the number the 5 s target is about)."""
        return median([t.timing.total_s for t in self.turns if t.timing])


def scripted_responder(persona: Persona):
    """Phase 2 stand-in for the dialog: cycles through fixed persona lines, ignoring what was said."""
    lines = [persona.say("ask_reason"), persona.say("ask_name"), persona.say("ask_number"), persona.say("anything_else")]
    state = {"i": 0}

    def respond(text: str, heard: Heard) -> str:
        if heard.ignored:
            return persona.say("repeat_request")
        reply = lines[state["i"] % len(lines)]
        state["i"] += 1
        return reply

    return respond


def dialog_responder(persona: Persona, faq, understand_fn, decide_fn=dialog.decide_a):
    """The real receptionist behind the page. `respond.state` is the call state (for the page's side panel)."""
    greeting, state = dialog.start_call(persona)

    def respond(text: str, heard: Heard) -> str:
        # A turn Whisper ignored (silence, noise) reaches the dialog as silence: it asks again, and ends after two in a row.
        reply, _ = dialog.next_reply(state, "" if heard.ignored else text, persona, faq, understand_fn, decide_fn, heard=heard)
        return reply

    respond.state, respond.greeting = state, greeting
    return respond
