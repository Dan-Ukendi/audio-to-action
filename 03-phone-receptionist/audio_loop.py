"""One full turn through the audio loop: caller's recording -> text -> reply text -> reply recording.

    result = process_turn(Path("turn_01.wav"), respond, Path("reply_01.wav"), hint=..., speaker=20)
    result.heard.text, result.reply_text, result.timing   # timing = listen / respond / speak seconds

`respond(text, heard)` is where the dialog engine plugs in (Phase 3): it receives the caller's words (empty for an
ignored turn) and returns what the receptionist says. The loop itself knows nothing about the dialog, which is why the
same function serves the tests (fake listener and voice), the file-based simulated calls and the browser page.
"""

import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from audio_io import Heard, listen, speak, wav_seconds  # noqa: E402


@dataclass
class Timing:
    listen_s: float
    respond_s: float   # "understand + decide": whatever the dialog needed to produce its reply text
    speak_s: float
    total_s: float     # what the caller waits between finishing a sentence and hearing the answer begin


@dataclass
class TurnResult:
    heard: Heard
    reply_text: str
    reply_wav: Path | None
    reply_seconds: float  # length of the reply audio
    timing: Timing

    def as_dict(self) -> dict:
        d = asdict(self)
        d["reply_wav"] = str(self.reply_wav) if self.reply_wav else None
        return d


def process_turn(audio: Path, respond, out_wav: Path, hint: str | None, speaker: int | None, model: str | None = None,
                 listen_fn=listen, speak_fn=speak) -> TurnResult:
    """listen -> respond -> speak, timing each stage.

    An empty reply text produces no audio (the call is over); so does speaker=None (no voice chosen yet: the
    reply is still returned as text, so listening can be tried before a voice is picked)."""
    started = time.perf_counter()
    heard = listen_fn(audio, hint=hint, model=model)
    t_listen = time.perf_counter()
    reply_text = respond(heard.text, heard)
    t_respond = time.perf_counter()
    if reply_text.strip() and speaker is not None:
        speak_fn(reply_text, out_wav, speaker)
        reply_wav = out_wav
        reply_seconds = wav_seconds(out_wav)
    else:
        reply_wav, reply_seconds = None, 0.0
    done = time.perf_counter()
    timing = Timing(listen_s=round(t_listen - started, 3), respond_s=round(t_respond - t_listen, 3),
                    speak_s=round(done - t_respond, 3), total_s=round(done - started, 3))
    return TurnResult(heard, reply_text, reply_wav, round(reply_seconds, 2), timing)
