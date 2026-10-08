"""One turn of audio in, one turn of audio out: LISTEN (Whisper) and SPEAK (Piper).

    heard = listen(Path("turn.wav"), hint=persona.hint)        # Heard(text, ignored, ...)
    seconds = speak("Could I take your name, please?", Path("reply.wav"), speaker=20)

listen() also decides whether a turn contains speech at all. Whisper is trained to always produce text, so on a
second of silence or breath noise it invents "Thank you." or "Thanks for watching."; acting on that would make the
receptionist answer a caller who said nothing. A turn is IGNORED (text = "") when it holds under MIN_SPEECH_S of
speech, when Whisper itself thinks every segment is silence, or when it is just one of the known invented phrases and
Whisper doubts it (under 1.5 s of speech with low confidence, or any clip with a high no-speech probability; a confident, clearly spoken "Thank you." at the end of a call is a real answer). The dialog treats an ignored turn as silence and never acts on its text.

Everything that needs a model is passed in (`transcribe_fn`, `synth_fn`), so the tests run with fakes and the
laptop uses the real Whisper and Piper by default.
"""

import re
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

MIN_SPEECH_S = 0.3       # less speech than this in a turn = nobody spoke (the plan says ~0.5; 0.3 keeps a bare "Yes.")
SILENCE_PROB = 0.6       # Whisper's own "this chunk is silence" probability
SHORT_CLIP_S = 1.5       # an invented phrase is only suspected in clips this short AND not confidently recognised ...
SHAKY_LOGPROB = -0.8     # ... (confidence below this), ...
SUSPECT_PROB = 0.3       # ... or in any clip where Whisper itself doubts there was speech

# What Whisper typically writes on silence or noise (lower case, punctuation removed).
INVENTED_PHRASES = {"thank you", "thanks", "thank you very much", "thanks for watching", "thank you for watching",
                    "please subscribe", "you", "bye"}
# ("okay" and "so" are NOT here: a caller may well answer a read-back with just "okay".)


@dataclass
class Heard:
    """What the listening step made of one turn."""
    text: str                 # "" when the turn was ignored
    ignored: bool
    why: str | None           # why it was ignored, or None
    raw_text: str             # what Whisper wrote (kept for the call record, never acted on when ignored)
    duration_s: float         # length of the recording
    speech_s: float           # seconds Whisper found speech in
    min_logprob: float | None  # least confident segment (closer to 0 = more confident)
    seconds: float            # how long listening took (the number the speed budget needs)


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z' ]", " ", text.lower()).split())


def listen(wav: Path, hint: str | None = None, model: str | None = None, transcribe_fn=None) -> Heard:
    """Transcribe one caller turn (no cache: every turn is new audio) and decide whether anyone spoke."""
    if transcribe_fn is None:
        from shared.transcribe import transcribe as transcribe_fn  # lazy: pulls in Whisper
    started = time.perf_counter()
    t = transcribe_fn(wav, cache_dir=None, model=model, hint=hint)
    seconds = time.perf_counter() - started

    speech_s = sum(max(0.0, s.end - s.start) for s in t.segments)
    min_logprob = min((s.avg_logprob for s in t.segments), default=None)
    why = None
    if not t.segments or not t.text.strip():
        why = "no speech found"
    elif speech_s < MIN_SPEECH_S:
        why = f"under {MIN_SPEECH_S} s of speech"
    elif all(s.no_speech_prob > SILENCE_PROB for s in t.segments):
        why = "Whisper thinks it is silence"
    elif normalize(t.text) in INVENTED_PHRASES and (
            (speech_s < SHORT_CLIP_S and min_logprob < SHAKY_LOGPROB)
            or max(s.no_speech_prob for s in t.segments) > SUSPECT_PROB):
        why = "probably a phrase Whisper invents on silence"
    return Heard(text="" if why else t.text.strip(), ignored=why is not None, why=why, raw_text=t.text.strip(),
                 duration_s=t.duration_s, speech_s=round(speech_s, 2), min_logprob=min_logprob,
                 seconds=round(seconds, 3))


def piper_synth(text: str, speaker: int, out_wav: Path) -> None:
    from shared.tts import load_voice, synthesize  # lazy: needs the Piper voice
    synthesize(load_voice(), text, speaker, out_wav)


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def speak(text: str, out_wav: Path, speaker: int, synth_fn=piper_synth) -> float:
    """Say `text` in the receptionist's voice into out_wav. Returns how long synthesis took in seconds."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    synth_fn(text, speaker, out_wav)
    return round(time.perf_counter() - started, 3)
