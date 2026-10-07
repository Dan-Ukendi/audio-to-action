"""Local text-to-speech with Piper: text -> wav. Used to build synthetic test sets (Parts 1-2)
and, later, for the receptionist's voice (Part 3).

    voice = load_voice()
    synthesize(voice, "Hello", speaker=7, out_wav=Path("hello.wav"))
"""

import subprocess
import sys
import wave
from functools import lru_cache
from pathlib import Path

from piper import PiperVoice, SynthesisConfig

ROOT = Path(__file__).resolve().parent.parent
VOICE_NAME = "en_GB-vctk-medium"  # one model file, 109 British speakers (VCTK corpus)
# Part 1 already downloaded the voice into its test set; reuse it instead of a second 77 MB copy.
VOICE_DIRS = [ROOT / "01-voicemail-triage" / "testset" / "voices", ROOT / "models" / "piper"]


def voice_path() -> Path:
    """Find the voice model, downloading it (~77 MB) into models/piper/ only if it's nowhere yet."""
    for folder in VOICE_DIRS:
        model = folder / f"{VOICE_NAME}.onnx"
        if model.exists():
            return model
    target = VOICE_DIRS[-1]
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Piper voice {VOICE_NAME} (~77 MB) to {target}...")
    subprocess.run([sys.executable, "-m", "piper.download_voices", VOICE_NAME, "--download-dir", str(target)],
                   check=True)
    return target / f"{VOICE_NAME}.onnx"


@lru_cache(maxsize=1)
def load_voice() -> PiperVoice:
    return PiperVoice.load(str(voice_path()))


def synthesize(voice: PiperVoice, text: str, speaker: int, out_wav: Path,
               speed: float = 1.0, volume: float = 0.7) -> None:
    """Text -> 22.05 kHz mono wav. speed = Piper length_scale (>1 slower).

    volume < 1 leaves headroom so mixing in noise later doesn't clip.
    """
    config = SynthesisConfig(speaker_id=speaker, length_scale=speed, volume=volume)
    with wave.open(str(out_wav), "wb") as wav_file:
        voice.synthesize_wav(text, wav_file, syn_config=config)
