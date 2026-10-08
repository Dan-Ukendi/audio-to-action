"""Pick Holly's voice (laptop; needs the Piper voice and a pair of ears).

    python 03-phone-receptionist/choose_voice.py [--count 12] [--ids 20 21 30]

Renders the greeting in a dozen candidate speakers of en_GB-vctk-medium into testset/voice_samples/speaker_NNN.wav (git-ignored).
Listen, then put the number you like into persona.json ("piper_speaker"). Candidates never include a voice already used
by Part 1 callers, Part 2 meetings or the Part 3 FAQ callers (the persona loader would refuse those anyway).
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from persona import MAX_SPEAKER, Persona, check_speaker, load_persona  # noqa: E402

SAMPLES = HERE / "testset" / "voice_samples"


def candidate_ids(persona: Persona, count: int = 12) -> list[int]:
    """`count` free speaker ids spread across the voice's whole range (neighbouring ids often sound alike)."""
    free = [i for i in range(MAX_SPEAKER + 1) if i not in persona.taken_voice_ids]
    if count < 1:
        raise ValueError("count must be at least 1")
    if count >= len(free):
        return free
    step = len(free) / count
    return [free[int(i * step)] for i in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--ids", type=int, nargs="*", help="render exactly these speaker ids")
    args = parser.parse_args()

    from audio_io import piper_synth, speak
    persona = load_persona()
    ids = args.ids or candidate_ids(persona, args.count)
    for speaker in ids:
        check_speaker(speaker)  # 0-108 only
    clash = [i for i in ids if i in persona.taken_voice_ids]
    if clash:
        sys.exit(f"speakers {clash} are used by Part 1/2/3 callers; the receptionist needs her own voice")
    for speaker in ids:
        out = SAMPLES / f"speaker_{speaker:03d}.wav"
        speak(persona.say("greeting"), out, speaker, synth_fn=piper_synth)
        print(f"wrote {out}")
    print('\nListen to them, then set "piper_speaker" in 03-phone-receptionist/persona.json to the number you like.')
    return 0


if __name__ == "__main__":
    sys.exit(main())
