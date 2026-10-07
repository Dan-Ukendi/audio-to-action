"""Regenerate the synthetic test voicemails from scripts.json.

    python 01-voicemail-triage/testset/generate.py            # all files
    python 01-voicemail-triage/testset/generate.py --only 05  # ids starting with "05"

Pipeline per voicemail:
    text --Piper--> clean.wav --ffmpeg (noise, phone band, encode)--> audio/<id>.<format>

The audio is git-ignored; this script + scripts.json are what make the set reproducible.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

from piper import PiperVoice, SynthesisConfig

HERE = Path(__file__).resolve().parent
SCRIPTS_FILE = HERE / "scripts.json"
AUDIO_DIR = HERE / "audio"
VOICES_DIR = HERE / "voices"
VOICE_NAME = "en_GB-vctk-medium"  # one model file, ~100 British speakers

# Explicit codecs so the output doesn't depend on ffmpeg's defaults.
CODECS = {
    "wav": ["-c:a", "pcm_s16le"],
    "mp3": ["-c:a", "libmp3lame", "-b:a", "32k"],
    "m4a": ["-c:a", "aac", "-b:a", "32k"],
    "ogg": ["-c:a", "libopus", "-b:a", "24k"],  # what WhatsApp voice notes use
}


def ensure_voice() -> Path:
    """Download the Piper voice once (~77 MB) if it isn't there yet."""
    model = VOICES_DIR / f"{VOICE_NAME}.onnx"
    if not model.exists():
        print(f"Downloading Piper voice {VOICE_NAME} (~77 MB)...")
        VOICES_DIR.mkdir(exist_ok=True)
        subprocess.run(
            [sys.executable, "-m", "piper.download_voices", VOICE_NAME,
             "--download-dir", str(VOICES_DIR)],
            check=True,
        )
    return model


def synthesize(voice: PiperVoice, text: str, speaker: int, speed: float, out_wav: Path) -> None:
    """Text -> clean studio-quality speech (Piper's native 22.05 kHz)."""
    # volume < 1 leaves headroom: Piper normalizes to full scale, and adding noise
    # on top would otherwise clip (distort) the loudest parts.
    config = SynthesisConfig(speaker_id=speaker, length_scale=speed, volume=0.7)
    with wave.open(str(out_wav), "wb") as wav_file:
        voice.synthesize_wav(text, wav_file, syn_config=config)


def build_filter(noise: float, phone: bool, seed: int) -> tuple[list[str], str]:
    """Build the extra ffmpeg inputs and the filter graph for one voicemail.

    Returns (extra_input_args, filter_complex). The graph always ends in [out].
    """
    extra_inputs: list[str] = []
    chain = "[0:a]"
    graph = []

    if noise > 0:
        # Pink noise sounds like traffic/room noise. A fixed seed keeps it reproducible.
        extra_inputs = ["-f", "lavfi", "-i",
                        f"anoisesrc=color=pink:amplitude={noise}:seed={seed}:sample_rate=22050"]
        graph.append(f"{chain}[1:a]amix=inputs=2:duration=first:normalize=0[mix]")
        chain = "[mix]"

    if phone:
        # Phone lines only carry ~300-3400 Hz; this is why voicemails sound 'thin'
        # and why consonants like f/s/th get confused in transcription.
        graph.append(f"{chain}highpass=f=300,lowpass=f=3400,aresample=8000[out]")
    else:
        graph.append(f"{chain}anull[out]")

    return extra_inputs, ";".join(graph)


def degrade(clean_wav: Path, out_file: Path, noise: float, phone: bool, seed: int) -> None:
    """Clean speech -> realistic voicemail file in the requested format."""
    extra_inputs, graph = build_filter(noise, phone, seed)
    fmt = out_file.suffix.lstrip(".")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(clean_wav), *extra_inputs,
        "-filter_complex", graph, "-map", "[out]",
        "-ac", "1", *CODECS[fmt], str(out_file),
    ]
    subprocess.run(cmd, check=True)


def check_labels() -> list[str]:
    """Cross-check labels.json against scripts.json and the generated audio.

    Catches the boring mistakes (typo in an id, wrong file extension) that would
    otherwise show up later as confusing eval errors.
    """
    script_ids = {v["id"] for v in json.loads(SCRIPTS_FILE.read_text(encoding="utf-8"))["voicemails"]}
    labels = json.loads((HERE / "labels.json").read_text(encoding="utf-8"))["items"]
    problems = []
    for label in labels:
        if label["id"] not in script_ids:
            problems.append(f"{label['id']}: label has no script")
        if not (AUDIO_DIR / label["file"]).exists():
            problems.append(f"{label['id']}: audio file {label['file']} missing")
    for missing in script_ids - {label["id"] for label in labels}:
        problems.append(f"{missing}: script has no label")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="only generate ids starting with this prefix")
    args = parser.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH (open a new terminal after installing it)")

    voicemails = json.loads(SCRIPTS_FILE.read_text(encoding="utf-8"))["voicemails"]
    if args.only:
        voicemails = [v for v in voicemails if v["id"].startswith(args.only)]

    voice = PiperVoice.load(str(ensure_voice()))
    AUDIO_DIR.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        for index, vm in enumerate(voicemails, start=1):
            clean = Path(tmp) / f"{vm['id']}_clean.wav"
            out = AUDIO_DIR / f"{vm['id']}.{vm['format']}"
            synthesize(voice, vm["text"], vm["speaker"], vm["speed"], clean)
            # Seed derived from the id number, so each file always gets the same noise.
            seed = int(vm["id"].split("_")[0])
            degrade(clean, out, vm["noise"], vm["phone"], seed)
            print(f"[{index:2}/{len(voicemails)}] {out.name}")

    print(f"\nDone. Audio in {AUDIO_DIR}")
    problems = check_labels()
    for p in problems:
        print(f"  LABEL PROBLEM: {p}")
    print("labels.json is consistent with scripts and audio." if not problems else "")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
