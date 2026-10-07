"""Build the synthetic meeting audio from scripts.json and check labels.json.

    python 02-meeting-action-agent/testset/generate.py            # all meetings
    python 02-meeting-action-agent/testset/generate.py --only m3  # ids starting with "m3"

Per meeting:
    each turn --Piper (that person's voice)--> wav --join with pauses--> clean.wav
    --ffmpeg (light room noise, encode)--> audio/<id>.<format>   + audio/<id>.turns.json (who spoke when)
The audio is git-ignored; this script + scripts.json make it reproducible in content, not bit for bit
(Piper varies slightly per run), so regenerating changes file hashes and Whisper output a little.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # repo root, for 'shared'

import answer_key  # noqa: E402
from shared.tts import load_voice, synthesize  # noqa: E402

SCRIPTS_FILE = HERE / "scripts.json"
AUDIO_DIR = HERE / "audio"
CODECS = {
    "wav": ["-c:a", "pcm_s16le"],
    "mp3": ["-c:a", "libmp3lame", "-b:a", "48k"],
    "m4a": ["-c:a", "aac", "-b:a", "48k"],
    "ogg": ["-c:a", "libopus", "-b:a", "32k"],
}


def join_turns(turn_wavs: list[tuple[str, Path]], pause_s: float, out_wav: Path) -> list[dict]:
    """Concatenate turn wavs with silence between them. Returns who spoke when (seconds)."""
    timeline, position = [], 0.0
    with wave.open(str(out_wav), "wb") as out:
        for i, (speaker, wav_path) in enumerate(turn_wavs):
            with wave.open(str(wav_path), "rb") as turn:
                if i == 0:
                    out.setparams(turn.getparams())  # same voice model -> same format for every turn
                rate = turn.getframerate()
                frames = turn.readframes(turn.getnframes())
                duration = turn.getnframes() / rate
            out.writeframes(frames)
            timeline.append({"speaker": speaker, "start": round(position, 2), "end": round(position + duration, 2)})
            position += duration
            silence = int(pause_s * rate)
            out.writeframes(b"\x00" * silence * out.getsampwidth() * out.getnchannels())  # zero samples = silence
            position += silence / rate
    return timeline


def degrade(clean_wav: Path, out_file: Path, noise: float, seed: int) -> None:
    """Add light pink noise (laptop mic in a room) and encode. No phone band: meetings aren't phone calls."""
    fmt = out_file.suffix.lstrip(".")
    if noise > 0:
        inputs = ["-i", str(clean_wav), "-f", "lavfi", "-i",
                  f"anoisesrc=color=pink:amplitude={noise}:seed={seed}:sample_rate=22050"]
        graph = ["-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:normalize=0[out]", "-map", "[out]"]
    else:
        inputs, graph = ["-i", str(clean_wav)], []
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, *graph, "-ac", "1", *CODECS[fmt], str(out_file)],
                   check=True)


def check_labels(script_files: dict[str, str]) -> list[str]:
    """script_files = {meeting id: audio file name that scripts.json produces}."""
    labels = answer_key.load()
    found = answer_key.problems(labels, set(script_files))
    for meeting in labels["meetings"]:
        expected = script_files.get(meeting["id"])
        # Compare names, not just existence: after a format change the old file would still be on disk.
        if expected and meeting["file"] != expected:
            found.append(f"{meeting['id']}: labels say {meeting['file']}, scripts.json makes {expected}")
        elif not (AUDIO_DIR / meeting["file"]).exists():
            found.append(f"{meeting['id']}: audio file {meeting['file']} missing")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="only generate ids starting with this prefix")
    args = parser.parse_args()
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH (open a new terminal after installing it)")

    config = json.loads(SCRIPTS_FILE.read_text(encoding="utf-8"))
    # Noise seed = position in the full series, so '--only m3' gets the same noise as a full run.
    # (Piper itself adds random variation, so a regenerated file never has the same hash anyway.)
    seeds = {m["id"]: number for number, m in enumerate(config["meetings"], start=1)}
    meetings = [m for m in config["meetings"] if not args.only or m["id"].startswith(args.only)]
    voice = load_voice()
    AUDIO_DIR.mkdir(exist_ok=True)

    for number, meeting in enumerate(meetings, start=1):
        with tempfile.TemporaryDirectory() as tmp:
            turn_wavs = []
            for i, (speaker, text) in enumerate(meeting["turns"]):
                wav = Path(tmp) / f"turn{i:02}.wav"
                synthesize(voice, text, config["voices"][speaker], wav)
                turn_wavs.append((speaker, wav))
            clean = Path(tmp) / "clean.wav"
            timeline = join_turns(turn_wavs, config["pause_s"], clean)
            out = AUDIO_DIR / f"{meeting['id']}.{meeting['format']}"
            degrade(clean, out, meeting["noise"], seed=seeds[meeting["id"]])
        (AUDIO_DIR / f"{meeting['id']}.turns.json").write_text(json.dumps(timeline, indent=1), encoding="utf-8")
        print(f"[{number}/{len(meetings)}] {out.name}  {timeline[-1]['end']:.0f} s, {len(timeline)} turns")

    found = check_labels({m["id"]: f"{m['id']}.{m['format']}" for m in config["meetings"]})
    for p in found:
        print(f"  LABEL PROBLEM: {p}")
    if not found:
        print("labels.json is consistent with scripts and audio.")
        final = answer_key.tracker_states(answer_key.load())
        last = list(final.values())[-1]
        counts = {s: sum(t["status"] == s for t in last.values()) for s in answer_key.STATUSES}
        print(f"Expected tracker after the series: {len(last)} tasks, {counts}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
