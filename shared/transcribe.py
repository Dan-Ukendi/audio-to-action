"""Transcription step: audio file (any format) -> Transcript.

    transcribe(path, cache_dir=...) -> Transcript

Steps, in order:
    1. file_sha256()   fingerprint the audio bytes
    2. cache lookup    <cache_dir>/<hash>_<model>[_h<hint hash>].json exists? return it, done
    3. to_wav_16k()    ffmpeg: any format -> 16 kHz mono wav (temporary)
    4. Whisper         read_wav_samples() -> numbers; faster-whisper turns them into timestamped segments
    5. save            write the Transcript JSON into the cache

Try it:  python -m shared.transcribe path\\to\\file.mp3 [--model large-v3-turbo] [--hint "Sam, Priya, Tom"]
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import tempfile
import time
import wave
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from faster_whisper import WhisperModel

from shared.schemas import Segment, Transcript

load_dotenv()  # reads .env if present; the defaults below apply otherwise


def _add_cuda_libraries() -> None:
    """Windows only: make the CUDA DLLs from the pip packages nvidia-cublas-cu12 / nvidia-cudnn-cu12 findable.
    Without this, Whisper on the GPU fails with 'cublas64_12.dll is not found'. Harmless if they are not installed."""
    if not hasattr(os, "add_dll_directory"):
        return
    import site
    for root in site.getsitepackages():
        for folder in (Path(root) / "nvidia").glob("*/bin"):
            os.add_dll_directory(str(folder))
            os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")


_add_cuda_libraries()


def settings() -> dict:
    """Whisper settings from .env, so switching model needs no code change."""
    return {
        "model": os.getenv("WHISPER_MODEL", "small"),
        "device": os.getenv("WHISPER_DEVICE", "auto"),
        "compute_type": os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        # 0 = library default (4). More threads = faster on CPU, up to about the number of cores.
        "cpu_threads": int(os.getenv("WHISPER_CPU_THREADS", "0")),
    }


def file_sha256(path: Path) -> str:
    """Fingerprint of the file's bytes. Same audio -> same hash, whatever the file name.

    Read in 1 MB chunks so a long recording never has to fit in memory at once.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_wav_16k(src: Path, dst: Path) -> None:
    """Convert any audio format to 16 kHz, mono, 16-bit wav: exactly what Whisper was trained on.

    Doing this explicitly (instead of letting the library decode) gives one place where
    format problems show up, with ffmpeg's own error message.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH (open a new terminal after installing it)")
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-i", str(src),
        "-ac", "1",  # mono
        "-ar", "16000",  # 16 kHz
        "-c:a", "pcm_s16le",  # plain uncompressed 16-bit samples
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg could not convert {src.name}: {result.stderr.strip()}")


def read_wav_samples(wav: Path) -> np.ndarray:
    """Read our 16 kHz mono 16-bit wav into floats between -1 and 1 (Whisper's input format).

    We hand Whisper the samples directly instead of a file path, so it never decodes audio
    itself (its built-in decoder, PyAV, broke with av 19: 'metadata_errors' error).
    """
    with wave.open(str(wav), "rb") as w:
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


@lru_cache(maxsize=2)
def load_model(name: str, device: str, compute_type: str, cpu_threads: int) -> WhisperModel:
    """Load a Whisper model once per process; loading takes seconds, so we reuse it.

    The first call for a model downloads it to the Hugging Face cache in your user folder.
    """
    return WhisperModel(name, device=device, compute_type=compute_type, cpu_threads=cpu_threads)


def cache_path(cache_dir: Path, audio_hash: str, model: str, hint: str | None = None) -> Path:
    # The model name is part of the key: 'small' and 'large-v3-turbo' give different text.
    # So is the hint (a short fingerprint of it): a different hint can give different text too.
    # No hint = the original Part 1 file name, so those caches stay valid.
    suffix = f"_h{hashlib.sha256(hint.encode('utf-8')).hexdigest()[:8]}" if hint else ""
    return cache_dir / f"{audio_hash[:16]}_{model}{suffix}.json"


def transcribe(path: str | Path, cache_dir: str | Path | None = None, model: str | None = None,
               hint: str | None = None) -> Transcript:
    """Transcribe one audio file. With cache_dir, a file already done is returned from disk.

    hint: text Whisper treats as "what was said just before" (its initial_prompt). Names spelled
    in it (team, customers, streets) become much more likely to be spelled that way in the output.
    """
    path = Path(path)
    cfg = settings()
    model = model or cfg["model"]
    # "" would still change Whisper's prompt but share the no-hint cache file, so treat it as no hint.
    hint = hint or None

    audio_hash = file_sha256(path)
    cached = cache_path(Path(cache_dir), audio_hash, model, hint) if cache_dir else None
    if cached and cached.exists():
        return Transcript.model_validate_json(cached.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio_16k.wav"
        to_wav_16k(path, wav)  # first: a broken file fails here, before seconds of model loading
        samples = read_wav_samples(wav)

        whisper = load_model(model, cfg["device"], cfg["compute_type"], cfg["cpu_threads"])

        started = time.perf_counter()
        # beam_size=5: consider 5 candidate word sequences instead of 1; slower, more accurate.
        # vad_filter: skip long silences first; Whisper tends to invent text in silence.
        # initial_prompt: only "remembered" for the first ~220 tokens of output (about a minute of speech),
        # then it scrolls out of Whisper's context; fine for ~30 s meetings, weaker on long recordings.
        segments_iter, info = whisper.transcribe(samples, beam_size=5, vad_filter=True, initial_prompt=hint)
        # segments_iter is lazy: the actual work happens while we loop over it.
        segments = [
            Segment(
                start=round(s.start, 2), end=round(s.end, 2), text=s.text.strip(),
                avg_logprob=round(s.avg_logprob, 3), no_speech_prob=round(s.no_speech_prob, 3),
            )
            for s in segments_iter
        ]
        elapsed = time.perf_counter() - started

    transcript = Transcript(
        source_file=path.name,
        audio_sha256=audio_hash,
        model=model,
        hint=hint,
        language=info.language,
        language_probability=round(info.language_probability, 3),
        duration_s=round(info.duration, 2),
        transcribe_s=round(elapsed, 2),
        text=" ".join(s.text for s in segments),
        segments=segments,
        created_at=datetime.now(timezone.utc),
    )

    if cached:
        cached.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp name, then rename: a crash mid-write never leaves a half JSON in the cache.
        tmp_file = cached.with_suffix(".tmp")
        tmp_file.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
        tmp_file.replace(cached)

    return transcript


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe one audio file and print the text.")
    parser.add_argument("path")
    parser.add_argument("--model", help="overrides WHISPER_MODEL from .env")
    parser.add_argument("--cache-dir", default="01-voicemail-triage/transcripts")
    parser.add_argument("--hint", help="names to spell right, passed to Whisper as its initial prompt")
    args = parser.parse_args()

    started = time.perf_counter()
    t = transcribe(args.path, cache_dir=args.cache_dir, model=args.model, hint=args.hint)
    total = time.perf_counter() - started
    print(f"{t.source_file}  model={t.model}  lang={t.language} ({t.language_probability:.0%})  "
          f"audio={t.duration_s:.1f}s  transcribe={t.transcribe_s:.1f}s  this call={total:.1f}s")
    for s in t.segments:
        print(f"  [{s.start:6.2f} -> {s.end:6.2f}] {s.text}")


if __name__ == "__main__":
    main()
