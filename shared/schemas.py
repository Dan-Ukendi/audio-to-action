"""Data shapes passed between pipeline steps.

Pydantic models instead of plain dicts: every step gets a checked, documented
object, and saving/loading to JSON is one call (model_dump_json / model_validate_json).
"""

from datetime import datetime

from pydantic import BaseModel


class Segment(BaseModel):
    """One chunk of speech as Whisper split it (roughly a sentence)."""

    start: float  # seconds from the start of the audio
    end: float
    text: str
    # Whisper's own confidence signals, useful later to flag doubtful transcripts:
    avg_logprob: float  # closer to 0 = more confident; below about -1.0 is shaky
    no_speech_prob: float  # high = this chunk is probably silence/noise, not words


class Transcript(BaseModel):
    """Output of the transcription step, saved to disk as JSON."""

    source_file: str  # file name only, not the full path (keeps folder names out of the record)
    audio_sha256: str  # fingerprint of the audio bytes; the cache key
    model: str  # e.g. "small", "large-v3-turbo"
    language: str  # detected, ISO 639-1 ("en")
    language_probability: float
    duration_s: float  # length of the audio
    transcribe_s: float  # how long transcription took (model loading excluded)
    text: str  # all segments joined
    segments: list[Segment]
    created_at: datetime
