"""Data shapes passed between pipeline steps.

Pydantic models instead of plain dicts: every step gets a checked, documented
object, and saving/loading to JSON is one call (model_dump_json / model_validate_json).
"""

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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


Category = Literal["urgent", "sales", "personal", "spam", "other"]


class Analysis(BaseModel):
    """What the LLM must fill in. Its JSON schema is sent to Ollama as the required output shape.

    Field ORDER matters: the model writes the fields top to bottom, so it summarizes and
    states its reason first, then commits to a category. Writing the reasoning before the
    decision tends to give better decisions than the other way round.

    The validators below are plain-code rules. If one fails, analyze() sends the error back
    to the model once and asks for a corrected answer.
    """

    summary: str = Field(description="One sentence, max 25 words: who called and what they want.")
    reason: str = Field(description="One short sentence: why this category and urgency.")
    category: Category
    urgency: int = Field(ge=1, le=3, description="1 = no time pressure, 2 = reply within a day or two, 3 = act today")
    caller_name: str | None = Field(description="Caller's name as they would spell it; null if not said.")
    callback_number: str | None = Field(description="UK phone number to call back, digits only; null if not said.")
    language: str = Field(description="ISO 639-1 code of the message language, e.g. 'en'.")

    @field_validator("callback_number")
    @classmethod
    def check_uk_number(cls, value: str | None) -> str | None:
        """Keep digits only, then check it looks like a UK national number (0 + 9 or 10 digits).

        Normalizing here (not in the prompt) means '07700 900-123' and '07700900123' are the
        same answer. Rejecting '1632960789' (leading 0 lost, our Phase 2 'oh'->'a' problem)
        gives the model one chance to fix it instead of us storing a wrong number.
        """
        if value is None:
            return None
        digits = re.sub(r"\D", "", value)
        if not digits:
            return None
        if not re.fullmatch(r"0\d{9,10}", digits):
            raise ValueError(
                f"'{value}' is not a UK phone number: it must be 10-11 digits starting with 0. "
                "Callers often say 'oh' for zero, which transcripts may write as 'o', 'oh' or 'a'."
            )
        return digits

    @field_validator("caller_name")
    @classmethod
    def empty_name_is_none(cls, value: str | None) -> str | None:
        # Models sometimes write "" or "unknown" instead of null; treat them the same.
        if value is None or value.strip().lower() in {"", "unknown", "none", "n/a", "null"}:
            return None
        return value.strip()

    @field_validator("language")
    @classmethod
    def two_letter_language(cls, value: str) -> str:
        # The model sometimes answers "en-GB"; we only keep the language part.
        return value.split("-")[0].strip().lower()

    @model_validator(mode="after")
    def urgent_means_today(self) -> "Analysis":
        """Category and urgency must agree: 'urgent' is defined as 'act today' (urgency 3)."""
        if (self.category == "urgent") != (self.urgency == 3):
            raise ValueError(
                f"category '{self.category}' with urgency {self.urgency} is inconsistent: "
                "use urgency 3 if and only if the category is 'urgent'."
            )
        return self


class Result(BaseModel):
    """Output of the analysis step, saved to disk as JSON (one per voicemail)."""

    source_file: str
    audio_sha256: str
    transcript_model: str  # which Whisper model produced the text we analyzed
    llm_model: str  # e.g. "qwen2.5:7b"
    prompt_version: str  # bump when the prompt changes, so old cached results aren't reused
    attempts: int  # 1 = valid first time, 2 = needed the retry
    # Kept when a retry happened, so we can see what the model first said and why it was rejected.
    rejected_reply: str | None = None
    rejected_because: str | None = None
    analyze_s: float
    analysis: Analysis
    created_at: datetime
