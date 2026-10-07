"""Data shapes passed between pipeline steps.

Pydantic models instead of plain dicts: every step gets a checked, documented
object, and saving/loading to JSON is one call (model_dump_json / model_validate_json).
"""

import re
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator


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
    hint: str | None = None  # Whisper initial_prompt used, if any (default keeps old cache files valid)
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


# ---------------------------------------------------------------- Part 2: meetings

Status = Literal["open", "done", "cancelled"]


class ActionItem(BaseModel):
    """One task as a meeting talks about it: what the LLM must fill in for each item.

    Field order: the evidence (exact words) first, then the interpretation. Validators that need
    outside facts read them from the validation context: {"team": [...], "transcript": "..."}.
    """

    evidence: str = Field(description="The exact words from the transcript that show this item (one short quote).")
    task: str = Field(description="Short description of the work itself, with customer and place names, "
                                  "e.g. 'Send Siobhan Gallagher the bathroom quote'.")
    owner: str | None = Field(description="Team member responsible; null if nobody is named or clearly implied.")
    due_text: str | None = Field(description="The words used for when it will be done, exactly as said "
                                             "(e.g. 'by Wednesday', 'this afternoon'); null if not said.")
    status: Status = Field(description="open = still to do; done = said to be finished; cancelled = called off.")

    @field_validator("owner")
    @classmethod
    def owner_on_team(cls, value: str | None, info: ValidationInfo) -> str | None:
        """Owners must be real team members: catches invented names and customers as owners."""
        if value is None or value.strip().lower() in {"", "unknown", "none", "null", "someone", "nobody"}:
            return None
        team = (info.context or {}).get("team")
        if team:
            match = next((member for member in team if member.lower() == value.strip().lower()), None)
            if match is None:
                raise ValueError(f"'{value}' is not on the team ({', '.join(team)}); use one of them or null.")
            return match
        return value.strip()

    @field_validator("due_text")
    @classmethod
    def empty_due_is_none(cls, value: str | None) -> str | None:
        # Models sometimes write the word "null" (or "none", "") instead of a real null.
        if value is None or value.strip().lower() in {"", "null", "none", "n/a", "not said", "unknown"}:
            return None
        return value.strip()

    @field_validator("evidence")
    @classmethod
    def evidence_in_transcript(cls, value: str, info: ValidationInfo) -> str:
        """Most words of the quote must really appear in the transcript: a cheap check against invented items."""
        transcript = (info.context or {}).get("transcript")
        if transcript:
            # Ignore '[00:23]' line times: the model often copies them, but the plain transcript has none
            # (in a short quote like "[00:26] We'll do." they'd be half the words).
            quote = re.sub(r"\[\d+:\d+\]", " ", value)
            words = re.findall(r"[a-z0-9']+", quote.lower())
            heard = set(re.findall(r"[a-z0-9']+", transcript.lower()))
            if words and sum(w in heard for w in words) / len(words) < 0.6:
                raise ValueError(f"the quote '{value}' is not in the transcript; quote the transcript's words exactly.")
        return value


class MeetingItems(BaseModel):
    """The LLM's whole answer for one meeting (or one chunk of a long meeting).

    jobs_mentioned comes first on purpose ("think first", like summary/reason before category in
    Part 1): listing every job separately before writing items stops the model merging two jobs.
    """

    jobs_mentioned: list[str] = Field(description="Every distinct job talked about, in order, one short phrase "
                                                  "each. A finished job and the new job that follows it are two entries.")
    items: list[ActionItem]


class ExtractedItem(ActionItem):
    """An ActionItem after plain code resolved due_text into a calendar date."""

    due: date | None = None
    owner_from_llm: str | None = None  # what the LLM said before ground_owner() (to see what grounding changed)


class MeetingResult(BaseModel):
    """Output of the Part 2 extraction step, saved to disk as JSON (one per meeting)."""

    source_file: str
    audio_sha256: str
    meeting_date: date
    transcript_model: str
    transcript_hint: bool  # was the Whisper name hint used?
    llm_model: str
    prompt_version: str
    chunks: int  # how many pieces the transcript was split into
    attempts: int  # total LLM calls (chunks + retries)
    rejected: list[str]  # why any first answer was rejected (empty = none)
    jobs_mentioned: list[str] = []  # the model's "think first" list (prompt x3+), kept for inspection
    extract_s: float
    items: list[ExtractedItem]
    # The model's raw items per chunk, before merge + ground_owner (x6+): to see what plain code changed.
    llm_items: list[list[ActionItem]] = []
    created_at: datetime
