"""Who answers the phone: loads persona.json and says its fixed sentences.

    persona = load_persona()
    persona.say("read_back", name="Siobhan Gallagher", number=speak_number("01632960501"), reason="a bathroom quote")

The dialog never writes free text. Every sentence the receptionist says is one of these lines or an
approved answer from faq.json, so nothing in a reply can be invented by a model. The loader checks the
file once at start-up (every line present, only known placeholders, no digits) so a typo fails here,
not in the middle of somebody's call.
"""

import json
import os
import re
import string
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_PERSONA = HERE / "persona.json"

# Every line the engine can say, and the placeholders it must contain (exactly these).
REQUIRED_LINES: dict[str, set[str]] = {
    "greeting": set(),
    "ask_reason": set(),
    "ask_name": set(),
    "ask_name_spelling": {"first_name"},
    "ask_number": set(),
    "number_refused": set(),
    "repeat_request": set(),
    "confirmed": set(),
    "read_back": {"name", "number", "reason"},
    "read_back_no_number": {"name", "reason"},
    "read_back_no_name": {"number", "reason"},
    "read_back_no_name_no_number": {"reason"},
    "reason_missing_phrase": set(),
    "correction": set(),
    "faq_unknown": set(),
    "anything_else": set(),
    "urgent_ack": set(),
    "goodbye": set(),
    "goodbye_urgent": set(),
    "goodbye_spam": set(),
    "silence_end": set(),
    "turn_limit": set(),
}
DETAILS = ("reason", "name", "number")


class PersonaError(Exception):
    """persona.json is incomplete or inconsistent."""


@dataclass(frozen=True)
class Persona:
    business: str
    receptionist: str
    speaker: int | None  # Piper speaker id; None until the owner picks a voice on the laptop
    taken_voice_ids: frozenset[int]  # every id the receptionist must not use (Part 1/2 voices + the Part 3 FAQ callers)
    part_voice_ids: frozenset[int]  # only the Part 1 callers and Part 2 meeting voices
    hint: str  # Whisper initial_prompt: business, team and places (never caller names)
    detail_order: tuple[str, ...]
    max_turns: int
    max_reasks: int
    max_silent_turns: int
    lines: dict[str, str]

    def say(self, key: str, **values: str) -> str:
        """One fixed sentence with its placeholders filled in. A missing value is a bug: it raises."""
        return self.lines[key].format(**values)


MAX_SPEAKER = 108  # en_GB-vctk-medium has speakers 0-108
SPOKEN_FORBIDDEN = re.compile(r"[\d£$€%&/@#*+=<>]")  # symbols the voice would read oddly or skip


def placeholders(text: str) -> set[str]:
    """Named placeholders in a line. A positional '{}' or '{0}' is a bug (say() uses names), so it is refused."""
    found = set()
    for _, name, _, _ in string.Formatter().parse(text):
        if name is None:
            continue
        if name == "" or name.isdigit():
            raise PersonaError(f"positional placeholder in {text!r}: use a name like {{name}}")
        found.add(name)
    return found


def check_speaker(value: int | None) -> None:
    if value is not None and not 0 <= value <= MAX_SPEAKER:
        raise PersonaError(f"speaker {value} does not exist: the voice has speakers 0-{MAX_SPEAKER}")


def load_persona(path: str | Path = DEFAULT_PERSONA) -> Persona:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    lines = data["lines"]

    missing = [key for key in REQUIRED_LINES if key not in lines]
    if missing:
        raise PersonaError(f"persona lines missing: {', '.join(missing)}")
    for key, wanted in REQUIRED_LINES.items():
        found = placeholders(lines[key])
        if found != wanted:
            raise PersonaError(f"line '{key}' has placeholders {sorted(found)}, expected {sorted(wanted)}")
        if SPOKEN_FORBIDDEN.search(lines[key]):
            raise PersonaError(f"line '{key}' contains a digit or symbol; write it in words (it is read aloud)")

    order = tuple(data["detail_order"])
    if sorted(order) != sorted(DETAILS):
        raise PersonaError(f"detail_order must be a permutation of {DETAILS}, got {order}")
    groups = data["voice_ids_taken"]
    taken = frozenset(i for ids in groups.values() for i in ids)
    part_ids = frozenset(i for name in ("part2_meeting_voices", "part1_callers") for i in groups[name])
    speaker = data["piper_speaker"]
    check_speaker(speaker)
    if speaker is not None and speaker in taken:
        raise PersonaError(f"speaker {speaker} is already used by a Part 1/2 voice; pick another")

    limits = data["limits"]
    names = data["whisper_hint"]
    team, places = names["team"], names["places"]
    if not team or not places:
        raise PersonaError("whisper_hint needs at least one team member and one place")
    # Reads like the start of a phone call, which is how Whisper uses its initial prompt.
    hint = (f"{data['business']} phone call, answered by {data['receptionist']}. "
            f"The team: {', '.join(team[:-1])} and {team[-1]}. Places: {', '.join(places)}.")
    return Persona(
        business=data["business"], receptionist=data["receptionist"], speaker=speaker, taken_voice_ids=taken, part_voice_ids=part_ids,
        hint=hint, detail_order=order, max_turns=limits["max_turns"], max_reasks=limits["max_reasks_per_detail"],
        max_silent_turns=limits["max_silent_turns"], lines=dict(lines),
    )


def speaker_id(persona: Persona) -> int:
    """The Piper voice to speak with: RECEPTIONIST_SPEAKER (one run) beats persona.json.

    Left unset on purpose until the owner has listened to the candidates (choose_voice.py): which voice
    sounds like a receptionist is a taste decision, and there is no ear in the cloud.
    """
    override = os.getenv("RECEPTIONIST_SPEAKER", "").strip()
    if override and not override.isdigit():
        raise PersonaError(f"RECEPTIONIST_SPEAKER must be a number, got {override!r}")
    chosen = int(override) if override else persona.speaker
    check_speaker(chosen)
    if chosen is None:
        raise PersonaError("no receptionist voice chosen yet: run 'python 03-phone-receptionist/choose_voice.py' (written in Phase 2), "
                           "listen, then put the number in persona.json (piper_speaker)")
    if chosen in persona.taken_voice_ids:
        raise PersonaError(f"speaker {chosen} belongs to a Part 1/2 voice; the receptionist needs her own")
    return chosen


# ---------------------------------------------------------------- numbers, names

DIGIT_WORDS = {"0": "oh", "1": "one", "2": "two", "3": "three", "4": "four",
               "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}


def group_number(digits: str) -> list[str]:
    """01632960501 -> ['01632', '960', '501'] (a UK number the way people say it)."""
    if len(digits) == 11:
        return [digits[:5], digits[5:8], digits[8:]]
    if len(digits) == 10:
        return [digits[:4], digits[4:7], digits[7:]]
    return [digits]


def format_number(digits: str) -> str:
    """For screens and logs: '01632 960 501'."""
    return " ".join(group_number(digits))


def speak_number(digits: str) -> str:
    """For the voice: digit by digit in groups, with a comma pause between groups.

    '01632960501' -> 'oh one six three two, nine six oh, five oh one'. Piper reads a bare '01632960501' as one
    huge number, and a caller who is checking their number needs to hear the groups.
    """
    return ", ".join(" ".join(DIGIT_WORDS[d] for d in group) for group in group_number(digits))


def first_name(name: str) -> str:
    return name.split()[0] if name.split() else name
