"""The caller cards (testset/callers.json): loading, merging with Part 1's labels, and checking them.

    cards = load_cards()                 # 24 Card objects, labels filled in
    problems = check_cards(cards)        # [] when the answer key is consistent

A card is one synthetic caller: what is true about them (facts), how they behave (quirks), what they say (script)
and the answer key (labels + expect). For the 18 Part 1 scenarios the category/urgency/name/number labels are READ
from Part 1's labels.json instead of copied, so there is one answer key, not two that can drift apart.

    python 03-phone-receptionist/cards.py      # prints a summary and any problems
"""

import json
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import faq as faq_module  # noqa: E402
import persona as persona_module  # noqa: E402

CARDS_FILE = HERE / "testset" / "callers.json"
PART1_LABELS = ROOT / "01-voicemail-triage" / "testset" / "labels.json"
PART1_SCRIPTS = ROOT / "01-voicemail-triage" / "testset" / "scripts.json"

Quirk = Literal["all_upfront", "hesitant", "refuses_number", "withholds_name", "withholds_number",
                "self_corrects_number", "wrong_number_first", "spells_name", "robocall", "rambles",
                "emergency", "question_only"]


class Voice(BaseModel):
    speaker: int = Field(ge=0, le=108)
    speed: float = 1.0
    noise: float = 0.0
    phone: bool = True


class Facts(BaseModel):
    name: str | None
    name_spoken: str | None
    number: str | None
    reason_keywords: list[str]


class Script(BaseModel):
    opening: str
    reason: str | None = None
    name: str | None = None
    name_again: str | None = None
    number: str | None = None
    number_again: str | None = None
    anything_else: list[str] = []


class Question(BaseModel):
    topic: str | None  # faq.json id, or None when faq.json has no answer
    text: str


class Labels(BaseModel):
    category: Literal["urgent", "sales", "personal", "spam", "other"]
    urgency: int = Field(ge=1, le=3)
    caller_name: str | None
    callback_number: str | None


class Expect(BaseModel):
    urgent_flag: bool
    outcome: Literal["completed", "spam", "info_only"]
    faq_topics: list[str]
    faq_unknown: bool = False
    safety: Literal["gas", "co", "water"] | None = None  # which safety advice the emergency fast path must give


class Card(BaseModel):
    id: str
    part1_id: str | None
    split: Literal["dev", "score"]
    about: str
    voice: Voice
    facts: Facts
    quirks: list[Quirk]
    script: Script
    question: Question | None
    labels: Labels | None = None  # FAQ cards carry their own; Part 1 cards get theirs from labels.json
    expect: Expect


def load_cards(path: str | Path = CARDS_FILE) -> list[Card]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))["cards"]
    part1 = {item["id"]: item for item in json.loads(PART1_LABELS.read_text(encoding="utf-8"))["items"]}
    cards = []
    for item in raw:
        card = Card.model_validate(item)
        if card.part1_id:
            label = part1[card.part1_id]  # a KeyError here = a card points at a voicemail that does not exist
            card.labels = Labels(category=label["category"], urgency=label["urgency"],
                                 caller_name=label["caller_name"], callback_number=label["callback_number"])
        cards.append(card)
    return cards


def check_cards(cards: list[Card]) -> list[str]:
    """Every inconsistency in the answer key as a readable line; an empty list means the key is sound."""
    problems: list[str] = []
    persona = persona_module.load_persona()
    faq_ids = set(faq_module.load_faq())
    part1_ids = [item["id"] for item in json.loads(PART1_LABELS.read_text(encoding="utf-8"))["items"]]
    part1_speakers = {v["id"]: v["speaker"] for v in json.loads(PART1_SCRIPTS.read_text(encoding="utf-8"))["voicemails"]}

    ids = [c.id for c in cards]
    if len(set(ids)) != len(ids):
        problems.append("duplicate card ids")
    linked = [c.part1_id for c in cards if c.part1_id]
    for missing in sorted(set(part1_ids) - set(linked)):
        problems.append(f"Part 1 scenario {missing} has no caller card")
    for dup in sorted({p for p in linked if linked.count(p) > 1}):
        problems.append(f"Part 1 scenario {dup} has more than one card")

    new_speakers: dict[int, str] = {}
    for c in cards:
        where = c.id
        if c.labels is None:
            problems.append(f"{where}: no labels")
            continue
        if c.facts.name != c.labels.caller_name:
            problems.append(f"{where}: facts.name {c.facts.name!r} != label {c.labels.caller_name!r}")
        if c.facts.number != c.labels.callback_number:
            problems.append(f"{where}: facts.number {c.facts.number!r} != label {c.labels.callback_number!r}")
        if c.facts.number and not re.fullmatch(r"0\d{9,10}", c.facts.number):
            problems.append(f"{where}: number {c.facts.number!r} is not a UK national number")
        if c.facts.number and not re.fullmatch(r"(07700900\d{3}|01632960\d{3})", c.facts.number):
            problems.append(f"{where}: number {c.facts.number!r} is outside Ofcom's fictional ranges")
        if (c.facts.name is None) != (c.facts.name_spoken is None):
            problems.append(f"{where}: name and name_spoken must both be set or both null")
        if c.expect.urgent_flag != (c.labels.category == "urgent"):
            problems.append(f"{where}: expect.urgent_flag disagrees with category {c.labels.category!r}")
        if c.labels.category == "spam" and c.expect.outcome != "spam":
            problems.append(f"{where}: a spam label needs outcome 'spam'")
        if c.expect.outcome == "spam" and "robocall" not in c.quirks:
            problems.append(f"{where}: outcome 'spam' needs the robocall quirk")
        if c.expect.outcome == "info_only" and (c.facts.name or c.facts.number):
            problems.append(f"{where}: an info-only caller leaves no name or number")
        if c.expect.safety and not c.expect.urgent_flag:
            problems.append(f"{where}: expect.safety needs urgent_flag")
        if c.expect.safety and f"safety_{c.expect.safety}" not in faq_ids:
            problems.append(f"{where}: no FAQ entry safety_{c.expect.safety}")
        script_lines = [c.script.opening, *(v for v in (c.script.reason, c.script.name, c.script.name_again, c.script.number,
                                                          c.script.number_again) if v), *c.script.anything_else]
        if any(re.search(r"\d", line) for line in script_lines):
            problems.append(f"{where}: script lines are spoken: write numbers as words or use {{number_words}}")
        if {"refuses_number", "withholds_number"} & set(c.quirks) and c.facts.number:
            problems.append(f"{where}: a caller who refuses the number has number null")
        if c.part1_id is None and not (c.question or c.expect.faq_topics or c.expect.faq_unknown):
            problems.append(f"{where}: an FAQ caller asks at least one question")
        if c.expect.urgent_flag and "emergency" not in c.quirks:
            problems.append(f"{where}: an urgent caller should carry the emergency quirk")
        # quirks must have something to act on
        if "spells_name" in c.quirks and not (c.facts.name and len(c.facts.name.split()) >= 2):
            problems.append(f"{where}: spells_name needs a full name")
        if {"self_corrects_number", "wrong_number_first"} & set(c.quirks) and not c.facts.number:
            problems.append(f"{where}: a number quirk needs a number")
        if "{number_words}" in " ".join([c.script.opening, c.script.number or ""]) and not c.facts.number:
            problems.append(f"{where}: script uses {{number_words}} but the caller has no number")
        for topic in [q for q in [c.question.topic if c.question else None] + c.expect.faq_topics if q]:
            if topic not in faq_ids:
                problems.append(f"{where}: FAQ topic {topic!r} is not in faq.json")
        if c.question and c.question.topic is None and not c.expect.faq_unknown:
            problems.append(f"{where}: a question without a topic needs expect.faq_unknown")
        if c.question and c.question.topic and c.question.topic not in c.expect.faq_topics:
            problems.append(f"{where}: question topic {c.question.topic!r} missing from expect.faq_topics")
        if c.question and c.question.text.lower() not in " ".join([c.script.opening, *c.script.anything_else]).lower():
            problems.append(f"{where}: the question text is not part of what the caller says")
        if c.facts.name and c.expect.outcome == "completed" and not c.facts.reason_keywords:
            problems.append(f"{where}: needs reason_keywords")
        # voices
        if c.part1_id:
            if c.voice.speaker != part1_speakers[c.part1_id]:
                problems.append(f"{where}: voice differs from the Part 1 voicemail's speaker")
        else:
            if c.voice.speaker in persona.part_voice_ids:
                problems.append(f"{where}: FAQ caller voice {c.voice.speaker} is already used by Part 1/2")
            if c.voice.speaker in new_speakers:
                problems.append(f"{where}: voice {c.voice.speaker} is also used by {new_speakers[c.voice.speaker]}")
            new_speakers[c.voice.speaker] = c.id
        if persona.speaker is not None and c.voice.speaker == persona.speaker:
            problems.append(f"{where}: caller voice equals the receptionist's voice")

    if len(cards) != 24:
        problems.append(f"expected 24 cards (18 + 6), found {len(cards)}")
    return problems


def split_cards(cards: list[Card], split: str) -> list[Card]:
    return [c for c in cards if c.split == split]


def main() -> int:
    cards = load_cards()
    problems = check_cards(cards)
    dev, score = split_cards(cards, "dev"), split_cards(cards, "score")
    print(f"{len(cards)} cards: {len(dev)} dev ({', '.join(c.id.split('_')[0] for c in dev)}), {len(score)} score")
    for p in problems:
        print("  PROBLEM:", p)
    print("caller cards are consistent with Part 1's labels and faq.json." if not problems else "")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
