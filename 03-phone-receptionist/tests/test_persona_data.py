"""persona.json and faq.json are data the whole call depends on: check them like code (no LLM, instant)."""

import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # repo root, for 'shared'
sys.path.insert(0, str(HERE.parent))  # 03-phone-receptionist

import faq  # noqa: E402
import persona as persona_module  # noqa: E402
from persona import PersonaError, load_persona, speak_number, speaker_id  # noqa: E402

def voices_in_use() -> tuple[set[int], set[int]]:
    """(Part 2 meeting voices, Part 1 caller voices), read from the real test-set files so they cannot drift."""
    import json
    root = HERE.parents[1]
    part1 = json.loads((root / "01-voicemail-triage/testset/scripts.json").read_text(encoding="utf-8"))["voicemails"]
    part2 = json.loads((root / "02-meeting-action-agent/testset/scripts.json").read_text(encoding="utf-8"))
    part2_ids = set()

    def walk(node):  # find every "speaker"-like integer in Part 2's script file without assuming its layout
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("speaker", "speaker_id") and isinstance(value, int):
                    part2_ids.add(value)
                elif key == "voices" and isinstance(value, dict):
                    part2_ids.update(v for v in value.values() if isinstance(v, int))
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(part2)
    return part2_ids, {v["speaker"] for v in part1}


PART2_VOICES, PART1_VOICES = voices_in_use()


def test_persona_file_is_valid_and_complete():
    p = load_persona()
    assert p.receptionist == "Holly"
    assert p.business.startswith("Brightwater")
    assert p.detail_order == ("reason", "name", "number")
    assert (p.max_turns, p.max_reasks, p.max_silent_turns) == (12, 2, 2)
    assert set(p.lines) == set(persona_module.REQUIRED_LINES)


def test_greeting_says_she_is_automated_and_the_call_is_recorded():
    greeting = load_persona().lines["greeting"].lower()
    assert "automated" in greeting and "recorded" in greeting


def test_taken_voice_ids_match_parts_1_and_2():
    assert PART2_VOICES == {7, 0, 9, 11}  # if this fails, the walk above no longer finds Part 2's voices
    assert load_persona().taken_voice_ids == frozenset(PART1_VOICES | PART2_VOICES)


def test_voice_is_not_chosen_yet_and_asking_for_it_explains_why(monkeypatch):
    monkeypatch.delenv("RECEPTIONIST_SPEAKER", raising=False)
    assert load_persona().speaker is None
    with pytest.raises(PersonaError, match="choose_voice"):
        speaker_id(load_persona())


def test_voice_override_must_not_be_a_part_1_or_2_voice(monkeypatch):
    monkeypatch.setenv("RECEPTIONIST_SPEAKER", "20")
    assert speaker_id(load_persona()) == 20
    monkeypatch.setenv("RECEPTIONIST_SPEAKER", "7")
    with pytest.raises(PersonaError, match="own"):
        speaker_id(load_persona())


def test_missing_line_or_wrong_placeholder_is_refused(tmp_path):
    import json
    data = json.loads(persona_module.DEFAULT_PERSONA.read_text(encoding="utf-8"))
    del data["lines"]["goodbye"]
    path = tmp_path / "p.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(PersonaError, match="goodbye"):
        load_persona(path)
    data = json.loads(persona_module.DEFAULT_PERSONA.read_text(encoding="utf-8"))
    data["lines"]["read_back"] = "So that's {name}. Is that right?"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(PersonaError, match="placeholders"):
        load_persona(path)


def test_say_fills_placeholders():
    p = load_persona()
    text = p.say("read_back", name="Siobhan Gallagher", number=speak_number("01632960501"), reason="a bathroom quote")
    assert text == ("So that's Siobhan Gallagher, on oh one six three two, nine six oh, five oh one, "
                    "about a bathroom quote. Is that right?")


def test_numbers_are_spoken_in_groups():
    assert speak_number("01632960501") == "oh one six three two, nine six oh, five oh one"
    assert speak_number("07700900123") == "oh seven seven oh oh, nine oh oh, one two three"
    assert persona_module.format_number("01632960501") == "01632 960 501"
    assert speak_number("0121496000") == "oh one two one, four nine six, oh oh oh"  # 10 digits: 4-3-3


def test_faq_has_the_planned_entries():
    entries = faq.load_faq()
    assert 20 <= len(entries) <= 25
    for wanted in ("hours", "emergency_callout", "area", "services", "not_offered", "boiler_service",
                   "landlord_certificate", "new_boilers", "bathrooms", "prices", "free_quotes", "booking_time",
                   "payment", "guarantee", "qualifications", "cancellation", "vulnerable_priority", "contact",
                   "callback_time", "safety_gas", "safety_co", "safety_water"):
        assert wanted in entries, wanted
    assert {e.safety for e in entries.values() if e.safety} == {"gas", "co", "water"}


def test_faq_facts_match_the_brief():
    a = {k: v.answer for k, v in faq.load_faq().items()}
    assert ("Monday to Friday from eight in the morning until half past five" in a["hours"]
            and "Saturdays from nine until one" in a["hours"]
            and "twenty-four hours a day, seven days a week" in a["hours"])
    assert "within two hours" in a["emergency_callout"]
    assert "ninety-five pounds" in a["emergency_callout"] and "includes the first hour" in a["emergency_callout"]
    assert "sixty-five pounds an hour" in a["emergency_callout"]
    assert "Kelmbridge" in a["area"] and "fifteen miles" in a["area"]
    assert "eighty-five pounds" in a["boiler_service"]
    assert "seventy pounds" in a["landlord_certificate"]
    assert "Worcester Bosch" in a["new_boilers"] and "Vaillant" in a["new_boilers"] and "free survey" in a["new_boilers"]
    assert "free" in a["bathrooms"] and "twenty-five percent deposit" in a["bathrooms"]
    assert "fifty-five pounds an hour" in a["prices"] and "office hours" in a["prices"] and "minimum charge of one hour" in a["prices"]
    assert "free" in a["free_quotes"]
    assert "three to five working days" in a["booking_time"]
    assert "card" in a["payment"] and "bank transfer" in a["payment"]
    assert "twelve-month labour guarantee" in a["guarantee"]
    assert "Gas Safe" in a["qualifications"] and "five million pounds" in a["qualifications"]
    assert "twenty-four hours" in a["cancellation"] and "thirty pound" in a["cancellation"]
    assert "priority" in a["vulnerable_priority"]
    assert speak_number("01632960000") in a["contact"]  # the business number, spoken exactly as the voice reads digits
    assert "brightwater dash plumbing dot example" in a["contact"]
    assert "within two hours" in a["callback_time"] and "office hours" in a["callback_time"]
    assert "Unit four, Mill Lane Trading Estate, Kelmbridge" in a["location"]


def test_safety_advice_uses_the_real_uk_gas_emergency_number():
    entries = faq.load_faq()
    for key in ("safety_gas", "safety_co"):
        assert "oh eight hundred, one one one, nine nine nine" in entries[key].answer
        assert "nine nine nine" in entries[key].answer
    assert "stopcock" in entries["safety_water"].answer and "electric" in entries["safety_water"].answer


def test_faq_keywords_are_unique_per_entry_but_may_overlap_across_entries():
    for entry in faq.load_faq().values():
        assert len(set(entry.keywords)) == len(entry.keywords), entry.id


def test_faq_domain_is_fictional():
    a = {k: v.answer for k, v in faq.load_faq().items()}
    # e-mail and website are spoken, and both live under the reserved .example domain
    assert a["contact"].count("dot example") == 2  # one for the e-mail, one for the website
    assert "hello at brightwater dash plumbing dot example" in a["contact"]
    assert "w w w dot brightwater dash plumbing dot example" in a["contact"]
    assert not re.search(r"https?://", (persona_module.HERE / "faq.json").read_text(encoding="utf-8"))


def test_faq_loader_rejects_digits_in_answers(tmp_path):
    import json
    path = tmp_path / "f.json"
    path.write_text(json.dumps({"entries": [{"id": "x", "topic": "t", "keywords": ["a"], "answer": "It costs 5."}]}),
                    encoding="utf-8")
    with pytest.raises(faq.FaqError, match="words"):
        faq.load_faq(path)


def test_whisper_hint_has_places_and_team_but_no_caller_names():
    """The caller's name is a scored field: a hint that contains test callers' names would leak the answer."""
    import json
    p = load_persona()
    assert "Kelmbridge" in p.hint and "Priya" in p.hint
    part1 = json.loads((HERE.parents[1] / "01-voicemail-triage" / "testset" / "labels.json").read_text(encoding="utf-8"))
    caller_names = {n for item in part1["items"] if item["caller_name"] for n in item["caller_name"].split()}
    # Team members double as caller names in Part 1 (Priya Shah, Tom Bradley, Jamie): the team may be hinted,
    # but no caller's surname may be, and neither may a caller first name that is not a team member (e.g. Dave).
    team = {"Sam", "Priya", "Tom", "Jamie"}
    leaked = [n for n in caller_names - team if re.search(rf"\b{re.escape(n)}\b", p.hint)]
    assert leaked == []


def test_loaders_refuse_symbols_and_bad_speakers(tmp_path, monkeypatch):
    import json
    base = json.loads(persona_module.DEFAULT_PERSONA.read_text(encoding="utf-8"))
    path = tmp_path / "p.json"

    def load_with(change):
        data = json.loads(json.dumps(base))
        change(data)
        path.write_text(json.dumps(data), encoding="utf-8")
        return load_persona(path)

    with pytest.raises(PersonaError, match="symbol"):
        load_with(lambda d: d["lines"].update(goodbye="Cheers & bye."))
    with pytest.raises(PersonaError, match="positional"):
        load_with(lambda d: d["lines"].update(goodbye="Bye {}."))
    with pytest.raises(PersonaError, match="0-108"):
        load_with(lambda d: d.update(piper_speaker=500))
    with pytest.raises(PersonaError, match="already used"):
        load_with(lambda d: d.update(piper_speaker=7))
    with pytest.raises(PersonaError, match="at least one"):
        load_with(lambda d: d["whisper_hint"].update(team=[]))
    assert load_with(lambda d: d.update(piper_speaker=20)).speaker == 20
    monkeypatch.setenv("RECEPTIONIST_SPEAKER", "abc")
    with pytest.raises(PersonaError, match="number"):
        speaker_id(load_persona())
    monkeypatch.setenv("RECEPTIONIST_SPEAKER", "500")
    with pytest.raises(PersonaError, match="0-108"):
        speaker_id(load_persona())


def test_faq_safety_field_must_be_known(tmp_path):
    import json
    path = tmp_path / "f.json"
    path.write_text(json.dumps({"entries": [{"id": "x", "topic": "t", "keywords": ["a"], "answer": "Fine.",
                                             "safety": "fire"}]}), encoding="utf-8")
    with pytest.raises(faq.FaqError, match="gas, co or water"):
        faq.load_faq(path)
