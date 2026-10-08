"""The caller cards are the answer key: check them like code, before any dialog exists."""

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

import cards as cards_module  # noqa: E402
from cards import check_cards, load_cards  # noqa: E402


@pytest.fixture(scope="module")
def cards():
    return load_cards()


def test_the_answer_key_is_consistent(cards):
    assert check_cards(cards) == []


def test_eighteen_part_1_scenarios_and_six_faq_callers(cards):
    assert len(cards) == 24
    assert sum(c.part1_id is not None for c in cards) == 18
    assert sum(c.part1_id is None for c in cards) == 6
    part1 = json.loads(cards_module.PART1_LABELS.read_text(encoding="utf-8"))["items"]
    assert {c.part1_id for c in cards if c.part1_id} == {item["id"] for item in part1}


def test_labels_come_from_part_1_not_from_a_copy(cards):
    raw = json.loads(cards_module.CARDS_FILE.read_text(encoding="utf-8"))["cards"]
    assert all("labels" not in item for item in raw if item["part1_id"])  # a copy could drift
    labels = {item["id"]: item for item in json.loads(cards_module.PART1_LABELS.read_text(encoding="utf-8"))["items"]}
    for c in cards:
        if c.part1_id:
            assert c.labels.category == labels[c.part1_id]["category"]
            assert c.labels.caller_name == labels[c.part1_id]["caller_name"]
            assert c.labels.callback_number == labels[c.part1_id]["callback_number"]


def test_dev_and_score_split(cards):
    dev = [c for c in cards if c.split == "dev"]
    score = [c for c in cards if c.split == "score"]
    assert len(dev) == 9 and len(score) == 15
    # The dev cards cover the core behaviours, so a developer can see each one without touching the scoring cards...
    quirks = {q for c in dev for q in c.quirks}
    assert {"robocall", "spells_name", "self_corrects_number", "wrong_number_first", "refuses_number", "emergency",
            "hesitant", "question_only"} <= quirks
    assert any(c.expect.faq_unknown for c in dev) and any(c.expect.faq_topics for c in dev)
    assert any(c.expect.outcome == "info_only" for c in dev) and any(c.expect.safety for c in dev)
    # ...and these are deliberately seen ONLY by the scoring run (documented in README decision 15).
    only_score = {q for c in score for q in c.quirks} - quirks
    assert only_score == {"all_upfront", "withholds_name", "withholds_number", "rambles"}


def test_safety_expectations(cards):
    by_id = {c.id.split("_")[0]: c for c in cards}
    assert by_id["c03"].expect.safety == "gas" and by_id["c03"].expect.faq_topics == ["safety_gas"]
    assert by_id["c01"].expect.safety == "water" and by_id["c05"].expect.safety == "water"
    assert all(c.expect.urgent_flag for c in cards if c.expect.safety)


def test_urgent_callers_are_exactly_the_urgent_labels(cards):
    urgent = {c.id for c in cards if c.expect.urgent_flag}
    assert urgent == {c.id for c in cards if c.labels.category == "urgent"}
    assert len(urgent) == 6


def test_spam_callers_are_robocalls_and_not_urgent(cards):
    spam = [c for c in cards if c.expect.outcome == "spam"]
    assert {c.id.split("_")[0] for c in spam} == {"c11", "c12"}
    assert all(not c.expect.urgent_flag and "robocall" in c.quirks for c in spam)


def test_faq_callers_have_new_distinct_voices(cards):
    faq_voices = [c.voice.speaker for c in cards if c.part1_id is None]
    assert len(set(faq_voices)) == 6
    taken = {3, 5, 11, 15, 25, 27, 33, 40, 45, 50, 55, 60, 64, 72, 80, 90, 7, 0, 9}  # Part 1 callers + Part 2 voices
    assert not set(faq_voices) & taken


def test_every_number_is_in_ofcoms_fictional_ranges(cards):
    for c in cards:
        if c.facts.number:
            assert c.facts.number.startswith(("07700900", "01632960")), c.id


@pytest.mark.parametrize("change, expected", [
    (lambda c: c["facts"].update(name="Someone Else"), "facts.name"),
    (lambda c: c["facts"].update(number="07700900999"), "facts.number"),
    (lambda c: c["expect"].update(faq_topics=["no_such_topic"]), "not in faq.json"),
    (lambda c: c["expect"].update(urgent_flag=True), "urgent_flag"),
    (lambda c: c["voice"].update(speaker=7), "Part 1/2"),
    (lambda c: c["script"].update(opening="Hi, my number is 07700900402. How soon could someone come out to look at it?"), "spoken"),
])
def test_the_checker_catches_a_broken_card(tmp_path, change, expected):
    """A checker that never complains proves nothing: break a FAQ card and see it say so."""
    raw = json.loads(cards_module.CARDS_FILE.read_text(encoding="utf-8"))
    target = next(c for c in raw["cards"] if c["id"] == "f01_radiator_how_soon")
    if expected == "not in faq.json":
        target["question"] = {"topic": "no_such_topic", "text": target["question"]["text"]}
    change(target)
    path = tmp_path / "callers.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    problems = check_cards(load_cards(path))
    assert any(expected in p for p in problems), problems


def test_a_card_pointing_at_a_missing_voicemail_fails_loudly(tmp_path):
    raw = json.loads(cards_module.CARDS_FILE.read_text(encoding="utf-8"))
    raw["cards"][0]["part1_id"] = "99_nothing"
    path = tmp_path / "callers.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(KeyError):
        load_cards(path)


def test_a_missing_part_1_scenario_is_reported(tmp_path):
    raw = json.loads(cards_module.CARDS_FILE.read_text(encoding="utf-8"))
    raw["cards"] = [c for c in raw["cards"] if c["part1_id"] != "07_personal_mum"]
    path = tmp_path / "callers.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert any("07_personal_mum" in p for p in check_cards(load_cards(path)))
