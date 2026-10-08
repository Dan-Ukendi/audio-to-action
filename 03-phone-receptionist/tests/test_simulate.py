"""The scripted caller answers by question type, deterministically, and its words parse back to the facts."""

import subprocess
import sys
import wave
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

import asks  # noqa: E402
import simulate  # noqa: E402
from cards import load_cards  # noqa: E402
from simulate import SimulatedCaller, number_words, render_turn, spell_out, wrong_number  # noqa: E402
from spoken import digit_runs, letter_runs  # noqa: E402

CARDS = {c.id.split("_")[0]: c for c in load_cards()}


def caller(prefix: str) -> SimulatedCaller:
    return SimulatedCaller(CARDS[prefix])


def test_number_words_roundtrip_through_the_digit_parser():
    for digits in ("07700900123", "01632960456", "07700900204", "01632960501", "07700900882", "0121496000"):
        assert digit_runs(number_words(digits)) == [digits], number_words(digits)


def test_number_words_sound_like_part_1_scripts():
    assert number_words("07700900123") == "oh seven seven double oh, nine hundred, one two three"
    assert number_words("01632960456") == "oh one six three two, nine six oh, four five six"
    assert number_words("07700900204") == "oh seven seven double oh, nine hundred, two oh four"


def test_wrong_number_is_a_different_valid_number():
    for digits in ("07700900123", "07700900204", "07700900111", "01632960500"):
        wrong = wrong_number(digits)
        assert wrong != digits and len(wrong) == len(digits) and wrong.startswith(digits[:8])


def test_spelling_roundtrip():
    assert spell_out("Siobhan Gallagher") == "S, I, O, B, H, A, N, G, A, double L, A, G, H, E, R"
    assert letter_runs(spell_out("Wojciech Nowak")) == ["WOJCIECHNOWAK"]


def test_every_card_has_an_opening_and_no_unfilled_placeholders():
    for card in load_cards():
        for asked in (asks.GREETING, asks.REASON, asks.NAME, asks.SPELLING, asks.NUMBER, asks.NUMBER_AGAIN,
                      asks.CONFIRM, asks.CORRECTION, asks.ANYTHING_ELSE):
            text = SimulatedCaller(card).reply(asked)
            assert "{" not in text and "}" not in text, (card.id, asked, text)
        assert SimulatedCaller(card).reply(asks.GREETING).strip()


def test_the_same_questions_give_the_same_words():
    for card in load_cards():
        a, b = SimulatedCaller(card), SimulatedCaller(card)
        for asked in (asks.GREETING, asks.NAME, asks.NUMBER, asks.CONFIRM, asks.ANYTHING_ELSE):
            assert a.reply(asked) == b.reply(asked)


def test_a_normal_caller_gives_the_labelled_name_and_number():
    c = caller("c08")
    assert "Jamie" in c.reply(asks.NAME)
    assert digit_runs(c.reply(asks.NUMBER), min_len=8) == ["07700900551"]


def test_every_caller_with_a_number_can_state_it_when_asked():
    for card in load_cards():
        if card.facts.number and not set(card.quirks) & {"refuses_number", "withholds_number", "wrong_number_first",
                                                         "self_corrects_number", "robocall"}:
            assert card.facts.number in digit_runs(SimulatedCaller(card).reply(asks.NUMBER), min_len=8), card.id


def test_refusing_callers_never_say_digits():
    for prefix in ("c04", "c07", "c17"):
        c = caller(prefix)
        # min_len: an interjection like "Oh he knows it" reads as a single zero, which is never a phone number
        assert digit_runs(c.reply(asks.NUMBER), min_len=6) == [] and digit_runs(c.reply(asks.NUMBER_AGAIN), min_len=6) == []
    assert caller("c04").reply(asks.NUMBER) == "You've got my number."
    assert caller("c04").reply(asks.NUMBER_AGAIN) != caller("c04").reply(asks.NUMBER)


def test_withholding_name_is_a_polite_no_not_a_made_up_name():
    c = caller("c10")
    assert "rather not" in c.reply(asks.NAME)
    assert "Top Rank" in c.reply(asks.NAME)  # the second ask gets the 'name_again' line


def test_self_correcting_caller_ends_on_the_right_number():
    runs = digit_runs(caller("c18").reply(asks.NUMBER), min_len=8)
    assert runs[-1] == "07700900349" and runs[0] != "07700900349"


def test_wrong_number_first_is_corrected_at_the_read_back():
    c = caller("c13")
    first = digit_runs(c.reply(asks.NUMBER), min_len=8)
    assert first and first != ["07700900204"]
    answer = c.reply(asks.CONFIRM, heard="So that's Laura Jenkins on whatever.")
    assert answer.startswith("No,") and digit_runs(answer, min_len=8) == ["07700900204"]
    right = "So that's Laura Jenkins on oh seven seven double oh, nine hundred, two oh four, about a radiator. Is that right?"
    assert c.reply(asks.CONFIRM, heard=right) == "Yes, that's right."  # the quirk objects once only


def test_spelling_is_only_given_by_callers_who_spell():
    assert letter_runs(caller("c14").reply(asks.SPELLING)) == ["SIOBHANGALLAGHER"]
    assert letter_runs(caller("c15").reply(asks.SPELLING)) == ["WOJCIECHNOWAK"]
    assert letter_runs(caller("c18").reply(asks.SPELLING)) == []


def test_an_attentive_caller_corrects_a_wrong_read_back_twice_then_gives_up():
    c = caller("c14")
    readback = "So that's Shivawn Gallagher, on oh one six three two, nine six oh, five oh one, about a bathroom quote. Is that right?"
    first = c.reply(asks.CONFIRM, heard=readback)
    assert first.startswith("No, my name is") and letter_runs(first) == ["SIOBHANGALLAGHER"]
    assert c.reply(asks.CONFIRM, heard=readback).startswith("No,")
    assert c.reply(asks.CONFIRM, heard=readback) == "Yes, that's right."  # patience is limited


def test_a_correct_read_back_gets_a_yes():
    c = caller("c08")
    heard = "So that's Jamie, on oh seven seven oh oh, nine oh oh, five five one, about five a side. Is that right?"
    assert c.reply(asks.CONFIRM, heard=heard) == "Yes, that's right."


def test_a_caller_who_gave_no_number_objects_to_an_invented_one():
    c = caller("c04")
    heard = "So that's Dave, on oh one six three two, nine six oh, five oh one, about a leak. Is that right?"
    assert c.reply(asks.CONFIRM, heard=heard) == "No, I didn't give you a number."


def test_an_invented_number_is_objected_to_and_the_correction_asks_to_remove_it():
    c = caller("c04")
    heard = "So that's Dave, on oh one six three two, nine six oh, five oh one, about a leak. Is that right?"
    assert c.reply(asks.CONFIRM, heard=heard) == "No, I didn't give you a number."
    answer = c.reply(asks.CORRECTION)
    assert "take that number off" in answer and digit_runs(answer, min_len=6) == []  # "one" alone is a 1


def test_the_name_in_the_read_back_is_compared_as_whole_words():
    for prefix, wrong in (("c01", "Mark Thompsonson"), ("c04", "Davey"), ("c13", "Laura Jenkinson")):
        c = caller(prefix)
        number = digit_runs(c.reply(asks.GREETING) + " " + (c.reply(asks.NUMBER) if prefix == "c13" else ""), min_len=8)
        digits = (CARDS[prefix].facts.number or "")
        heard = f"So that's {wrong}, on {number_words(digits) if digits else 'nothing'}, about it. Is that right?"
        reply = c.reply(asks.CONFIRM, heard=heard)
        assert reply.startswith("No,") and "my name is" in reply, (prefix, reply)  # the NAME is what is objected to
    # and a read-back with the right name and number is accepted
    c = caller("c01")
    ok = f"So that's Mark Thompson, on {number_words('07700900123')}, about a pipe. Is that right?"
    assert c.reply(asks.CONFIRM, heard=ok) == "Yes, that's right."


def test_a_wrong_name_and_an_invented_number_are_objected_to_together():
    c = caller("c04")
    heard = "So that's Davey, on oh one six three two, nine six oh, five oh one, about a leak. Is that right?"
    assert c.reply(asks.CONFIRM, heard=heard) == "No, my name is Dave, and I didn't give you a number."
    answer = c.reply(asks.CORRECTION)
    assert "My name is Dave" in answer and "take that number off" in answer


def test_non_spellers_never_spell_even_when_correcting_their_name():
    c = caller("c18")  # not a speller
    reply = c.reply(asks.CONFIRM, heard="So that's Chris Okaforr, on whatever.")
    assert reply.startswith("No, my name is Chris Okafor") and letter_runs(reply) == []
    assert letter_runs(caller("c14").reply(asks.CONFIRM, heard="So that's Shivawn Gallagher.")) == ["SIOBHANGALLAGHER"]


def test_the_noise_filter_matches_part_1s_generator():
    pytest.importorskip("piper")  # Part 1's generate.py imports Piper at load time
    import importlib.util
    spec = importlib.util.spec_from_file_location("part1_generate", HERE.parents[1] / "01-voicemail-triage/testset/generate.py")
    part1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(part1)
    for noise, phone in ((0, True), (0.09, True), (0.03, False), (0, False)):
        assert simulate.build_filter(noise, phone, 5) == part1.build_filter(noise, phone, 5)


def test_correction_question_is_answered_with_what_was_wrong():
    c = caller("c13")
    c.reply(asks.NUMBER)
    c.reply(asks.CONFIRM)  # the wrong_number_first quirk objects
    assert digit_runs(c.reply(asks.CORRECTION), min_len=8) == ["07700900204"]


def test_robocall_plays_once_and_never_answers():
    c = caller("c11")
    assert "automated message" in c.reply(asks.GREETING)
    assert c.reply(asks.REASON) == "" and c.reply(asks.NAME) == "" and c.reply(asks.CONFIRM) == ""


def test_anything_else_follows_the_script_then_says_no():
    c = caller("c15")
    assert c.reply(asks.ANYTHING_ELSE) == "That's all, thank you."
    c = caller("f05")
    assert "quote for a new bathroom" in c.reply(asks.ANYTHING_ELSE)
    assert "everything" in c.reply(asks.ANYTHING_ELSE)
    assert c.reply(asks.ANYTHING_ELSE) == "That's all, thank you."


def test_repeat_says_the_last_thing_again():
    c = caller("c08")
    said = c.reply(asks.NAME)
    assert c.reply(asks.REPEAT) == said
    assert c.reply(asks.REPEAT) == said


def test_hesitant_caller_prefixes_answers_with_a_filler():
    assert caller("c03").reply(asks.NAME).startswith("Um, yeah, ")
    assert not caller("c03").reply(asks.GREETING).startswith("Um, yeah, ")


def test_nothing_to_say_after_the_call_and_unknown_questions_are_bugs():
    assert caller("c08").reply(asks.NOTHING) == ""
    with pytest.raises(ValueError):
        caller("c08").reply("what_is_your_favourite_colour")


def test_the_gas_caller_asks_what_to_do():
    assert "should I turn it off" in caller("c03").reply(asks.GREETING)


def test_faq_callers_ask_their_questions():
    assert "How soon could someone come out" in caller("f01").reply(asks.GREETING)
    assert "opening hours" in caller("f03").reply(asks.GREETING)
    assert "solar panels" in caller("f04").reply(asks.REASON)  # the card's own reason line after the unknown question


# ---------------------------------------------------------------- audio (fake voice, real ffmpeg)

def fake_synth(text, speaker, speed, out_wav):
    """Stand-in for Piper: half a second of a 220 Hz tone at 22.05 kHz."""
    import math
    import struct
    rate = 22050
    frames = b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 220 * n / rate))) for n in range(rate // 2))
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)


def wav_info(path: Path) -> tuple[int, int, float]:
    with wave.open(str(path), "rb") as w:
        return w.getnchannels(), w.getframerate(), w.getnframes() / w.getframerate()


def test_render_turn_makes_a_16k_mono_wav_with_noise_and_phone_band(tmp_path):
    out = render_turn(CARDS["c05"], "The water heater is leaking.", tmp_path / "t.wav", synth=fake_synth)  # noise 0.09
    channels, rate, seconds = wav_info(out)
    assert (channels, rate) == (1, 16000) and 0.4 < seconds < 0.7


def test_noise_is_reproducible_for_the_same_card(tmp_path):
    a = render_turn(CARDS["c05"], "Hello there.", tmp_path / "a.wav", synth=fake_synth)
    b = render_turn(CARDS["c05"], "Hello there.", tmp_path / "b.wav", synth=fake_synth)
    assert a.read_bytes() == b.read_bytes()


def test_silence_is_a_second_of_nothing(tmp_path):
    out = render_turn(CARDS["c08"], "", tmp_path / "s.wav", synth=fake_synth)
    channels, rate, seconds = wav_info(out)
    assert (channels, rate, round(seconds, 1)) == (1, 16000, 1.0)
    assert set(out.read_bytes()[44:]) == {0}


def test_preview_command_runs(capsys):
    simulate.preview(CARDS["c14"])
    out = capsys.readouterr().out
    assert "spelling" in out and "double L" in out


def test_ffmpeg_is_available_for_the_audio_tests():
    assert subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0
