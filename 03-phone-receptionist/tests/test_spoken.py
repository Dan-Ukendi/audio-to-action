"""Numbers said as words and names spelled out: the conversion the dialog and the simulated callers share."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

from spoken import digit_runs, letter_runs  # noqa: E402


def test_digits_written_as_digits_in_any_grouping():
    assert digit_runs("my number is 01632 960 501") == ["01632960501"]
    assert digit_runs("07700-900-123.") == ["07700900123"]
    assert digit_runs("+44 1632 960501") == ["01632960501"]


def test_uk_number_words_oh_double_and_hundred():
    assert digit_runs("oh seven seven double oh, nine hundred, one two three") == ["07700900123"]
    assert digit_runs("oh one six three two, nine six oh, five oh one") == ["01632960501"]
    assert digit_runs("zero one six three two nine six zero four five six") == ["01632960456"]
    assert digit_runs("oh seven seven triple oh nine nine nine") == ["077000999"]
    assert digit_runs("double seven") == ["77"]


def test_a_correction_splits_the_runs_so_nothing_is_glued_together():
    text = ("oh seven seven double oh, nine hundred, four three, no sorry, three four nine. "
            "So that's oh seven seven double oh, nine hundred, three four nine")
    assert digit_runs(text) == ["0770090043", "349", "07700900349"]
    assert digit_runs(text, min_len=8) == ["0770090043", "07700900349"]


def test_other_numbers_in_speech_are_separate_short_runs():
    assert digit_runs("the water heater at forty two Mill Lane") == ["42"]
    assert digit_runs("error code F twenty eight") == ["28"]
    assert digit_runs("tenants at nine o'clock tomorrow") == ["9"]


def test_a_lone_o_is_a_letter_not_a_zero():
    assert digit_runs("o k then") == []
    assert digit_runs("o seven seven") == ["077"]  # but inside a number it is a zero


def test_teens_and_tens():
    assert digit_runs("sixteen") == ["16"]
    assert digit_runs("forty-five") == ["45"]
    assert digit_runs("twenty") == ["20"]


def test_no_digits_no_runs():
    assert digit_runs("You've got my number.") == []
    assert digit_runs("") == []


def test_spelled_names():
    assert letter_runs("that's S, I, O, B, H, A, N, G, A, double L, A, G, H, E, R") == ["SIOBHANGALLAGHER"]
    assert letter_runs("S-I-O-B-H-A-N.") == ["SIOBHAN"]
    assert letter_runs("S. I. O. B. H. A. N. Gallagher") == ["SIOBHAN"]


def test_ordinary_speech_has_no_letter_runs():
    assert letter_runs("it's a bit of a long story, I'm sure") == []
    assert letter_runs("") == []


def test_double_u_is_the_letter_w_not_two_us():
    assert letter_runs("W, O, J, C, I, E, C, H") == ["WOJCIECH"]
    assert letter_runs("double u, O, J, C, I, E, C, H") == ["WOJCIECH"]
    assert letter_runs("double-u O J C") == ["WOJC"]
    assert letter_runs("double you, O, J, C") == ["WOJC"]
    assert letter_runs("G, A, double L, A, G, H, E, R") == ["GALLAGHER"]  # other doubles still double


def test_plus_44_in_all_its_spellings():
    assert digit_runs("+44 (0) 7700 900123") == ["07700900123"]
    assert digit_runs("plus four four 1632 960501") == ["01632960501"]
    assert digit_runs("plus forty four 1632 960501") == ["01632960501"]


def test_known_limits_lose_a_number_but_do_not_invent_one():
    """Written down so nobody is surprised: the UK number rule rejects these, the dialog re-asks."""
    assert digit_runs("oh, oh one six three two, nine six oh, five oh one") == ["001632960501"]  # 12 digits
    assert digit_runs("two hundred and five") == ["200", "5"]
