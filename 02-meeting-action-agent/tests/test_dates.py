"""The date resolver against every deadline phrase in the meeting series (+ a few edge cases).

    python -m pytest 02-meeting-action-agent/tests -v
"""

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 02-meeting-action-agent

from dates import resolve_due  # noqa: E402

M1, M2, M3, M4, M5 = (date(2026, 9, 7), date(2026, 9, 14), date(2026, 9, 21), date(2026, 9, 28), date(2026, 10, 5))


@pytest.mark.parametrize("words, meeting, expected", [
    ("by Wednesday", M1, date(2026, 9, 9)),
    ("this afternoon", M1, M1),
    ("on Thursday", M1, date(2026, 9, 10)),
    ("at the end of the month", M1, date(2026, 9, 30)),
    ("Friday", M2, date(2026, 9, 18)),
    ("so Friday", M2, date(2026, 9, 18)),
    ("tomorrow", M2, date(2026, 9, 15)),
    ("before the end of the month", M2, date(2026, 9, 30)),
    ("by Thursday", M3, date(2026, 9, 24)),
    ("by Wednesday", M3, date(2026, 9, 23)),
    ("due on the 30th", M3, date(2026, 9, 30)),
    ("on the thirtieth", M3, date(2026, 9, 30)),
    ("by Wednesday", M4, date(2026, 9, 30)),
    ("on the 2nd", M4, date(2026, 10, 2)),
    ("Friday the 2nd of October", M4, date(2026, 10, 2)),
    ("starting Monday the 12th", M5, date(2026, 10, 12)),
    ("on the 12th", M5, date(2026, 10, 12)),
    ("by the end of the month", M5, date(2026, 10, 31)),
    ("at some point", M3, None),
    ("next week", M3, None),
    (None, M3, None),
    # edge cases
    ("Monday", M1, date(2026, 9, 14)),            # said on a Monday: next Monday, not today
    ("next Friday", M1, date(2026, 9, 18)),       # the Friday after this week's
    ("the 5th", M2, date(2026, 10, 5)),           # day already past this month: next month
    ("the 31st", M3, date(2026, 10, 31)),         # September has no 31st: October
    ("a second", M3, None),                       # "second" without "the" isn't a date
    ("I may do it by the 25th", M3, date(2026, 9, 25)),  # "may" the verb, not the month
    ("the 3rd of May", M3, date(2027, 5, 3)),     # the month
    ("for now", M3, None),                        # not "today"
    ("the day after tomorrow", M3, date(2026, 9, 23)),
])
def test_resolve_due(words, meeting, expected):
    assert resolve_due(words, meeting) == expected
