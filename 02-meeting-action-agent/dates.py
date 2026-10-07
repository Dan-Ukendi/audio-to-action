"""Turn the words for a deadline ("by Wednesday", "on the 30th") into a date, relative to the meeting.

    resolve_due("by Wednesday", date(2026, 9, 7)) -> date(2026, 9, 9)

Plain code on purpose: calendar arithmetic is exactly the kind of thing LLMs get wrong now and then,
and code never does. The LLM copies the words; this file decides what they mean. Same rules as the
answer key's conventions (labels.json). Returns None when the words don't name a day ("at some point").
"""

import calendar
import re
from datetime import date, timedelta

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = [m.lower() for m in calendar.month_name if m]  # "january" ... "december"
ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
            "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
            "eighteenth", "nineteenth", "twentieth", "twenty first", "twenty second", "twenty third",
            "twenty fourth", "twenty fifth", "twenty sixth", "twenty seventh", "twenty eighth",
            "twenty ninth", "thirtieth", "thirty first"]
# Not a bare "now": "let's leave it for now" means the opposite of "today".
SAME_DAY = ("today", "this afternoon", "this morning", "this evening", "tonight", "right away", "right now")


def normalise(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower().replace("-", " ")).split())


def month_pattern(name: str) -> str:
    """'may' is also a verb ("I may do it by the 25th"): count it only next to a day ('of May', 'May 3rd')."""
    if name == "may":
        return r"\bof may\b|\b\d{1,2}(st|nd|rd|th) may\b|\bmay (the )?\d"
    return rf"\b{name}\b"


def day_of_month(text: str) -> tuple[int, int | None] | None:
    """(day, month or None) for '30th', 'the thirtieth', 'the 2nd of October', 'October 2nd'."""
    month = next((i for i, name in enumerate(MONTHS, start=1) if re.search(month_pattern(name), text)), None)
    digits = re.search(r"\b(\d{1,2})(st|nd|rd|th)\b", text)
    if digits:
        return int(digits.group(1)), month
    # Ordinal words only after "the" ("the second"), so "a second" or "second time" don't count.
    for day in range(len(ORDINALS), 0, -1):  # longest first: "twenty second" before "second"
        if re.search(rf"\bthe {ORDINALS[day - 1]}\b", text):
            return day, month
    return None


def from_day_of_month(day: int, month: int | None, meeting: date) -> date | None:
    """The first such day on or after the meeting: this month or the next (or that month, this year or next)."""
    if month:
        candidates = [(meeting.year, month), (meeting.year + 1, month)]
    else:
        next_month = meeting.month % 12 + 1
        candidates = [(meeting.year, meeting.month), (meeting.year + (next_month == 1), next_month)]
    for year, candidate_month in candidates:
        try:
            result = date(year, candidate_month, day)
        except ValueError:  # e.g. "the 31st" in a 30-day month
            continue
        if result >= meeting:
            return result
    return None


def resolve_due(text: str | None, meeting: date) -> date | None:
    if not text:
        return None
    t = normalise(text)

    explicit = day_of_month(t)  # "Monday the 12th": the number wins over the weekday
    if explicit:
        return from_day_of_month(*explicit, meeting)
    if any(re.search(rf"\b{phrase}\b", t) for phrase in SAME_DAY):
        return meeting
    if re.search(r"\bday after tomorrow\b", t):
        return meeting + timedelta(days=2)
    if re.search(r"\btomorrow\b", t):
        return meeting + timedelta(days=1)
    if re.search(r"\bend of (the )?month\b", t):
        return date(meeting.year, meeting.month, calendar.monthrange(meeting.year, meeting.month)[1])
    if re.search(r"\bend of (the )?week\b", t):
        return meeting + timedelta(days=(4 - meeting.weekday()) % 7)  # Friday
    for index, name in enumerate(WEEKDAYS):
        if re.search(rf"\b{name}\b", t):
            ahead = (index - meeting.weekday()) % 7 or 7  # "Monday" said on a Monday = next Monday
            if re.search(rf"\bnext {name}\b", t) and ahead < 7:
                ahead += 7  # "next Friday" = the Friday after this week's
            return meeting + timedelta(days=ahead)
    return None
