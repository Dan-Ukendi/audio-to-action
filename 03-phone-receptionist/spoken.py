"""Turning what people SAY into data: numbers said as words, letters spelled out.

    digit_runs("oh seven seven double oh, nine hundred, one two three")   -> ["07700900123"]
    digit_runs("my number is 01632 960 501")                              -> ["01632960501"]
    letter_runs("that's S, I, O, B, H, A, N")                             -> ["SIOBHAN"]

Whisper writes numbers either as digits or as words, and callers say "oh", "double oh" and "nine hundred".
Both the simulated callers (to check what the receptionist read back) and the dialog (to check that a number
the model reports was really said) need the same conversion, so it lives in one place and is tested on its own.

Known limits (they lose a number, they never invent one; the UK number rule rejects the result): "oh, oh one six..."
(a doubled leading zero) gives 12 digits, and "two hundred and five" gives 200 and 5. "Oh" as an interjection reads as a
zero, so always ask for runs of at least 8 digits when looking for a phone number.

A "run" is a stretch of consecutive number words/digits. Any other word ends it ("no sorry" splits a corrected
number into two runs), so a run is never glued together from separate statements.
"""

import re
from difflib import SequenceMatcher

UNITS = {"zero": "0", "oh": "0", "o": "0", "nought": "0", "naught": "0", "one": "1", "two": "2", "three": "3",
         "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
TEENS = {"ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
         "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19"}
TENS = {"twenty": "2", "thirty": "3", "forty": "4", "fifty": "5", "sixty": "6", "seventy": "7", "eighty": "8",
        "ninety": "9"}
REPEATERS = {"double": 2, "triple": 3}


def number_at(tokens: list[str], i: int, in_run: bool) -> tuple[str, int] | None:
    """The digits that start at tokens[i] and how many tokens they use, or None if it is not a number."""
    token = tokens[i]
    after = tokens[i + 1] if i + 1 < len(tokens) else ""
    if token.isdigit():
        return token, 1
    if token in REPEATERS and after in UNITS | {d: d for d in "0123456789"}:
        digit = UNITS.get(after, after)
        return digit * REPEATERS[token], 2  # "double oh" -> 00
    if token in UNITS:
        if token == "o" and not in_run and not (after in UNITS or after.isdigit() or after in REPEATERS):
            return None  # a lone "o" is a letter, not a zero
        if after == "hundred" and UNITS[token] != "0":
            return UNITS[token] + "00", 2  # "nine hundred" -> 900
        return UNITS[token], 1
    if token in TEENS:
        return TEENS[token], 1
    if token in TENS:
        if after in UNITS and UNITS[after] != "0":
            return TENS[token] + UNITS[after], 2  # "forty five" -> 45
        return TENS[token] + "0", 1
    return None


def digit_runs(text: str, min_len: int = 1) -> list[str]:
    """Every run of spoken or written digits in the text, in order, keeping runs of at least min_len digits."""
    # "+44 1632 960501", "+44 (0) 1632 960501" and "plus four four 1632 960501" are all 01632 960501.
    lowered = re.sub(r"\(\s*0\s*\)", " ", text.lower())
    lowered = lowered.replace("+44", " 0 ")
    lowered = re.sub(r"\bplus\s+(four\s+four|forty[\s-]+four)\b", " 0 ", lowered)
    tokens = re.findall(r"\d+|[a-z']+", lowered)
    runs: list[str] = []
    current = ""
    i = 0
    while i < len(tokens):
        found = number_at(tokens, i, in_run=bool(current))
        if found:
            digits, used = found
            current += digits
            i += used
        else:
            if current:
                runs.append(current)
            current = ""
            i += 1
    if current:
        runs.append(current)
    return [r for r in runs if len(r) >= min_len]


def letter_runs(text: str, min_len: int = 3) -> list[str]:
    """Names spelled letter by letter: 'S, I, O, B, H, A, N' -> 'SIOBHAN'; 'G, A, double L, A' -> 'GALLA'.

    At least min_len single letters in a row, so the words "a" and "I" in ordinary speech never count.
    """
    tokens = re.findall(r"[a-z']+", text.lower())
    runs: list[str] = []
    current = ""
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if len(token) == 1:
            current += token.upper()
            i += 1
        elif token == "double" and i + 1 < len(tokens) and tokens[i + 1] in ("u", "you"):
            current += "W"  # "double u" is the NAME of the letter W (what Piper says and Whisper may write), not UU
            i += 2
        elif token == "double" and i + 1 < len(tokens) and len(tokens[i + 1]) == 1:
            current += tokens[i + 1].upper() * 2
            i += 2
        else:
            if len(current) >= min_len:
                runs.append(current)
            current = ""
            i += 1
    if len(current) >= min_len:
        runs.append(current)
    return runs


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def apply_spelling(heard: str | None, letters: str) -> str:
    """Combine the name Whisper heard with the letters the caller spelled; the spelling wins where it can be placed.

        apply_spelling("Shivawn Gallagher", "SIOBHANGALLAGHER") -> "Siobhan Gallagher"

    The caller may spell the whole name in one run of letters (the receptionist asks for the full name), or only a
    part. The last word heard is taken as the surname (Whisper is usually right about it); everything before it is
    the first name. Which part the letters are:
      - about as long as the whole name: split them where they best match the first name and the surname heard;
      - about as long as the surname only (or the first name only): replace just that part.
    Letters that fit nothing are not used (the heard name is returned unchanged): better a misheard name than a
    mangled one.
    """
    spelled = letters.strip().lower()
    tokens = (heard or "").split()
    if not spelled:
        return heard or ""
    if not tokens:
        return spelled.title()
    if len(tokens) == 1:
        return spelled.title() if abs(len(spelled) - len(tokens[0])) <= 3 else heard

    first, last = " ".join(tokens[:-1]), tokens[-1]
    first_flat = first.replace(" ", "")
    total = len(spelled)
    off_full = abs(total - (len(first_flat) + len(last)))
    off_last = abs(total - len(last))
    off_first = abs(total - len(first_flat))
    nearest = min(off_full, off_last, off_first)
    if nearest > 3:
        return heard
    if off_full == nearest:
        best_k, best_score = None, -1.0
        for k in range(2, total - 1):
            score = similarity(spelled[:k], first_flat) + similarity(spelled[k:], last) - 0.01 * abs(k - len(first_flat))
            if score > best_score:
                best_k, best_score = k, score
        return f"{spelled[:best_k].title()} {spelled[best_k:].title()}"
    if off_last == nearest:
        return f"{first} {spelled.title()}"
    return f"{spelled.title()} {last}"
