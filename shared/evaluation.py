"""Small, reusable scoring helpers: compare predictions with labels.

Nothing here knows about voicemails; Part 2 (meeting action items) can score its own fields
with the same functions.
"""

import re
from collections import Counter


def norm_text(value: str | None) -> str:
    """Lowercase, punctuation to spaces, single spaces: 'O'Brien,' -> 'o brien'."""
    return " ".join(re.sub(r"[^\w\s]", " ", (value or "").lower()).split())


def same_value(expected: str | None, got: str | None) -> bool:
    """Exact match after normalizing; null must match null."""
    if expected is None or got is None:
        return expected is got
    return norm_text(expected) == norm_text(got)


def same_first_word(expected: str | None, got: str | None) -> bool:
    """Lenient name check: 'Helen' for 'Helen Carter' counts. Null must still match null."""
    if expected is None or got is None:
        return expected is got
    a, b = norm_text(expected).split(), norm_text(got).split()
    return bool(a and b) and a[0] == b[0]


def null_aware(expected: str | None, got: str | None) -> str:
    """Classify an extracted value: correct | wrong | missed (said, not found) | invented (not said, but given)."""
    if expected is None and got is None:
        return "correct"
    if expected is None:
        return "invented"
    if got is None:
        return "missed"
    return "correct" if same_value(expected, got) else "wrong"


def confusion(pairs: list[tuple[str, str]], classes: list[str]) -> str:
    """Markdown confusion matrix: rows = true label, columns = prediction."""
    counts = Counter(pairs)
    lines = ["| label \\ predicted | " + " | ".join(classes) + " |", "|---" * (len(classes) + 1) + "|"]
    for true in classes:
        cells = [str(counts[(true, pred)]) if counts[(true, pred)] else "." for pred in classes]
        lines.append(f"| **{true}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pct(part: int, whole: int) -> str:
    return f"{part}/{whole} ({part / whole:.0%})" if whole else "n/a"
