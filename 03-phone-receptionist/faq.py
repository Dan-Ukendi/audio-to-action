"""The approved answers (faq.json): loading and checking. Matching a caller's question comes in Phase 3.

The receptionist may only state facts from this file, word for word. Models are allowed to pick WHICH entry
answers a question, never to write the answer: that keeps invented prices and opening hours out of the call.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_FAQ = HERE / "faq.json"


class FaqError(Exception):
    """faq.json is inconsistent."""


@dataclass(frozen=True)
class FaqEntry:
    id: str
    topic: str
    keywords: tuple[str, ...]
    answer: str
    safety: str | None  # "gas" | "co" | "water" for the emergency advice entries, else None


def load_faq(path: str | Path = DEFAULT_FAQ) -> dict[str, FaqEntry]:
    """id -> entry, in file order. Fails loudly on duplicates, empty fields, upper case or digits in answers."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))["entries"]
    entries: dict[str, FaqEntry] = {}
    for item in raw:
        entry = FaqEntry(id=item["id"], topic=item["topic"], keywords=tuple(item["keywords"]),
                         answer=item["answer"], safety=item.get("safety"))
        if entry.id in entries:
            raise FaqError(f"duplicate FAQ id '{entry.id}'")
        if not entry.keywords or not entry.answer.strip():
            raise FaqError(f"FAQ entry '{entry.id}' needs keywords and an answer")
        if any(k != k.lower() or k != k.strip() for k in entry.keywords):
            raise FaqError(f"FAQ entry '{entry.id}': keywords must be lowercase and trimmed")
        if re.search(r"[\d£$€%&/@#*+=<>]", entry.answer):
            raise FaqError(f"FAQ entry '{entry.id}': write numbers and symbols as words in the spoken answer")
        if entry.safety not in (None, "gas", "co", "water"):
            raise FaqError(f"FAQ entry '{entry.id}': safety must be gas, co or water")
        entries[entry.id] = entry
    return entries
