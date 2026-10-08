"""Emergency detection for the fast path: which safety words were said, and which advice entry applies.

    emergency_from_text("There's a smell of gas in the hallway")  -> Emergency(hits=["smell of gas"], kinds=["gas"])

Part 1's SAFETY_PATTERNS (routing.py) are the safety net, reused as they are, so a voicemail and a phone call agree
on what counts as danger. The kinds map those words to the three pieces of real UK advice in faq.json:
gas, carbon monoxide ("co") and water. Sparks, "electrics", "no heating" and "no hot water" raise the alarm
without a specific advice entry: the urgent acknowledgement is all Holly says about them.
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-voicemail-triage"))
sys.path.insert(0, str(HERE.parent))

from routing import safety_hits  # noqa: E402  (Part 1's rules: the same words decide for voicemails and calls)

KIND_PATTERNS = {
    "gas": r"\bsmell(s|ing)? (of )?gas\b|\bgas (leak|smell)\b|\bsmells? like gas\b",
    "co": r"\bcarbon monoxide\b|\bco (alarm|detector)\b",
    "water": r"\bflood(ed|ing)?\b|\bburst\b|\bwater (is )?everywhere\b|\bthrough the ceiling\b|\bwater leak\b|\bleaking (badly|everywhere)\b",
}


@dataclass
class Emergency:
    hits: list[str] = field(default_factory=list)   # the safety words found (Part 1's patterns)
    kinds: list[str] = field(default_factory=list)  # which advice entries apply: gas / co / water

    def __bool__(self) -> bool:
        return bool(self.hits or self.kinds)


def emergency_from_text(text: str) -> Emergency:
    lowered = text.lower()
    kinds = [kind for kind, pattern in KIND_PATTERNS.items() if re.search(pattern, lowered)]
    return Emergency(hits=safety_hits(text), kinds=kinds)
