"""What the business already knows before any meeting: who's on the team, regular customers, places.

Used as a spelling hint for Whisper (Phase 2) and as context for extraction (Phase 3: owners must
be team members). Only proper nouns: putting task words like "quote" in the hint would make the
transcripts look better on exactly the words we score, which is cheating ourselves.
"""

BUSINESS = "Brightwater Plumbing & Heating"
TEAM = ["Sam", "Priya", "Tom", "Jamie"]
CUSTOMERS = ["Siobhan Gallagher", "Margaret Ellis", "Dave"]
PLACES = ["Mill Lane", "Station Road"]


def whisper_hint() -> str:
    """Reads like the start of a meeting, which is how Whisper uses its initial prompt."""
    return (f"{BUSINESS} weekly team meeting with {', '.join(TEAM[:-1])} and {TEAM[-1]}. "
            f"Customers: {', '.join(CUSTOMERS)}. Jobs at {' and '.join(PLACES)}.")
