"""Phase 3 check: analyze every test voicemail and compare with labels.json.

    python 01-voicemail-triage/analyze_testset.py                       # small transcripts
    python 01-voicemail-triage/analyze_testset.py --whisper large-v3-turbo

A quick look, not the full evaluation (that's Phase 6). Transcripts and results are cached
in testset/transcripts/ and testset/results/, so a second run is instant.
"""

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # so 'shared' is importable when run as a script

from shared.analyze import analyze  # noqa: E402
from shared.transcribe import transcribe  # noqa: E402

TESTSET = HERE / "testset"


def same_name(expected: str | None, got: str | None) -> bool:
    """Case and punctuation don't matter; null must match null."""
    if expected is None or got is None:
        return expected is got
    norm = lambda s: " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())  # noqa: E731
    return norm(expected) == norm(got)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper", default="small", help="which cached transcripts to analyze")
    args = parser.parse_args()

    labels = json.loads((TESTSET / "labels.json").read_text(encoding="utf-8"))["items"]
    totals = {"category": 0, "urgency": 0, "name": 0, "number": 0}
    urgent_missed, retries, seconds = [], 0, 0.0

    print(f"{'id':32} {'category':20} {'urg':5} {'name':4} {'number':6} try")
    for label in labels:
        t = transcribe(TESTSET / "audio" / label["file"], cache_dir=TESTSET / "transcripts", model=args.whisper)
        r = analyze(t, cache_dir=TESTSET / "results")
        a = r.analysis
        ok = {
            "category": a.category == label["category"],
            "urgency": a.urgency == label["urgency"],
            "name": same_name(label["caller_name"], a.caller_name),
            "number": a.callback_number == label["callback_number"],
        }
        for key, good in ok.items():
            totals[key] += good
        if label["category"] == "urgent" and a.category != "urgent":
            urgent_missed.append(label["id"])
        retries += r.attempts - 1
        seconds += r.analyze_s

        cat = a.category if ok["category"] else f"{a.category}!={label['category']}"
        urg = str(a.urgency) if ok["urgency"] else f"{a.urgency}!={label['urgency']}"
        print(f"{label['id']:32} {cat:20} {urg:5} {'ok' if ok['name'] else 'NO':4} "
              f"{'ok' if ok['number'] else 'NO':6} {r.attempts}", flush=True)
        if not (ok["name"] and ok["number"]):
            print(f"{'':34}expected name={label['caller_name']!r} number={label['callback_number']!r}")
            print(f"{'':34}got      name={a.caller_name!r} number={a.callback_number!r}")

    n = len(labels)
    print("\n" + "  ".join(f"{k} {v}/{n}" for k, v in totals.items()))
    print(f"urgent voicemails missed: {len(urgent_missed)} {urgent_missed}")
    print(f"retries needed: {retries}   LLM time: {seconds:.0f}s ({seconds / n:.0f}s per voicemail)")


if __name__ == "__main__":
    main()
