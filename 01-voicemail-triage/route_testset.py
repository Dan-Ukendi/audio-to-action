"""Phase 4 check: route the 18 analyzed test voicemails and check the safety rule.

    python 01-voicemail-triage/route_testset.py

Uses the cached transcripts and analysis results (instant, no LLM calls), pushes in dry-run
mode, and writes a throwaway database testset/triage_test.db (rebuilt on every run).
Fails (exit code 1) if any voicemail labelled urgent ends up in the archive.
"""

import json
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from deliver import deliver  # noqa: E402
from shared.analyze import analyze  # noqa: E402
from shared.transcribe import transcribe  # noqa: E402
from store import connect, summary  # noqa: E402

TESTSET = HERE / "testset"
TEST_DB = TESTSET / "triage_test.db"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="  %(message)s")  # shows the [ntfy dry run] lines
    TEST_DB.unlink(missing_ok=True)  # start clean so the summary only shows this run
    conn = connect(TEST_DB)
    labels = json.loads((TESTSET / "labels.json").read_text(encoding="utf-8"))["items"]

    urgent_archived = []
    print(f"{'id':32} {'label':9} {'llm':9} {'route':11} push review")
    for label in labels:
        print(f"{label['id']}:")
        t = transcribe(TESTSET / "audio" / label["file"], cache_dir=TESTSET / "transcripts", model="small")
        r = analyze(t, cache_dir=TESTSET / "results")
        d = deliver(conn, t, r)
        print(f"{'':32} {label['category']:9} {r.analysis.category:9} {d.route:11} "
              f"{'yes' if d.notify else '-':4} {'yes' if d.review else '-'}")
        for reason in d.reasons[1:]:  # [0] is just the category rule
            print(f"{'':34}- {reason}")
        if label["category"] == "urgent" and d.route == "archive":
            urgent_archived.append(label["id"])

    print()
    summary(conn)
    print(f"\nUrgent voicemails routed to the archive: {len(urgent_archived)} {urgent_archived}")
    return 1 if urgent_archived else 0


if __name__ == "__main__":
    sys.exit(main())
