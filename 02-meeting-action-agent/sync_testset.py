"""Phase 4 check: run the sync agent over the whole meeting series and compare the tracker with the answer key.

    python 02-meeting-action-agent/sync_testset.py --items gold        # perfect items from labels.json
    python 02-meeting-action-agent/sync_testset.py --items extracted   # the real extraction (x7 cache)

'gold' tests the agent alone: extraction mistakes can't hide (or cause) agent mistakes.
Starts from an empty tracker (testset/tracker_<items>.db), meeting by meeting, then after each meeting
checks every expected task: exactly one tracker task must match it, with the right owner, due and status.
Traces go to testset/traces_<items>/.
"""

import argparse
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "testset"))

import answer_key  # noqa: E402
import context  # noqa: E402
import tracker  # noqa: E402
from agent import sync_meeting  # noqa: E402
from extract import extract  # noqa: E402
from shared.schemas import ExtractedItem  # noqa: E402
from shared.transcribe import transcribe  # noqa: E402

TESTSET = HERE / "testset"


def gold_items(meeting: dict, tasks: dict) -> list[ExtractedItem]:
    """The items a perfect extraction would produce, straight from the answer key."""
    return [ExtractedItem(evidence="(answer key)", task=tasks[m["task"]]["task"], owner=m["owner"], due_text=None,
                          status=m["status"], due=date.fromisoformat(m["due"]) if m["due"] else None)
            for m in meeting["mentions"]]


def extracted_items(meeting: dict) -> list[ExtractedItem]:
    t = transcribe(TESTSET / "audio" / meeting["file"], cache_dir=TESTSET / "transcripts", hint=context.whisper_hint())
    return extract(t, date.fromisoformat(meeting["date"]), cache_dir=TESTSET / "results").items


def score_tracker(conn, expected: dict[str, dict], tasks: dict) -> tuple[int, list[str], list[str]]:
    """(tasks fully right, problems, tracker tasks matching no expected task)."""
    rows = tracker.all_tasks(conn)
    right, problems, claimed = 0, [], set()
    for key, want in expected.items():
        hits = [r for r in rows if answer_key.matches(tasks[key]["match"], r["task"])]
        claimed.update(r["id"] for r in hits)
        if not hits:
            problems.append(f"{key} missing")
            continue
        if len(hits) > 1:
            problems.append(f"{key} duplicated as {[f'#{h['id']}' for h in hits]}")
            continue
        got = hits[0]
        diffs = [f"{f} {got[f]!r} != {want[f]!r}" for f in ("owner", "due", "status") if got[f] != want[f]]
        if diffs:
            problems.append(f"{key} #{got['id']}: " + ", ".join(diffs))
        else:
            right += 1
    spurious = [f"#{r['id']} {r['task']} [{r['status']}]" for r in rows if r["id"] not in claimed]
    return right, problems, spurious


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", choices=["gold", "extracted"], default="gold")
    parser.add_argument("--only", help="stop after the meeting whose id starts with this (e.g. m2)")
    args = parser.parse_args()

    labels = answer_key.load()
    expected_states = answer_key.tracker_states(labels)
    db = TESTSET / f"tracker_{args.items}.db"
    db.unlink(missing_ok=True)  # every run starts from an empty tracker
    conn = tracker.connect(db)
    calls = seconds = 0
    for meeting in labels["meetings"]:
        items = gold_items(meeting, labels["tasks"]) if args.items == "gold" else extracted_items(meeting)
        run = sync_meeting(conn, meeting["id"], date.fromisoformat(meeting["date"]), items,
                           trace_dir=TESTSET / f"traces_{args.items}")
        calls += run.llm_calls
        seconds += run.seconds
        expected = expected_states[meeting["id"]]
        right, problems, spurious = score_tracker(conn, expected, labels["tasks"])
        print(f"\n{meeting['id']}: {len(items)} items, {run.llm_calls} LLM calls, {run.seconds:.0f} s, "
              f"finished={run.finished}, unhandled={run.unhandled()}", flush=True)
        for item_id, action in run.handled.items():
            print(f"  {item_id} -> {action:10} {run.items[item_id].task}")
        print(f"  tracker: {right}/{len(expected)} tasks right" + (f"; spurious: {spurious}" if spurious else ""))
        for p in problems:
            print(f"    {p}")
        if args.only and meeting["id"].startswith(args.only):
            break
    print(f"\nTotal: {calls} LLM calls, {seconds:.0f} s ({seconds / max(calls, 1):.0f} s per call)")


if __name__ == "__main__":
    main()
