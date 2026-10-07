"""Phase 4 check: run the sync agent over the whole meeting series and compare the tracker with the answer key.

    python 02-meeting-action-agent/sync_testset.py --items gold        # perfect items from labels.json
    python 02-meeting-action-agent/sync_testset.py --items extracted   # the real extraction (x7 cache)
    python 02-meeting-action-agent/sync_testset.py --items gold --sync rules   # plain-code sync, no LLM

'gold' tests the agent alone: extraction mistakes can't hide (or cause) agent mistakes.
Starts from an empty tracker (testset/tracker_<items>_<sync>.db), meeting by meeting, then after each meeting
checks every expected task: exactly one tracker task must match it, with the right owner, due and status.
Agent traces go to testset/traces_<items>/ (an --only run overwrites them and the db). A summary of every
full run is saved to testset/sync_runs/<items>_<sync>.json for evaluate.py (agent runs take ~40 min on CPU,
so the report reads saved runs instead of re-running them).
"""

import argparse
import json
import sys
import time
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
from extract import PROMPT_VERSION, extract  # noqa: E402
from rules_sync import rules_sync  # noqa: E402
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
    parser.add_argument("--sync", choices=["agent", "rules"], default="agent")
    parser.add_argument("--only", help="stop after the meeting whose id starts with this (e.g. m2)")
    args = parser.parse_args()

    labels = answer_key.load()
    expected_states = answer_key.tracker_states(labels)
    db = TESTSET / f"tracker_{args.items}_{args.sync}.db"
    db.unlink(missing_ok=True)  # every run starts from an empty tracker
    conn = tracker.connect(db)
    calls = seconds = 0
    summary = {"items": args.items, "sync": args.sync, "extraction_prompt": PROMPT_VERSION, "meetings": []}
    for meeting in labels["meetings"]:
        items = gold_items(meeting, labels["tasks"]) if args.items == "gold" else extracted_items(meeting)
        day = date.fromisoformat(meeting["date"])
        started = time.perf_counter()
        if args.sync == "agent":
            run = sync_meeting(conn, meeting["id"], day, items, trace_dir=TESTSET / f"traces_{args.items}")
            run_calls, run_seconds, review = run.llm_calls, run.seconds, run.unhandled()
            actions = [f"{i} -> {a:10} {run.items[i].task}" for i, a in run.handled.items()]
        else:
            review = rules_sync(conn, meeting["id"], day, items)
            run_calls, run_seconds = 0, time.perf_counter() - started
            actions = [f"{c['action']:6} #{c['task_id']}: {c['reason']}" for c in tracker.changes(conn, meeting["id"])]
        calls += run_calls
        seconds += run_seconds
        expected = expected_states[meeting["id"]]
        right, problems, spurious = score_tracker(conn, expected, labels["tasks"])
        print(f"\n{meeting['id']}: {len(items)} items, {run_calls} LLM calls, {run_seconds:.0f} s, review={review}",
              flush=True)
        for line in actions:
            print(f"  {line}")
        print(f"  tracker: {right}/{len(expected)} tasks right" + (f"; spurious: {spurious}" if spurious else ""))
        for p in problems:
            print(f"    {p}")
        summary["meetings"].append({"id": meeting["id"], "right": right, "expected": len(expected),
                                    "problems": problems, "spurious": spurious, "llm_calls": run_calls,
                                    "seconds": round(run_seconds, 1), "review": review})
        if args.only and meeting["id"].startswith(args.only):
            break
    print(f"\nTotal: {calls} LLM calls, {seconds:.0f} s ({seconds / max(calls, 1):.0f} s per call)")
    if not args.only:  # only full series are comparable
        out = TESTSET / "sync_runs" / f"{args.items}_{args.sync}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved: {out.relative_to(HERE)}")


if __name__ == "__main__":
    main()
