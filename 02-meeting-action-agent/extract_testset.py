"""Phase 3 check: extract action items from the 5 meetings and compare with the answer key.

    python 02-meeting-action-agent/extract_testset.py

Uses the default transcripts (Whisper small + name hint, cached) and caches results in testset/results/.
A quick look, not the full evaluation (that's Phase 6).
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
from extract import PROMPT_VERSION, extract  # noqa: E402
from shared.evaluation import pct  # noqa: E402
from shared.transcribe import transcribe  # noqa: E402

TESTSET = HERE / "testset"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=PROMPT_VERSION, help="re-score an older cached prompt version")
    args = parser.parse_args()
    labels = answer_key.load()
    tasks = labels["tasks"]
    n_items = n_mentions = n_pairs = 0
    owner_ok = owner_text = owner_text_ok = due_ok = status_ok = calls = 0
    seconds = 0.0
    for meeting in labels["meetings"]:
        t = transcribe(TESTSET / "audio" / meeting["file"], cache_dir=TESTSET / "transcripts", hint=context.whisper_hint())
        r = extract(t, date.fromisoformat(meeting["date"]), cache_dir=TESTSET / "results", prompt_version=args.version)
        pairs, extra, missed = answer_key.match_items([i.task for i in r.items], meeting["mentions"], tasks)
        print(f"\n{meeting['id']}: {len(r.items)} items, {len(pairs)} matched, {len(extra)} extra, "
              f"{len(missed)} missed  ({r.attempts} call(s), {r.extract_s:.0f} s)", flush=True)
        for index, m in pairs:
            item = r.items[index]
            due = None if item.due is None else item.due.isoformat()
            got = {"owner": item.owner, "due": due, "status": item.status}
            checks = {field: got[field] == m[field] for field in got}
            wrong = ", ".join(f"{field}: got {got[field]!r} want {m[field]!r}" for field in got if not checks[field])
            print(f"  {m['task']:4} {'ok' if all(checks.values()) else 'XX'}  {item.task}" + (f"  [{wrong}]" if wrong else ""))
            owner_ok += checks["owner"]
            due_ok += checks["due"]
            status_ok += checks["status"]
            if m.get("owner_from_text", True):
                owner_text += 1
                owner_text_ok += checks["owner"]
        for index in extra:
            print(f"  EXTRA    {r.items[index].task}  (status {r.items[index].status})")
        for m in missed:
            print(f"  MISSED {m['task']:4} {tasks[m['task']]['task']}")
        n_items += len(r.items)
        n_mentions += len(meeting["mentions"])
        n_pairs += len(pairs)
        calls += r.attempts
        seconds += r.extract_s

    print(f"\nprecision {pct(n_pairs, n_items)}   recall {pct(n_pairs, n_mentions)}   (task matches)")
    # The Definition of done counts an item only if task AND owner match: the stricter number.
    print(f"precision {pct(owner_ok, n_items)}   recall {pct(owner_ok, n_mentions)}   (task + owner, as in the DoD)")
    print(f"on matched items: owner {pct(owner_ok, n_pairs)} (where the text says who: {pct(owner_text_ok, owner_text)}), "
          f"due {pct(due_ok, n_pairs)}, status {pct(status_ok, n_pairs)}")
    print(f"LLM calls {calls}, {seconds:.0f} s total ({seconds / len(labels['meetings']):.0f} s per meeting)")


if __name__ == "__main__":
    main()
