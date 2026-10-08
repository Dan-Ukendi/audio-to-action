"""Phase 6: score the receptionist on the caller cards, and decide version A or B by the rule written down BEFORE any run.

    python 03-phone-receptionist/evaluate.py --split score --understand model --decide both --repeat-a 2 --write-docs
    python 03-phone-receptionist/evaluate.py --split dev --understand rules      # no model: only checks the harness itself

What is scored, per call, against the answer key in testset/callers.json (written before any dialog code):
  - name and number: correct / wrong / missed / invented (shared.evaluation.null_aware; numbers compared as digits)
  - invented numbers: a recorded number that is not made of digits the caller said. Must be 0.
  - urgent: flagged during the call or not (missed / false alarm), and the safety advice that card expects
  - FAQ: the expected topics were answered, nothing else was answered, an unknown question was passed on
  - sentences: every reply is made of approved pieces only (persona.json lines, faq.json answers)
  - outcome of the call, number of turns, time per turn (understand + decide, plus listen + speak in audio mode)
  - agent only: turns where the agent failed and the state machine decided (fallbacks)
  - hand-off: the call goes through handoff.py into an in-memory copy of Part 1's table; routing and push are compared with the
    card's Part 1 label (urgent calls pushed, nothing else pushed)

Numbers from a run WITHOUT a model (--understand rules) only check that this harness works; they are never written to
docs/part3-eval-results.md. That file holds numbers only after a model run on the laptop (--write-docs).
"""

import argparse
import json
import logging
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "01-voicemail-triage"))

import call as call_module  # noqa: E402
import dialog  # noqa: E402
import handoff  # noqa: E402
from cards import Card, load_cards  # noqa: E402
from faq import FaqEntry, load_faq  # noqa: E402
from persona import Persona, load_persona  # noqa: E402
from shared.evaluation import median, null_aware, pct, percentile  # noqa: E402
from spoken import digit_runs  # noqa: E402
from store import connect  # noqa: E402  (Part 1)

RESULTS_FILE = ROOT / "docs" / "part3-eval-results.md"
LOGS = HERE / "logs"

# The pre-registered rule (README section 4). Constants here, so the code cannot drift from the text without a test noticing.
MIN_EXTRA_DETAILS = 2        # B must get at least this many more correct details (name + number) than A
MAX_EXTRA_SECONDS = 3.0      # B's median latency may be at most A's + this
MAX_FALLBACK_SHARE = 0.10    # B's turns decided by the state machine because the agent failed
DOD_PERCENT = 80             # name, number, FAQ: at least this share correct
TARGET_TURN_SECONDS = 5.0    # median receptionist reply (audio runs only)


# ---------------------------------------------------------------- scoring one call

def digits_of(value: str | None) -> str | None:
    return re.sub(r"\D", "", value) or None if value else None


def approved_pattern(persona: Persona, faq: dict[str, FaqEntry]) -> re.Pattern:
    """A reply is approved if it is made of persona lines (placeholders may be anything) and faq answers, nothing else."""
    pieces = [re.escape(e.answer) for e in faq.values()]
    pieces += [re.escape(re.sub(r"\{(\w+)\}", "\x00", t)).replace("\x00", ".+?") for t in persona.lines.values()]
    one = "(?:" + "|".join(pieces) + ")"
    return re.compile(f"{one}(?: {one})*")


def invented_number(record) -> bool:
    """True when the recorded number is not made of digits the caller said, in order (a number corrected over several turns is
    built from digits of several turns, so all turns are joined)."""
    recorded = digits_of(record.message.get("number"))
    if recorded is None:
        return False
    runs_per_turn = [digit_runs(t.get("caller_text") or "", 1) for t in record.turns]
    said = [run for runs in runs_per_turn for run in runs]
    return not (recorded in said or recorded in "".join(said))


def turn_seconds(entry: dict) -> float:
    """One reply's time: understanding + deciding, plus listening and speaking when the call was audio."""
    return sum(entry.get(k) or 0.0 for k in ("understand_s", "decide_s", "listen_s", "speak_s"))


def answered_topics(message: dict) -> set[str]:
    return set(message["faq_answered"]) | {f"safety_{k}" for k in message["safety_advised"]}


def expected_topics(card: Card) -> set[str]:
    topics = set(card.expect.faq_topics)
    if card.expect.safety:
        topics.add(f"safety_{card.expect.safety}")
    return topics


def score_call(card: Card, record, pattern: re.Pattern, hand: handoff.HandOff | None = None, pushes: int = 0) -> dict:
    m = record.message
    name_class = null_aware(card.facts.name, m.get("name"))
    number_class = null_aware(digits_of(card.facts.number), digits_of(m.get("number")))
    want, got = expected_topics(card), answered_topics(m)
    faq_card = bool(want) or card.expect.faq_unknown
    unknown_ok = bool(m["unanswered_questions"]) == card.expect.faq_unknown
    traces = [t["decide_trace"] for t in record.turns if t.get("decide_trace")]
    bad_replies = [t["reply"] for t in record.turns if t.get("reply") and not pattern.fullmatch(t["reply"])]
    score = {
        "card": card.id, "split": card.split, "outcome": record.outcome, "outcome_ok": record.outcome == card.expect.outcome,
        "name": name_class, "number": number_class, "details_correct": (name_class == "correct") + (number_class == "correct"),
        "invented_number": invented_number(record), "urgent_expected": card.expect.urgent_flag, "urgent_flagged": m["urgent"],
        "missed_urgent": card.expect.urgent_flag and not m["urgent"], "false_urgent": m["urgent"] and not card.expect.urgent_flag,
        "faq_card": faq_card, "faq_ok": faq_card and want <= got and unknown_ok,
        "faq_missing": sorted(want - got), "faq_extra": sorted(got - want), "unknown_ok": unknown_ok,
        "non_approved_replies": bad_replies, "turns": len(record.turns), "cut_off": record.cut_off,
        "turn_seconds": [round(turn_seconds(t), 3) for t in record.turns],
        "agent_turns": len(traces), "fallback_turns": sum(1 for t in traces if t["fallback"]),
        "recorded": {"name": m.get("name"), "number": m.get("number"), "reason": m.get("reason")},
    }
    if hand is not None and card.labels is not None:
        want_push = card.labels.category == "urgent"
        score["handoff"] = {"route": hand.decision.route, "notify": hand.decision.notify, "review": hand.decision.review,
                            "category": hand.result.analysis.category, "category_ok": hand.result.analysis.category == card.labels.category,
                            "pushes_sent": pushes, "missed_push": want_push and not hand.decision.notify,
                            "false_push": hand.decision.notify and not want_push}
    return score


# ---------------------------------------------------------------- running a whole set

def run_set(cards: list[Card], persona: Persona, faq: dict[str, FaqEntry], understand_fn, decide_label: str, make_decide,
            understand_label: str = "rules", prompt_version: str | None = None, audio: bool = False,
            save_dir: Path | None = None, with_handoff: bool = True) -> list[dict]:
    """Run every card once. `make_decide()` returns a fresh decide function per call (the agent keeps a trace)."""
    pattern = approved_pattern(persona, faq)
    conn = connect(":memory:") if with_handoff else None
    scores = []
    for card in cards:
        pushes: list[str] = []
        sender = lambda title, message, priority="high", tags="": (pushes.append(title), "sent")[1]  # noqa: E731
        live = handoff.LivePush(sender)
        channel = call_module.audio_channel_for(card, persona) if audio else None
        record = call_module.run_call(card, persona, faq, understand_fn, channel=channel, decide_fn=make_decide(), on_urgent=live,
                                      understand_label=understand_label, prompt_version=prompt_version, decide_label=decide_label)
        hand = handoff.hand_off(record, conn, live=live, sender=sender, use_model=understand_label == "model") if conn else None
        if save_dir:
            call_module.save_record(record, save_dir, name=f"eval-{decide_label}-{record.call_id}.json")
        scores.append(score_call(card, record, pattern, hand, len(pushes)))
    return scores


def summarize(scores: list[dict], audio: bool = False) -> dict:
    n = len(scores)
    seconds = [s for sc in scores for s in sc["turn_seconds"]]
    agent_turns = sum(sc["agent_turns"] for sc in scores)
    faq_cards = [sc for sc in scores if sc["faq_card"]]
    hands = [sc["handoff"] for sc in scores if "handoff" in sc]
    return {
        "calls": n, "audio": audio,
        "name_correct": sum(sc["name"] == "correct" for sc in scores), "number_correct": sum(sc["number"] == "correct" for sc in scores),
        "details_correct": sum(sc["details_correct"] for sc in scores),
        "name_classes": count_by(scores, "name"), "number_classes": count_by(scores, "number"),
        "invented_numbers": sum(sc["invented_number"] for sc in scores),
        "missed_urgent": sum(sc["missed_urgent"] for sc in scores), "false_urgent": sum(sc["false_urgent"] for sc in scores),
        "faq_cards": len(faq_cards), "faq_ok": sum(sc["faq_ok"] for sc in faq_cards),
        "faq_extra_answers": sum(len(sc["faq_extra"]) for sc in scores),
        "non_approved_replies": sum(len(sc["non_approved_replies"]) for sc in scores),
        "outcome_ok": sum(sc["outcome_ok"] for sc in scores), "cut_off": sum(sc["cut_off"] for sc in scores),
        "turns": sum(sc["turns"] for sc in scores),
        "latency_p50": median(seconds), "latency_p95": percentile(seconds, 0.95),
        "agent_turns": agent_turns, "fallback_turns": sum(sc["fallback_turns"] for sc in scores),
        "fallback_share": (sum(sc["fallback_turns"] for sc in scores) / agent_turns) if agent_turns else None,
        "handoff_calls": len(hands), "missed_push": sum(h["missed_push"] for h in hands), "false_push": sum(h["false_push"] for h in hands),
        "category_ok": sum(h["category_ok"] for h in hands),
        "completed_calls": sum(sc["outcome"] == "completed" for sc in scores),
    }


def count_by(scores: list[dict], key: str) -> dict:
    counts: dict[str, int] = {}
    for sc in scores:
        counts[sc[key]] = counts.get(sc[key], 0) + 1
    return counts


# ---------------------------------------------------------------- the decisions

def share(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0


def dod_table(s: dict) -> list[tuple[str, str, str]]:
    """(criterion, result, evidence): PASS / FAIL, or NOT MEASURED where this run cannot tell."""
    def check(ok: bool) -> str:
        return "PASS" if ok else "FAIL"
    rows = [
        (f"Callback number exactly right >= {DOD_PERCENT} %", check(share(s["number_correct"], s["calls"]) >= DOD_PERCENT),
         pct(s["number_correct"], s["calls"])),
        (f"Name right >= {DOD_PERCENT} %", check(share(s["name_correct"], s["calls"]) >= DOD_PERCENT),
         pct(s["name_correct"], s["calls"])),
        ("0 invented numbers", check(s["invented_numbers"] == 0), str(s["invented_numbers"])),
        ("Every urgent caller flagged during the call (0 missed)", check(s["missed_urgent"] == 0), f"missed {s['missed_urgent']}"),
        ("A push for every urgent call (and none for others)", check(s["missed_push"] == 0 and s["false_push"] == 0),
         f"missed {s['missed_push']}, false {s['false_push']}"),
        (f"FAQ questions answered right >= {DOD_PERCENT} %", check(share(s["faq_ok"], s["faq_cards"]) >= DOD_PERCENT),
         f"{s['faq_ok']}/{s['faq_cards']}"),
        ("0 answers that are not in the FAQ", check(s["non_approved_replies"] == 0 and s["faq_extra_answers"] == 0),
         f"non-approved replies {s['non_approved_replies']}, unexpected answers {s['faq_extra_answers']}"),
        ("Every completed call lands in voicemails.db", check(s["handoff_calls"] == s["calls"]), f"{s['handoff_calls']}/{s['calls']} handed off"),
    ]
    if s["audio"] and s["latency_p50"] is not None:
        rows.append((f"Median reply <= {TARGET_TURN_SECONDS:g} s", check(s["latency_p50"] <= TARGET_TURN_SECONDS), f"{s['latency_p50']:.2f} s"))
    else:
        rows.append((f"Median reply <= {TARGET_TURN_SECONDS:g} s", "NOT MEASURED", "needs an audio run on the laptop (--audio)"))
    rows.append(("The owner can explain why the dialog is a state machine and what the agent costs", "OWNER", "not a measurement"))
    return rows


def decide_ab(a_runs: list[dict], b: dict) -> dict:
    """The pre-registered rule, literally (README section 4). `a_runs` = summaries of A's runs (the first is the main one)."""
    a = a_runs[0]
    reasons: list[tuple[str, bool, str]] = []
    gates = b["invented_numbers"] == 0 and b["missed_urgent"] == 0 and b["non_approved_replies"] == 0
    reasons.append(("1. Safety gates: B has 0 invented numbers, 0 missed urgent, 0 non-approved sentences", gates,
                    f"invented {b['invented_numbers']}, missed urgent {b['missed_urgent']}, non-approved {b['non_approved_replies']}"))
    margin = b["details_correct"] - a["details_correct"]
    reasons.append((f"2. Quality: B has at least {MIN_EXTRA_DETAILS} more correct details than A", margin >= MIN_EXTRA_DETAILS,
                    f"B {b['details_correct']} vs A {a['details_correct']} (margin {margin})"))
    a_lat, b_lat = a["latency_p50"], b["latency_p50"]
    slow_ok = a_lat is not None and b_lat is not None and b_lat <= a_lat + MAX_EXTRA_SECONDS
    reasons.append((f"3a. Cost: B's median latency <= A's + {MAX_EXTRA_SECONDS:g} s", slow_ok, f"B {b_lat} vs A {a_lat}"))
    fb = b["fallback_share"]
    reasons.append((f"3b. Cost: B's fallbacks <= {MAX_FALLBACK_SHARE:.0%} of its turns", fb is not None and fb <= MAX_FALLBACK_SHARE, f"{fb}"))
    noise = abs(a_runs[0]["details_correct"] - a_runs[1]["details_correct"]) if len(a_runs) > 1 else None
    noisy = noise is not None and margin >= MIN_EXTRA_DETAILS and noise >= margin
    reasons.append(("4. Noise: A's two runs differ by less than B's margin", noise is not None and not noisy,
                    "A was run once: no noise estimate, so the rule cannot be applied" if noise is None else f"A's runs differ by {noise}"))
    if all(ok for _, ok, _ in reasons):
        verdict = "B replaces A"
    elif noise is None or noisy:
        verdict = "inconclusive, A stays" if (gates and margin >= MIN_EXTRA_DETAILS) else "A stays"
    else:
        verdict = "A stays"
    return {"verdict": verdict, "rules": [{"rule": r, "ok": ok, "evidence": e} for r, ok, e in reasons]}


# ---------------------------------------------------------------- reports

def fmt(value, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def summary_markdown(label: str, s: dict) -> str:
    lines = [f"### {label}", "",
             f"- calls {s['calls']}, turns {s['turns']}, outcome as expected {s['outcome_ok']}/{s['calls']}, cut off {s['cut_off']}",
             f"- name {s['name_classes']}, number {s['number_classes']}; correct details {s['details_correct']}",
             f"- invented numbers {s['invented_numbers']}; missed urgent {s['missed_urgent']}; false urgent {s['false_urgent']}",
             f"- FAQ cards right {s['faq_ok']}/{s['faq_cards']}; unexpected answers {s['faq_extra_answers']}; non-approved replies {s['non_approved_replies']}",
             f"- hand-off: category right {s['category_ok']}/{s['handoff_calls']}, missed push {s['missed_push']}, false push {s['false_push']}",
             f"- time per reply: median {fmt(s['latency_p50'])} s, p95 {fmt(s['latency_p95'])} s"
             + ("" if s["audio"] else " (text run: no listening or speaking time)"),
             f"- agent fallbacks: {s['fallback_turns']}/{s['agent_turns']} turns" if s["agent_turns"] else "", ""]
    return "\n".join(line for line in lines if line is not None)


def results_markdown(meta: dict, summaries: dict[str, dict], verdict: dict | None) -> str:
    out = ["# Part 3 evaluation results", "",
           f"Run {meta['when']}: split `{meta['split']}`, {meta['calls']} caller cards, understanding = `{meta['understand']}`"
           f"{' (' + meta['models'] + ')' if meta.get('models') else ''}, mode = {'audio' if meta['audio'] else 'text'}.", ""]
    if meta["understand"] == "rules":
        out += ["> **Harness check only.** This run used the plain-code baseline instead of the model (and the caller cards it was built on), so",
                "> none of these numbers say anything about the real receptionist. They are never written to the results file.", ""]
    for label, s in summaries.items():
        out.append(summary_markdown(label, s))
    main = summaries.get("A") or next(iter(summaries.values()))
    out += ["## Definition of done (run of the chosen default)", "", "| Criterion | Result | Evidence |", "|---|---|---|"]
    out += [f"| {c} | {r} | {e} |" for c, r, e in dod_table(main)]
    if verdict:
        out += ["", f"## A or B? **{verdict['verdict']}**", "", "| Rule | Met | Evidence |", "|---|---|---|"]
        out += [f"| {r['rule']} | {'yes' if r['ok'] else 'no'} | {r['evidence']} |" for r in verdict["rules"]]
    out += ["", "The rule was written before any run (README section 4). Per-call details: the JSON next to this run in `03-phone-receptionist/logs/`.", ""]
    return "\n".join(out)


def placeholder_markdown() -> str:
    return """# Part 3 evaluation results

**NOT MEASURED YET.** The cloud session that built Part 3 has no GPU, no Ollama model, no Whisper model and no Piper voice, so no
evaluation number exists and none is written here. Runs without a model (`--understand rules`) only test the harness and are never
written to this file.

To produce the numbers, on the laptop (after the speed measurements and the voice choice, see README section 8):

```powershell
python 03-phone-receptionist\\evaluate.py --split score --understand model --decide both --repeat-a 2 --write-docs
python 03-phone-receptionist\\evaluate.py --split score --understand model --decide a --audio --write-docs   # adds the real speed (audio) run
```

The command rewrites this file with the counts per version, the definition-of-done table (PASS / FAIL / NOT MEASURED) and the
verdict of the pre-registered A-vs-B rule. The metrics and the rule are explained in `docs/part3-eval-notes.md`.
Tune prompts only on the `dev` cards (`--split dev`); the `score` cards are for this run.
"""


# ---------------------------------------------------------------- command line

def build_understand(kind: str):
    from turn import PROMPT_VERSION, Understanding, understand
    if kind == "rules":
        from rules_turn import rules_understand
        return (lambda text, ctx: Understanding(rules_understand(text, ctx))), "rules", None
    return understand, "model", PROMPT_VERSION


def choose_cards(split: str) -> list[Card]:
    cards = load_cards()
    chosen = [c for c in cards if split in ("all", c.split) or c.id.startswith(split)]
    if not chosen:
        sys.exit(f"no card matches {split!r}")
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the receptionist on the caller cards; decide A or B by the pre-registered rule.")
    parser.add_argument("--split", default="score", help="dev | score | all | a card id prefix (default: score)")
    parser.add_argument("--understand", choices=["model", "rules"], default="model")
    parser.add_argument("--decide", choices=["a", "b", "both"], default="both")
    parser.add_argument("--repeat-a", type=int, default=2, help="how many times version A is run (2 = the noise check of rule 4)")
    parser.add_argument("--audio", action="store_true", help="speak and listen for real (Piper + Whisper): laptop only")
    parser.add_argument("--no-handoff", action="store_true", help="skip the hand-off into an in-memory copy of Part 1's table")
    parser.add_argument("--write-docs", action="store_true", help="write docs/part3-eval-results.md (only after a --understand model run)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    persona, faq, cards = load_persona(), load_faq(), choose_cards(args.split)
    understand_fn, label, version = build_understand(args.understand)
    runs: dict[str, list[dict]] = {}
    summaries: dict[str, dict] = {}

    def one_run(decide_label: str, make_decide) -> dict:
        scores = run_set(cards, persona, faq, understand_fn, decide_label, make_decide, label, version, audio=args.audio,
                         with_handoff=not args.no_handoff)
        runs.setdefault(decide_label, []).append({"scores": scores, "summary": summarize(scores, args.audio)})
        return runs[decide_label][-1]["summary"]

    if args.decide in ("a", "both"):
        a_summaries = [one_run("a", lambda: dialog.decide_a) for _ in range(max(1, args.repeat_a))]
        summaries["A"] = a_summaries[0]
    if args.decide in ("b", "both"):
        from agent_dialog import make_decide_b
        summaries["B"] = one_run("b", make_decide_b)
    verdict = decide_ab([r["summary"] for r in runs["a"]], summaries["B"]) if args.decide == "both" else None

    when = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta = {"when": when, "split": args.split, "calls": len(cards), "understand": label, "audio": args.audio}
    if args.understand == "model":
        from shared.llm import settings
        meta["models"] = f"LLM {settings()['model']}"
    report = results_markdown(meta, summaries, verdict)
    print(report)
    LOGS.mkdir(exist_ok=True)
    path = LOGS / f"eval-{when.replace(':', '')}.json"
    path.write_text(json.dumps({"meta": meta, "runs": runs, "verdict": verdict}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("details saved to", path)
    if args.write_docs:
        if args.understand != "model":
            print("NOT written to docs/part3-eval-results.md: a run without a model only tests the harness.")
        else:
            RESULTS_FILE.write_text(report, encoding="utf-8")
            print("wrote", RESULTS_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
