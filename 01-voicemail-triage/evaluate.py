"""Phase 6: evaluate the pipeline on the labelled test set and write docs/eval-results.md.

    python 01-voicemail-triage/evaluate.py                          # default config
    python 01-voicemail-triage/evaluate.py --configs small:v2 small:v3

A config is <whisper model>:<prompt version>. For each config, every test file goes through
transcribe -> analyze -> route (cached, so re-runs are instant; a new config costs one LLM call
per file). Routing is the pure route() function: no pushes, no database.

Metrics, most important first:
  - urgent false negatives  urgent voicemails NOT classified urgent (target: 0)
  - urgent archived         urgent voicemails routed to the archive (must be 0, even if misclassified)
  - category accuracy       target >= 90%
  - urgency, name (exact / first word), number (correct / wrong / missed / invented)
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from routing import route  # noqa: E402
from shared.analyze import DEFAULT_PROMPT_VERSION, PROMPTS  # noqa: E402
from shared.analyze import analyze  # noqa: E402
from shared.evaluation import confusion, null_aware, pct, same_first_word, same_value  # noqa: E402
from shared.transcribe import settings as whisper_settings  # noqa: E402
from shared.transcribe import transcribe  # noqa: E402

TESTSET = HERE / "testset"
REPORT = ROOT / "docs" / "eval-results.md"
NOTES = ROOT / "docs" / "eval-notes.md"  # hand-written conclusions, inserted after the metrics
CATEGORIES = ["urgent", "other", "personal", "sales", "spam"]


def load_labels() -> list[dict]:
    """Synthetic set + your own recordings (testset/my_recordings/labels.json) if you added any."""
    labels = []
    for folder, audio_dir in ((TESTSET, TESTSET / "audio"), (TESTSET / "my_recordings", TESTSET / "my_recordings")):
        label_file = folder / "labels.json"
        if label_file.exists():
            for item in json.loads(label_file.read_text(encoding="utf-8"))["items"]:
                labels.append({**item, "path": audio_dir / item["file"]})
    return labels


def run_config(whisper: str, prompt: str, labels: list[dict]) -> list[dict]:
    """One row per test file: the label, what the pipeline produced, and the routing decision."""
    rows = []
    for label in labels:
        t = transcribe(label["path"], cache_dir=TESTSET / "transcripts", model=whisper)
        r = analyze(t, cache_dir=TESTSET / "results", prompt_version=prompt)
        rows.append({"label": label, "result": r, "decision": route(t, r)})
    return rows


def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    lab = lambda row: row["label"]  # noqa: E731
    got = lambda row: row["result"].analysis  # noqa: E731
    urgent = [r for r in rows if lab(r)["category"] == "urgent"]
    numbers = [null_aware(lab(r)["callback_number"], got(r).callback_number) for r in rows]
    return {
        "urgent false negatives": pct(sum(got(r).category != "urgent" for r in urgent), len(urgent)),
        "urgent archived": str(sum(r["decision"].route == "archive" for r in urgent)),
        "false urgent (non-urgent called urgent)": str(
            sum(got(r).category == "urgent" for r in rows if lab(r)["category"] != "urgent")),
        "category accuracy": pct(sum(got(r).category == lab(r)["category"] for r in rows), n),
        "urgency accuracy": pct(sum(got(r).urgency == lab(r)["urgency"] for r in rows), n),
        "name exact": pct(sum(same_value(lab(r)["caller_name"], got(r).caller_name) for r in rows), n),
        "name first word": pct(sum(same_first_word(lab(r)["caller_name"], got(r).caller_name) for r in rows), n),
        "number correct": pct(numbers.count("correct"), n),
        "number wrong / missed / invented": f"{numbers.count('wrong')} / {numbers.count('missed')} / {numbers.count('invented')}",
        "pushes sent": str(sum(r["decision"].notify for r in rows)),
        "flagged for review": str(sum(r["decision"].review for r in rows)),
        "analysis retries": str(sum(r["result"].attempts - 1 for r in rows)),
        "LLM seconds per voicemail (CPU)": f"{sum(r['result'].analyze_s for r in rows) / n:.0f}",
    }


def errors_table(rows: list[dict]) -> str:
    out = ["| id | field | expected | got |", "|---|---|---|---|"]
    for row in rows:
        lab, a = row["label"], row["result"].analysis
        checks = [("category", lab["category"], a.category, lab["category"] == a.category),
                  ("urgency", lab["urgency"], a.urgency, lab["urgency"] == a.urgency),
                  ("name", lab["caller_name"], a.caller_name, same_value(lab["caller_name"], a.caller_name)),
                  ("number", lab["callback_number"], a.callback_number,
                   null_aware(lab["callback_number"], a.callback_number) == "correct")]
        for field, expected, value, ok in checks:
            if not ok:
                out.append(f"| {lab['id']} | {field} | {expected} | {value} |")
    return "\n".join(out)


def build_report(results: dict[str, list[dict]]) -> str:
    configs = list(results)
    all_metrics = {c: metrics(rows) for c, rows in results.items()}
    out = [
        "# Evaluation results (Part 1, Phase 6)", "",
        f"Generated by `01-voicemail-triage/evaluate.py` on {date.today()}. "
        f"{len(results[configs[0]])} labelled voicemails (`testset/labels.json` + `my_recordings/` if present), "
        "Ollama `qwen2.5:7b` on CPU, temperature 0. Config = `<whisper model>:<prompt version>`.",
        "", "## Metrics", "",
        "| metric | " + " | ".join(f"`{c}`" for c in configs) + " |",
        "|---" * (len(configs) + 1) + "|",
    ]
    for name in all_metrics[configs[0]]:
        out.append(f"| {name} | " + " | ".join(all_metrics[c][name] for c in configs) + " |")

    # Conclusions are written by a human, so they live in their own file and survive re-generation.
    if NOTES.exists():
        out += ["", NOTES.read_text(encoding="utf-8").strip()]

    for c, rows in results.items():
        pairs = [(r["label"]["category"], r["result"].analysis.category) for r in rows]
        out += ["", f"## `{c}`: category confusion matrix", "", confusion(pairs, CATEGORIES),
                "", f"## `{c}`: every error", "", errors_table(rows)]
    return "\n".join(out) + "\n"


def main() -> None:
    # Windows consoles default to cp1252, which can't print characters like "≥" or "→".
    sys.stdout.reconfigure(encoding="utf-8")
    default =f"{whisper_settings()['model']}:{DEFAULT_PROMPT_VERSION}"
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=[default], help="e.g. small:v2 small:v3")
    parser.add_argument("--no-write", action="store_true", help="print only, don't overwrite the report")
    args = parser.parse_args()

    labels = load_labels()
    print(f"{len(labels)} labelled voicemails")
    results = {}
    for config in args.configs:
        whisper, prompt = config.split(":")
        assert prompt in PROMPTS, f"unknown prompt version {prompt}; known: {list(PROMPTS)}"
        print(f"Evaluating {config} ...")
        results[config] = run_config(whisper, prompt, labels)

    report = build_report(results)
    if not args.no_write:  # save first: a printing problem must never lose the report
        REPORT.write_text(report, encoding="utf-8")
    print(report.split("## Conclusions")[0].split("## `")[0])  # just the metrics table
    if not args.no_write:
        print(f"Written: {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
