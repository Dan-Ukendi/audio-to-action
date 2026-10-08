"""How long does each stage of a turn take on THIS machine? (laptop only: needs Whisper, Ollama and the Piper voice)

    python 03-phone-receptionist/measure_speed.py --device cpu  --whisper base small --llm qwen2.5:7b
    python 03-phone-receptionist/measure_speed.py --device cuda --whisper base small --llm qwen2.5:7b qwen2.5:3b

Per turn the caller waits for  listen (Whisper)  +  understand (LLM)  +  speak (Piper).  This script times each stage on
short, realistic turns (the `dev` caller cards only, never the scoring cards), keeps the first call of every model apart
(loading it is not a per-turn cost), and writes docs/part3-speed.md + docs/part3-speed-results.json. Runs with different
--device / models are MERGED into the same files, so CPU and GPU end up side by side.

The default-picking rule was written down before any measurement (README section 4):
  Whisper: the smallest model (base before small) whose name hit-rate on the dev clips is within 1 of `small`.
  LLM: qwen2.5:7b if its median understand time is <= 3 s, otherwise qwen2.5:3b (asking before that download).
  Target: a median turn (listen + understand + speak) of 5 s or less.
Nothing in this file produces a number without running the models: the tests only exercise the arithmetic.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from audio_io import normalize  # noqa: E402
from shared.evaluation import median, percentile  # noqa: E402

DOCS = ROOT / "docs"
REPORT = DOCS / "part3-speed.md"
RESULTS = DOCS / "part3-speed-results.json"
TARGET_TURN_S = 5.0
LLM_BUDGET_S = 3.0
WHISPER_ORDER = ["base", "small"]  # smallest first: the rule prefers the smaller model


# ---------------------------------------------------------------- the three measurements

def stats(values: list[float]) -> dict:
    return {"n": len(values), "median_s": round(median(values), 3) if values else None,
            "p95_s": round(percentile(values, 95), 3) if values else None}


def measure_whisper(clips: list[dict], model: str, listen_fn) -> dict:
    """clips: [{"path", "name_tokens" or None}]. The first clip is a warm-up (model load) and is not averaged."""
    first, *rest = clips
    warm = listen_fn(first["path"], model=model)
    times, hits, named = [], 0, 0
    for clip in rest:
        heard = listen_fn(clip["path"], model=model)
        times.append(heard.seconds)
        if clip["name_tokens"]:
            named += 1
            words = set(normalize(heard.raw_text).split())
            hits += all(token in words for token in clip["name_tokens"])
    return {"load_and_first_s": round(warm.seconds, 3), **stats(times), "name_hits": hits, "name_clips": named}


def measure_llm(model: str, repeats: int, chat_fn) -> dict:
    """chat_fn(model) -> seconds for one structured reply. The first call (loads the model onto the GPU) is kept apart."""
    first = chat_fn(model)
    return {"load_and_first_s": round(first, 3), **stats([chat_fn(model) for _ in range(repeats)])}


def measure_piper(lines: dict[str, str], speaker: int, repeats: int, speak_fn) -> dict:
    """Time the synthesis of typical receptionist sentences (short ask, read-back, long FAQ answer)."""
    speak_fn(next(iter(lines.values())), speaker)  # warm-up: loads the voice
    out = {}
    for name, text in lines.items():
        out[name] = stats([speak_fn(text, speaker) for _ in range(repeats)])
    every = [v["median_s"] for v in out.values()]
    return {"per_line": out, "median_s": round(median(every), 3)}


# ---------------------------------------------------------------- the rule

def turn_estimate(listen_s: float | None, understand_s: float | None, speak_s: float | None) -> float | None:
    parts = [listen_s, understand_s, speak_s]
    return None if any(p is None for p in parts) else round(sum(parts), 2)


def recommend(results: dict) -> dict:
    """Apply the pre-registered rule to the merged results. Pure arithmetic: no model runs here."""
    whisper = results.get("whisper", {})  # key "model/device"
    llm = results.get("llm", {})
    piper = results.get("piper", {})
    out: dict = {"whisper": None, "llm": None, "estimated_turn_s": None, "meets_target": None, "notes": []}

    for device in sorted({k.split("/")[1] for k in whisper}):
        measured = [m for m in WHISPER_ORDER if f"{m}/{device}" in whisper]
        if not measured:
            continue
        best_hits = max(whisper[f"{m}/{device}"]["name_hits"] for m in measured)
        small_hits = whisper[f"small/{device}"]["name_hits"] if "small" in measured else best_hits
        if "small" not in measured:
            out["notes"].append(f"small was not measured on {device}: the rule compares against the best measured model instead.")
        choice = next(m for m in measured if whisper[f"{m}/{device}"]["name_hits"] >= small_hits - 1)
        out.setdefault("whisper_by_device", {})[device] = choice
    if out.get("whisper_by_device"):
        device = "cuda" if "cuda" in out["whisper_by_device"] else sorted(out["whisper_by_device"])[0]
        out["whisper"] = f"{out['whisper_by_device'][device]}/{device}"

    seven, three = llm.get("qwen2.5:7b"), llm.get("qwen2.5:3b")
    if seven and seven["median_s"] is not None and seven["median_s"] <= LLM_BUDGET_S:
        out["llm"] = "qwen2.5:7b"
    elif three and three["median_s"] is not None:
        out["llm"] = "qwen2.5:3b"
    elif seven:
        out["llm"] = "qwen2.5:7b"
        out["notes"].append(f"7b needs {seven['median_s']} s per understanding step (budget {LLM_BUDGET_S} s) "
                            "and 3b was not measured: measure it (ask before the ~1.9 GB download).")

    if out["whisper"] and out["llm"] and piper:
        out["estimated_turn_s"] = turn_estimate(whisper[out["whisper"]]["median_s"], llm[out["llm"]]["median_s"],
                                                piper["median_s"])
        out["meets_target"] = out["estimated_turn_s"] <= TARGET_TURN_S if out["estimated_turn_s"] is not None else None
    return out


def render_markdown(results: dict, rec: dict) -> str:
    if not any(results.get(k) for k in ("whisper", "llm", "piper", "whisper_errors", "llm_errors")):
        return PLACEHOLDER
    lines = ["# Part 3 speed budget (measured)", "",
             f"Measured on: {results.get('machine', 'unknown machine')} · last run {results.get('date', '?')}", "",
             "A turn = listen (Whisper) + understand (LLM) + speak (Piper). Target: median turn <= 5 s.", "",
             "## Listen (Whisper)", "", "| model/device | first call (load) | median | p95 | clips | name hits |", "|---|---|---|---|---|---|"]
    for key, v in sorted(results.get("whisper", {}).items()):
        lines.append(f"| {key} | {v['load_and_first_s']} s | {v['median_s']} s | {v['p95_s']} s | {v['n']} | "
                     f"{v['name_hits']}/{v['name_clips']} |")
    lines += ["", "## Understand (LLM, one structured reply)", "", "| model | first call (load) | median | p95 | calls | processor |", "|---|---|---|---|---|---|"]
    for key, v in sorted(results.get("llm", {}).items()):
        lines.append(f"| {key} | {v['load_and_first_s']} s | {v['median_s']} s | {v['p95_s']} s | {v['n']} | {v.get('processor', '?')} |")
    if results.get("piper"):
        lines += ["", "## Speak (Piper)", "", "| line | median | p95 |", "|---|---|---|"]
        for name, v in results["piper"]["per_line"].items():
            lines.append(f"| {name} | {v['median_s']} s | {v['p95_s']} s |")
    for label, errors in (("Whisper runs", results.get("whisper_errors")), ("LLM models", results.get("llm_errors"))):
        if errors:
            lines += ["", f"{label} that could not be measured: " + "; ".join(f"{k} ({v})" for k, v in errors.items())]
    estimate = f"**{rec['estimated_turn_s']} s**" if rec["estimated_turn_s"] is not None else "not available (a stage is missing)"
    lines += ["", "## Decision (pre-registered rule, README section 4)", "",
              f"- Whisper: **{rec['whisper'] or 'not decided'}**",
              f"- LLM: **{rec['llm'] or 'not decided'}**",
              f"- Estimated median turn: {estimate} (sum of the three stage medians, not a measured median of whole turns)" + (
                  ", target met" if rec["meets_target"] else ", target NOT met" if rec["meets_target"] is False else "")]
    lines += [f"- Note: {n}" for n in rec["notes"]]
    if results.get("gpu"):
        lines += ["", "## GPU check", "", "```", results["gpu"].strip(), "```"]
    return "\n".join(lines) + "\n"


PLACEHOLDER = """# Part 3 speed budget

**NOT MEASURED YET.** The cloud session that built Part 3 has no GPU, no Whisper model, no Ollama model and no Piper voice,
so no number is written here. To produce them, on the laptop:

```powershell
nvidia-smi
python 03-phone-receptionist\\choose_voice.py             # pick Holly's voice first (persona.json piper_speaker)
python 03-phone-receptionist\\measure_speed.py --device cpu  --whisper base small --llm qwen2.5:7b
python 03-phone-receptionist\\measure_speed.py --device cuda --whisper base small --llm qwen2.5:7b qwen2.5:3b   # after the GPU is fixed
```

This file is then rewritten with the medians, p95 and the decision of the pre-registered rule (README section 4):
the smallest Whisper model within 1 name-hit of `small`; `qwen2.5:7b` if its understanding step takes <= 3 s, else `qwen2.5:3b`;
target = median turn (listen + understand + speak) of 5 s or less.
"""


# ---------------------------------------------------------------- real runs (laptop)

class ProbeTurn(BaseModel):
    """Same size and shape as the per-turn form of Phase 3, so the timing is representative of the real step."""
    heard_summary: str = Field(description="One short sentence: what the caller just said.")
    name: str | None = Field(description="The caller's name if they said it, else null.")
    number: str | None = Field(description="A phone number the caller said, digits only, else null.")
    reason: str | None = Field(description="Why they are calling, a short phrase, else null.")
    emergency: bool = Field(description="True only for danger now: gas, flooding, sparks, no heating for a vulnerable person.")


PROBE_SYSTEM = ("You help the receptionist of Brightwater Plumbing & Heating, a small UK plumbing business. Read what the caller "
                "just said (speech recognition, so it may contain mistakes) and fill the form. Never guess: a detail the caller did "
                "not say is null. A phone number is digits only; callers say 'oh' for zero. 'Emergency' means danger now. " * 2)


def run_command(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def gpu_report() -> str:
    smi = run_command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    return "nvidia-smi: " + (smi or "not available (no NVIDIA driver, or Code 43)") + "\n\nollama ps:\n" + (run_command(["ollama", "ps"]) or "(nothing loaded)")


def sample_texts(cards) -> list[str]:
    """Openings of the dev cards: realistic caller sentences that were never used for scoring."""
    return [c.script.opening.format(name_spoken=c.facts.name_spoken or "", number_words="oh seven seven double oh, nine hundred, one two three")
            for c in cards if c.split == "dev" and c.script.opening]


def build_clips(cards) -> list[dict]:
    """Render the dev cards' opening and name-answer turns (Piper + phone-line sound) and return the clip list."""
    import asks
    from simulate import SimulatedCaller, render_turn
    out_dir = HERE / "testset" / "audio" / "speed"
    clips = []
    for card in cards:
        if card.split != "dev" or "robocall" in card.quirks:
            continue
        caller = SimulatedCaller(card)
        for asked in (asks.GREETING, asks.NAME):
            text = caller.reply(asked)
            if asked == asks.NAME and not card.facts.name:
                continue
            path = render_turn(card, text, out_dir / f"{card.id}_{asked}.wav")
            tokens = [t.lower() for t in (card.facts.name or "").split()] if asked == asks.NAME else None
            clips.append({"path": path, "name_tokens": tokens})
    return clips


def save(results: dict, results_path: Path, report_path: Path) -> dict:
    """Write the JSON and the report NOW. Called after every stage, so a crash later loses nothing already measured."""
    rec = recommend(results)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    report_path.write_text(render_markdown(results, rec), encoding="utf-8")
    return rec


def run_measurements(args, persona, clips, texts, *, listen_fn, llm_call, speak_fn, speaker, processor_fn, gpu_text,
                     results_path: Path = RESULTS, report_path: Path = REPORT) -> dict:
    """All three measurements, merged into the saved results. Dependencies are passed in so a test can drive the whole flow."""
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
    results.setdefault("whisper", {})
    results.setdefault("llm", {})
    results.setdefault("llm_errors", {})
    results.setdefault("whisper_errors", {})
    results["machine"] = platform.platform()  # no host name: this file is meant to be committed
    results["date"] = date.today().isoformat()
    results["gpu"] = gpu_text

    for model in args.whisper:
        print(f"Whisper {model} on {args.device}...")
        key = f"{model}/{args.device}"
        try:
            results["whisper"][key] = measure_whisper(clips, model, lambda path, model: listen_fn(path, model=model))
            results["whisper_errors"].pop(key, None)
        except Exception as error:  # e.g. the GPU build cannot load the model: keep the other measurements
            results["whisper"].pop(key, None)  # an old row must not sit next to the new failure
            results["whisper_errors"][key] = f"{type(error).__name__}: {error}"[:300]
            print(f"  {key} failed ({results['whisper_errors'][key]}); continuing")
        save(results, results_path, report_path)

    counter = {"i": 0}

    def one_call(model: str) -> float:
        text = texts[counter["i"] % len(texts)]
        counter["i"] += 1
        return llm_call(model, text)

    for model in args.llm:
        print(f"LLM {model}...")
        try:
            results["llm"][model] = measure_llm(model, args.repeats, one_call)
            results["llm"][model]["processor"] = processor_fn()
            results["llm_errors"].pop(model, None)
        except Exception as error:  # e.g. the model is not pulled yet: record it, keep everything else
            results["llm"].pop(model, None)
            results["llm_errors"][model] = f"{type(error).__name__}: {error}"[:300]
            print(f"  {model} failed ({results['llm_errors'][model]}); continuing")
        save(results, results_path, report_path)

    if not args.skip_piper:
        lines = {"short question": persona.say("ask_name"), "greeting": persona.say("greeting"),
                 "read-back": persona.say("read_back", name="Siobhan Gallagher", number="oh one six three two, nine six oh, five oh one",
                                          reason="a bathroom quote")}
        print("Piper...")
        results["piper"] = measure_piper(lines, speaker, args.repeats, speak_fn)
    return {"results": results, "recommendation": save(results, results_path, report_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Whisper device for this run")
    parser.add_argument("--whisper", nargs="*", default=["base", "small"])
    parser.add_argument("--llm", nargs="*", default=["qwen2.5:7b"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--skip-piper", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    os.environ["WHISPER_DEVICE"] = args.device
    os.environ["WHISPER_COMPUTE_TYPE"] = "float16" if args.device == "cuda" else "int8"
    from audio_io import listen, piper_synth, speak
    from cards import load_cards
    from choose_voice import candidate_ids
    from persona import PersonaError, load_persona, speaker_id
    from shared.llm import structured_chat

    persona = load_persona()
    try:
        speaker = speaker_id(persona)
    except PersonaError:
        # Speed barely depends on WHICH speaker speaks, so a missing choice must not stop the whole measurement.
        speaker = candidate_ids(persona, 1)[0]
        print(f"No receptionist voice chosen yet: timing Piper with speaker {speaker} (choose_voice.py picks the real one).")

    cards = load_cards()
    print("Rendering the dev caller clips (Piper)...")
    clips = build_clips(cards)
    out_wav = HERE / "testset" / "audio" / "speed" / "piper.wav"

    def llm_call(model: str, text: str) -> float:
        messages = [{"role": "system", "content": PROBE_SYSTEM}, {"role": "user", "content": f"Caller said:\n<<<\n{text}\n>>>"}]
        return structured_chat(ProbeTurn, messages, llm=model).seconds

    outcome = run_measurements(
        args, persona, clips, sample_texts(cards),
        listen_fn=lambda path, model: listen(path, hint=persona.hint, model=model), llm_call=llm_call,
        speak_fn=lambda text, spk: speak(text, out_wav, spk, synth_fn=piper_synth), speaker=speaker,
        processor_fn=lambda: run_command(["ollama", "ps"]).replace("\n", " | ")[:200] or "?", gpu_text=gpu_report())
    print(f"\nWrote {REPORT} and {RESULTS}\nDecision: {json.dumps(outcome['recommendation'], indent=2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
