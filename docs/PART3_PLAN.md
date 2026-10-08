# Part 3 plan: phone receptionist (`03-phone-receptionist/`)

Status: **planned, not started** (2026-10-08). This file is the brief for whoever builds Part 3. Read it with
`PROJECT_RULES.md` (rules, phase list), `docs/HANDOVER.md` (architecture and contracts of Parts 1-2) and the
Part 2 entries of `docs/learning-log.md` (what went wrong with the agent and why).

> Committed as the owner wrote it, with two edits: the author e-mail in rule 7 is the one the git history uses
> (`dan.ukendi1@gmail.com`), and rule 7 no longer names any kind of tool (the no-mention rule applies to this file too).
> What was built from it, and every decision taken where the plan left it to the owner, is in
> `03-phone-receptionist/README.md`.

---

## 1. Rules that apply (from PROJECT_RULES.md and the owner)

1. **One phase at a time**, then stop for the owner's "next". If the owner authorises several phases in one go,
   run a **separate reviewer agent after each phase** to verify and optimise before continuing (it found real bugs
   in every Part 2 phase).
2. Explain in 2-4 sentences before writing code; small readable functions; comments explain *why*.
3. **Ask before installing anything or downloading models, and state the size first.**
4. Free and local: no paid APIs, no cloud telephony. Privacy (GDPR): call audio, transcripts and messages are other
   people's data: local and git-ignored, never committed.
5. Every step saves its output (JSON); everything inspectable.
6. **Commit after each phase** with a clear message. At the end of each phase: the 9-section recap in chat **and**
   appended to `docs/learning-log.md` (format in PROJECT_RULES.md).
7. **Git authorship (absolute):** the only author/committer is **Dan-Ukendi <dan.ukendi1@gmail.com>**. No
   co-author trailers, no "generated with" lines, no mention of any AI tool or vendor in commits, PRs,
   or committed files. Check `git log origin/main..HEAD --format='%an <%ae>%n%B'` before any push. **Push only
   when the owner asks.** Tool-specific instruction files stay local and git-excluded; the tracked rules
   file is `PROJECT_RULES.md`.
8. Always open text files with `encoding="utf-8"`; run from the repo root with `.venv` (Python 3.13).

## 2. Owner's decisions

| Topic | Decision |
|---|---|
| How calls reach the receptionist | **Local simulation** (no phone number). Live use: **push-to-talk in the browser app** (Streamlit `st.audio_input`, available in the installed 1.65). Automated tests: a scripted synthetic caller (audio files, no browser). |
| Receptionist's job | **Take a message** (name, callback number, reason, urgency) **and answer simple questions** (opening hours, services, area covered, ...) from a local FAQ file |
| Order | **State machine first** (version A); a tool-calling agent (version B) later, as a measured experiment |
| Speed | The owner is **fixing the GPU now** (RTX 5050 showed "Code 43": driver problem). Re-measure everything in Phase 2 |

Persona and wording (greeting, receptionist name, sign-off) and the FAQ content are still to be confirmed with the
owner in Phase 0.

## 3. Environment (state on 2026-10-08; verify in Phase 0)

- Windows 11, i5-13420H, 16 GB RAM (often only ~3 GB free: close browsers/Discord before runs), RTX 5050 8 GB.
- Ollama with `qwen2.5:7b`. On CPU it needs 30-200 s per reply: **unusable for a live call**. With a working GPU,
  the 7B model (4.7 GB) fits in 8 GB VRAM and should answer in ~1-3 s. Fallback: `qwen2.5:3b` (~1.9 GB download,
  ask first).
- faster-whisper 1.2.1 on CPU (`small` ≈ 1× real time; `base` ~3× faster). Whisper on the GPU additionally needs
  `nvidia-cublas-cu12` + `nvidia-cudnn-cu12` (~1.5-2 GB, ask first); RTX 50-series support in this CTranslate2 build
  is unverified: test in Phase 2, keep CPU `base` as fallback.
- Piper TTS (`shared/tts.py`, voice `en_GB-vctk-medium`, 109 speakers) ~0.2 s per sentence on CPU.
- ffmpeg 9 on PATH (new terminals). Local app: `start_app.bat` / `python -m streamlit run app\app.py`.

## 4. Goal and scope

A caller "phones" Brightwater Plumbing & Heating (the fictional business of Parts 1-2: owner Sam, team Priya, Tom,
Jamie). The receptionist:
1. greets and asks how it can help;
2. **answers simple questions** from `faq.json` (and only from it: no invented facts);
3. **takes a message**: name, callback number, reason; reads them back; accepts corrections;
4. **flags emergencies immediately** (safety words or the LLM's urgency) and tells the caller someone will call back;
5. ends politely, then hands the message to the **Part 1 pipeline** (analyze → route → push → `voicemails.db`).

Out of scope: real telephony, booking appointments, transferring calls, multiple languages, barge-in (talking over
the receptionist).

## 5. Architecture

```
 caller turn (browser push-to-talk recording, or a synthetic caller's audio file)
    │
    ▼
 1. LISTEN      ffmpeg → 16 kHz → Whisper (hint = team, customers, places) → text   [shared/transcribe.py]
                ignore empty/very short turns (Whisper invents "Thank you." on silence)
 2. UNDERSTAND  LLM fills a per-turn form (CallerTurn, below) via shared/llm.structured_chat:
                what the caller gave (name / number / reason), corrections, a question?, emergency?, wants to end?
                validators = plain-code rules (UK number check reused from Part 1, owner-style grounding)
 3. DECIDE      DIALOG MANAGER, version A = state machine in plain code (Phase 3):
                missing details → next question; question → FAQ answer; all details → read-back; confirmed → goodbye
                safety net (Part 1's SAFETY_PATTERNS) at ANY turn → urgent fast path + immediate push
 4. SPEAK       reply text → Piper (receptionist voice) → wav → played in the browser            [shared/tts.py]
    ⋮ repeat until done or a turn limit
 5. HAND OFF    call record (JSON) → Part 1 analyze() → routing.route() → deliver() (push if urgent, save row)
```

Key principle: the **dialog engine is UI-independent** (pure Python: `next_reply(call_state, caller_text) ->
(reply_text, new_state)`), so the synthetic-caller tests run headless and the app is only a thin UI.

### Proposed files (adjust in Phase 0)
```
03-phone-receptionist/
├── README.md
├── context.py        business facts for the hint + persona text (or import Part 2's context.py)
├── faq.json          question topics and the approved answers (hours, services, area, emergency call-out, ...)
├── turn.py           understand(): one caller turn → CallerTurn (LLM form + validators)
├── dialog.py         version A: CallState + state machine (next_reply), read-back, corrections, limits
├── faq.py            match a caller question to faq.json (keywords first; LLM only to pick among entries)
├── call.py           one call end to end: listen → understand → decide → speak; saves calls/<id>.json + audio
├── handoff.py        finished call → Part 1 analyze / route / deliver
├── agent_dialog.py   version B (Phase 5): tool-calling receptionist with Part 2's guardrails
├── simulate.py       scripted synthetic caller: answers the receptionist from a caller card
├── evaluate.py       runs every caller card through A (and B) → docs/part3-eval-results.md
├── tests/            dialog, faq, validators, handoff, simulator (no LLM)
└── testset/          callers.json (cards), labels (reuse Part 1 labels.json), generated audio (ignored)
app/                  new "Receptionist" tab: push-to-talk, live transcript, message card, latency per turn
```

### Data shapes (sketch)
- `CallerTurn` (LLM form, field order = think first): `heard_summary`, `name: str|None`, `number: str|None`
  (digits; Part 1's `check_uk_number` rules; null if not said, never guessed), `reason: str|None`,
  `is_correction: bool` (+ which field), `question_topic: str|None`, `emergency: bool`, `wants_to_end: bool`.
- `CallState`: collected slots + confidence/confirmed flags, `turns`, `urgent`, `questions_answered`, `state`
  (GREETING, COLLECTING, READ_BACK, CORRECTING, GOODBYE, ENDED), full turn log with timings.
- `CallRecord` (saved JSON, one per call): caller turns (text + audio file), receptionist replies, final message,
  urgency, per-turn latency (listen / understand / decide / speak), outcome.

## 6. Dialog rules (version A, plain code)

- Ask for one missing detail at a time, in the order: reason → name → number (or whatever the caller already gave).
- **Never invent:** a detail is stored only if the caller said it in this call; the number must pass the UK check.
  "You've got my number" → ask again once, then record "no number given".
- **Read-back:** "So that's Siobhan Gallagher on 01632 960 501 about a bathroom quote. Is that right?" Digits are
  read in groups. "No, it's 349" → correction flow for that field only.
- **Questions:** answered only from `faq.json`; unknown → "I'll pass your question on" and add it to the message.
- **Emergency fast path:** safety words (reuse `01-voicemail-triage/routing.py` `SAFETY_PATTERNS`) or
  `emergency=True` → say the safety line (e.g. gas: leave the property, call the gas emergency number), push at once
  (minimal text, as in Part 1), still collect name/number.
- **Limits:** max ~12 turns, max 2 re-asks per detail, then wrap up with what we have; silence/empty turns twice →
  end politely. Robocall/spam → short goodbye, still logged.

## 7. Phases

| # | Phase | Deliverables | Done when |
|---|---|---|---|
| 0 | Plan & setup | confirm persona + FAQ content with the owner; check GPU (`nvidia-smi`, `ollama ps` shows GPU); folders, `.gitignore` for call audio/records; PROJECT_RULES.md Part 3 section | owner says "next" |
| 1 | Test set: synthetic callers | `testset/callers.json`: one **caller card** per Part 1 scenario (18) + 4-6 FAQ callers. Card = facts (name, spelling, number, reason, quirks: spells name, corrects number, refuses number, emergency, robocall, asks a question) + labels (Part 1 `labels.json` fields + expected FAQ topic). `simulate.py` answers the receptionist's question type from the card, rendered by Piper in the caller's voice | labels checked by code; a reviewer re-reads cards vs Part 1 labels |
| 2 | Audio loop + speed budget | browser push-to-talk prototype (record → wav → reply audio autoplay); file-based loop for tests; measure per turn: Whisper `base`/`small` (CPU vs GPU), LLM 7B (GPU) vs 3B, Piper; pick defaults | median turn ≤ 5 s measured and written down (target ~2-3 s with GPU) |
| 3 | Dialog version A | `turn.py`, `dialog.py`, `faq.py`, `call.py`; unit tests for every state transition, correction, limit, emergency, FAQ fallback | all simulated callers complete without crash; tests pass |
| 4 | Hand-off to Part 1 | `handoff.py`: caller-side text → `shared/analyze.analyze()` → `routing.route()` → `deliver()`; calls stored like voicemails (source marked as call); urgent pushes (ntfy dry run by default) | every completed call appears in `voicemails.db` with route + reasons |
| 5 | Dialog version B (agent) | `agent_dialog.py`: tools `ask(detail)`, `answer_faq(topic)`, `read_back()`, `flag_urgent()`, `take_message()`, `end_call()`; Part 2 guardrails (state message, values copied by code, validated args, step limit, no-progress stop, trace) | same interface as A; tests pass |
| 6 | Evaluation | `evaluate.py`: all callers through A and B → slot accuracy, never-invented numbers, urgent flagged, FAQ correct, turns, latency p50/p95; **experiment A vs B** with a decision rule written BEFORE the runs | `docs/part3-eval-results.md` + notes; DoD table |
| 7 | Polish | "Receptionist" tab in the app; README; learning log; HANDOVER.md §13 | reviewer fact-checks docs vs code |

## 8. Definition of done (draft; confirm in Phase 0)
- Synthetic callers: callback number exactly right ≥ 80 %, name ≥ 80 %, **0 invented numbers**.
- **Every urgent caller flagged during the call** (0 missed) and a push sent (or dry-run logged).
- FAQ questions answered correctly from `faq.json` ≥ 80 %; 0 answers that aren't in the FAQ.
- Every completed call lands in `voicemails.db` through the Part 1 pipeline.
- Median receptionist reply ≤ 5 s on this laptop (measured in Phase 2/6).
- The owner can explain why the dialog is a state machine (or not) and what the agent version costs.

## 9. Reuse map
| Need | Reuse | Note |
|---|---|---|
| Speech-to-text + name hint | `shared/transcribe.py` (`hint=`) | short turns: consider `base`; hint fades after ~220 tokens (fine for short turns) |
| LLM form + validation + 1 retry | `shared/llm.structured_chat` | keep `temperature 0`; keep prompts short for speed |
| UK number rules | `shared/schemas.py` `Analysis.check_uk_number` logic | factor it into a reusable function rather than copying |
| Voice | `shared/tts.py` | pick a receptionist speaker id distinct from Part 2's (7, 0, 9, 11) and Part 1's callers |
| Routing, push, storage | Part 1 `routing.py`, `deliver.py`, `store.py`, `shared/notify.py` | minimal push text, dry run by default |
| Analysis of the whole call | `shared/analyze.analyze()` | its prompt says "Voicemail transcript": decide in Phase 4 whether a call prompt version is needed. **Its cache key has no hint fingerprint** (HANDOVER §9): add one, or pass `cache_dir=None` |
| Retries, inbox helpers, logging | `shared/retry.py`, `shared/pipeline.py` | |
| Agent guardrail patterns | Part 2 `agent.py` | state message, letters vs numbers, required args with advice, relevance checks, no-progress stop |
| App | `app/helpers.py` + `app/app.py` | add a tab; keep logic out of the page |

## 10. Lessons from Parts 1-2 to apply from day one
- Write the answer key (caller cards + labels) **before** the dialog code; never tune prompts on the same cards you
  score with (the Part 1/2 leak and overfitting episodes).
- Models satisfy rules the cheapest way (delete a value, guess an id, skip, rewrite a list): every validator needs a
  check that the "fix" is real; error messages must say what to do instead.
- Calendar/number arithmetic and dialog flow belong in code; the LLM only understands each turn.
- Build the simple baseline first and decide with a pre-registered rule on the realistic input.
- Temperature 0 isn't bit-exact on CPU; small test sets are noisy: report counts, not just percentages.
- Look for bugs in your own glue code before blaming the model; keep the raw model outputs for inspection.

## 11. Risks
| Risk | Mitigation |
|---|---|
| Latency too high for a call | GPU fix (in progress); small prompts; `qwen2.5:3b` fallback (ask before download); Whisper `base` |
| Low free RAM | warn in the app (already there); close apps; smaller model |
| Whisper hallucinations on silence/short clips | ignore turns under ~0.5 s of speech; VAD; never act on an empty turn |
| Echo / feedback | push-to-talk (decided), headphones for demos |
| FAQ answers drifting from the facts | answer text comes from `faq.json` verbatim; the LLM only picks the entry |
| Privacy of recordings | local, git-ignored folders; a real deployment would need an "automated and recorded" announcement |
