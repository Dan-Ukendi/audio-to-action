# Project rules: audio-to-action

## Project goal
A 3-part **learning** repo about using AI to handle phone calls and audio.
Understanding matters more than speed.

- **Part 1 (`01-voicemail-triage/`)**: voicemail triage as a *fixed workflow*:
  audio → transcribe → classify/summarize/extract (LLM, structured output) → route (plain code rules).
- **Part 2 (`02-meeting-action-agent/`)**: meeting recordings → action-item agent (workflow + agent).
- **Part 3 (`03-phone-receptionist/`)**: real-time AI phone receptionist (full agent).

Reusable code (transcription, schemas, eval helpers) lives in `shared/` so Parts 2 and 3 can reuse it.

## Machine & chosen stack
- Windows 11, i5-13420H, 16 GB RAM, RTX 5050 Laptop GPU (8 GB VRAM).
- Python 3.13 in `.venv` (3.14 is the system default but too new for some wheels).
- Transcription: `faster-whisper`, default model `small` (compare sizes in Phase 2).
- Analysis: Ollama, model `qwen2.5:7b`.
- Pydantic for validation, SQLite for storage, ntfy for notifications, watchdog/polling for the watcher.
- Repo lives at `C:\Users\danuk\code\audio-to-action`, deliberately **outside OneDrive** so personal data is not synced to the cloud.

## Rules for every session
1. **One phase at a time.** Do only the current phase, then stop and wait for the user to say "next". Never start the next phase on your own.
2. **Explain as you go.** Before writing code, say in 2-4 sentences what you're about to do and why. Prefer small, readable functions over clever code; add short comments explaining the *why*.
3. **Ask, don't assume,** when a choice is the user's. Ask before installing anything large or downloading models, and state the size first.
4. **Free and local by default.** Python, ffmpeg, faster-whisper, Ollama, Pydantic, SQLite, ntfy, watchdog/polling. No paid APIs unless asked.
5. **Privacy.** Voicemails contain other people's personal data (GDPR). Keep audio, transcripts and the database local. Never commit audio files or `.env`. (Note for Phase 4: the public ntfy.sh server sees message contents, so keep notification text minimal or self-host.)
6. **Workflow, not agent.** The pipeline is a fixed sequence of plain functions. Routing rules are plain code, not LLM decisions. If tempted to give the model control of the flow, tell the user instead.
7. **Everything inspectable.** Each step saves its output to disk (transcript, JSON result).
8. **Commit after each phase** with a clear message.
9. **Run things yourself and show results,** and also give the exact commands so the user can run them.

## Phase recap format (end of EVERY phase)
Give this recap in chat **and** append it to `docs/learning-log.md` under a heading for the phase:

1. **What we built:** plain-language summary, no unexplained jargon.
2. **Where it fits in the pipeline:** ASCII diagram of the full pipeline with the current phase marked.
3. **How it works, step by step:** main code path in run order, naming files and functions.
4. **Key concepts I should understand:** 3-5 ideas, each with a short example.
5. **Files created or changed:** one line each.
6. **Try it yourself:** exact commands and expected output.
7. **What can go wrong:** likely failure modes.
8. **Check my understanding:** 2-3 short questions (no answers unless asked).
9. **Next phase preview:** 2-3 sentences.

Keep it clear and friendly rather than long.

## Phases (Part 1)
0. Setup (done)
1. Build the test set (done: 18 English synthetic voicemails, en_GB-vctk-medium, UK fictional numbers; user chose English only)
2. Transcription step (done: `shared/transcribe.py`, CPU int8, 8 threads; small vs large-v3-turbo compared in `docs/transcription-comparison.md`; user kept `small` as default for speed. Known: "oh" often transcribed as "a", numbers sometimes as words, so Phase 3 must normalize numbers and prefer spelled-out names)
3. Analysis step (done: `shared/analyze.py`, `Analysis`/`Result` in `shared/schemas.py`, prompt v2. On small transcripts: category 15/18, 0 urgent missed, 1 false urgent (12, scam with "deadline today"). v2's "Rachel from Acme" example made names first-name-only: candidate Phase 6 experiment. Ollama runs CPU-only (~1-2 min/voicemail) because the RTX 5050 shows Code 43 in Device Manager; user to update NVIDIA driver)
4. Routing step (done: `routing.py` pure `route()` + safety-word net + review flags, `deliver.py`, `store.py` SQLite, `shared/notify.py`. User chose: minimal push text (no caller data), sales -> archive, ntfy dry run (`NTFY_DRY_RUN=1`) until a real random topic is set. Tests: `python -m pytest 01-voicemail-triage/tests`)
5. Glue & reliability (done: `run.py` [--watch, --retry-failed, --base], skip if hash in SQLite, `shared/retry.py` retries only transient errors 2 s/4 s, push-before-save = at-least-once, `failed/` + `.error.txt`, `logs/run.log`. Tested end-to-end in a scratch folder)
6. Evaluation (done: `evaluate.py` + `shared/evaluation.py`, `docs/eval-results.md` generated, conclusions in `docs/eval-notes.md`. small:v2 = category 15/18, urgent FN 0/6, names 10/18, numbers 16/18. Experiment v2->v3 (name example only) failed its pre-set rule: v2 stays. DoD 90% category NOT met; next: held-out data, turbo transcripts)
7. Polish (done: `digest.py` daily page + counts-only push, full README with architecture/results/Task Scheduler commands, `docs/shared-for-part2.md`, `docs/HANDOVER.md` technical report). Part 1 complete. Phase 7 of Part 2 fixed Part 1's run.py file move (rename, no copy)

## Definition of done (Part 1)
- ≥ 90% category accuracy on the test set.
- No urgent voicemail routed to the archive.
- Re-running on the same file does nothing.
- The user can explain why each step is a workflow step and not an agent.

## Part 2: meeting-action-agent (`02-meeting-action-agent/`)
Meeting recordings → action items (workflow) → task tracker kept in sync across meetings (agent).
Full plan and design: `02-meeting-action-agent/README.md`.

User's choices (Phase 0):
- Test data: short **synthetic** meetings (~30 s each in practice, 4 Piper voices), a *series* of weekly team meetings
  at Brightwater Plumbing & Heating, so later meetings refer back to earlier tasks. Answer key written first.
- Speakers: **no diarization**. Owners come from what is said ("Priya, can you…", "I'll take that, Tom here").
- Agent job: **task tracker sync** only. Compare a meeting's extracted items with open tasks in a local
  SQLite tracker and add / update / close / mark duplicate, using a fixed tool allowlist.

The boundary to keep: transcription, extraction and storage are **workflow** steps (reuse Part 1 patterns).
Only the tracker sync is an agent, because it needs lookups and judgement over existing state. The agent
has a step limit, only allowlisted tools with Pydantic-validated arguments, a saved trace of every step,
and every tracker change recorded with its reason (reversible).

## Phases (Part 2)
0. Plan & setup (done: plan in `02-meeting-action-agent/README.md`, folders, this section)
1. Test set (done: 5 meetings ~30 s, voices Sam 7 / Priya 0 / Tom 9 / Jamie 11, `testset/{scripts,labels}.json`, `answer_key.py` folds mentions into tracker states, `generate.py`, `shared/tts.py`; 14 tasks, final 10 done / 1 cancelled / 3 open; 3 owners only knowable from voice)
   Original plan: series of synthetic meetings (scripts with speaker turns → Piper multi-voice audio), answer key
   per meeting (action items: task, owner, due, status change) + expected tracker state after each meeting
2. Transcription (done: `transcribe(..., hint=)` = Whisper initial_prompt, part of the cache key; `02-meeting-action-agent/context.py`
   holds team/customers/places (proper nouns only, never scored task words). `compare_transcription.py` ->
   `docs/part2-transcription-comparison.md`: small 8.5% WER / names 71%, **small+hint 5.3% / 98% (Part 2 default)**,
   turbo+hint 2.9% / 100% at 2x the time. Remaining misses: 'quote'->'court/call' (m2), 'suite'->'Sweet' (m4))
3. Extraction (done: `shared/llm.structured_chat()` = generic fill-form + validate + 1 retry, Part 1 `analyze()` now
   uses it; `ActionItem`/`MeetingItems`/`MeetingResult` in `shared/schemas.py` (owner must be on team, evidence must be
   in transcript, via validation context); `02-meeting-action-agent/extract.py` prompt x7 (worked example with invented
   jobs, `jobs_mentioned` think-first list, `ground_owner()` drops owners not named nearby, done/cancelled -> no due,
   raw model items kept as `llm_items`), chunking + merge (unit-tested only), `dates.py` resolves due words in code.
   x7 (`extract_testset.py`): task-only precision 84 % / recall 75 %; task+owner (DoD) 64 % / 57 %; owner 84 % where
   named, due 95 %, status 95 %; ~235 s/meeting on CPU. Lessons: a merge bug in OUR code looked like model error
   (found by the reviewer); a coverage check + retry was tried and removed (model rewrote its job list instead);
   a test-set example leaked into the prompt and was removed (x7). Variance between runs is large on 28 mentions.
   Remaining misses: transcription ('court', 'Sweet') -> turbo is a Phase 6 experiment; 3 owners only from voice.
   Temperature 0 is not bit-exact on CPU: decisions mostly stable, wording varies.)
4. The agent (done: `tracker.py` SQLite tasks + change log with reasons + undo; `agent.py` tool loop on qwen2.5:7b,
   STATE-based (fresh state message each step, not chat history), tools search/add/update/skip/finish, item ids are
   letters, values copied from items by code, guards in `run_tool` (relevance >= 0.2, one item per task per meeting,
   duplicate add needs confirm_new, repeat >= 0.5, done items not 'not_work', blank reasons refused, no-progress
   stop, failed LLM reply = one step), trace JSON per meeting; `sync_testset.py --items gold|extracted`.
   Gold items (agent alone): 12/14 tracker tasks right after the series (86 %), 25 LLM calls, ~47 min CPU.
   Story: chat-history version replayed failing calls and guessed ids (0/5); optional task_id got omitted;
   'I5' was linked to task '#5'; temperature-0 fixed point in the state loop. KEY FINDING from the reviewer:
   a plain-code rules sync (no LLM) gets 14/14 on gold -> Phase 6 must compare agent vs rules on realistic wording)
5. Glue & reliability (done: `02-meeting-action-agent/run.py` [--watch, --retry-failed, --sync, --base], oldest meeting
   first (date from YYYY-MM-DD in the name, else mtime), idempotent by hash via tracker `meetings` table, sync wrapped in
   `sync_with_rollback` (starts by undoing leftovers of an interrupted run; any error incl. Ctrl+C undoes the
   meeting's changes), same-named recordings get `stem_<hash8>` ids, `shared/pipeline.py` helpers (rename-based
   move_to, error notes, logging). E2E in a scratch folder passed; review fixed double-apply after a crash, same-name
   wipe, bad-date crash, Windows open-file copy. Part 1 run.py had the same shutil.move bug: fixed in Phase 7.)
6. Evaluation (done: `rules_sync.py` = plain-code sync baseline (`--sync rules`), `sync_testset.py --sync` saves
   `testset/sync_runs/*.json` (committed), `evaluate.py` -> `docs/part2-eval-results.md` + `docs/part2-eval-notes.md`.
   Experiment agent vs rules (same items): extracted 7/14 vs 6/14, gold 11/14 vs 14/14; agent 51 calls ~94 min, rules
   instant. Pre-registered rule -> default stays agent (margin within noise; both tuned on this set; perfect sync of
   extracted items = 9/14). DoD: extraction P/R and tracker >= 80 % NOT met; guardrails + idempotency met.
   After the run: not_work allowed for cancelled items with no similar task (m1 extracted 16 -> 2 calls))
7. Polish (done: Part 2 README rewritten, main README + docs/HANDOVER.md §12 updated, Part 1 run.py move fix,
   learning log corrected; Part 2 complete)

## Part 3: phone-receptionist (not started)
User's early choices (2026-10-07, confirm before starting): calls arrive by **local simulation** (laptop mic/speakers
or a scripted synthetic caller; no Twilio or other paid/cloud telephony), and the receptionist's job is **take a
message** (name, number, reason) and hand it to the Part 1 pipeline (routing, urgent push). Start with a Phase 0 plan.
When the user authorises several phases at once: after each phase, a separate reviewer agent verifies and optimises.

## Definition of done (Part 2)
- Action items: precision ≥ 80 % and recall ≥ 80 % on the test set (item = task + owner match).
- Tracker state after the whole meeting series matches the answer key for ≥ 80 % of tasks.
- The agent never exceeds its step limit or calls a tool outside the allowlist; every change has a logged reason.
- Re-running on the same meeting does nothing.
- The user can explain which steps are workflow, why the tracker sync is an agent, and what the agent costs.

## Commands
- Activate venv: `.\.venv\Scripts\Activate.ps1`
- Verify setup: `python scripts\check_setup.py`
- Regenerate test audio: `python 01-voicemail-triage\testset\generate.py`
- Always open text files with `encoding="utf-8"` (Windows defaults to cp1252).
