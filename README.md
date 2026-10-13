# audio-to-action

Three small systems that turn speech into action for a fictional UK plumbing business (Brightwater Plumbing & Heating):
a **voicemail triage** pipeline, a **meeting action-item** pipeline that keeps a task tracker in sync, and **Holly**, a
simulated real-time **phone receptionist**. Everything runs **locally and for free** (faster-whisper, Ollama `qwen2.5:7b`,
Pydantic, SQLite, Piper TTS, Streamlit): no cloud APIs, and no audio leaves the machine. The project is about *where* an
LLM belongs in such a system: the model only fills in forms, plain code checks every answer and makes the decisions, an
agent is used only where judgement over existing state is needed, and every choice is measured against a simpler baseline
with a decision rule written down before the run, and the negative results are reported next to the positive ones.

| Part | Folder | Input → output | Style | Status |
|---|---|---|---|---|
| 1 | [`01-voicemail-triage/`](01-voicemail-triage/) | voicemail → category, urgency, caller, number, summary → routed + push | fixed workflow | done |
| 2 | [`02-meeting-action-agent/`](02-meeting-action-agent/README.md) | meeting recordings → action items → task tracker kept in sync | workflow + one bounded agent | done |
| 3 | [`03-phone-receptionist/`](03-phone-receptionist/README.md) | a live (simulated) call → a message handed to Part 1 | state machine, agent as experiment | done |

Status: all three parts are built and working, with a local app to try them and **513 automated tests** (none need a model).

## Key ideas and decisions (the 3-minute version)

1. **Workflow vs agent, decided per step.** Transcribing, extracting and routing are fixed jobs, so they are plain
   functions in a fixed order; the routing in Part 1 is readable `if` rules, not an LLM decision. An agent is used only
   for the Part 2 tracker sync (it must look up existing tasks and judge "same task, reworded?"), and in Part 3 the
   agent only chooses among the moves the state machine already allows. Each agent is boxed in: step limit, allowlisted
   tools, Pydantic-validated arguments, a trace of every step, and in Part 2 a logged reason for every change.
2. **The LLM proposes, plain code verifies.** The model fills a Pydantic form (`shared/llm.py` `structured_chat()`:
   schema-forced JSON, temperature 0, validators, one retry with the error as feedback). Code then checks the answer
   against the source: a UK phone number rule, an owner must be on the team, a quote must really be in the transcript,
   every name and number Holly stores must be words the caller said. Safety does not depend on the model: a
   safety-word net means "smell of gas" or "burst" is never archived, and Holly's safety lines are fixed text that the
   model never sees.
3. **Honest, pre-registered evaluation.** Answer keys were written before the code, decision rules before the runs,
   one variable per experiment. Negative results are kept: Part 1's category target was not met and a prompt change
   was rejected by its own rule; in Part 2 a plain-code rules sync beat the agent on perfect input (14/14 vs 11/14),
   and the agent's one-task lead on real input is within noise.
4. **Reliability like a real service.** Idempotent by SHA-256 of the audio (re-running does nothing), retries only on
   transient errors (2 s, 4 s), at-least-once push (push before save), failed files go to `failed/` with an error
   note, and a failed or interrupted tracker sync is rolled back.
5. **Privacy by design (GDPR).** Voicemails and calls contain other people's personal data: audio, transcripts,
   databases and logs stay on the machine and are git-ignored, logs never contain transcript text, and push
   notifications carry no names, numbers or content.
6. **Tested and reviewed.** 513 tests run without any model (routing, guards, rollback, dialog invariants such as
   "nothing is ever invented"). After each phase an independent review checked the work against the plan; it found real
   bugs, for example a merge bug in the extraction code that looked like a model error, a double-apply of tracker
   changes after a crash, and safety text reaching the rewording model as context.

### Where to look in the code
| To see | Look at |
|---|---|
| Routing as plain, testable rules + the safety-word net | `01-voicemail-triage/routing.py` (`route()`, `SAFETY_PATTERNS`) |
| "Fill a form, validate, retry once" | `shared/llm.py` (`structured_chat()`), `shared/schemas.py` |
| Idempotency, retries, `failed/` | `01-voicemail-triage/run.py`, `shared/retry.py`, `shared/pipeline.py` |
| A bounded agent and its guards | `02-meeting-action-agent/agent.py` (`run_tool()`), traces in `traces/` |
| Plain-code baseline the agent is measured against | `02-meeting-action-agent/rules_sync.py` |
| Rollback of a failed sync | `02-meeting-action-agent/run.py` (`sync_with_rollback()`), `tracker.py` (change log + undo) |
| Grounding the model's answer in the caller's words | `03-phone-receptionist/turn.py` (`ground()`) |
| The dialog as a UI-independent state machine | `03-phone-receptionist/dialog.py` (`next_reply()`, `apply_turn()`, `decide_a()`) |
| Agent version of the dialog, boxed by the state | `03-phone-receptionist/agent_dialog.py` |
| Checks on every model-reworded sentence | `03-phone-receptionist/phrase.py` (`check_sentence()`) |

## Architecture

```
                 shared/  transcribe (Whisper + name hint) · llm.structured_chat · schemas · analyze
                          tts (Piper) · retry · notify (ntfy) · pipeline (inbox / failed / logs)
                                         │ reused by all three parts
     ┌───────────────────────────────────┼───────────────────────────────────┐
     ▼                                   ▼                                   ▼
 PART 1 voicemail                  PART 2 meeting                     PART 3 phone call
 transcribe → analyze (form)       transcribe (+ name hint)           listen → understand (form + grounding)
 → route (plain rules)             → extract items (workflow)         → decide (A state machine | B agent)
 → push / voicemails.db            → sync tracker (agent | rules)     → speak (approved sentences)
     ▲                             → tracker.db                              │
     └──────────────── hand-off of the finished message (analyze, route, push, voicemails.db) ──┘
```

### Part 1: voicemail triage (fixed workflow)

```
 voicemail file (wav/mp3/m4a/ogg/...)
        │  dropped into 01-voicemail-triage/inbox/
        ▼
 run.py ── hash (SHA-256) ── already in SQLite? ──yes──► processed/   (idempotent: nothing else happens)
        │no
        ▼
 1. TRANSCRIBE  shared/transcribe.py   ffmpeg → 16 kHz mono wav → faster-whisper `small`
        │                              cached: transcripts/<hash>_<model>.json
        ▼
 2. ANALYZE     shared/analyze.py      Ollama qwen2.5:7b, JSON forced by the Pydantic schema,
        │                              temperature 0, validated by plain-code rules, 1 retry with feedback
        │                              cached: results/<hash>_<whisper>_<llm>_<prompt>.json
        ▼
 3. ROUTE       routing.py (pure)      category → notify_now | inbox | personal | archive
        │                              + safety-word net (never archive "smell of gas", "burst", ...)
        │                              + review flags (no number/name, retry needed, unclear audio)
        ▼
 4. DELIVER     deliver.py             ntfy push if needed (minimal text, retried) → save row (store.py)
        ▼
 processed/                            any error at any step → failed/<file> + <file>.error.txt
 digest.py                             daily page of callbacks, review items and failures
```

**Why a workflow, not an agent:** the order of steps is fixed in code. The LLM fills in one form and never chooses what
happens next; routing is a set of `if` rules you can read and test, so every outcome can be explained from the logs and
JSON files. Measured on 18 synthetic voicemails (CPU): **0 of 6 urgent voicemails missed**, callback number right 16/18
with none invented, category 15/18 (83 %, the 90 % target is **not met**). A one-line prompt change (v3) brought surnames
back but changed answers on unrelated files and failed its pre-set rule, so v2 stayed. Details:
[docs/eval-results.md](docs/eval-results.md), [docs/eval-notes.md](docs/eval-notes.md).

### Part 2: meeting action items + tracker sync (workflow + agent)

```
 inbox/<meeting>.m4a  (date from YYYY-MM-DD in the name; oldest first)       run.py
   0. hash already in tracker.meetings? ── yes ──► processed/ (nothing else)
   1. TRANSCRIBE  Whisper small + a hint with the team's and customers' names (context.py)
   2. EXTRACT     workflow: LLM fills MeetingItems; validators: owner on the team, quote in the transcript;
                  ground_owner() drops owners not named nearby; dates.py turns "by Wednesday" into a date in code
   3. SYNC        --sync agent (default): tool loop on qwen2.5:7b, fresh state message each step, guards in code
                  --sync rules:           rules_sync.py, plain code, no LLM
                  tracker.py: tasks + change log with a reason per change; a failed sync is rolled back
   4. RECORD      tracker.meetings + move to processed/   (any error: failed/ + .error.txt)
```

The agent only decides **relationships** (this item updates task 3, this one is new, this one is not work): its tools
take an item letter and a task number, and code copies owner, date and status from the item, so it cannot invent
values. Measured on 5 synthetic weekly meetings with 14 tasks (CPU):

| Step | Result |
|---|---|
| Transcription, `small` + name hint | 98 % of names right (71 % without the hint), 5.3 % word errors |
| Extraction, task + owner | precision 64 %, recall 57 % (task only: 84 % / 75 %); 80 % target **not met** |
| Sync on perfect (answer-key) items | rules **14/14**, agent 11/14 |
| Sync on the real extracted items | agent **7/14**, rules 6/14 (a perfect sync of these items would reach 9/14) |
| Cost of the agent | 51 LLM calls, ~94 min for both runs; rules: under a second |

By the rule written before the agent runs, the agent stays the default, but its +1 is within run-to-run noise and the
real bottleneck is extraction. Step limits, the tool allowlist, logged reasons and idempotency were all met. Details:
[02-meeting-action-agent/README.md](02-meeting-action-agent/README.md), [docs/part2-eval-results.md](docs/part2-eval-results.md).

### Part 3: Holly, the phone receptionist (state machine first, agent as an experiment)

Holly says she is an automated assistant and that the call is recorded, answers simple questions from `faq.json` only,
takes a message (reason, name, callback number, urgency), flags emergencies at once, and hands the message to Part 1.
Calls come from the browser microphone (push-to-talk) or from scripted synthetic callers; there is no telephony provider.

```
 caller's words ─► UNDERSTAND  (LLM fills a small form; plain code checks every name / number against the words)
                ─► APPLY       (shared rules: never invent, emergencies, FAQ only from faq.json, spam)
                ─► DECIDE      (version A: state machine in plain code | version B: tool-calling agent, boxed by the state)
                ─► RENDER      (fixed approved sentences from persona.json / faq.json)
                ─► REWORD      (optional: the LLM makes the sentence sound natural; code checks it, any doubt = fixed sentence)
                ─► speak ─► at the end: hand-off to Part 1 (analyze, route, live urgent push, voicemails.db)
```
- **A model never decides what is said.** The engine is plain Python (`next_reply(state, caller_text)`), with no screen
  or microphone, so it is tested headless. Every sentence is an approved line or a checked rewording of one.
- **Safety lines** (urgent acknowledgement, gas / water / carbon monoxide advice, "nine nine nine", the urgent goodbye)
  are never reworded and never sent to the model. An urgent call triggers a push during the call.
- **Version B** may only pick among the legal next moves; no tool takes a value, and any failure falls back to the
  state machine's choice and is logged.
- **Evaluation:** the harness and the A-vs-B decision rule (written before any run) are built; the formal scored run has
  not been done yet, so there are no Part 3 accuracy numbers. Version A is the default. It was tested live with the
  real models and by 424 automated tests.
- Details, every design decision and the pre-registered rule: [03-phone-receptionist/README.md](03-phone-receptionist/README.md).

## Quick start (Windows / PowerShell)
Needs Python 3.13, [ffmpeg](https://ffmpeg.org) on PATH and [Ollama](https://ollama.com).
```powershell
py -V:3.13-64 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
ollama pull qwen2.5:7b            # ~4.7 GB
python scripts\check_setup.py     # all [OK]? then you're ready
```
The first transcription downloads the Whisper model (`small`, ~480 MB) to your user cache. Rebuilding the synthetic
test audio downloads the Piper voice `en_GB-vctk-medium` (~73 MB).

**GPU (optional, NVIDIA).** Ollama uses the GPU by itself. For Whisper on the GPU, `requirements.txt` also installs the CUDA libraries
(`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, ~1.2 GB; `shared/transcribe.py` finds their DLLs) and `.env` needs `WHISPER_DEVICE=cuda` and
`WHISPER_COMPUTE_TYPE=float16`. Measured on the RTX 5050: a 30 s clip with `small` takes 9.8 s on the CPU and 1.7 s on the GPU.
Without the libraries, set `WHISPER_DEVICE=cpu` (`auto` picks the GPU and then fails).

## Try it

### The local app (easiest)
Double-click **`start_app.bat`** (or run `python -m streamlit run app/app.py`): a page opens at
http://localhost:8501, reachable only from this computer.
- **Voicemails** tab: drop audio files, or record one with the microphone → *Process the inbox* → a table of results (route, category, caller,
  number, summary, review flags); click a row for the transcript, the reasons for the route and the audio.
- **Meetings** tab: drop recordings (pick the meeting date if the file name has none) → choose *agent* or
  *rules* → *Process the inbox* → the task tracker, the items found in each meeting, every tracker change
  with its reason, and the agent's steps.
- **Receptionist** tab (Part 3): press the microphone, speak, and Holly answers (version A state machine or B
  agent; model or rules understanding; natural or fixed wording). Rehearsal only: calls are not saved. Replies show as text until a voice
  is chosen (`choose_voice.py`). The models load in the background when a call starts.
- Runs happen in the background with a live log; failed files show the step and error, with *Retry failed*.
- The sidebar warns if ffmpeg or Ollama is missing, or if there's too little free memory (~6 GB needed).

The app uses the same inbox folders, databases and `run.py` scripts as the command line. Its code is in `app/`
(`helpers.py` = the logic, `app.py` = the page).

### Part 1 from the command line
```powershell
python 01-voicemail-triage\testset\generate.py     # build the 18 synthetic test voicemails (Piper TTS)
copy 01-voicemail-triage\testset\audio\01_urgent_burst_pipe.wav 01-voicemail-triage\inbox\
python 01-voicemail-triage\run.py                  # process everything in inbox\ once
# copy the same file in again and re-run: "already processed (same audio ...)", 1 skipped
python 01-voicemail-triage\run.py --watch          # keep watching inbox\ (Ctrl+C stops)
python 01-voicemail-triage\run.py --retry-failed   # move failed\ files back and try again
python 01-voicemail-triage\store.py                # counts per route + review list
python 01-voicemail-triage\digest.py               # today's digest (saved in digests\)
```
Run it automatically with Windows Task Scheduler (adjust paths; run these yourself):
```powershell
schtasks /Create /TN "Voicemail watcher" /SC ONLOGON /TR "\"C:\Users\danuk\code\audio-to-action\.venv\Scripts\python.exe\" \"C:\Users\danuk\code\audio-to-action\01-voicemail-triage\run.py\" --watch"
schtasks /Create /TN "Voicemail digest" /SC DAILY /ST 18:00 /TR "\"C:\Users\danuk\code\audio-to-action\.venv\Scripts\python.exe\" \"C:\Users\danuk\code\audio-to-action\01-voicemail-triage\digest.py\" --push"
```

### Part 2 from the command line
```powershell
python 02-meeting-action-agent\testset\generate.py   # build the 5 synthetic meetings (4 Piper voices)
copy 02-meeting-action-agent\testset\audio\m1_2026-09-07.wav 02-meeting-action-agent\inbox\
python 02-meeting-action-agent\run.py                # agent sync (default); or --sync rules, --watch, --retry-failed
```
Look inside: `tracker.db` (tables `tasks`, `changes` with the reason for every change, `meetings`) and the agent's
trace per meeting in `02-meeting-action-agent/traces/`.

### Part 3 from the command line
```powershell
python 03-phone-receptionist\call.py --card c03 --understand rules     # a simulated call, no model needed
python 03-phone-receptionist\call.py --card c03 --phrase natural       # the LLM understands and rewords (needs Ollama)
python 03-phone-receptionist\call.py --card dev --decide b             # version B, the agent (needs Ollama)
```
More (audio loop, hand-off to Part 1, evaluation): section 8 of [03-phone-receptionist/README.md](03-phone-receptionist/README.md).

### Push notifications (ntfy)
Pushes are **dry runs** (printed, not sent) until you configure them. To go live: install the ntfy app,
pick a long random topic, subscribe to it in the app, then in `.env` set `NTFY_TOPIC=<that topic>` and
`NTFY_DRY_RUN=0`. Push texts never contain names, numbers or content ("Urgent voicemail, received 14:32"),
because the public ntfy.sh server and anyone guessing the topic can read them.

## Configuration (`.env`)
| Variable | Default | Meaning |
|---|---|---|
| `WHISPER_MODEL` | `small` | `large-v3-turbo` is more accurate, ~3× slower on CPU ([comparison](docs/transcription-comparison.md)) |
| `WHISPER_DEVICE` / `WHISPER_COMPUTE_TYPE` | `auto` / `int8` | `cuda` / `float16` for the GPU (needs the CUDA libraries above); `cpu` / `int8` otherwise |
| `WHISPER_CPU_THREADS` | `8` | 0 = library default (4) |
| `OLLAMA_HOST` / `OLLAMA_MODEL` | `http://localhost:11434` / `qwen2.5:7b` | |
| `ANALYSIS_PROMPT` | `v2` | prompt version in `shared/analyze.py` (`PROMPTS`) |
| `NTFY_SERVER` / `NTFY_TOPIC` / `NTFY_DRY_RUN` | `https://ntfy.sh` / placeholder / `1` | see above |

## Tests and evaluation
```powershell
python -m pytest 01-voicemail-triage\tests 02-meeting-action-agent\tests 03-phone-receptionist\tests app\tests -q   # 513 tests, no model needed
python 01-voicemail-triage\evaluate.py              # Part 1 metrics on the labelled test set → docs\eval-results.md
python 01-voicemail-triage\route_testset.py         # routes the test set, fails if an urgent one is archived
python 01-voicemail-triage\compare_models.py        # Whisper small vs large-v3-turbo
python 02-meeting-action-agent\evaluate.py          # Part 2 report from saved runs → docs\part2-eval-results.md
```
| Suite | Tests | Covers |
|---|---|---|
| `01-voicemail-triage/tests` | 11 | routing rules, safety net, retries |
| `02-meeting-action-agent/tests` | 70 | dates, extraction checks, agent guards, rules sync, rollback |
| `03-phone-receptionist/tests` | 424 | dialog flow, grounding, emergencies, FAQ, agent fallback, wording checks, hand-off |
| `app/tests` | 8 | the app's helper logic |

The test sets are synthetic (Piper voices, fictional UK numbers, no personal data), and each answer key was written
before the code it scores: 18 voicemails ([testset/README](01-voicemail-triage/testset/README.md)), a series of 5 weekly
meetings ([testset/README](02-meeting-action-agent/testset/README.md)) and 24 caller cards split into tuning and scoring
cards ([testset/README](03-phone-receptionist/testset/README.md)).

## Repository layout
```
01-voicemail-triage/     Part 1: run.py, routing.py, deliver.py, store.py, digest.py, evaluate.py, testset/, tests/
02-meeting-action-agent/ Part 2: run.py, extract.py, agent.py, rules_sync.py, tracker.py, dates.py, testset/, tests/
03-phone-receptionist/   Part 3: dialog.py, turn.py, agent_dialog.py, phrase.py, handoff.py, call.py, persona.json, faq.json
shared/                  transcribe, llm (structured_chat), schemas, analyze, tts, retry, notify, pipeline, evaluation
app/                     local Streamlit app (helpers.py = logic, app.py = page); start_app.bat launches it
docs/                    evaluation reports, experiment notes, learning log, technical handover
scripts/check_setup.py   checks Python, ffmpeg, faster-whisper and Ollama
```

## Privacy
Voicemails and calls contain other people's personal data (GDPR). Audio, transcripts, results, call records, logs,
digests and the databases stay on this machine and are git-ignored; keep the repo outside synced folders
(OneDrive, Dropbox). Logs record file names, steps and timings, never transcript content. Holly announces that the
call is recorded.

## Docs
- [docs/learning-log.md](docs/learning-log.md): every phase explained (what, how, concepts, questions)
- [docs/eval-results.md](docs/eval-results.md) and [docs/eval-notes.md](docs/eval-notes.md): Part 1 metrics, every error, the prompt experiment
- [docs/transcription-comparison.md](docs/transcription-comparison.md): Whisper model comparison on voicemails
- [docs/part2-eval-results.md](docs/part2-eval-results.md): Part 2 metrics, the agent-vs-rules experiment, definition of done
- [docs/part2-transcription-comparison.md](docs/part2-transcription-comparison.md): Whisper models × name hint on meetings
- [docs/part3-eval-notes.md](docs/part3-eval-notes.md): Part 3 metrics and the A-vs-B rule
