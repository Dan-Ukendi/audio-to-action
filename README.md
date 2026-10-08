# audio-to-action

A learning project about using AI to handle phone calls and audio, **fully local and free**:
no cloud APIs, no audio leaving the machine.

| Part | Folder | What | Style | Status |
|---|---|---|---|---|
| 1 | `01-voicemail-triage/` | Voicemails → transcript → category, urgency, caller, number, summary → routed | Workflow | done |
| 2 | `02-meeting-action-agent/` | Meeting recordings → action items → task tracker sync | Workflow + agent | done |
| 3 | `03-phone-receptionist/` | Real-time AI phone receptionist | State machine, then agent | in progress (Phase 0 built) |

`shared/` holds code reused across parts: transcription (+ name hint), schemas, the "fill a form with the LLM,
validate, retry once" helper (`llm.py`), text-to-speech (`tts.py`), eval helpers, retries, notifications and
inbox-pipeline helpers (`pipeline.py`).

## Part 1: how it works

```
 voicemail file (wav/mp3/m4a/ogg/...)
        │  dropped into 01-voicemail-triage/inbox/
        ▼
 run.py ── hash (SHA-256) ── already in SQLite? ──yes──► processed/   (idempotent: nothing else happens)
        │no
        ▼
 1. TRANSCRIBE  shared/transcribe.py   ffmpeg → 16 kHz mono wav → faster-whisper `small` (CPU, int8)
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

**Why a workflow, not an agent:** the order of steps is fixed in code. The LLM fills in one form
(category, urgency, name, number, summary) and never chooses what happens next. Routing is a set of
`if` rules you can read and test. That makes every outcome explainable from the logs and the JSON files.

## Part 2: how it works (summary)
Weekly meeting recordings go into `02-meeting-action-agent/inbox/`; `run.py` transcribes them (Whisper + a hint with
the team's and customers' names), **extracts** action items (workflow: the LLM fills a form, code checks owners and
quotes and computes due dates) and **syncs** a SQLite task tracker across meetings, either with a bounded
tool-calling **agent** (default) or with plain-code **rules** (`--sync rules`). A failed sync is rolled back.
Measured: the agent beat the rules by one task on real extracted items (7/14 vs 6/14) at ~50 minutes of CPU per
five meetings; on perfect input the rules won (14/14 vs 11/14). Details: [02-meeting-action-agent/README.md](02-meeting-action-agent/README.md).

## Setup (Windows / PowerShell)
Needs Python 3.13, [ffmpeg](https://ffmpeg.org) on PATH and [Ollama](https://ollama.com).
```powershell
py -V:3.13-64 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
ollama pull qwen2.5:7b            # ~4.7 GB
python scripts\check_setup.py     # all [OK]? then you're ready
```
The first transcription downloads the Whisper model (`small`, ~480 MB) to your user cache.

## Using it
```powershell
python 01-voicemail-triage\run.py                  # process everything in inbox\ once
python 01-voicemail-triage\run.py --watch          # keep watching inbox\ (Ctrl+C stops)
python 01-voicemail-triage\run.py --retry-failed   # move failed\ files back and try again
python 01-voicemail-triage\digest.py               # today's digest (saved in digests\)
python 01-voicemail-triage\store.py                # counts per route + review list
```
Run it automatically with Windows Task Scheduler (adjust paths; run these yourself):
```powershell
schtasks /Create /TN "Voicemail watcher" /SC ONLOGON /TR "\"C:\Users\danuk\code\audio-to-action\.venv\Scripts\python.exe\" \"C:\Users\danuk\code\audio-to-action\01-voicemail-triage\run.py\" --watch"
schtasks /Create /TN "Voicemail digest" /SC DAILY /ST 18:00 /TR "\"C:\Users\danuk\code\audio-to-action\.venv\Scripts\python.exe\" \"C:\Users\danuk\code\audio-to-action\01-voicemail-triage\digest.py\" --push"
```

### Push notifications (ntfy)
Pushes are **dry runs** (printed, not sent) until you configure them. To go live: install the ntfy app,
pick a long random topic, subscribe to it in the app, then in `.env` set `NTFY_TOPIC=<that topic>` and
`NTFY_DRY_RUN=0`. Push texts never contain names, numbers or content ("Urgent voicemail, received 14:32"),
because the public ntfy.sh server and anyone guessing the topic can read them.

## Configuration (`.env`)
| Variable | Default | Meaning |
|---|---|---|
| `WHISPER_MODEL` | `small` | `large-v3-turbo` is more accurate, ~3× slower on CPU ([comparison](docs/transcription-comparison.md)) |
| `WHISPER_DEVICE` / `WHISPER_COMPUTE_TYPE` | `auto` / `int8` | `cuda` / `float16` once the GPU works |
| `WHISPER_CPU_THREADS` | `8` | 0 = library default (4) |
| `OLLAMA_HOST` / `OLLAMA_MODEL` | `http://localhost:11434` / `qwen2.5:7b` | |
| `ANALYSIS_PROMPT` | `v2` | prompt version in `shared/analyze.py` (`PROMPTS`) |
| `NTFY_SERVER` / `NTFY_TOPIC` / `NTFY_DRY_RUN` | `https://ntfy.sh` / placeholder / `1` | see above |

## Testing and evaluation
```powershell
python -m pytest 01-voicemail-triage\tests          # 11 unit tests: routing rules, retries (instant)
python -m pytest 02-meeting-action-agent\tests      # 70 unit tests: dates, extraction, agent guards, rollback
python 02-meeting-action-agent\evaluate.py          # Part 2 report from saved runs → docs\part2-eval-results.md
python 01-voicemail-triage\evaluate.py              # metrics on the labelled test set → docs\eval-results.md
python 01-voicemail-triage\route_testset.py         # routes the test set, fails if an urgent one is archived
python 01-voicemail-triage\compare_models.py        # Whisper small vs large-v3-turbo
python 01-voicemail-triage\testset\generate.py      # rebuild the synthetic test audio (Piper TTS)
```
The test set is 18 synthetic English voicemails for a fictional UK plumbing business, with an answer
key written before the pipeline existed ([testset/README](01-voicemail-triage/testset/README.md)).

## Results (Part 1)
Default config `small:v2` (Whisper `small`, prompt v2, `qwen2.5:7b`, CPU) on the 18-file test set
([full report](docs/eval-results.md)):

| Metric | Result | Target |
|---|---|---|
| Urgent voicemails missed | **0/6** | 0 |
| Urgent voicemails archived | **0** | 0 |
| Category accuracy | 15/18 (83 %) | ≥ 90 %, **not met** |
| Callback number correct | 16/18 (89 %), none invented | |
| Caller name exact / first name | 10/18 / 14/18 | |
| Re-running the same file | does nothing (hash check) | ✔ |
| Time per new voicemail | ~20 s Whisper + ~1-4 min LLM on CPU | GPU would make this seconds |

Most remaining errors come from transcription (misheard names and digits over a phone-quality line) and
one scam that sounds urgent. A one-line prompt experiment (v3) changed answers on unrelated files and was
not adopted; further gains need more test data, not more prompt tweaks. See [eval notes](docs/eval-notes.md).

## Privacy
Voicemails contain other people's personal data (GDPR). Audio, transcripts, results, logs, digests and
the database stay on this machine and are git-ignored; keep the repo outside synced folders
(OneDrive, Dropbox). Logs record file names, steps and timings, never transcript content.

## Docs
- [docs/learning-log.md](docs/learning-log.md): every phase explained (what, how, concepts, questions)
- [docs/eval-results.md](docs/eval-results.md): metrics, confusion matrix, every error, experiment
- [docs/transcription-comparison.md](docs/transcription-comparison.md): Whisper model comparison
- [docs/HANDOVER.md](docs/HANDOVER.md): full technical report (architecture, code, decisions) for another developer or agent
- [docs/shared-for-part2.md](docs/shared-for-part2.md): what Part 2 could reuse (written before Part 2)
- [docs/part2-eval-results.md](docs/part2-eval-results.md): Part 2 metrics, the agent-vs-rules experiment, definition of done
- [docs/part2-transcription-comparison.md](docs/part2-transcription-comparison.md): Whisper models × name hint on meetings

## Status
- [x] Phase 0: Setup
- [x] Phase 1: Test set (18 synthetic English voicemails)
- [x] Phase 2: Transcription (`shared/transcribe.py`)
- [x] Phase 3: Analysis (`shared/analyze.py`, structured output + validation + 1 retry)
- [x] Phase 4: Routing (`routing.py`, ntfy dry run, SQLite)
- [x] Phase 5: Glue & reliability (`run.py`: idempotent, retries, `failed/`, logs)
- [x] Phase 6: Evaluation (`evaluate.py`, `docs/eval-results.md`, one experiment)
- [x] Phase 7: Polish (digest, README, handover report)

Part 2 (meetings): [x] 0 plan · [x] 1 test set · [x] 2 transcription + hint · [x] 3 extraction · [x] 4 agent ·
[x] 5 glue & rollback · [x] 6 evaluation (agent vs rules) · [x] 7 polish.
Part 3 (phone receptionist): [~] 0 plan + persona + FAQ (built; owner confirmation of wording/FAQ and the laptop GPU check pending) · [x] 1 caller cards · [ ] 2 audio loop · [ ] 3 dialog A ·
[ ] 4 hand-off · [ ] 5 dialog B · [ ] 6 evaluation · [ ] 7 polish. Plan: `docs/PART3_PLAN.md`; build notes and decisions:
`03-phone-receptionist/README.md`.
