# audio-to-action: technical handover (Part 1 complete)

Purpose of this document: give another developer or AI agent everything needed to understand,
run, modify and extend this repository without reading the whole conversation history.
It covers goals and constraints, architecture, every module's contract, data formats, decisions
with their reasons, measured results, known issues and the next steps.

---

## 1. Project goal and ground rules

A 3-part **learning** repository about using AI on phone calls and audio. The owner values
understanding over speed. Everything runs **locally and for free**.

- **Part 1, `01-voicemail-triage/` (DONE):** voicemail triage as a *fixed workflow*:
  audio → transcribe → analyze (LLM fills a schema) → route (plain-code rules) → notify/store.
- **Part 2, `02-meeting-action-agent/` (not started):** meeting recordings → action items (workflow + agent).
- **Part 3, `03-phone-receptionist/` (not started):** real-time AI phone receptionist (full agent).

Rules from `PROJECT_RULES.md` that any agent working here must follow:
1. One phase at a time; stop after each and wait for the user to say "next" (unless told otherwise).
2. Explain in 2-4 sentences before writing code; small readable functions; comments explain *why*.
3. Ask before installing anything large or downloading models (state the size first). Ask, don't
   assume, when a choice is the user's.
4. Free and local by default: Python, ffmpeg, faster-whisper, Ollama, Pydantic, SQLite, ntfy. No paid APIs.
5. **Privacy (GDPR):** voicemails contain third parties' personal data. Audio, transcripts, results,
   DB, logs, digests stay local and git-ignored. Never commit audio or `.env`. Push text carries no
   caller data (public ntfy.sh can read it).
6. **Workflow, not agent:** fixed sequence of plain functions; routing rules are code, never LLM decisions.
7. Everything inspectable: each step saves its output to disk (JSON).
8. Commit after each phase. 9. Run things yourself, show results, and give the exact commands.
10. At the end of every phase: a 9-section recap in chat **and** appended to `docs/learning-log.md`.
11. Always open text files with `encoding="utf-8"` (Windows defaults to cp1252).

## 2. Environment

| Item | Value |
|---|---|
| Machine | Windows 11 25H2 (build 26200), Medion Erazer Scout 15 E1 (Clevo board V2xxRNP), i5-13420H (8 cores / 12 threads), 16 GB RAM |
| GPU | RTX 5050 Laptop 8 GB: **not usable**, Device Manager "Code 43", NVIDIA driver 572.90 (April 2025). Fix = install current NVIDIA notebook driver (clean install) + restart; the user has not done this yet |
| Python | 3.13 in `.venv` (created with `py -V:3.13-64 -m venv .venv`; system default 3.14 is too new for some wheels) |
| Repo | `C:\Users\danuk\code\audio-to-action` (deliberately outside OneDrive) |
| ffmpeg | 9.0.2 via winget, on the user PATH. Terminals opened before the install don't see it |
| Ollama | 0.35.0, model `qwen2.5:7b` (4.7 GB, Q4). Runs **100% CPU** because of the GPU issue: ~50-220 s per voicemail depending on machine load |
| Whisper | faster-whisper 1.2.1 / CTranslate2 4.8.2, CPU int8, models in `%USERPROFILE%\.cache\huggingface\hub` (`small` 484 MB, `large-v3-turbo` 1.6 GB) |
| Key packages | pydantic 2.13, ollama (python) 0.6.3, requests, python-dotenv, numpy, av 19 (installed but bypassed, see §9), piper-tts (test-set generation only), pytest 9 |

## 3. Repository layout

```
audio-to-action/
├── PROJECT_RULES.md                     rules + phase list (the project's working rules)
├── README.md                     user-facing overview, setup, usage, results
├── requirements.txt / .env.example / .gitignore
├── scripts/check_setup.py        sanity check: python/venv, ffmpeg, faster-whisper(+CUDA?), Ollama+model
├── shared/                       reusable across Parts 1-3
│   ├── schemas.py                Segment, Transcript, Category, Analysis (+validators), Result
│   ├── transcribe.py             transcribe(path, cache_dir, model) -> Transcript
│   ├── analyze.py                analyze(transcript, cache_dir, llm, prompt_version) -> Result; PROMPTS
│   ├── evaluation.py             scoring helpers (same_value, same_first_word, null_aware, confusion, pct)
│   ├── retry.py                  with_retries(fn, ...), is_transient(exc)
│   └── notify.py                 send_push(title, message, priority, tags) -> "sent" | "dry_run"
├── 01-voicemail-triage/
│   ├── run.py                    THE PIPELINE: inbox/ -> ... -> processed/ | failed/
│   ├── routing.py                route(transcript, result) -> Decision   (pure)
│   ├── deliver.py                deliver(conn, transcript, result) -> Decision (push + save)
│   ├── store.py                  SQLite: connect(), save(), summary()
│   ├── digest.py                 daily digest page + optional counts-only push
│   ├── evaluate.py               metrics on the labelled set -> docs/eval-results.md
│   ├── route_testset.py          routes the test set into a throwaway DB; exit 1 if urgent archived
│   ├── compare_models.py         Whisper model comparison -> docs/transcription-comparison.md
│   ├── tests/test_routing.py     7 tests;  tests/test_retry.py  4 tests
│   ├── testset/                  scripts.json, labels.json, generate.py, README.md (+ ignored audio/, voices/,
│   │                             transcripts/, results/, my_recordings/, triage_test.db)
│   └── inbox/ processed/ failed/ transcripts/ results/   (contents ignored; .gitkeep committed)
│       logs/ digests/ voicemails.db                       (ignored, created at runtime)
└── docs/
    ├── learning-log.md           per-phase recaps (the pedagogical record)
    ├── eval-results.md           generated metrics + confusion matrices + every error
    ├── eval-notes.md             hand-written conclusions, inserted into eval-results.md by evaluate.py
    ├── transcription-comparison.md
    ├── shared-for-part2.md       reuse guide for Part 2
    └── HANDOVER.md               this file
```

Import convention: the folder `01-voicemail-triage` starts with a digit, so it can't be a package.
Each script there does `sys.path.insert(0, <repo root>)` and `sys.path.insert(0, <its own folder>)`,
then imports `shared.*` and sibling modules (`routing`, `store`, `deliver`, `run`) by bare name.
`shared` modules are run as `python -m shared.transcribe` / `python -m shared.analyze` from the repo root.

## 4. Architecture and data flow

```
inbox/<file>
  │ run.find_ready_files(): audio extension, mtime >= 3 s old (may still be copying), oldest first
  ▼
run.process_file(path, folders, conn)            never raises; returns processed | skipped | failed
  0. file_sha256(path) → already_processed(conn, hash)?  yes → move to processed/, "skipped"
  1. transcribe(path, cache_dir=transcripts/)           → Transcript   (JSON cache)
  2. with_retries(analyze, transcript, cache_dir=results/) → Result    (JSON cache)
  3. deliver(conn, transcript, result)                  → Decision
        route() → if notify: with_retries(send_push, ...) → store.save(... notified_at)
  4. move to processed/
  on any exception: log, move to failed/, write failed/<file>.error.txt (step + traceback)
```

Properties this design guarantees:
- **Idempotent by content:** the SQLite primary key is the SHA-256 of the audio bytes. The same audio
  under any file name is skipped (no LLM call, no push, no DB change). Caches are also keyed by hash,
  so a re-run after a failure reuses finished steps.
- **Failure isolation:** one bad file never stops the batch (`process_file` catches `Exception`).
- **At-least-once push:** push happens *before* save. A crash between them causes a duplicate push on
  the next run (acceptable); the opposite order could mark an urgent voicemail done with no push (not acceptable).
- **Retries only for transient errors:** builtin `ConnectionError` (what the ollama client raises when
  Ollama is down), `TimeoutError`, `requests.ConnectionError/Timeout`, `httpx.TransportError`, HTTP 429/5xx.
  3 attempts, waits 2 s then 4 s. Validation errors, ffmpeg errors and `AnalysisError` fail immediately.
- **Workflow, not agent:** the LLM only fills the `Analysis` form; it never selects steps or routes.

## 5. Module reference

### `shared/schemas.py`
- `Segment(start, end, text, avg_logprob, no_speech_prob)`.
- `Transcript(source_file, audio_sha256, model, language, language_probability, duration_s,
  transcribe_s, text, segments, created_at)`. `source_file` is the bare name (no folders, for privacy).
- `Category = Literal["urgent","sales","personal","spam","other"]`.
- `Analysis` (the LLM's form; its JSON schema is sent to Ollama). **Field order is deliberate**:
  `summary, reason` first, then `category, urgency`, then `caller_name, callback_number, language`
  (the model writes its justification before committing to a decision). Validators (= plain-code rules):
  - `check_uk_number`: strip non-digits; empty → None; must match `^0\d{9,10}$` else ValueError
    (message tells the model that "oh"/"o"/"a" may stand for 0).
  - `empty_name_is_none`: "", "unknown", "none", "n/a", "null" → None; strip.
  - `two_letter_language`: "en-GB" → "en".
  - `urgent_means_today` (model validator): `category == "urgent"` **iff** `urgency == 3`.
- `Result(source_file, audio_sha256, transcript_model, llm_model, prompt_version, attempts,
  rejected_reply, rejected_because, analyze_s, analysis, created_at)`.

### `shared/transcribe.py`
`transcribe(path, cache_dir=None, model=None) -> Transcript`
1. `settings()` from env: `WHISPER_MODEL` (small), `WHISPER_DEVICE` (auto), `WHISPER_COMPUTE_TYPE` (int8), `WHISPER_CPU_THREADS` (0 → library default 4; `.env` sets 8).
2. `file_sha256()` (1 MB chunks) → cache file `<cache_dir>/<hash[:16]>_<model>.json`; hit → return.
3. `to_wav_16k()`: `ffmpeg -nostdin -y -loglevel error -i src -ac 1 -ar 16000 -c:a pcm_s16le dst` in a
   TemporaryDirectory. Raises `RuntimeError` with ffmpeg's stderr. Runs **before** model loading so broken files fail in ~0 s.
4. `read_wav_samples()`: int16 → float32 / 32768 (numpy). Samples, not a path, are passed to Whisper (§9 PyAV bug).
5. `load_model()` (`@lru_cache(maxsize=2)`), then `whisper.transcribe(samples, beam_size=5, vad_filter=True)`;
   the segment iterator is lazy (work happens while iterating).
6. Write JSON to `.tmp`, then `replace()` (atomic: no half-written cache files).
CLI: `python -m shared.transcribe <file> [--model M] [--cache-dir D]`.

### `shared/analyze.py`
`analyze(transcript, cache_dir=None, llm=None, prompt_version=None) -> Result`
- `PROMPTS = {"v2": PROMPT_V2, "v3": PROMPT_V2.replace(V2_EXAMPLE, V3_EXAMPLE)}` (v3 differs in exactly one line,
  asserted at import). `DEFAULT_PROMPT_VERSION = os.getenv("ANALYSIS_PROMPT", "v2")`. Never edit a version in place.
- Cache file `<hash[:16]>_<whisper model>_<llm with ':'→'-'>_<prompt version>.json` (':' is illegal in Windows names).
- Messages: system = prompt (business context: Brightwater Plumbing & Heating, owner Sam; definitions for each
  category, urgency scale, name rules, number rules incl. "a 1632…" = "01632…", corrected numbers, never guess);
  user = `Voicemail transcript:\n<<<\n{text}\n>>>`.
- `ask_llm()`: `ollama.Client(host).chat(model, messages, format=Analysis.model_json_schema(), options={"temperature": 0})`.
- Validate with `Analysis.model_validate_json(raw)`. On `ValidationError`, append the assistant's reply and a user
  message: "Your answer broke these rules: <short_errors> Fix the values using the transcript; do not replace a
  value the caller actually said with null. Return the corrected JSON." Second failure → `AnalysisError`.
  The rejected first reply and the reason are stored in `Result`.
CLI: `python -m shared.analyze <audio> [--transcripts D] [--results D]`.

### `01-voicemail-triage/routing.py` (pure)
`route(transcript, result) -> Decision(route, notify, review, reasons)`
1. `CATEGORY_ROUTE`: urgent→`notify_now`, other→`inbox`, personal→`personal`, spam→`archive`, sales→`archive` (user's choice).
2. Safety net: `safety_hits(text)` over `SAFETY_PATTERNS` (regex phrases: smell(s/ing) (of) gas, gas leak/smell,
   carbon monoxide, spark(s/ing), electrics, flood(ed/ing), burst, water (is) everywhere, through the ceiling,
   no heating, no hot water). If hits **and** category ≠ urgent: archive→inbox, `notify=True`, `review=True`.
   Phrases, not words: "gas boiler service" must not trigger.
3. Review flags (never change the route): urgent/other without number; urgent/other without name;
   `attempts > 1`; min segment `avg_logprob < -1.0`; no segments.
`push_text(decision, received)` → (title, message, priority): "Urgent voicemail" / "Voicemail needs a look",
"Received HH:MM. …", priority `urgent` / `high`. Contains no caller data (tested).

### `01-voicemail-triage/deliver.py`
`deliver(conn, transcript, result)`: `route()` → if notify: `with_retries(send_push, …)`; `notified_at` set only
when status is `"sent"` (not for dry runs) → `store.save()`.

### `01-voicemail-triage/store.py`
Table `voicemails` (one row per audio hash):
`audio_sha256 TEXT PK, source_file, processed_at (ISO UTC), route, review INT 0/1, reasons (JSON list),
notified_at (NULL = no push), category, urgency INT, caller_name, callback_number, summary, transcript,
whisper_model, llm_model, prompt_version`.
`save()` = `INSERT … ON CONFLICT(audio_sha256) DO UPDATE SET …, notified_at = COALESCE(voicemails.notified_at, excluded.notified_at)`;
all values via `?` placeholders (transcripts are untrusted text). `summary()` prints counts per route + review list.
CLI: `python 01-voicemail-triage/store.py [db]`.

### `01-voicemail-triage/run.py`
`Folders(base)` (inbox, processed, failed, transcripts, results, logs/run.log, voicemails.db; created if missing).
`setup_logging()` console + file, format `time LEVEL message`, never transcript content; `httpx`/`faster_whisper` at WARNING.
`find_ready_files()`, `already_processed()`, `move_to()` (timestamp prefix on name clash), `timed()`, `process_file()`,
`retry_failed()` (moves `failed/` audio back to `inbox/`, deletes `.error.txt`), `run_once()`.
CLI: `run.py [--watch] [--interval 10] [--retry-failed] [--base DIR]`. `--base` points everything at another
folder (used for end-to-end tests in a scratch copy so the real DB stays clean). Not safe to run two instances at once.

### `01-voicemail-triage/digest.py`
Rows with `processed_at >= now-hours` (ISO UTC strings compare correctly), sorted notify_now, inbox, personal,
archive; plus `failed/` files with the step from their error note. Writes `digests/YYYY-MM-DD.md` (ignored) and
prints. `--push` sends "N to call back, N to review, N failed." (counts only). `--hours`, `--base`.

### `shared/notify.py`, `shared/retry.py`, `shared/evaluation.py`
- `send_push()`: POST `NTFY_SERVER/NTFY_TOPIC`, headers Title/Priority/Tags, `timeout=10`, `raise_for_status()`.
  Dry run (logged, returns "dry_run") when `NTFY_DRY_RUN=1` (default) or topic empty/placeholder.
- `with_retries(fn, *args, attempts=3, base_delay=2.0, sleep=time.sleep, **kwargs)`; `sleep` injectable for tests.
- `same_value` (normalized exact, null==null), `same_first_word`, `null_aware` → correct/wrong/missed/invented,
  `confusion(pairs, classes)` → markdown matrix, `pct`.

## 6. Test set and evaluation

`testset/scripts.json` (input side: text written for TTS, Piper speaker 0-108, speed, noise amplitude, phone
band, output format) and `testset/labels.json` (answer key, written before any pipeline code). 18 English
synthetic voicemails for "Brightwater Plumbing & Heating" (owner Sam), Ofcom fictional numbers
(07700 900xxx, 01632 960xxx). `generate.py`: Piper `en_GB-vctk-medium` → ffmpeg (pink noise with seed = id
number, 300-3400 Hz band-pass, 8 kHz) → wav/mp3/m4a/ogg.
Label conventions: category definitions as in the prompt; urgency 1/2/3; caller name as spelled, null if not
said, relationship word ("Mum") allowed; callback number digits only, UK national format, null if not said.
Composition: urgent 6, other 6, personal 2, sales 2, spam 2. Each file targets a trap (see `tests` field), e.g.
04 "you've got my number" → null; 06 error code "F28" near the number; 12 spam saying "urgent"; 14 name spelled
letter by letter; 15 Polish name never spelled; 18 caller corrects the number.
`evaluate.py` also loads `testset/my_recordings/labels.json` if present (real recordings, never committed).

## 7. Measured results

**Transcription (Phase 2, `docs/transcription-comparison.md`)**, strict substring checks on 18 files, CPU int8:

| model | names found | numbers found | real-time factor |
|---|---|---|---|
| small (default) | 9/14 | 8/13 | 1.7 |
| large-v3-turbo | 11/14 | 11/13 | 5.7 |

**End-to-end analysis (Phase 6, `docs/eval-results.md`)**, Whisper small transcripts:

| metric | small:v1 (historic) | **small:v2 (default)** | small:v3 (experiment) |
|---|---|---|---|
| urgent false negatives | 0/6 | **0/6** | 0/6 |
| urgent archived | n/a | **0** | 0 |
| false urgent | 0 | 1 (file 12 scam) | 1 |
| category accuracy | 12/18 | **15/18 (83 %)** | 15/18 |
| urgency accuracy | 12/18 | 11/18 | 10/18 |
| name exact / first word | 11/18 / n/a | 10/18 / 14/18 | 10/18 / 12/18 |
| number correct (wrong/missed/invented) | 15/18 | 16/18 (2/0/0) | 16/18 (2/0/0) |
| analysis retries | 2 | 2 | 1 |

v1 → v2 fixed the "sales" definition (+3 category) but its "Rachel from Acme → Rachel" example shortened
surnames and file 12 flipped to urgent. v2 → v3 (one-line example change, pre-registered rule: adopt only if
exact names improve with no safety/category loss) restored surnames on 13/16 but broke 07 (Mum → null) and
17 ("Yak" invented): exact names 10 → 10, rule not met, **v2 kept**. Takeaway: on 18 files, single prompt
sentences move unrelated answers; further tuning needs held-out data.

**Definition of done:** no urgent archived ✔; re-run does nothing ✔; **≥ 90 % category ✘ (83 %)**; the user's
ability to explain workflow vs agent is covered by the learning log.

**Routing on the test set (v2):** 7 pushes (6 urgent + scam 12), inbox 6, personal 2, archive 3 (09, 10 sales;
11 spam), review 4 (04 no number; 11, 15 retry; 12 no name/number). The safety-word net never fired (the LLM
caught all urgent calls) and had no false hits; unit tests prove it fires when the LLM misses.

**Reliability (Phase 5, scratch-folder run):** broken file → `failed/` with ffmpeg error, batch continued;
duplicate audio under a new name → skipped; Ollama unreachable → retries at 2 s, 4 s → `failed/`;
`--retry-failed` → processed (one real CPU LLM call: 202 s); `--watch` picked up a dropped file in ~4 s.

## 8. Decision log (what was chosen and why)

| Phase | Decision | Reason |
|---|---|---|
| 1 | English-only synthetic set, Piper TTS, phone-band + noise degradation | reproducible, no real personal data; labels before code to avoid grading on a curve |
| 2 | Whisper `small` default (user's choice) over `large-v3-turbo` | turbo: names 11/14, numbers 11/13, RTF 5.7; small: 9/14, 8/13, RTF 1.7. User preferred speed |
| 2 | CPU int8, 8 threads | GPU unusable (Code 43); 8 threads ~25% faster than default 4 |
| 2 | ffmpeg conversion + numpy samples instead of letting faster-whisper decode | single place for format errors; works around faster-whisper 1.2.1 × av 19 `metadata_errors` bug |
| 3 | Ollama structured output (`format=<schema>`), temperature 0, Pydantic validators, 1 retry with feedback | shape guaranteed by the server, content checked by code, repeatable runs; feedback needed because temperature 0 would repeat the same answer |
| 3 | Prompt v2 (user approved fixing v1 now) | v1 read "sales" as "sales opportunity" (quotes/bookings → sales, 12/18 category); v1's retry deleted a number instead of fixing the missing 0 |
| 4 | Minimal push text, sales → archive, ntfy dry run (all user's choices) | GDPR: ntfy.sh can read pushes; trade offers stay findable; no phone app needed yet |
| 4 | Pure `route()` + separate `deliver()`; regex safety net; review flags | testable without I/O; independent second filter for the costly error (missed emergency) |
| 5 | Idempotency by audio hash in SQLite; push before save (at-least-once); retry only transient errors | re-runs are free and safe; duplicate push is the safer failure; no wasted retries on permanent errors |
| 5 | Polling (`--watch`) instead of watchdog | no extra dependency; mtime ≥ 3 s check handles files still being copied |
| 6 | Single-variable experiment v2 → v3 with a pre-registered adoption rule | see §7 |

## 9. Known issues, gotchas and limitations

- **GPU Code 43** (see §2). Until fixed, the LLM takes ~1-4 min per voicemail. faster-whisper on GPU would
  additionally need `nvidia-cublas-cu12` + `nvidia-cudnn-cu12` (~1.5-2 GB, ask the user first); RTX 50-series
  (Blackwell, sm_120) support in the CTranslate2 build is unverified.
- **faster-whisper 1.2.1 × PyAV 19:** `av.open(..., metadata_errors=...)` raises `TypeError`. Avoided by passing
  numpy samples. Don't "simplify" back to passing a file path.
- **Windows:** always `encoding="utf-8"`; ':' illegal in file names; huggingface_hub warns about symlinks (harmless,
  set `HF_HUB_DISABLE_SYMLINKS_WARNING=1`); new PATH entries need a new terminal; git warns about LF→CRLF (harmless).
- **Whisper systematically writes the UK "oh" (zero) as "a"** ("a 1632-960-789"), sometimes writes numbers as words
  (turbo: "oh seven seven double o …"), mishears names over the phone band (Carter→Carver, "calling"→"Colleen").
  The prompt tells the LLM about "a"; the number validator forces a leading 0 and the retry fixes it.
- **Whisper confidence is not an error signal:** lowest `avg_logprob` in the set is -0.38 even on wrong transcripts.
- **Scams sound urgent:** file 12 ("suspended today, final notice") is classified urgent → false push. Accepted
  (asymmetric errors) but noted.
- **Cache invalidation:** caches key on hash + model (+ llm + prompt version). Changing Whisper options (beam size,
  VAD) without changing the model name reuses stale transcripts: delete the cache folder. A voicemail already in
  `voicemails.db` is never re-analyzed after a prompt change.
- **Idempotency is by exact bytes**; two encodings of the same message are two voicemails.
- **Single instance only:** two `run.py --watch` processes can race on the same file.
- **The test set is small (18) and synthetic** (one TTS engine, perfect pacing): scores are optimistic and noisy
  (one urgent file = 17 percentage points of urgent recall). Repeated prompt tuning against it risks overfitting.

## 10. How to run everything

```powershell
cd C:\Users\danuk\code\audio-to-action
.\.venv\Scripts\Activate.ps1
python scripts\check_setup.py
python -m pytest 01-voicemail-triage\tests                         # 11 tests, instant
python 01-voicemail-triage\evaluate.py --configs small:v2 small:v3 # cached → instant
python 01-voicemail-triage\route_testset.py                        # exit 1 if an urgent voicemail is archived
python 01-voicemail-triage\run.py [--watch] [--retry-failed] [--base DIR]
python 01-voicemail-triage\digest.py [--push] [--hours 24]
python -m shared.transcribe <audio> ; python -m shared.analyze <audio>
```

## 11. Suggested next steps

1. Fix the GPU driver; then re-time the pipeline and consider `large-v3-turbo` + GPU Whisper (needs CUDA libs).
2. Raise category accuracy toward the 90 % target **without** tuning on the same 18 files: add real recordings to
   `my_recordings/` (with consent) or a second held-out synthetic set, then try one variable at a time
   (turbo transcripts; a scam/automated-message rule for urgency; a larger LLM).
3. Go live with ntfy: random topic, `NTFY_DRY_RUN=0`, Task Scheduler entries from the README.
4. Part 2: start from `docs/shared-for-part2.md` (generalize `analyze()` to take schema + prompt; chunk long
   transcripts; set-based evaluation of action items).
