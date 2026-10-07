# audio-to-action: technical handover (Parts 1 and 2 complete)

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
- **Part 2, `02-meeting-action-agent/` (DONE, see §12):** meeting recordings → action items (workflow) → a task
  tracker synced across meetings by a bounded agent (or plain-code rules).
- **Part 3, `03-phone-receptionist/` (not started):** real-time AI phone receptionist (full agent). User's early
  choices: local simulation (laptop mic/speakers or a scripted synthetic caller, no telephony provider), job =
  take a message (name, number, reason) into the Part 1 pipeline. Confirm before starting.

Rules from `PROJECT_RULES.md` that any agent working here must follow:
1. One phase at a time; stop after each and wait for the user to say "next" (unless told otherwise). When the
   user authorises several phases in one go, spawn a separate reviewer agent after each phase to verify and optimise
   before continuing (that review found real bugs in every Part 2 phase).
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
│   ├── schemas.py                Segment, Transcript, Category, Analysis (+validators), Result (+ Part 2 models, §12.4)
│   ├── transcribe.py             transcribe(path, cache_dir, model, hint) -> Transcript
│   ├── analyze.py                analyze(transcript, cache_dir, llm, prompt_version) -> Result; PROMPTS
│   ├── llm.py                    structured_chat(schema, messages, ...) (Part 2; used by analyze.py too)
│   ├── evaluation.py             scoring helpers (same_value, same_first_word, null_aware, confusion, word_errors, pct)
│   ├── retry.py                  with_retries(fn, ...), is_transient(exc)
│   ├── notify.py                 send_push(title, message, priority, tags) -> "sent" | "dry_run"
│   ├── pipeline.py               inbox helpers used by Part 2 (Part 1's run.py keeps its own copies)
│   └── tts.py                    Piper text-to-speech (test-set generation; Part 3's voice)
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
├── 02-meeting-action-agent/      Part 2, see §12.2
└── docs/
    ├── learning-log.md           per-phase recaps (the pedagogical record)
    ├── eval-results.md           generated metrics + confusion matrices + every error
    ├── eval-notes.md             hand-written conclusions, inserted into eval-results.md by evaluate.py
    ├── transcription-comparison.md
    ├── shared-for-part2.md       reuse guide for Part 2 (written before Part 2; see its note)
    ├── part2-eval-results.md     Part 2 report, generated by 02-meeting-action-agent/evaluate.py
    ├── part2-eval-notes.md       hand-written Part 2 conclusions, inserted into part2-eval-results.md
    ├── part2-transcription-comparison.md
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
  on any exception: log, move to failed/, write failed/<file>.error.txt (step + traceback);
  if that move fails too (file open in another program), the file stays in inbox/ for the next run
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
- `Transcript(source_file, audio_sha256, model, hint, language, language_probability, duration_s,
  transcribe_s, text, segments, created_at)`. `source_file` is the bare name (no folders, for privacy).
  `hint` (Part 2) defaults to None so Part 1's cached transcripts still load.
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
`transcribe(path, cache_dir=None, model=None, hint=None) -> Transcript`
1. `settings()` from env: `WHISPER_MODEL` (small), `WHISPER_DEVICE` (auto), `WHISPER_COMPUTE_TYPE` (int8), `WHISPER_CPU_THREADS` (0 → library default 4; `.env` sets 8).
2. `file_sha256()` (1 MB chunks) → cache file `<cache_dir>/<hash[:16]>_<model>[_h<sha256(hint)[:8]>].json`; hit → return.
   No hint (or "") = Part 1's original file name, so its caches stay valid.
3. `to_wav_16k()`: `ffmpeg -nostdin -y -loglevel error -i src -ac 1 -ar 16000 -c:a pcm_s16le dst` in a
   TemporaryDirectory. Raises `RuntimeError` with ffmpeg's stderr. Runs **before** model loading so broken files fail in ~0 s.
4. `read_wav_samples()`: int16 → float32 / 32768 (numpy). Samples, not a path, are passed to Whisper (§9 PyAV bug).
5. `load_model()` (`@lru_cache(maxsize=2)`), then `whisper.transcribe(samples, beam_size=5, vad_filter=True,
   initial_prompt=hint)`; the segment iterator is lazy (work happens while iterating).
6. Write JSON to `.tmp`, then `replace()` (atomic: no half-written cache files).
CLI: `python -m shared.transcribe <file> [--model M] [--cache-dir D] [--hint TEXT]`.

### `shared/analyze.py`
`analyze(transcript, cache_dir=None, llm=None, prompt_version=None) -> Result`
- `PROMPTS = {"v2": PROMPT_V2, "v3": PROMPT_V2.replace(V2_EXAMPLE, V3_EXAMPLE)}` (v3 differs in exactly one line,
  asserted at import). `DEFAULT_PROMPT_VERSION = os.getenv("ANALYSIS_PROMPT", "v2")`. Never edit a version in place.
- Cache file `<hash[:16]>_<whisper model>_<llm with ':'→'-'>_<prompt version>.json` (':' is illegal in Windows names).
- Messages: system = prompt (business context: Brightwater Plumbing & Heating, owner Sam; definitions for each
  category, urgency scale, name rules, number rules incl. "a 1632…" = "01632…", corrected numbers, never guess);
  user = `Voicemail transcript:\n<<<\n{text}\n>>>`.
- Since Part 2 the LLM call is `shared/llm.structured_chat(Analysis, messages, llm=llm, fix_hint=FIX_HINT)` (§12.4):
  `ollama.Client(host).chat(model, messages, format=Analysis.model_json_schema(), options={"temperature": 0})`.
  (There is no `ask_llm()` any more.)
- Validate with `Analysis.model_validate_json(raw)`. On `ValidationError`, append the assistant's reply and a user
  message: "Your answer broke these rules: <short_errors> Fix the values using the transcript; do not replace a
  value the caller actually said with null. Return the corrected JSON." Second failure → `LLMFormError`, re-raised as
  `AnalysisError`. The rejected first reply and the reason are stored in `Result`.
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
`find_ready_files()`, `already_processed()`, `move_to()` (a plain `Path.rename`, never copy + delete; timestamp prefix
on name clash), `timed()`, `process_file()` (step labels hash/transcribe/analyze/deliver/move),
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
- `norm_text`, `same_value` (normalized exact, null==null), `same_first_word`, `null_aware` → correct/wrong/missed/invented,
  `confusion(pairs, classes)` → markdown matrix, `word_errors(ref, hyp)` → (edits, ref length) for WER (Part 2), `pct`.

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
- **Cache invalidation:** caches key on hash + model (+ hint fingerprint, + llm + prompt version). Changing Whisper options (beam size,
  VAD) without changing the model name reuses stale transcripts: delete the cache folder. A voicemail already in
  `voicemails.db` is never re-analyzed after a prompt change. `analyze()`'s cache key has **no** hint fingerprint
  (Part 1 never uses a hint; `extract.py`'s key has one): if Part 3 analyzes hinted transcripts, add it first.
- **Idempotency is by exact bytes**; two encodings of the same message are two voicemails.
- **Single instance only:** two `run.py --watch` processes can race on the same file.
- **The test set is small (18) and synthetic** (one TTS engine, perfect pacing): scores are optimistic and noisy
  (one urgent file = 17 percentage points of urgent recall). Repeated prompt tuning against it risks overfitting.

## 10. How to run everything

```powershell
cd C:\Users\danuk\code\audio-to-action
.\.venv\Scripts\Activate.ps1
python scripts\check_setup.py
python -m pytest 01-voicemail-triage\tests                         # 11 tests, instant (Part 2: 70 more, §12.10)
python 01-voicemail-triage\evaluate.py --configs small:v2 small:v3 # cached → instant
python 01-voicemail-triage\route_testset.py                        # exit 1 if an urgent voicemail is archived
python 01-voicemail-triage\run.py [--watch] [--retry-failed] [--base DIR]
python 01-voicemail-triage\digest.py [--push] [--hours 24]
python -m shared.transcribe <audio> ; python -m shared.analyze <audio>
```

## 11. Suggested next steps (Part 1; Part 2 in §12.9)

1. Fix the GPU driver; then re-time the pipeline and consider `large-v3-turbo` + GPU Whisper (needs CUDA libs).
2. Raise category accuracy toward the 90 % target **without** tuning on the same 18 files: add real recordings to
   `my_recordings/` (with consent) or a second held-out synthetic set, then try one variable at a time
   (turbo transcripts; a scam/automated-message rule for urgency; a larger LLM).
3. Go live with ntfy: random topic, `NTFY_DRY_RUN=0`, Task Scheduler entries from the README.
4. (Done in Part 2: `analyze()` generalized into `shared/llm.structured_chat()`, chunking, set-based evaluation; see §12.)


---

## 12. Part 2: meeting-action-agent (complete)

### 12.1 Goal and the user's decisions
Weekly team-meeting recordings → action items → a SQLite task tracker that stays correct across meetings. The point of
Part 2 is the **workflow vs agent boundary**: transcription and extraction are workflow steps; only the tracker sync
may be an agent, and it was measured against plain rules. User's choices (Phase 0): short **synthetic** meeting series;
**no speaker diarization** (owners come from names in the words); agent job = **tracker sync only**; Whisper `small`
for speed. The user later authorised running all phases in one go with an independent review between phases.

### 12.2 Files
```
02-meeting-action-agent/
├── README.md            what it does, architecture, usage, results, decisions
├── context.py           BUSINESS, TEAM, CUSTOMERS, PLACES, whisper_hint()  (edit for a real business)
├── dates.py             resolve_due(words, meeting_date): calendar arithmetic in code
├── extract.py           extraction workflow (prompt x7), chunk/merge, ground_owner, finish_items, cache
├── extract_testset.py   extraction scores per prompt version (--version re-scores cached older versions)
├── tracker.py           SQLite: tasks, changes (before/after + reason), meetings; similarity, undo
├── agent.py             the bounded tool-calling agent (sync_meeting) + guards (run_tool)
├── rules_sync.py        the plain-code sync (no LLM), same signature as run.py's agent_sync
├── run.py               inbox pipeline: hash → transcribe → extract → sync_with_rollback → record → move
├── sync_testset.py      runs a sync method over the series (gold or extracted items), saves sync_runs/*.json
├── evaluate.py          docs/part2-eval-results.md from cached extraction + saved sync runs + eval notes
├── compare_transcription.py   Phase 2 experiment → docs/part2-transcription-comparison.md
├── tests/               70 tests, no LLM: dates, extract, agent guards, rules, run/rollback
└── testset/             scripts.json, labels.json, answer_key.py, generate.py, README.md, sync_runs/ (committed);
                         audio/, transcripts/, results/, traces_*/, *.db ignored
shared/ (new in Part 2): llm.py (structured_chat), tts.py (Piper), pipeline.py (inbox helpers);
transcribe.py gained `hint`; schemas.py gained ActionItem/MeetingItems/ExtractedItem/MeetingResult; evaluation.py
gained word_errors().
```

### 12.3 Data flow (`run.py`)
1. `find_ready_files()` → sorted by `(date_for_sorting, name)`: oldest meeting first (later meetings refer to earlier
   tasks). `meeting_date()`: `YYYY-MM-DD` in the file name, else mtime; impossible dates fail that file (step "date").
   A meeting older than the newest recorded one is still synced, with a warning (its news may overwrite newer news).
2. `file_sha256` → `tracker.meeting_done()` → skip. `meeting_id()`: file stem, or `stem_<hash8>` if a recorded
   meeting already used that stem (weekly `standup.m4a`).
3. `transcribe(path, cache_dir, hint=context.whisper_hint())` (cache key includes an 8-char hint fingerprint).
4. `with_retries(extract, transcript, day, cache_dir)`.
5. `with_retries(sync_with_rollback, SYNC_MODES[mode], conn, meeting, day, items, folders)`:
   `start_clean()` first undoes changes left by an interrupted run of this meeting (raises if a later meeting
   changed the tracker after them); any exception incl. `KeyboardInterrupt` → `tracker.undo_meeting(meeting)`, re-raise.
6. `tracker.record_meeting(...)` (items, sync mode, review list) → `move_to(processed/)` (a plain rename).
Any error → `failed/<file>` + `.error.txt`; if even that move fails (file open elsewhere), the file stays in inbox.

### 12.4 Module reference (Part 2)
- **`shared/llm.py`** `structured_chat(schema, messages, llm, context, fix_hint, options) -> StructuredReply(value,
  attempts, rejected_reply, rejected_because, seconds)`: Ollama `format=schema.model_json_schema()`, temperature 0,
  `schema.model_validate_json(raw, context=context)`, one retry with the errors + `fix_hint`, else `LLMFormError`.
  Part 1's `analyze()` now uses it (identical messages and caches).
- **`shared/schemas.py`** `ActionItem(evidence, task, owner, due_text, status)`: validators read the validation
  context: `owner` must be in `context["team"]` (case-normalised) or null; `due_text` "null"-strings → None;
  ≥ 60 % of the evidence words (ignoring `[mm:ss]`) must be in `context["transcript"]`. `MeetingItems(jobs_mentioned,
  items)`: `jobs_mentioned` first = "think first". `ExtractedItem` adds `due: date` and `owner_from_llm`.
  `MeetingResult` stores items, `llm_items` (raw model output, so post-processing can be re-applied), jobs, attempts.
- **`extract.py`** prompt x7 (history in the comment above `PROMPT_VERSION`): rules for what is / isn't an item, field
  rules, a worked example with invented jobs. `format_transcript()` = one `[mm:ss] text` line per segment.
  `chunk_segments(max_words=1200)` with one-segment overlap; `merge_items()`: exact repeats anywhere, similar items only
  across chunks. `ground_owner()`: owner kept only if named (`\bname\b`) in the evidence, the segment that best matches
  the evidence, or the 2 segments before it.
  `finish_items()`: merge → ground → `resolve_due()`; done/cancelled items get no due date. Cache:
  `<hash16>_<whisper>[_h<hint8>]_<llm>_<prompt>.json`.
- **`tracker.py`** tables `tasks(id, task, owner, due, status, created_in, updated_in)`, `changes(id, task_id, meeting,
  action add|update, before JSON, after JSON, reason, at)`, `meetings(audio_sha256 PK, meeting, meeting_date,
  source_file, processed_at, items, sync_mode, review JSON)`. `update_task` uses `COALESCE` (null never wipes a
  value). `similarity()` = Jaccard of meaningful words (len > 2, minus STOPWORDS). `undo_last`, `undo_meeting`.
- **`agent.py`** `sync_meeting(conn, meeting, day, items, trace_dir, llm, max_steps)`. Items get letters (A, B, ...).
  Each step: `state_message()` (items with TO DO / HANDLED, open + touched tasks, feedback from the last calls, what's
  left) → `client.chat(tools=tool_schemas(), options={"temperature": 0, "num_predict": 1024})` → `run_tool()` per call.
  Tools (Pydantic args): `search_tasks(query)`, `add_task(item_id, reason, confirm_new)`, `update_task(task_id, item_id,
  reason)` (task_id REQUIRED: optional made the model omit it), `skip_item(item_id, kind not_work|repeat_of_item,
  other_item, reason)`, `finish(summary)`. Guards: unknown tool; bad args answered with advice (`bad_arguments`); blank
  reason; unknown item id; item handled once; `update` needs an existing task, similarity ≥ `MIN_SIMILARITY` 0.2, and a task changed
  only once per meeting; `add` refuses an open task with similarity ≥ `DUPLICATE_SIMILARITY` 0.6 unless `confirm_new`;
  `repeat_of_item` needs ≥ `REPEAT_SIMILARITY` 0.5 with the other item; `not_work` refused for done/cancelled items
  that resemble a tracker task; `finish` refused while items are left. Loop stops at finish, at `2n+4` steps, or when
  the state message is identical to the previous step (temperature-0 fixed point); a failed LLM reply
  (`ollama.ResponseError`, e.g. "token repeat limit reached") costs one step and changes the feedback.
  Trace: `<trace_dir>/<meeting>.trace.json` (items, handled, unhandled_for_review, every call/result).
- **`rules_sync.py`** per item: same-status repeat (≥ 0.5) → skip; done/cancelled → update most similar open task if
  ≥ `LINK` 0.2; open → update if ≥ `SAME` 0.6; cancelled with no match → skip; else add. Tasks changed today excluded.
  `rules_sync(conn, meeting, day, items, folders=None) -> []`: same signature as `run.py`'s `agent_sync` wrapper (not
  `agent.sync_meeting`), so both sit in `run.SYNC_MODES`.
- **`dates.py`** day-of-month first ("Monday the 12th" → the 12th), then today/this afternoon, day after tomorrow,
  tomorrow, end of month/week, weekday ("Monday" said on a Monday = next Monday; "next Friday" = the week after);
  "may" only as a month next to a day; bare "now" isn't today. 30 tests.

### 12.5 Test set (`testset/`)
5 meetings (Mondays 2026-09-07 … 10-05), ~30 s each, voices Sam 7 / Priya 0 / Tom 9 / Jamie 11 (en_GB-vctk-medium),
light pink noise, wav/m4a/mp3/wav/ogg. `labels.json`: 14 tasks (T1-T14) with `match` rules (`all` word groups +
`none`, substring matching, self-checked by `answer_key.match_problems()`), and per meeting the `mentions` extraction
should output (task, status, owner, due resolved from the meeting date, `owner_from_text` false for the 3 voice-only
owners T5/m1, T7/m2, T8/m3). `answer_key.tracker_states()` folds mentions into the expected tracker after each meeting
(final: 10 done, 1 cancelled, 3 open). TTS spellings "Preeya", "Shiv awn" are deliberate. **Piper output isn't
bit-identical between runs: regenerating changes file hashes and invalidates all caches; don't, once results exist.**

### 12.6 Measured results
| what | result |
|---|---|
| Transcription (`docs/part2-transcription-comparison.md`) | small 8.5 % WER / 71 % names; **small+hint 5.3 % / 98 %** (default); turbo+hint 2.9 % / 100 % at 2× time. The hint lists exactly the test names (best case; ~93 % with a realistic 17-name list) |
| Extraction x7 (`extract_testset.py`) | task only P 84 % / R 75 %; task + owner (DoD) 64 % / 57 %; owner 84 % where named, due 95 %, status 95 %; ~235 s/meeting |
| Sync, gold items | rules 14/14, agent 11/14 (an earlier Phase 4 run: 12/14) |
| Sync, extracted items | agent 7/14, rules 6/14; perfect hand-linking would reach 9/14 (extraction loses 5) |
| Cost | agent 51 LLM calls / ~94 min for both runs; rules < 1 s |
| Definition of done | extraction ≥ 80 % and tracker ≥ 80 %: **not met**; guardrails and idempotency: **met** |
Default sync stays `agent` by the pre-registered rule; the margin is within noise and both methods were tuned on this
set. Saved agent runs predate the last agent fix (not_work for cancelled ideas). Full analysis: `docs/part2-eval-results.md`.

### 12.7 Decision log (Part 2)
| Phase | Decision | Reason |
|---|---|---|
| 1 | meeting *series* with an answer key of mentions + derived tracker states | the agent's job only exists across meetings; derived states can't drift |
| 2 | `small` + known-names hint as default | names 71 % → 98 % for ~15 % more time; never hint scored task words |
| 3 | `structured_chat` shared helper; evidence + team checks via validation context; dates in code | ground the model in its input; calendar maths in code |
| 3 | `jobs_mentioned` think-first list; owner grounding; prompt frozen at x7 | x1-x5 history; a job-coverage retry was removed (the model rewrote its list instead) |
| 4 | state message instead of chat history; letters for items; values copied by code | history version replayed failing calls and guessed ids; "I5" got linked to "#5" |
| 4 | guards in code, errors that say what to do instead | prompts didn't stop wrong links; bare type errors were repeated forever |
| 5 | rollback via the change log; start_clean; hash identity; oldest meeting first | a meeting is all-or-nothing; reruns safe after crashes/Ctrl+C |
| 6 | rules baseline + pre-registered decision on extracted items | an agent must beat the simple alternative on the real pipeline |

### 12.8 Gotchas (Part 2)
- **Schema = prompt:** an optional tool argument gets omitted; required + good error messages work better.
- **Models satisfy errors the cheapest way** (guess an id, delete a value, rewrite a list, skip an item): every guard
  needs a check that the "fix" is real.
- **State-based loops have a fixed point at temperature 0** (same state → same answer): the no-progress stop.
- **Temperature 0 isn't bit-exact on CPU:** decisions mostly stable, wording varies; ±1 task between agent runs.
- **Look for bugs in your own code before blaming the model:** `merge_items` once merged "order" + "fit" items.
- **Test-set leakage:** an example copied from a test transcript was in the prompt until x7.
- **`sync_testset.py --only` overwrites the full run's DB and traces** (`tracker_<items>_<sync>.db`, `traces_<items>/`).
- **Long meetings:** chunking is only unit-tested; the hint (`initial_prompt`) fades after ~220 tokens (`hotwords=`
  would be needed); prompts may need `num_ctx` > 4096.
- **CPU timings:** Whisper ~1-2× real time, extraction ~4 min/meeting, agent ~2 min per LLM call (1-16 per meeting).

### 12.9 Next steps (Part 2)
1. Fix the GPU (Code 43), then re-run everything with `large-v3-turbo` + hint and faster LLM calls.
2. Improve extraction first (it caps the sync at 9/14): turbo transcripts; more and real (consented) meetings.
3. Try a hybrid sync: rules first, the agent only for items no rule matches confidently.
4. Re-run the agent series twice with the current `agent.py` to measure its noise; add the 9/14 oracle to evaluate.py.
5. Part 3 (receptionist): start with a Phase 0 plan from the choices in §1; reuse `shared/tts.py`, `transcribe.py`,
   `llm.py`, `pipeline.py` and Part 1's routing.

### 12.10 How to run (Part 2)
```powershell
python -m pytest 01-voicemail-triage\tests 02-meeting-action-agent\tests -q   # 81 tests (11 + 70), no LLM
python 02-meeting-action-agent\evaluate.py           # docs\part2-eval-results.md from caches + sync_runs (seconds, no LLM
                                                     # while the x7 caches exist); always rewrites the report
python 02-meeting-action-agent\sync_testset.py --items gold|extracted --sync agent|rules [--only m2]
                                                     # rules: instant; agent: ~45-50 min per series on CPU;
                                                     # rewrites testset\sync_runs\<items>_<sync>.json (timings change)
python 02-meeting-action-agent\extract_testset.py [--version x6]   # extraction scores (cached; --version re-scores old caches)
python 02-meeting-action-agent\extract.py <audio> <YYYY-MM-DD>     # one meeting: transcript + items
python 02-meeting-action-agent\compare_transcription.py [--configs ...]
python 02-meeting-action-agent\run.py [--watch] [--interval 10] [--retry-failed] [--sync agent|rules] [--base DIR]
python 02-meeting-action-agent\testset\generate.py [--only m3]     # DON'T once results exist (§12.5)
```
Caches are found by hashing the (git-ignored) audio in `testset/audio/`, so those files must exist locally. ffmpeg is
needed only when a transcript cache is missing; a missing x7 extraction cache makes `evaluate.py` call the LLM.
