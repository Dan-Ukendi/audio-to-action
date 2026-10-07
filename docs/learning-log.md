# Learning log: audio-to-action

## Part 1, Phase 0: Setup (2026-10-01)

### 1. What we built
The empty but well-organised workbench for the project: folders, a Python environment with the
libraries we'll need, the tools (ffmpeg, Ollama), privacy rules in `.gitignore`, a `PROJECT_RULES.md`
that remembers our rules across sessions, and a script that checks all the pieces can talk to each other.
No pipeline code yet.

### 2. Where it fits in the pipeline
```
 [Phase 0: SETUP]  <-- you are here (tools + folders everything below depends on)

 audio file ──► inbox/
                  │
                  ▼
           ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
           │ 1. transcribe│──►│ 2. analyze   │──►│ 3. route     │──►│ 4. store     │
           │ ffmpeg +     │   │ Ollama LLM → │   │ plain-code   │   │ SQLite +     │
           │ faster-      │   │ JSON checked │   │ rules → ntfy │   │ move file to │
           │ whisper      │   │ by Pydantic  │   │ / archive /  │   │ processed/ or│
           │ (Phase 2)    │   │ (Phase 3)    │   │ review queue │   │ failed/      │
           └──────────────┘   └──────────────┘   │ (Phase 4)    │   │ (Phase 4-5)  │
                                                 └──────────────┘   └──────────────┘
 measured against testset/ (Phase 1) by the eval script (Phase 6)
```

### 3. How it works, step by step
1. `py -V:3.13-64 -m venv .venv` creates an isolated Python 3.13 just for this project.
2. `pip install -r requirements.txt` installs faster-whisper, ollama, pydantic, python-dotenv, requests.
3. `scripts/check_setup.py` → `main()` runs four independent checks:
   - `check_python()`: right version, and running inside the venv?
   - `check_ffmpeg()`: is `ffmpeg` on PATH? (`shutil.which`)
   - `check_faster_whisper()`: can we import it, and does CTranslate2 see a CUDA GPU?
   - `check_ollama()`: does `GET http://localhost:11434/api/tags` answer, and is our model in the list?
4. Settings come from `.env` (loaded by `python-dotenv`), with `.env.example` as the template.

### 4. Key concepts I should understand
- **Virtual environment:** a private copy of Python plus packages per project. Example: our `.venv`
  uses 3.13 even though your system default is 3.14, because CTranslate2 needs a version it has builds for.
- **Local model server:** Ollama runs in the background and exposes an HTTP API on port 11434. Our code
  doesn't load the LLM itself; it sends requests. Example: `/api/tags` lists the downloaded models.
- **CPU vs GPU inference:** the same model runs on either; GPU is much faster but needs extra
  libraries (CUDA). Example: we currently see "CPU only" because the CUDA runtime libs aren't installed yet.
- **Configuration outside code:** model names, hosts and secrets live in `.env`, so swapping
  `WHISPER_MODEL=small` → `large-v3-turbo` needs no code change.
- **Privacy by default:** `.gitignore` stops audio, transcripts and the DB reaching git, and keeping the repo
  outside OneDrive stops it reaching the cloud. Two different leaks, two different fixes.

### 5. Files created or changed
- `.gitignore`: keeps audio, transcripts, results, DB, `.env` and `.venv` out of git.
- `requirements.txt`: the Python packages for now (more added per phase).
- `.env.example`: template for local settings (models, Ollama host, ntfy topic).
- `PROJECT_RULES.md`: project goal, rules, recap format, phase list; read at the start of every work session.
- `README.md`: skeleton with setup steps and phase checklist.
- `scripts/check_setup.py`: verifies Python, ffmpeg, faster-whisper and Ollama.
- `docs/learning-log.md`: this file.
- `01-voicemail-triage/{inbox,processed,failed,transcripts,results,testset}/`: working folders.
- `02-meeting-action-agent/.gitkeep`, `03-phone-receptionist/.gitkeep`: placeholders for later parts.

### 6. Try it yourself
```powershell
cd C:\Users\danuk\code\audio-to-action
.\.venv\Scripts\Activate.ps1
copy .env.example .env
python scripts\check_setup.py
```
Expected: `[OK]` for python, ffmpeg and faster-whisper ("CPU only"). The ollama line says the server is
running but `qwen2.5:7b` isn't pulled yet; after `ollama pull qwen2.5:7b` it turns `[OK]` and prints "All good."

### 7. What can go wrong
- **"ffmpeg not on PATH":** terminals opened before the install don't see the new PATH. Open a new one.
- **"Ollama not reachable":** the Ollama app isn't running (check the tray icon) or something else uses port 11434.
- **Activate.ps1 blocked:** PowerShell execution policy. Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.
- **Wrong Python in venv:** creating the venv with plain `python` would use 3.14; always use `py -V:3.13-64`.
- **Low free RAM:** only ~2.7 GB was free at check time; a 7B model needs ~5 GB, so close heavy apps.

### 8. Check my understanding
1. Why does keeping audio out of git *not* fully solve the privacy problem on its own?
2. Our code never loads the LLM weights directly. Then what does `check_ollama()` actually talk to?
3. Why might faster-whisper say "CPU only" even though this laptop has an NVIDIA GPU?

### 9. Next phase preview
Phase 1 builds the test set *before* any pipeline code: 15-20 synthetic voicemails generated with a
local text-to-speech tool (Piper), covering urgent, sales, spam, noisy, rambling and multilingual calls,
plus `labels.json` with the right answers. That's our ruler for measuring every later phase.

---

## Part 1, Phase 1: Test set (2026-10-07)

### 1. What we built
18 fake-but-realistic voicemails for a fictional UK plumbing business, plus an answer key.
Each one is designed to test something specific (a hidden emergency, a spam call that *says* "urgent",
a name spelled letter by letter, a corrected phone number...). A script rebuilds all the audio from
text, so the set is reproducible and no audio ever needs to go into git.

### 2. Where it fits in the pipeline
```
 audio ─► inbox/ ─► 1.transcribe ─► 2.analyze ─► 3.route ─► 4.store + move file
                    (Phase 2)        (Phase 3)    (Phase 4)   (Phase 4-5)
                         ▲               ▲             ▲
                         └───────────────┴─────────────┘
                     compared against ┌──────────────────────────┐
                                      │ [Phase 1: TEST SET]      │ <-- you are here
                                      │ testset/audio + labels   │
                                      └──────────────────────────┘
                                      by the eval script (Phase 6)
```

### 3. How it works, step by step
`01-voicemail-triage/testset/generate.py` → `main()`:
1. Reads `scripts.json` (what to say, which speaker, speed, noise, phone band, format).
2. `ensure_voice()` loads the Piper voice from `voices/` (downloads it if missing).
3. For each voicemail:
   - `synthesize()`: Piper turns `text` into clean speech (22.05 kHz wav) in a temp folder.
   - `build_filter()`: builds an ffmpeg filter graph: optional pink noise (fixed seed), then a
     300–3400 Hz band-pass and resample to 8 kHz (the "phone line" sound).
   - `degrade()`: runs ffmpeg with that graph and encodes to wav/mp3/m4a/ogg in `audio/`.
4. `check_labels()`: cross-checks `labels.json` ↔ `scripts.json` ↔ `audio/` and reports mismatches.

### 4. Key concepts I should understand
- **Test set first:** if the answers are written after seeing the pipeline's output, we unconsciously
  grade it on a curve. Example: deciding *now* that `04_urgent_vague_leak` is urgent means a model
  that calls it "other" is simply wrong, not "arguably right".
- **Inputs vs labels are separate:** `scripts.json` says "Shiv awn" (how it sounds); `labels.json` says
  "Siobhan" (the truth). Real callers don't hand you spellings, so this mismatch is the test.
- **Coverage over volume:** 18 files is tiny, but each targets one failure mode. Example: `12` checks
  that the *word* "urgent" doesn't fool the classifier, `06` checks that "F28" isn't read as part of a number.
- **Narrowband audio:** phone lines carry ~300–3400 Hz at 8 kHz sampling. "f", "s" and "th" live higher,
  so they blur. Example: "fifteen"/"fifty" confusions are much more likely in a voicemail than in a podcast.
- **Label conventions make metrics possible:** "digits only, null if not said" means `07700 900 123`,
  `07700-900123` and `07700900123` all compare equal after normalization, and a guessed number
  where none was spoken counts as wrong.

### 5. Files created or changed
- `01-voicemail-triage/testset/scripts.json`: the 18 voicemail texts + audio settings.
- `01-voicemail-triage/testset/labels.json`: answer key with conventions at the top.
- `01-voicemail-triage/testset/generate.py`: rebuilds audio (Piper + ffmpeg) and checks labels.
- `01-voicemail-triage/testset/README.md`: what's in the set and how to add your own recordings.
- `01-voicemail-triage/testset/my_recordings/`: (git-ignored) place for real recordings.
- `.gitignore`: added `testset/audio/`, `testset/voices/`, `*.onnx`.
- `requirements.txt`: added `piper-tts`.
- `PROJECT_RULES.md`, `README.md`: status and commands updated.

### 6. Try it yourself
```powershell
cd C:\Users\danuk\code\audio-to-action
.\.venv\Scripts\Activate.ps1
python 01-voicemail-triage\testset\generate.py
start 01-voicemail-triage\testset\audio\05_urgent_noisy_water_heater.wav
```
Expected: 18 lines `[ 1/18] 01_urgent_burst_pipe.wav` … then `labels.json is consistent with scripts and audio.`
The noisy file should sound like a thin phone line with hiss behind it.

### 7. What can go wrong
- **TTS mispronunciations:** Piper may say "F twenty-eight" or "Shiv awn" oddly. That's fine as long as
  a human could still understand it; if not, rewrite the `text`, not the label.
- **Synthetic is too clean:** one engine, perfect pacing, no real accents. Scores here will be
  *optimistic* compared to real voicemails, which is why `my_recordings/` exists.
- **Small set = noisy metrics:** with 6 urgent files, one mistake moves the urgent recall by ~17 points.
- **Label mistakes:** a wrong label looks exactly like a model error. `check_labels()` catches structural
  mistakes, but not wrong answers; reread `labels.json` once yourself.
- **Encoding on Windows:** reading JSON without `encoding="utf-8"` fails (we hit it once).

### 8. Check my understanding
1. Why does `05_urgent_noisy_water_heater` have both noise *and* the phone band, and what does each test?
2. For `04_urgent_vague_leak`, why is `callback_number: null` the correct label, even though Dave says "you've got my number"?
3. Why should the labels be written before we see any pipeline output?

### 9. Next phase preview
Phase 2 writes `transcribe(path) -> Transcript` in `shared/`: ffmpeg converts any format to 16 kHz mono,
faster-whisper turns it into text with timestamped segments, and results are cached by file hash.
We'll run all 18 files, put transcripts next to the scripts, and compare `small` with a bigger model
on accuracy (especially numbers and names) and speed.

---

## Part 1, Phase 2: Transcription (2026-10-07)

### 1. What we built
A reusable function `transcribe(path) -> Transcript` that turns any voicemail file (wav, mp3, m4a, ogg)
into text with timestamps, plus a script that ran both `small` and `large-v3-turbo` over the 18 test files
and compared them. The result is saved as JSON, and a file that was already transcribed is never done twice.
Decision: `small` stays the default (about 3x faster); turbo is more accurate.

### 2. Where it fits in the pipeline
```
 audio ─► inbox/ ─► [1. TRANSCRIBE] ─► 2.analyze ─► 3.route ─► 4.store + move file
                     (Phase 2)          (Phase 3)    (Phase 4)   (Phase 4-5)
                     <-- you are here
                     any format ─ffmpeg─► 16 kHz wav ─Whisper─► Transcript JSON
                                                                (cached by file hash)
```

### 3. How it works, step by step
`shared/transcribe.py` → `transcribe(path, cache_dir, model)`:
1. `settings()` reads `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`, `WHISPER_CPU_THREADS` from `.env`.
2. `file_sha256(path)` reads the file's bytes in 1 MB chunks and computes a SHA-256 fingerprint
   (64 hex characters). Same bytes = same fingerprint, whatever the file is called.
3. `cache_path()` builds `<cache_dir>/<first 16 chars of hash>_<model>.json`. If that file exists, we
   load it with `Transcript.model_validate_json(...)` and return. **Nothing else runs.**
4. `load_model()` loads Whisper. It's decorated with `@lru_cache`, so the second call with the same
   arguments returns the already-loaded model instead of loading it again (loading takes 4-8 s).
5. `to_wav_16k()` runs `ffmpeg -i <input> -ac 1 -ar 16000 -c:a pcm_s16le <tmp>.wav`: mono, 16 000 samples
   per second, plain 16-bit numbers. That's the exact format Whisper was trained on. The wav goes into a
   `TemporaryDirectory`, which Python deletes when the `with` block ends (so no extra copy of private audio stays around).
6. `read_wav_samples()` turns the wav into a numpy array of floats between -1 and 1 (each 16-bit sample
   divided by 32768). We give Whisper these numbers instead of a file path, which sidesteps a bug between
   faster-whisper 1.2.1 and PyAV 19 (`metadata_errors` error).
7. `whisper.transcribe(samples, beam_size=5, vad_filter=True)` returns a *lazy* iterator: no work happens
   until the list comprehension loops over it. Each item becomes a `Segment` (start, end, text, confidence).
8. A `Transcript` (defined in `shared/schemas.py`) is built and written to the cache: first to `.tmp`,
   then renamed, so a crash never leaves a half-written JSON behind.

`01-voicemail-triage/compare_models.py` → `main()`: for each model, `evaluate()` calls `transcribe()` on
every labelled file, then `name_found()` and `number_found()` compare against `labels.json`.
`build_report()` writes `docs/transcription-comparison.md`.

### 4. Key concepts I should understand
- **Content hash as cache key:** the cache doesn't care about file names. Example: copy
  `06_urgent_landlord_deadline.ogg` to `test.ogg` and transcribe it: it returns instantly, because the bytes,
  and so the hash, are identical. Change one byte and it's a "new" file. Phase 5 uses the same idea for
  "re-running on the same file does nothing".
- **Model name in the key:** `small` and `large-v3-turbo` produce different text for the same audio, so
  each gets its own cache file (`364ca7..._small.json` vs `364ca7..._large-v3-turbo.json`).
- **Normalize the input once:** four formats in, one format out of ffmpeg. Whisper (and any later
  step) only ever has to deal with 16 kHz mono wav. A broken file fails *here*, with ffmpeg's message.
- **Speed vs accuracy:** real-time factor (RTF) = processing time / audio length. `small` ran at 1.7
  (a 30 s voicemail takes ~50 s), turbo at 5.7 (~3 min). Bigger model, fewer mistakes, more waiting.
- **Transcription errors are not all equal:** "Carver" for "Carter" is annoying; "07700**94**0349"
  for "07700900349" (small, file 18) sends Sam to a stranger. The second kind is what we must catch.

### 5. Files created or changed
- `shared/__init__.py`: makes `shared` an importable package.
- `shared/schemas.py`: `Segment` and `Transcript` Pydantic models.
- `shared/transcribe.py`: the transcription step + a small command-line entry point.
- `01-voicemail-triage/compare_models.py`: runs models over the test set and writes the report.
- `docs/transcription-comparison.md`: the results (synthetic data only).
- `.env.example` (+ your local `.env`): added `WHISPER_CPU_THREADS=8`.
- `.gitignore`: added `testset/transcripts/`.
- `requirements.txt`: added `numpy`.
- `PROJECT_RULES.md`, `README.md`: status updated.

### 6. Try it yourself
Open a **new** terminal first (so ffmpeg is on PATH), then:
```powershell
cd C:\Users\danuk\code\audio-to-action
.\.venv\Scripts\Activate.ps1
python -m shared.transcribe 01-voicemail-triage\testset\audio\06_urgent_landlord_deadline.ogg --cache-dir 01-voicemail-triage\testset\transcripts
python -m shared.transcribe 01-voicemail-triage\testset\audio\18_other_number_corrected.wav --cache-dir 01-voicemail-triage\testset\transcripts --model large-v3-turbo
python 01-voicemail-triage\compare_models.py
```
Expected: each command prints the file name, model, `audio=…s transcribe=…s this call=0.1s` (cached:
`transcribe` is the original time, `this call` is how long it took now) and the timestamped segments.
The last command finishes in seconds (all cached) with `small names 9/14 numbers 8/13` and
`large-v3-turbo names 11/14 numbers 11/13`. Delete `testset\transcripts\` to force a fresh (slow) run.

### 7. What can go wrong
- **"ffmpeg not found on PATH":** the terminal was opened before ffmpeg was installed. Open a new one.
- **Very slow transcription:** other heavy work on the laptop (a download, a build, many browser tabs).
  The first run of file 06 took 86 s during the model download; on a quiet machine, ~20 s.
- **"oh" written as "a":** "a 1632-960-789" loses the leading zero. Whisper's English model hears "oh"
  as a word. Phase 3 must repair this, not this step.
- **Numbers written as words:** turbo once wrote "oh seven seven double o nine hundred…". Correct, but
  any code that only looks for digits misses it.
- **Stale cache:** if you change Whisper settings (beam size, VAD) but not the model name, the cache still
  returns old results. Delete the cache folder after changing settings.

### 8. Check my understanding
1. You rename `05_urgent_noisy_water_heater.wav` to `voicemail.wav` and transcribe it again with the same model. Is Whisper run again? Which function decides that, and why?
2. Why does the cache file name contain the model name as well as the hash?
3. In file 18, `small` wrote "07700940349". Is that a problem for the transcription step to fix, the analysis step, or neither? What could a later step do about a number it isn't sure of?

### 9. Next phase preview
Phase 3 adds the analysis step: a Pydantic `Result` schema (category, urgency, name, callback number,
summary) in `shared/schemas.py`, and `analyze(transcript) -> Result`, which asks `qwen2.5:7b` in Ollama for JSON
in exactly that shape and checks it, with one retry if the JSON is invalid. That's also where "a 1632…",
"double o" and "spelled S-I-O-B-H-A-N" get turned into clean answers.

---

## Part 1, Phase 3: Analysis (2026-10-07)

### 1. What we built
A function `analyze(transcript) -> Result` that asks the local LLM (`qwen2.5:7b` in Ollama) to read a
transcript and fill in a fixed form: summary, reason, category, urgency (1-3), caller name, callback number,
language. The form is a Pydantic schema; Ollama is forced to answer in exactly that JSON shape, and plain-code
rules check the answer. If a rule fails, the model gets the error back **once** to correct itself.

### 2. Where it fits in the pipeline
```
 audio ─► inbox/ ─► 1.transcribe ─► [2. ANALYZE] ─► 3.route ─► 4.store + move file
                     (Phase 2)       (Phase 3)       (Phase 4)   (Phase 4-5)
                                     <-- you are here
           Transcript ─► prompt + schema ─► Ollama ─► JSON ─► Pydantic rules ─ok─► Result JSON
                                               ▲                     │
                                               └── error, 1 retry ◄──┘ fail twice ─► AnalysisError
```

### 3. How it works, step by step
`shared/analyze.py` → `analyze(transcript, cache_dir, llm)`:
1. `cache_path()` builds `<hash>_<whisper model>_<llm>_<PROMPT_VERSION>.json` (':' in "qwen2.5:7b" becomes
   '-' because Windows forbids ':' in file names). If it exists, load and return it.
2. `messages` = the `SYSTEM_PROMPT` (business context + rules for each field) and the transcript between `<<< >>>`.
3. `ask_llm()` calls `client.chat(..., format=Analysis.model_json_schema(), options={"temperature": 0})`.
   `format=<schema>` makes Ollama only produce tokens that fit the JSON shape; `temperature=0` makes it pick
   the most likely answer every time, so runs are repeatable.
4. `Analysis.model_validate_json(raw)` parses and checks the reply (`shared/schemas.py`):
   - `category` must be one of 5 `Literal` values; `urgency` must be 1-3 (`Field(ge=1, le=3)`).
   - `check_uk_number()`: keep digits only, then require `0` + 9-10 digits.
   - `empty_name_is_none()`: "", "unknown" → null. `two_letter_language()`: "en-GB" → "en".
   - `urgent_means_today()`: category urgent **if and only if** urgency 3.
5. On `ValidationError`: `short_errors()` turns it into readable lines; we append the model's own reply and
   "Your answer broke these rules: … do not replace a value the caller said with null" and ask again.
   Second failure → `AnalysisError` (Phase 5 will move such files to `failed/`).
6. A `Result` wraps the `Analysis` with bookkeeping (models, prompt version, attempts, the rejected first reply
   and why, time) and is written to the cache via `.tmp` + rename.

`01-voicemail-triage/analyze_testset.py` runs this over the 18 test files and compares with `labels.json`.

### Results (small transcripts, CPU)
| | prompt v1 | prompt v2 |
|---|---|---|
| urgent missed | 0/6 | 0/6 |
| category | 12/18 | 15/18 |
| urgency | 12/18 | 11/18 |
| name | 11/18 | 10/18 |
| number | 15/18 | 16/18 |
| false "urgent" | 0 | 1 (12, scam) |

v1 → v2 changes: "sales = selling TO Brightwater" (v1 put quotes/bookings in sales), retry message "fix, don't
delete" (v1's retry on 09 replaced a number missing its 0 by null), "a name after 'Hi' is usually who is being
called", and the example "Rachel from Acme → Rachel". Side effects: names shortened to first names (caused by
that example), and scam 12 became urgent because its "deadline today" matched the urgent definition.

### 4. Key concepts I should understand
- **Structured output:** the schema goes *into* the request, so the model can't answer in prose. Example: it
  cannot reply "This seems urgent!"; it must produce `{"summary": …, "category": "urgent", …}`.
- **Validation = plain-code rules on model output:** the model proposes, code checks. Example: "1632960222"
  is rejected by `check_uk_number()` no matter how confident the model is.
- **Retry with feedback, and its risk:** at temperature 0, asking again without new info gives the same
  answer, so the retry includes the error. But the easiest way to satisfy a rule is to delete the value:
  v1 turned a fixable number into null. Rules + feedback must say what a *good* fix looks like.
- **Field order is a thinking order:** `summary` and `reason` come before `category`, so the model writes
  its justification first and then commits.
- **Prompts are code without tests:** one example sentence ("Rachel from Acme → Rachel") fixed one file and
  shortened three other names. Every prompt change needs the whole test set re-run, hence `PROMPT_VERSION`
  in the cache key.

### 5. Files created or changed
- `shared/analyze.py`: prompt, Ollama call, validation + one retry, cache, command-line demo.
- `shared/schemas.py`: added `Category`, `Analysis` (with validators) and `Result`.
- `01-voicemail-triage/analyze_testset.py`: runs analysis on the test set and compares with labels.
- `.gitignore`: added `testset/results/`.
- `PROJECT_RULES.md`, `README.md`: status updated.

### 6. Try it yourself
In a new terminal (ffmpeg on PATH), with Ollama running:
```powershell
cd C:\Users\danuk\code\audio-to-action
.\.venv\Scripts\Activate.ps1
python 01-voicemail-triage\analyze_testset.py
python -m shared.analyze 01-voicemail-triage\testset\audio\06_urgent_landlord_deadline.ogg --transcripts 01-voicemail-triage\testset\transcripts --results 01-voicemail-triage\testset\results
```
Expected: the first prints one line per file and ends with `category 15/18 urgency 11/18 name 10/18
number 16/18`, `urgent voicemails missed: 0 []`, in seconds because everything is cached. The second prints
the transcript and a JSON result with `"category": "urgent"` and `"callback_number": "01632960789"`.
To see a retry, open `testset\results\*_v2.json` files with `"attempts": 2` and read `rejected_because`.

### 7. What can go wrong
- **Slow (1-2 min per voicemail):** Ollama runs on CPU because the RTX 5050 shows Code 43 in Device Manager.
  Updating the NVIDIA driver should bring it to a few seconds.
- **"Ollama not reachable":** the Ollama app isn't running.
- **Urgent-sounding scams:** "suspended today, final notice" matches the urgent definition (file 12 in v2).
- **The LLM can't fix what Whisper misheard:** "Carver", "Colleen" (from "calling"), "0770090618".
- **Retry that deletes data:** the model may "fix" a rule violation by returning null.
- **Over-tuning on 18 files:** each prompt tweak can just memorize the test set. Phase 6 needs held-out data.

### 8. Check my understanding
1. Ollama already forces the JSON shape with `format=<schema>`. Why do we still validate with Pydantic afterwards?
2. Why would retrying with exactly the same messages be pointless at `temperature=0`?
3. In v1, file 09 ended with `callback_number: null` after a retry. Which is worse for Sam: a number missing its first 0, or no number at all? How did v2 change the outcome?

### 9. Next phase preview
Phase 4 adds routing in plain code: rules like "urgent → ntfy notification now", "spam → archive",
"missing number or name → review queue". Every result is stored in SQLite. The model doesn't decide
any of this; `if` statements do, and you'll be able to read every rule.

---

## Part 1, Phase 4: Routing (2026-10-07)

### 1. What we built
The part that *acts* on the analysis. Plain `if` rules decide where each voicemail goes (push now,
inbox, personal, archive), whether a human should double-check it, and why. Urgent ones trigger an
ntfy push with deliberately boring text ("Urgent voicemail, check the laptop"), and every voicemail
becomes one row in a local SQLite database. The LLM decides nothing here; its answer is just input.
Your choices: minimal push text, sales → archive, pushes in dry-run mode for now.

### 2. Where it fits in the pipeline
```
 audio ─► inbox/ ─► 1.transcribe ─► 2.analyze ─► [3. ROUTE] ─► 4.store + move file
                     (Phase 2)       (Phase 3)    (Phase 4)     (store: Phase 4, move: Phase 5)
                                                  <-- you are here
   Result + Transcript ─► route() ─► Decision ─► send_push() if notify ─► save() to SQLite
                          (pure rules)           (ntfy, dry run)          (voicemails.db)
```

### 3. How it works, step by step
`01-voicemail-triage/deliver.py` → `deliver(conn, transcript, result)`:
1. `routing.route(transcript, result)` (pure, no side effects):
   - `CATEGORY_ROUTE[category]`: urgent → `notify_now`, other → `inbox`, personal → `personal`,
     spam/sales → `archive`.
   - Safety net: `safety_hits(text)` searches `SAFETY_PATTERNS` (regex phrases like `smell(s)? of gas`,
     `burst`, `sparking`, `no heating`). If any match and the LLM did **not** say urgent: never archive,
     push anyway, flag review.
   - Review flags: customer call (urgent/other) without number or name, analysis needed a retry,
     Whisper confidence below -1.0, no speech at all.
   - Returns a `Decision(route, notify, review, reasons)`; `reasons` is the plain-language trail.
2. If `decision.notify`: `routing.push_text()` builds title/message (no caller data), and
   `shared/notify.send_push()` POSTs it to `NTFY_SERVER/NTFY_TOPIC`, or prints it when `NTFY_DRY_RUN=1`.
3. `store.save()` does `INSERT ... ON CONFLICT(audio_sha256) DO UPDATE`: one row per audio hash, and
   `notified_at` keeps the *first* push time (`COALESCE`).

`route_testset.py` runs this on the 18 cached results (instant) into `testset/triage_test.db`;
`tests/test_routing.py` forces each rule with hand-made cases.

### Results
- Test set: 0 urgent archived. 7 pushes (6 real urgent + scam 12, which the LLM called urgent).
  Archive: 09, 10 (sales), 11 (spam). Review: 04 (no number), 11 and 15 (retry), 12 (no name/number).
- The safety net never fired on the test set (LLM caught every urgent call) and had no false alarms;
  the unit tests prove it works when the LLM misses.
- Whisper confidence was high everywhere (lowest -0.38), *including* the misheard files, so the
  confidence flag can't catch "Carver" or the wrong number in 18.

### 4. Key concepts I should understand
- **Pure decision, separate actions:** `route()` only returns a decision; `deliver()` does the sending
  and saving. Example: the 7 tests run in 0.4 s with no network, database or LLM.
- **Defense in depth:** the LLM is the first filter; dumb keyword rules are a second, independent one.
  Example: if the LLM calls "smell of gas" `other`, the regex still pushes. Two filters that fail in
  *different* ways miss less than one clever filter.
- **Asymmetric errors:** a false alarm (scam 12 pushed) costs Sam a glance; a missed emergency could cost
  a flooded house or worse. So the rules lean towards pushing and reviewing, never towards archiving.
- **Parameterized SQL:** `?` placeholders, never f-strings. Example: a caller saying
  `'); DROP TABLE voicemails; --` is stored as text, not run as SQL.
- **Privacy by content design:** the push says *that* something happened, not *what*. Example: a stranger
  who guesses the topic learns "urgent voicemail at 14:32", not Tom Bradley's address.

### 5. Files created or changed
- `01-voicemail-triage/routing.py`: `route()`, `Decision`, safety patterns, `push_text()`.
- `01-voicemail-triage/deliver.py`: `deliver()`: decide, push, save.
- `01-voicemail-triage/store.py`: SQLite table, `connect()`, `save()`, `summary()` (+ command line).
- `01-voicemail-triage/route_testset.py`: routes the test set, fails if an urgent one is archived.
- `01-voicemail-triage/tests/test_routing.py`: 7 tests for the rules.
- `shared/notify.py`: `send_push()` with dry-run mode.
- `.env.example` / `.env`: `NTFY_DRY_RUN=1`. `requirements.txt`: `pytest`. `.gitignore`: `.pytest_cache/`.

### 6. Try it yourself
```powershell
cd C:\Users\danuk\code\audio-to-action
.\.venv\Scripts\Activate.ps1
python 01-voicemail-triage\route_testset.py
python -m pytest 01-voicemail-triage\tests -v
python 01-voicemail-triage\store.py 01-voicemail-triage\testset\triage_test.db
```
Expected: 7 `[ntfy dry run]` lines, a table per file, `Urgent voicemails routed to the archive: 0 []`;
then `7 passed`; then counts `archive 3, inbox 6, notify_now 7, personal 2` and the 4 review items.
Try it: in `routing.py` change `"sales": "archive"` to `"inbox"`, re-run the first command, and watch 09/10 move.

### 7. What can go wrong
- **False alarms:** scams written to sound urgent (12) get pushed. Accepted on purpose (asymmetric errors).
- **Safety words missing a phrasing:** "I can smell something like gas" doesn't match `smell of gas`.
  Regex nets are simple but never complete; add phrases when real voicemails slip through.
- **Double push:** if saving crashes right after a real push, a re-run pushes again (Phase 5 fixes this).
- **Topic left as placeholder:** pushes silently stay dry runs. That's intended until you set a real,
  random `NTFY_TOPIC` and `NTFY_DRY_RUN=0`.
- **ntfy.sh unreachable:** `send_push` raises after 10 s; Phase 5 adds retries with backoff.

### 8. Check my understanding
1. Why is `route()` written so it never sends or saves anything itself? What would be harder if it did?
2. The LLM already classifies urgency. What does the regex safety net add, and why use phrases like
   "smell of gas" instead of the single word "gas"?
3. Scam 12 caused a push. Why do the rules accept that kind of mistake but not the opposite one?

### 9. Next phase preview
Phase 5 glues everything into `run.py`: watch `inbox/`, and for each new file run transcribe → analyze →
deliver, then move it to `processed/`. Re-running on the same file does nothing (checked by hash in
SQLite), network calls get retries with backoff, files that fail go to `failed/` with the error, and every
step is logged.

---

## Part 1, Phase 5: Glue & reliability (2026-10-07)

### 1. What we built
`run.py`, the program that actually runs the pipeline. Drop a voicemail into `inbox/` and it gets
transcribed, analyzed, routed, stored and moved to `processed/`. It's built to survive real life:
the same audio twice does nothing the second time, a temporarily missing Ollama or ntfy gets retried
with growing waits, a broken file ends up in `failed/` with a note saying why (the rest carry on),
and every step is written to `logs/run.log`.

### 2. Where it fits in the pipeline
```
 [Phase 5: run.py glues it all together]  <-- you are here
 inbox/ ─► hash ─► seen before? ─yes─► processed/ (nothing else happens)
              │no
              ▼
          1.transcribe ─► 2.analyze ─────► 3.deliver ─────────► processed/
          (cache)         (cache, retries)  (route, push w/ retries, save)
              │               │                 │
              └───────────────┴─── any error ───┴──► failed/ + .error.txt
 everything logged to logs/run.log;  --retry-failed puts failed/ files back in inbox/
```

### 3. How it works, step by step
`01-voicemail-triage/run.py` → `main()`:
1. `Folders(base)` creates/locates `inbox/ processed/ failed/ transcripts/ results/ logs/` and `voicemails.db`.
   `--base` points it at another folder (we tested in a scratch copy, so the real DB stays clean).
2. `setup_logging()` sends log lines to the console and `logs/run.log` (file names, steps, timings,
   routes; never transcript text or numbers).
3. `run_once()` → `find_ready_files()`: audio files in `inbox/`, oldest first, skipping anything changed in
   the last 3 s (it may still be copying in) → `process_file()` for each.
4. `process_file()`:
   - `file_sha256()` → `already_processed()` checks SQLite. Found → move to `processed/`, return "skipped".
   - `transcribe()` (cached) → `with_retries(analyze, ...)` → `deliver()` (whose push is also wrapped in
     `with_retries`), each wrapped in `timed()` for the log.
   - Success → `move_to(processed/)`. Any exception → log it, `move_to(failed/)`, write `<name>.error.txt`
     with the step and traceback. It never raises, so one bad file can't stop the batch.
5. `--watch` repeats `run_once()` every `--interval` seconds until Ctrl+C.
   `--retry-failed` first moves `failed/` files back to `inbox/` (caches make the retry cheap).

`shared/retry.py` → `with_retries(fn, ...)`: up to 3 attempts, waits 2 s then 4 s, but **only** if
`is_transient(exc)`: connection errors, timeouts, HTTP 429/5xx. Anything else is raised immediately.

### Results (scratch-folder test)
| scenario | outcome |
|---|---|
| 3 voicemails + `broken.mp3` | 3 processed (1 push), broken → `failed/` with ffmpeg's error, batch continued |
| same audio as 01, renamed | skipped: "already processed (same audio)", no push, no new row |
| Ollama unreachable | retry after 2 s, after 4 s, then `failed/` at step "analyze" |
| `--retry-failed`, Ollama back | real LLM call (202 s on CPU) → `inbox`; broken file failed again, instantly |
| `--watch`, file dropped in while running | picked up ~4 s later and processed |

### 4. Key concepts I should understand
- **Idempotency:** doing it twice = doing it once. Example: `copy_of_pipe.wav` has the same bytes as
  `01_urgent_burst_pipe.wav`, so its hash is already in SQLite: no LLM call, no second push.
- **Transient vs permanent errors:** retry what might fix itself, fail fast on what won't. Example:
  "can't connect to Ollama" → retried; "this .mp3 is text" → straight to `failed/` (retrying = wasted time).
- **Exponential backoff:** wait 2 s, then 4 s (then 8 s...). Example: if Ollama is restarting, hammering it
  every 0.1 s doesn't help; giving it more time each round does.
- **At-least-once vs at-most-once:** across two systems (ntfy and SQLite) you can't guarantee "exactly once".
  Example: push → crash → no row → next run pushes again. We chose that over save → crash → never pushed.
- **Isolate failures:** `process_file()` catches everything for *one* file. Example: `broken.mp3` failed
  between three good files and all three still went through.

### 5. Files created or changed
- `01-voicemail-triage/run.py`: the pipeline (once / `--watch` / `--retry-failed` / `--base`).
- `shared/retry.py`: `is_transient()` and `with_retries()`.
- `01-voicemail-triage/tests/test_retry.py`: 4 tests (backoff waits, give up, no retry on permanent, HTTP codes).
- `01-voicemail-triage/deliver.py`: push wrapped in retries; at-least-once explained.
- `shared/transcribe.py`: convert with ffmpeg *before* loading Whisper (broken files fail in 0 s, not 8 s).
- `shared/notify.py`: dry-run message via `logging` instead of `print`. `route_testset.py`: logging setup.
- `.gitignore`: `01-voicemail-triage/logs/`. `PROJECT_RULES.md`, `README.md`: status.

### 6. Try it yourself
In a new terminal (ffmpeg on PATH), Ollama running:
```powershell
cd C:\Users\danuk\code\audio-to-action
.\.venv\Scripts\Activate.ps1
copy 01-voicemail-triage\testset\audio\13_other_appointment_confirm.wav 01-voicemail-triage\inbox\
python 01-voicemail-triage\run.py
python 01-voicemail-triage\run.py
copy 01-voicemail-triage\processed\13_other_appointment_confirm.wav 01-voicemail-triage\inbox\again.wav
python 01-voicemail-triage\run.py
python 01-voicemail-triage\store.py
```
Expected: the first run logs `transcribe` (~20 s), `analyze` (~1-3 min on CPU), `-> inbox`, `1 processed`.
The second run: `0 processed, 0 skipped, 0 failed` (inbox empty). The third: `again.wav already processed
(same audio)`, `1 skipped`. `store.py` shows `inbox 1`. This uses the real `voicemails.db`; delete it
afterwards for a clean start. Then try `python 01-voicemail-triage\run.py --watch` and copy a file in.

### 7. What can go wrong
- **Slow:** ~2-4 min per new voicemail while Ollama is on CPU (GPU Code 43).
- **Duplicate push** after a crash between push and save: by design (at-least-once).
- **Same message, different bytes = new voicemail:** idempotency is by exact file content.
- **Failed files wait silently:** nothing alerts you about `failed/`; check it or the log (Phase 7 digest).
- **Two runs at once** (e.g. two `--watch` terminals) can grab the same file. Run one at a time.
- **Prompt change and old rows:** a voicemail already in the DB is never re-analyzed, even with a new prompt.

### 8. Check my understanding
1. You copy the same voicemail into `inbox/` twice under different names. What exactly happens to the second one, and which function decides that?
2. Why does `with_retries` retry "can't connect to Ollama" but not an `AnalysisError` (invalid answer twice)?
3. We push *before* saving. Describe the crash that causes a duplicate push, and the crash we avoided by not saving first.

### 9. Next phase preview
Phase 6 is evaluation: an eval script that runs the steps over the whole test set and computes metrics
(category accuracy, urgent false-negative rate, name/number accuracy), writes `docs/eval-results.md`,
and runs one single-variable experiment (candidate: the "Rachel from Acme" example that shortened names).

---

## Part 1, Phase 6: Evaluation (2026-10-07)

### 1. What we built
A measuring tool. `evaluate.py` runs every labelled voicemail through transcribe → analyze → route and
compares the output with the answer key, producing one table of numbers per configuration (Whisper model
+ prompt version) plus a confusion matrix and a list of every single error, saved in `docs/eval-results.md`.
Then we ran one controlled experiment: change one line of the prompt, keep everything else the same, and
decide with a rule written down *before* seeing the result.

### 2. Where it fits in the pipeline
```
 audio ─► 1.transcribe ─► 2.analyze ─► 3.route            (the real pipeline, unchanged)
              │               │            │
              ▼               ▼            ▼
 ┌──────────────────────────────────────────────────────┐
 │ [Phase 6: EVALUATE]  compare with testset/labels.json │ <-- you are here
 │ metrics per config → docs/eval-results.md             │
 └──────────────────────────────────────────────────────┘
```

### 3. How it works, step by step
`01-voicemail-triage/evaluate.py` → `main()`:
1. `load_labels()`: `testset/labels.json` + `testset/my_recordings/labels.json` if you add real recordings.
2. For each config `whisper:prompt` (e.g. `small:v2`), `run_config()` calls `transcribe()`, then
   `analyze(..., prompt_version=prompt)`, then the pure `route()` for every file. Cached, so a known config is instant.
3. `metrics()` computes, using `shared/evaluation.py`:
   urgent false negatives, urgent archived, false urgent, category / urgency accuracy, name exact
   (`same_value`) and first word (`same_first_word`), numbers as correct / wrong / missed / invented
   (`null_aware`), pushes, review flags, retries, LLM time.
4. `build_report()` writes the metrics table, the hand-written conclusions from `docs/eval-notes.md`,
   and per config a confusion matrix (`confusion()`) and every error (`errors_table()`).

`shared/analyze.py` now holds `PROMPTS = {"v2": ..., "v3": ...}`, where v3 is built as
`PROMPT_V2.replace(V2_EXAMPLE, V3_EXAMPLE)` with an `assert`, so the code itself guarantees exactly one
line differs. `ANALYSIS_PROMPT` in `.env` picks the default (v2).

### Results
| metric | small:v2 | small:v3 |
|---|---|---|
| urgent false negatives | 0/6 | 0/6 |
| urgent archived | 0 | 0 |
| category accuracy | 15/18 (83 %) | 15/18 (83 %) |
| urgency accuracy | 11/18 | 10/18 |
| name exact / first word | 10/18 / 14/18 | 10/18 / 12/18 |
| number correct (wrong / missed / invented) | 16/18 (2/0/0) | 16/18 (2/0/0) |

Experiment: v3 brought surnames back (Laura Jenkins, Brenda Walsh) but broke two unrelated answers (Mum → null,
"Yak" invented) → exact names unchanged → rule not met → **v2 stays**.
Definition of done: no urgent archived ✔, re-run does nothing ✔, **90 % category ✘ (83 %)**.

### 4. Key concepts I should understand
- **Metrics that match the cost of mistakes:** "urgent false negatives" is listed first because missing a gas
  leak is the expensive error. Example: 0/6 here matters more than the 83 % overall accuracy.
- **Confusion matrix:** rows = truth, columns = prediction, so you see *which* mistakes happen. Example:
  the spam row has a 1 under "urgent": scam 12, not a random error.
- **Single-variable experiment:** change one thing, so any difference has one cause. Example: v3 differs from
  v2 in one line, enforced by an `assert` in the code.
- **Pre-registered decision rule:** decide what "better" means before looking. Example: v3 improved surnames,
  and it would be tempting to call that a win, but the rule said exact names must go up, and they didn't (10 → 10).
- **Small samples are noisy:** with 18 files, one sentence moved answers on unrelated files (Mum, Yak). Example:
  a ±2 difference can't be distinguished from that noise; more data is the fix, not more tweaking.

### 5. Files created or changed
- `01-voicemail-triage/evaluate.py`: the evaluation (configs, metrics, report).
- `shared/evaluation.py`: reusable scoring helpers.
- `shared/analyze.py`: `PROMPTS` dict (v2, v3), `prompt_version` parameter, `ANALYSIS_PROMPT` default.
- `docs/eval-results.md` (generated) and `docs/eval-notes.md` (hand-written conclusions, inserted into it).
- `01-voicemail-triage/analyze_testset.py`: removed (replaced by `evaluate.py`).
- `.env.example` / `.env`: `ANALYSIS_PROMPT=v2`. `PROJECT_RULES.md`: status.

### 6. Try it yourself
```powershell
cd C:\Users\danuk\code\audio-to-action
.\.venv\Scripts\Activate.ps1
python 01-voicemail-triage\evaluate.py --configs small:v2 small:v3
python 01-voicemail-triage\evaluate.py --configs large-v3-turbo:v2 --no-write
```
Expected: the first prints the metrics table above in ~2 s (all cached) and rewrites `docs\eval-results.md`.
The second is the next experiment (turbo transcripts are cached from Phase 2, but the analysis isn't): 18 LLM
calls, about 30-60 min on CPU; `--no-write` keeps the saved report unchanged.

### 7. What can go wrong
- **Overfitting the test set:** every tweak judged on the same 18 files makes the numbers look better than
  reality. Hold some data back.
- **Label mistakes look like model mistakes:** 08 (friend's football: personal or other?) is arguably a label question.
- **Synthetic audio is too clean:** real voicemails will score lower; add `my_recordings/` to find out by how much.
- **Windows console encoding:** printing "≥" crashed on cp1252; fixed with `sys.stdout.reconfigure(encoding="utf-8")`
  and by saving the report before printing.
- **Slow experiments on CPU:** a new prompt version = 18 LLM calls = 25-60 min until the GPU works.

### 8. Check my understanding
1. Why is "urgent false negatives" more important than "category accuracy" for this project?
2. v3 got more full surnames right. Why did we still keep v2?
3. Why does the `assert V2_EXAMPLE in PROMPT_V2` line matter for the experiment?

### 9. Next phase preview
Phase 7 is polish: a daily digest (callbacks, review items, failed files) with a counts-only push, a full
README with the architecture and results, a summary of what `shared/` gives Part 2, and a technical handover report.

---

## Part 1, Phase 7: Polish (2026-10-07)

### 1. What we built
The finishing touches that make Part 1 usable day to day and understandable by someone else: a daily
digest (one page with who to call back, what needs review and what failed, plus an optional counts-only
push), a complete README (architecture, setup, usage, configuration, results, privacy, how to automate it
with Task Scheduler), a guide to what Part 2 can reuse from `shared/`, and a technical handover report.

### 2. Where it fits in the pipeline
```
 inbox/ ─► run.py: transcribe ─► analyze ─► route ─► deliver ─► processed/ | failed/
                                                        │
                                                        ▼
                                                  voicemails.db ─► [digest.py] ─► digests/<date>.md
                                                                   <-- you are here   (+ counts-only push)
```

### 3. How it works, step by step
`01-voicemail-triage/digest.py` → `main()`:
1. `recent_rows(conn, hours)`: rows with `processed_at >= now - hours`. `processed_at` is ISO text in UTC,
   so comparing strings compares times correctly. Sorted urgent → inbox → personal → archive.
2. `failed_files(failed/)`: every audio file still in `failed/` and the step from its `.error.txt`.
3. `build()`: a Markdown page with one section per route; each line is
   `name, number: summary` plus the review reasons when flagged (`line()`).
4. Saves `digests/<date>.md` (git-ignored: it has names and numbers) and prints it.
5. `--push`: `send_push("Voicemail digest", "2 to call back, 1 to review, 0 failed.")`: counts only.

### 4. Key concepts I should understand
- **A pipeline needs a "what happened" view:** files in `failed/` are silent otherwise. Example: the digest's
  "Failed, needs attention" section is how `broken.mp3` gets noticed.
- **Same privacy rule everywhere:** the detailed page stays local; anything that leaves the machine carries
  counts only. Example: the digest push says "1 failed", not which caller.
- **Documentation for different readers:** README = how to use it, learning log = how to understand it,
  HANDOVER = how to change it. Example: another agent should start with `docs/HANDOVER.md`.
- **Scheduling outside the code:** Task Scheduler runs `run.py --watch` at logon and `digest.py --push` daily;
  the code itself stays simple (no built-in scheduler).

### 5. Files created or changed
- `01-voicemail-triage/digest.py`: the daily digest.
- `README.md`: rewritten (architecture, usage, config, results, privacy, docs).
- `docs/shared-for-part2.md`: what Part 2 reuses and what it must change.
- `docs/HANDOVER.md`: full technical report for another developer or agent.
- `.gitignore`: `01-voicemail-triage/digests/`. `PROJECT_RULES.md`: status.

### 6. Try it yourself
```powershell
cd C:\Users\danuk\code\audio-to-action
.\.venv\Scripts\Activate.ps1
python 01-voicemail-triage\digest.py --hours 168 --push
```
Expected: a `[ntfy dry run] … 'N to call back, N to review, N failed.'` line, then the page with sections
"Urgent", "Call back", "Personal", "Archived", "Failed", and `Saved: …\digests\<date>.md`. If you haven't run
the pipeline on real files yet, the sections are empty: process a test file with `run.py` first.

### 7. What can go wrong
- **Empty digest:** the time window (`--hours`) is based on processing time, not on when the call came in.
- **Task Scheduler runs without your terminal's PATH:** if ffmpeg isn't found in scheduled runs, add its folder
  to the *system* PATH or call the task through a small `.cmd` that sets PATH first.
- **Laptop asleep at 18:00:** the scheduled digest doesn't run; tick "run as soon as possible after a missed start".

### 8. Check my understanding
1. Why does the digest page stay on the laptop while the digest push may go through ntfy.sh?
2. Which file would you give to someone who wants to *use* the system, to *learn* from it, or to *change* it?
3. The definition of done says ≥ 90 % category accuracy, and we got 83 %. What would you try first, and why not another prompt tweak?

### 9. Next phase preview
Part 1 is complete. Part 2 (`02-meeting-action-agent/`) turns meeting recordings into action items. It reuses
`shared/transcribe.py`, `retry.py`, `notify.py` and `evaluation.py` as they are, and the analysis *pattern*
with a new schema; long recordings will need the GPU or chunking (see `docs/shared-for-part2.md`). It is also
the first place where a small agent part is allowed, so deciding what stays workflow will be the main design question.

---

## Part 1 summary
| Step | Code | Workflow or agent? | Why |
|---|---|---|---|
| Transcribe | `shared/transcribe.py` | workflow | always the same conversion; no decision to make |
| Analyze | `shared/analyze.py` | workflow step that *uses* an LLM | the LLM fills a fixed form; code validates it; it never picks the next step |
| Route | `routing.py` | workflow | business rules as `if` statements: testable, explainable, same input → same output |
| Deliver / store | `deliver.py`, `store.py` | workflow | side effects in a fixed order (push before save, at-least-once) |
| Glue | `run.py` | workflow | fixed sequence, retries and failure handling in plain code |
