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

---

# Part 2: meeting-action-agent

## Part 2, Phase 0: Plan & setup (2026-10-07)

### 1. What we built
No code yet: a plan. We decided what Part 2 is for (meeting recordings → action items → a task tracker that
stays correct across meetings), where the line between workflow and agent goes, how we'll test it, and what
"done" means. The plan lives in `02-meeting-action-agent/README.md` and `PROJECT_RULES.md`, and the working folders exist.

### 2. Where it fits in the pipeline
```
 [Phase 0: PLAN]  <-- you are here
 meeting audio ─► 1. transcribe ─► 2. extract items ─► 3. SYNC with tracker ─► tracker.db + trace
                  (workflow,        (workflow,           (AGENT: tool-calling
                   reused)           Part 1 pattern)      loop, bounded)
 measured against a synthetic meeting series + answer key (Phase 1) by the eval script (Phase 6)
```

### 3. How it works, step by step
Nothing runs yet. The decisions, in the order they shape the work:
1. **Test data:** a *series* of short synthetic weekly meetings (2-4 min, 2-4 Piper voices) at the Part 1
   plumbing business. A series, because the agent's whole job is relating new items to old tasks.
2. **Speakers:** no diarization; owners come from what people say. Cheaper, and its weak spot
   ("I'll do it" from an unknown speaker) is something we'll measure, not guess about.
3. **Agent:** only the tracker sync. Tools: `list_open_tasks`, `search_tasks`, `add_task`, `update_task`,
   `close_task`, `mark_duplicate`, `finish`. Guardrails: step limit, allowlist, validated arguments, a reason
   stored with every change, a saved trace.
4. **Done =** item precision and recall ≥ 80 %, tracker state ≥ 80 % correct after the series, no guardrail
   broken, re-runs do nothing, and you can explain the workflow/agent boundary.

### 4. Key concepts I should understand
- **Workflow vs agent:** in a workflow, *code* decides the next step; in an agent, the *model* decides which
  tool to call next, in a loop, until it says it's finished. Example: "extract items from this transcript" is
  always one step (workflow); "is this item the same as task #12, or an update to it, or new?" may need
  1 lookup or 5 (agent).
- **Agents need state to reason about:** without earlier tasks, there's nothing to decide. Example: the test
  set is a meeting *series* so that "did Priya send the quote?" refers to a real task from an earlier meeting.
- **Bounded autonomy:** an agent's path isn't fixed, so we fix its limits instead. Example: at most N steps,
  only 7 tools, every argument checked by Pydantic, every change logged with a reason.
- **Precision vs recall:** precision = of the items we found, how many are real; recall = of the real items,
  how many we found. Example: inventing tasks hurts precision, missing "someone should look at the van" hurts recall.
- **Decide what to measure before building:** like Part 1, the answer key (items per meeting, tracker state
  after each) comes before any pipeline code.

### 5. Files created or changed
- `02-meeting-action-agent/README.md`: problem, planned architecture, decisions, phases.
- `02-meeting-action-agent/{inbox,processed,failed,transcripts,results,testset}/.gitkeep`: working folders.
- `02-meeting-action-agent/.gitkeep`: removed (the folder has real content now).
- `.gitignore`: Part 2 working folders, logs, test audio/transcripts/results.
- `PROJECT_RULES.md`: Part 2 section, phases, definition of done.

### 6. Try it yourself
```powershell
cd C:\Users\danuk\code\audio-to-action
type 02-meeting-action-agent\README.md
git check-ignore -v 02-meeting-action-agent\inbox\test.wav
```
Expected: the plan prints; the second command shows the `.gitignore` rule that keeps meeting audio out of git.

### 7. What can go wrong
- **Synthetic meetings are too tidy:** real meetings have crosstalk, interruptions and half-sentences. Scores
  will be optimistic, as in Part 1.
- **Long audio on CPU:** a 4-min meeting ≈ 7 min of Whisper `small` plus several LLM minutes while the GPU shows
  Code 43. That's why the meetings are kept short.
- **Context length:** Ollama gave us a 4096-token context; a long transcript plus a prompt can exceed it
  (Phase 3 handles this by chunking).
- **Agent loops:** a model can keep calling tools without finishing; the step limit is the backstop.
- **Owners without diarization:** "I'll handle it" means nothing without knowing who spoke.

### 8. Check my understanding
1. Why is "extract action items from this meeting" a workflow step, while "sync them with the tracker" is an agent?
2. Why does the test set need several meetings in a row instead of independent ones?
3. Give one example of an extraction mistake that hurts precision and one that hurts recall.

### 9. Next phase preview
Phase 1 builds the meeting series: scripts with speaker turns (who says what), rendered with several Piper
voices into one audio file per meeting, and the answer key: action items per meeting and the expected
tracker state after each meeting, written before any pipeline code.

---

## Part 2, Phase 1: Test set: a meeting series (2026-10-07)

### 1. What we built
Five fake-but-realistic weekly team meetings (Mondays 7 Sep - 5 Oct 2026) at the plumbing business, each
~30 s, four distinct voices (Sam, Priya, Tom, Jamie), plus an answer key: 14 tasks and, per meeting, what is
said about which task. Because later meetings close, reassign, cancel or repeat earlier tasks, the answer key
also defines what the tracker should look like after every meeting (final: 10 done, 1 cancelled, 3 open).

### 2. Where it fits in the pipeline
```
 meeting audio ─► 1. transcribe ─► 2. extract items ─► 3. SYNC with tracker (agent)
                        ▲                  ▲                      ▲
                        └──────────────────┴──────────────────────┘
               compared against ┌──────────────────────────────────────────┐
                                │ [Phase 1: MEETING SERIES + ANSWER KEY]   │ <-- you are here
                                │ mentions per meeting, tracker after each │
                                └──────────────────────────────────────────┘
```

### 3. How it works, step by step
`02-meeting-action-agent/testset/generate.py` → `main()`:
1. Reads `scripts.json`: per meeting a list of `[speaker, text]` turns; `voices` maps people to Piper speaker ids.
2. `shared/tts.py` → `load_voice()` finds the Piper model Part 1 already downloaded; `synthesize()` speaks each turn.
3. `join_turns()` glues the turn wavs together with 0.5 s pauses and records who spoke when (`audio/<id>.turns.json`:
   ground truth we deliberately don't give the pipeline, since we chose no diarization).
4. `degrade()` mixes in light pink noise (seed = meeting number) and encodes wav / m4a / mp3 / ogg.
5. `check_labels()` → `answer_key.problems()`: unknown task keys or fields, bad statuses, owners not in the team,
   due dates before the meeting, unused tasks, file names, and `match_problems()`: each task's description must
   fit its own match rule and no other's (catches T1/T8, T7/T11 mix-ups in the rules).
6. `answer_key.tracker_states()` folds the mentions meeting by meeting into the expected tracker.

### Independent review (separate reviewer agent)
Re-ran everything, checked every label against the conversation and the calendar, transcribed all meetings.
Found and fixed: an unintended "owner only from the voice" in m5 (script now says "And Tom, the Ellis gas check?");
Piper mispronouncing "Priya" so Whisper heard "Freya"/"prayer" (TTS spelling "Preeya", like "Shiv awn");
two lines Whisper garbled beyond recovery ("Renewed on Friday" → "We need on Friday"; "Gallagher suite" →
"sweeped"), reworded; a noise-seed bug with `--only`; silence size assumed 16-bit; stale-file check; typo'd
fields; "quotation" added to quote rules. No answers changed. Its biggest finding for Phase 2: giving Whisper
the known names (team, customers, streets) as a hint fixed nearly all name errors in a quick test.

### 4. Key concepts I should understand
- **State over time is what makes an agent necessary:** each meeting alone is easy; knowing that "did Priya send
  the quote?" closes T1 requires remembering T1. Example: T5 insurance is re-mentioned in m3 with no change, and
  the right action is *nothing*.
- **Mentions vs tracker state:** extraction should output what *this* meeting says (mentions); the tracker is the
  accumulation. Example: m2 says T1 is done but gives no due date, so the mention's due is null and the tracker
  keeps 2026-09-09.
- **Derived labels can't drift:** the expected tracker is computed from the mentions by one rule, instead of being
  written by hand a second time. Example: change one mention and every later tracker state updates with it.
- **Fix the input, not the answer:** when TTS or Whisper makes a label impossible to recover, reword the script.
  Example: "Preeya" in the TTS text, "Priya" in the labels, the same idea as "Shiv awn" in Part 1.
- **Synthetic audio isn't bit-for-bit reproducible:** Piper adds a little randomness, so regenerating changes file
  hashes (and caches). Example: once Phase 2+ results exist, don't regenerate casually.

### 5. Files created or changed
- `02-meeting-action-agent/testset/scripts.json`: 5 meetings as speaker turns + voices, noise, formats.
- `02-meeting-action-agent/testset/labels.json`: 14 tasks with match rules, mentions per meeting, conventions.
- `02-meeting-action-agent/testset/answer_key.py`: load, `matches()`, `tracker_states()`, `problems()`, `match_problems()`.
- `02-meeting-action-agent/testset/generate.py`: per-turn voices, joining, noise, encoding, checks.
- `02-meeting-action-agent/testset/README.md`: what each trap tests.
- `shared/tts.py`: reusable Piper wrapper (Part 3's receptionist will speak with it).
- `PROJECT_RULES.md`: Phase 1 status.

### 6. Try it yourself
```powershell
cd C:\Users\danuk\code\audio-to-action
.\.venv\Scripts\Activate.ps1
python 02-meeting-action-agent\testset\generate.py
start 02-meeting-action-agent\testset\audio\m3_2026-09-21.mp3
type 02-meeting-action-agent\testset\audio\m3_2026-09-21.turns.json
```
Expected: 5 lines like `[3/5] m3_2026-09-21.mp3  33 s, 11 turns`, then `labels.json is consistent…` and
`Expected tracker after the series: 14 tasks, …10 done, 1 cancelled, 3 open`. The mp3 plays four different voices.
(Regenerating makes slightly different audio; that's fine before Phase 2, avoid it afterwards.)

### 7. What can go wrong
- **Too tidy:** one TTS engine, no interruptions or crosstalk, short turns. Real meetings will be harder.
- **Small:** 5 meetings, 28 mentions. One mistake moves a percentage a lot.
- **Match rules are word lists:** a reasonable description like "carry out the booked check" can hit the wrong
  task (T7 vs T11). Phase 6 should list items that match several tasks or none for a human look.
- **Owners from the voice:** T5, T7, T8 can't be recovered from text alone, by design; they cap the owner score.

### 8. Check my understanding
1. In m3 Priya says the insurance is "still on my list, due on the thirtieth". What should the tracker do, and why is that a trap?
2. Why are T1 (send quote) and T8 (send revised quote) two tasks and not a duplicate?
3. Why does the answer key store *mentions* per meeting instead of the full tracker state per meeting?

### 9. Next phase preview
Phase 2 runs `shared/transcribe.py` on the meetings: speed, quality, and the main question the reviewer raised:
how much does giving Whisper the known names (team, customers, streets) as a hint help, measured on the
names and words our match rules depend on.

---

## Part 2, Phase 2: Transcription of meetings (2026-10-07)

### 1. What we built
We reused Part 1's `transcribe()` for meetings and gave it one new ability: a **hint**, a short text with the
names the business already knows (team, customers, streets) that Whisper treats as "what was said just before".
Then we measured four setups on the five meetings. The hint took correct names from 71 % to 98 % with `small`,
at almost no extra cost, so **`small` + hint** is the Part 2 default.

### 2. Where it fits in the pipeline
```
 meeting audio ─► [1. TRANSCRIBE + name hint] ─► 2. extract items ─► 3. sync tracker (agent)
                   <-- you are here
                   ffmpeg → 16 kHz → Whisper small, initial_prompt = context.whisper_hint()
                   cache: transcripts/<hash>_<model>_h<hint hash>.json
```

### 3. How it works, step by step
1. `02-meeting-action-agent/context.py`: `TEAM`, `CUSTOMERS`, `PLACES` and `whisper_hint()` →
   "Brightwater Plumbing & Heating weekly team meeting with Sam, Priya, Tom and Jamie. Customers: …".
   Proper nouns only: task words such as "quote" are what we score, so hinting them would be cheating.
2. `shared/transcribe.py` → `transcribe(path, cache_dir, model, hint)`: the hint goes to faster-whisper as
   `initial_prompt`, and into the cache file name as an 8-character fingerprint (`_h3c46c32f`). No hint = the
   old file name, so Part 1's caches still work; an empty hint counts as no hint.
3. `02-meeting-action-agent/compare_transcription.py` → `run_config()` for each of `small`, `small+hint`,
   `large-v3-turbo`, `large-v3-turbo+hint`, against the scripts (TTS spellings mapped back, "Preeya" → Priya):
   - `word_errors()` (new in `shared/evaluation.py`): word-level edit distance → WER;
   - `name_hits()`: each name occurrence said vs heard (word boundaries, so "Tom" ≠ "tomato");
   - `recoverable()`: is each mentioned task still recognisable (`answer_key.matches`) in the transcript?
4. `build_report()` → `docs/part2-transcription-comparison.md` (table, caveat, every transcript).

### Results
| config | WER | names | task words recoverable | real-time factor |
|---|---|---|---|---|
| small | 8.5 % | 32/45 (71 %) | 25/28 | 1.11 |
| **small + hint (default)** | **5.3 %** | **44/45 (98 %)** | 26/28 | 1.27 |
| large-v3-turbo | 3.8 % | 43/45 | 27/28 | 2.55 |
| large-v3-turbo + hint | 2.9 % | 45/45 | 27/28 | 2.51 |

Still lost with the default: "Gallagher **quote**" → "court"/"call" (m2, every model) and "bathroom **suite**" →
"buff from Sweet" (m4). Phase 3 has to recover those from context.

### Independent review (separate reviewer agent)
Reproduced every number; checked `word_errors` on 9 hand cases, name counting, cache-key compatibility.
Fixed: empty hint sharing the no-hint cache; duplicated matching code; docstring + a `--hint` CLI option.
Findings: (1) the hint lists exactly the test names, so 98 % is a best case; a realistic 17-name list gave 93 %
and once swapped in a look-alike ("Ellie"). (2) `initial_prompt` fades after ~220 tokens, so on long recordings
`hotwords=` would be needed. (3) Beam 5 vs greedy: greedy 33 % faster but WER 3.9 % → 7.1 %, keep beam 5.
(4) VAD trims nothing here. (5) Give the LLM one segment per line with timestamps in Phase 3.

### 4. Key concepts I should understand
- **Context helps recognition:** speech recognition guesses words from sound *and* from what seems likely. Telling
  it the names makes them likely. Example: "Chavang Gallagher" → "Siobhan Gallagher" with the hint.
- **Don't hint what you score:** if "quote" were in the hint, "task words recoverable" would go up because of the
  hint, not because the system got better at listening. Example: we left "court" for "quote" as an honest miss.
- **Best case vs realistic case:** a test that knows the answers in advance flatters you. Example: 98 % with the
  exact names, 93 % with a realistic customer list.
- **Cache keys must include every input that changes the output:** model *and* hint. Example: the same audio has
  four cached transcripts, one per config, and none can be confused with another.
- **WER is a summary, not the goal:** "7th" vs "seventh" counts as an error but harms nothing; "court" vs "quote"
  is one word but breaks a task. That's why we also measure names and task words.

### 5. Files created or changed
- `shared/transcribe.py`: `hint` parameter (initial_prompt), hint in cache key, `--hint` CLI option.
- `shared/schemas.py`: `Transcript.hint`. `shared/evaluation.py`: `word_errors()`.
- `02-meeting-action-agent/context.py`: business knowledge + `whisper_hint()`.
- `02-meeting-action-agent/compare_transcription.py` → `docs/part2-transcription-comparison.md`.
- `PROJECT_RULES.md`: Phase 2 status.

### 6. Try it yourself
```powershell
cd C:\Users\danuk\code\audio-to-action
.\.venv\Scripts\Activate.ps1
python 02-meeting-action-agent\compare_transcription.py
python -m shared.transcribe 02-meeting-action-agent\testset\audio\m1_2026-09-07.wav --cache-dir 02-meeting-action-agent\testset\transcripts
```
Expected: the four result lines above in ~2 s (cached), then the m1 transcript *without* hint, with
"Chavang Gallagher" and "boiler milling". Add `--hint "Siobhan Gallagher, Mill Lane"` to see a new (slow) run
with those names fixed.

### 7. What can go wrong
- **Look-alike names in the hint:** "Mill Road" in the list can turn "Mill Lane" into the wrong street.
- **Long recordings:** the hint fades after ~220 tokens; consider `hotwords=` then.
- **Words outside the hint:** task words ("quote", "suite") get no help; extraction must cope.
- **Noisy timings:** single runs; differences of ~15 % in speed are noise.

### 8. Check my understanding
1. Why is the hint allowed to contain "Siobhan Gallagher" but not "quote"?
2. Why does the cache file name contain a fingerprint of the hint?
3. small+hint has a lower WER than large-v3-turbo without hint on names, but a higher WER overall. Which matters more for this project, and why?

### 9. Next phase preview
Phase 3 extracts action items from each transcript: a new schema (task, owner, due as said, status, evidence),
the Part 1 "fill the form, validate, retry once" pattern generalised to any schema, owners checked against the
team, due dates resolved by plain code from the words ("by Wednesday" + meeting date), and chunking for long meetings.

---

## Part 2, Phase 3: Extraction of action items (2026-10-07)

### 1. What we built
The step that reads one meeting's transcript and writes down every action item: what the work is, who owns it,
when it's due, and whether it's open, done or cancelled, plus the exact quote it came from. It's a workflow step:
one meeting in, one list out, no memory of earlier meetings (that's the agent's job in Phase 4). Along the way we
turned Part 1's "fill the form, validate, retry once" into a shared helper that both parts now use.

### 2. Where it fits in the pipeline
```
 meeting audio ─► 1. transcribe ─► [2. EXTRACT] ─────────────────────────────► 3. sync tracker (agent)
                  (small + hint)    <-- you are here
                  segments ─► [mm:ss] lines ─► LLM fills MeetingItems ─► validators (team, quote in transcript)
                            ─► merge repeats ─► ground_owner() ─► resolve_due() ─► MeetingResult JSON
```

### 3. How it works, step by step
`02-meeting-action-agent/extract.py` → `extract(transcript, meeting_date, cache_dir)`:
1. `cache_path()`: `<audio hash>_<whisper model>_h<hint>_<llm>_<prompt version>.json`; hit → return.
2. `chunk_segments()`: pieces of ≤ 1200 words on segment boundaries, overlapping by one segment (our meetings are 1 chunk).
3. For each chunk, `shared/llm.structured_chat(MeetingItems, messages, context=...)`:
   - messages: `SYSTEM_PROMPT` (business, team, what is / isn't an item, field rules, a worked example with
     invented jobs) + "Meeting date: Monday 2026-09-14" + the transcript as `[mm:ss] text` lines (`format_transcript()`).
   - Ollama `format=<schema>` forces `{"jobs_mentioned": [...], "items": [...]}`; `jobs_mentioned` comes first as a
     "think first" list.
   - Validators (`shared/schemas.py`, via the validation `context`): owner must be in `context["team"]` or null;
     `due_text` "null" → None; at least 60 % of the evidence words must be in the transcript (invented items fail).
   - Invalid → one retry with the errors; still invalid → `LLMFormError`.
4. `finish_items()`: `merge_items()` (exact repeats anywhere, similar items only across chunks), `ground_owner()`
   (owner kept only if their name is in the quote or the 2 lines before), `resolve_due()` (`dates.py`, plain code:
   "by Wednesday" + meeting date → 2026-09-09; done/cancelled items get no due date).
5. Save `MeetingResult` with the items, the model's raw items (`llm_items`), job list, attempts and timing.

`extract_testset.py` scores it against `labels.json` using `answer_key.match_items()` (closest one-to-one pairs).

### Results (prompt x7, Whisper small + hint, CPU)
| | task only | task + owner (DoD) |
|---|---|---|
| precision | 21/25 (84 %) | 16/25 (64 %) |
| recall | 21/28 (75 %) | 16/28 (57 %) |

On matched items: owner 84 % where the words name someone, due 95 %, status 95 %. ~235 s per meeting.
DoD (precision and recall ≥ 80 %, task + owner) **not met**.

How we got there (each version re-scored on the same 5 meetings):
x1 rules only (recall 71 %) → x2 worked example + owner grounding (75 %) → x3 "jobs_mentioned" first (75 %, owners
better) → x4 done items get no due date → x5 a "every job needs an item" check + retry (tried and removed: the model
rewrote its job list instead of adding items) → x6 **code bug fixed**: `merge_items` was merging similar items
*within* one chunk ("order the suite" + "fit the suite"), which we had been blaming on the model → x7 removed an
example copied from a test transcript ("the Gallagher court" = quote): a test-set leak.

### Independent reviews (separate reviewer agent)
Found the merge bug (high), an unfair T12 match rule, a `[mm:ss]` false-reject in the evidence check, date edge
cases ("I **may** do it by the 25th" → May 2027; "for **now**" → today), "Tom's" not counting as Tom; pointed out that
the headline was task-only (the DoD needs task + owner) and that the prompt leaked a test example. All fixed.

### 4. Key concepts I should understand
- **Validation context:** a validator can check facts from outside the answer. Example: `owner_on_team()` reads
  `context["team"]`, so "Siobhan" (a customer) is rejected as an owner.
- **Ground the model's claims in the input:** every item must quote the transcript, and an owner must be *named*
  nearby. Example: three invented "Sam" owners were dropped by `ground_owner()`, never a correct one.
- **Do deterministic work in code:** the LLM copies "by Wednesday"; `resolve_due()` does the calendar. Example: 30 date
  tests, including "Monday" said on a Monday and "the 31st" in September.
- **Look for bugs in your own code before blaming the model:** the "merged jobs" we tried to fix with three prompt
  versions were partly made by `merge_items()`. Saving the raw model output (`llm_items`) makes this visible.
- **Test-set leakage:** a prompt example taken from a test meeting makes scores look better than they are. Example:
  "the Gallagher court" in the prompt; removed in x7.

### 5. Files created or changed
- `shared/llm.py`: `structured_chat()`, `short_errors()`, `LLMFormError`.
- `shared/analyze.py`: Part 1 now uses `structured_chat()` (same messages, cache, errors).
- `shared/schemas.py`: `ActionItem`, `MeetingItems`, `ExtractedItem`, `MeetingResult` + validators.
- `02-meeting-action-agent/extract.py`, `dates.py`, `extract_testset.py`.
- `02-meeting-action-agent/tests/test_dates.py`, `tests/test_extract.py`.
- `02-meeting-action-agent/testset/answer_key.py` (`match_items`, `overlap`), `labels.json` (T9, T12 match rules).
- `PROJECT_RULES.md`: Phase 3 status.

### 6. Try it yourself
```powershell
cd C:\Users\danuk\code\audio-to-action
.\.venv\Scripts\Activate.ps1
python -m pytest 02-meeting-action-agent\tests -q
python 02-meeting-action-agent\extract_testset.py
python 02-meeting-action-agent\extract.py 02-meeting-action-agent\testset\audio\m5_2026-10-05.ogg 2026-10-05
```
Expected: `41 passed`; then the per-meeting table ending in `precision 21/25 (84%) recall 21/28 (75%)` (cached,
seconds); then m5's transcript and its 6 items, e.g. `[done] Order the Gallagher bathroom suite` and
`[open] Fit the Gallagher bathroom suite | owner=Tom due=2026-10-12`.

### 7. What can go wrong
- **Run-to-run variance:** m3 found 5 items with x6's prompt and 3 with x7's after a one-line change. With 28
  mentions, a few points either way is noise.
- **Owners from the voice:** "I'll handle that" stays unowned (by design): caps task + owner recall at 25/28.
- **Misheard task words:** "Gallagher court invoice", "buff from Sweet" can't be matched to the right task.
- **Slow:** ~4 min per meeting on CPU; prompt experiments take 20 min each.
- **Long meetings:** chunking + merging is only unit-tested; real long recordings need a live check.

### 8. Check my understanding
1. Why does `resolve_due()` exist instead of asking the LLM for an ISO date?
2. `ground_owner()` sets an owner to null even when the LLM gave a name. When is that the right call, and what does it cost?
3. Why was copying "the Gallagher court = quote" into the prompt a problem even though it's a real mishearing?

### 9. Next phase preview
Phase 4 builds the agent: a SQLite task tracker and a small set of tools (list/search tasks, add, update, close,
finish) that the LLM calls in a loop to relate each meeting's items to existing tasks, with a step limit, validated
tool arguments, a reason stored with every change, and a saved trace of every step.

---

## Part 2, Phase 4: The sync agent (2026-10-07/08)

### 1. What we built
The one part of the project where the model decides what to do next. After each meeting, an agent looks at
the items extracted from it and at the task tracker (a small SQLite database), and for every item chooses:
update an existing task, add a new one, or skip it (not real work / a repeat). It does this by calling tools
in a loop until it says it's finished. Code boxes it in: it can only choose *which* task an item belongs to;
the owner, date and status are copied from the item by code, and every choice passes plain-code checks.

### 2. Where it fits in the pipeline
```
 meeting audio ─► 1. transcribe ─► 2. extract items ─► [3. SYNC AGENT] ─► tracker.db (+ trace JSON)
                                                        <-- you are here
   each step:  state message (items + what's done, tracker tasks, feedback) ─► qwen2.5:7b ─► tool calls
               ─► run_tool(): guards ─► tracker.add_task / update_task (with reason) ─► next state ...
               until finish (refused while items are unhandled) / step limit / no progress
```

### 3. How it works, step by step
`02-meeting-action-agent/agent.py` → `sync_meeting(conn, meeting, date, items)`:
1. Items get letters (A, B, C...): with "I5" the model linked item I5 to task #5 because the numbers matched.
2. Each step builds a fresh `state_message()`: every item with "TO DO" or "HANDLED (update #3)", the open tasks
   (plus tasks changed today), the results of the last tool calls, and what is still to do. No chat history.
3. Ollama `chat(..., tools=tool_schemas())`: the tool list is generated from Pydantic models (`AddArgs`,
   `UpdateArgs`, `SkipArgs`, ...), so the JSON schema the model sees is the same one that validates its calls.
4. `run_tool()` checks every call before anything changes:
   unknown tool / bad arguments (with advice, e.g. "update_task needs task_id... if new work use add_task");
   item handled already; `update_task` only to a task with enough words in common (`similarity >= 0.2`) and only
   one item per task per meeting; `add_task` refuses a near-duplicate of an open task unless `confirm_new`;
   `skip_item` must say `not_work` or `repeat_of_item` (which must mostly match the other item), and a done item
   is never "not work"; `finish` is refused while items are left; blank reasons are refused.
5. `tracker.py` applies the change with `COALESCE` (a null never wipes a known owner or date) and logs before/after
   + reason in `changes` (`undo_last()` reverses it).
6. Stops on finish, on the step limit (2 × items + 4), or when a step changes nothing (no progress); anything left
   goes to the review list in `traces_<mode>/<meeting>.trace.json`. A failed model reply costs one step, not the run.

`sync_testset.py --items gold` feeds the agent the answer key's items (so extraction errors can't interfere) and
scores the tracker after every meeting against `answer_key.tracker_states()`.

### How the design evolved (each version run on the gold series)
| version | change | result |
|---|---|---|
| v1 | chat history, tools as planned | m1: 0/5, 14 calls: replayed failing calls 10×, then **guessed task ids** to get past the errors |
| v2 | state message instead of history; helpful error texts; relevance check | m1 5/5 (1 call), m2 6/7 |
| v3 | `task_id` optional (for a nicer error) | the model **omitted it** and wrote "task #1" in the reason: the schema is part of the prompt |
| v4 | `task_id` required again; letters for items; typed `skip_item` | m2 7/7 in 1 call |
| v5 | one item per task per meeting; a repeat must share words | m3 8/9; series 11/14 |
| v6 | a repeat must mostly match; done news is never "not work" | **12/14 (86 %)**, 25 calls, ~47 min; m3 stuck on an identical state 10× |
| v6 + review | no-progress stop, blank reasons refused, readability refactor | same behaviour, fewer wasted calls |

### Independent review (separate reviewer agent)
Found the temperature-0 fixed point (same state → same refused answer, 10 times), blank reasons accepted, `undo_last`
not restoring `updated_in`; refactored `run_tool` into small functions (3000 random call sequences: identical
behaviour). Its key finding: **a plain-code sync with no LLM** (match each item to the most similar open task, use the
item's status, skip repeats) scores **14/14 on gold**, better than the agent, instantly and for free. On real extracted
items both are capped by extraction (perfect links would give only 7/14; corrected in Phase 6: linking by hand gives 9/14). Phase 6 compares them properly.

### 4. Key concepts I should understand
- **Agent = model chooses the next action in a loop.** Example: in m4 it called update_task five times and add_task
  once, in an order it picked itself; the workflow parts of the project always run the same steps in the same order.
- **Bound the agent with code, not with prompts.** Example: "item C is NOT task #3" in the prompt didn't stop the
  I5→#5 mistake; the relevance check (`similarity < 0.2` → refused) did.
- **Errors must say what to do instead.** Example: "Input should be a valid integer" was repeated forever;
  "update_task needs task_id... if new work use add_task" was followed on the next step.
- **Models satisfy errors the cheapest way:** guess an id, omit a field, skip an item, rewrite a list. Every guard
  needs a check that the "fix" is real (Part 1 Phase 3 and Part 2 Phase 3 showed the same pattern).
- **An agent isn't automatically better.** Example: for these items, about 20 lines of matching rules beat the agent.
  The agent only earns its cost where rules can't decide (fuzzy wording, merged items, history lookups).

### 5. Files created or changed
- `02-meeting-action-agent/tracker.py`: tasks + changes tables, search, similarity, add/update/undo.
- `02-meeting-action-agent/agent.py`: tools, guards, state message, loop, trace.
- `02-meeting-action-agent/sync_testset.py`: series runner + tracker scoring (gold / extracted).
- `02-meeting-action-agent/tests/test_agent.py`: 16 guard tests (no LLM).
- `.gitignore`: `testset/traces_*/`. `PROJECT_RULES.md`: Phase 4 status.

### 6. Try it yourself
```powershell
cd C:\Users\danuk\code\audio-to-action
.\.venv\Scripts\Activate.ps1
python -m pytest 02-meeting-action-agent\tests -q
python 02-meeting-action-agent\sync_testset.py --items gold --only m1
type 02-meeting-action-agent\testset\traces_gold\m1_2026-09-07.trace.json
```
Expected: `68 passed` (with Part 1: run both test folders); then m1 with 5 `add` actions and `tracker: 5/5 tasks right`
after ~3 min (one LLM call); the trace shows each tool call, its arguments, the agent's reason and the result.

### 7. What can go wrong
- **Slow and costly:** 1-16 LLM calls per meeting, ~2 min each on CPU. The step limit is also a time budget.
- **The model ignores advice:** it plans all calls at once from the item list; refused calls are repeated.
- **Plausible wrong links pass the guards:** "order the valve" vs "fit the valve" share 57 % of their words.
- **Garbled task text sticks:** an update never changes a task's description, so "Gallagher court invoice" stays.
- **Gold is easy mode:** answer-key wording makes every true update a 100 % word match.

### 8. Check my understanding
1. Why does each step get a fresh state message instead of the conversation so far? What new problem did that create?
2. The agent can't set owners or dates itself. Why is that a good design, and what does it cost?
3. If simple rules score 14/14 on gold, when would an agent still be worth it?

### 9. Next phase preview
Phase 5 glues Part 2 into `run.py` like Part 1: watch `inbox/`, transcribe with the name hint, extract, sync (agent or
rules), move files, skip meetings already processed, retry Ollama hiccups, failed files to `failed/`, logs.

---

## Part 2, Phase 5: Glue & reliability (2026-10-08)

### 1. What we built
`02-meeting-action-agent/run.py`: drop meeting recordings into `inbox/` and each one is transcribed (with the name
hint), its action items extracted, and the tracker synced by the agent, oldest meeting first. A recording already
processed is skipped. If anything fails, the file goes to `failed/` with a note, and, new compared with Part 1,
**the tracker is rolled back** so a meeting is never half-applied.

### 2. Where it fits in the pipeline
```
 inbox/ ─► sort by meeting date ─► hash in tracker.meetings? ─yes─► processed/ (nothing else)
                                       │no
                                       ▼
          transcribe (+hint) ─► extract (retries) ─► sync_with_rollback(agent) ─► record_meeting ─► processed/
                                                       │ undo leftovers first;            │
                                                       │ any error/Ctrl+C: undo meeting   │
                                                       └──────────── failed/ + .error.txt ◄┘
```

### 3. How it works, step by step
`run.py` → `run_once()` sorts ready files by `(date_for_sorting, name)`: meeting 3 refers to tasks from 1 and 2.
`process_file()`:
1. `file_sha256()` → `tracker.meeting_done()`: already processed → move, done.
2. `meeting_id()`: the file name without extension, or `name_<hash8>` if another recording already used it.
   `meeting_date()`: `YYYY-MM-DD` in the name, else the file's modified date (impossible dates → that file fails).
3. `transcribe(..., hint=context.whisper_hint())` → `with_retries(extract, ...)` (both cached).
4. `with_retries(sync_with_rollback, agent_sync, ...)`: `start_clean()` first undoes changes left by an interrupted run
   of this meeting; then the agent runs; on *any* exception (Ctrl+C too) `tracker.undo_meeting()` reverses this
   meeting's changes from the change log, newest first, and the error is re-raised (temporary ones are retried).
5. `tracker.record_meeting()` (items, sync mode, items left for review) → move to `processed/`.

`shared/pipeline.py`: `find_ready_files`, `move_to` (a plain rename: either it moves or it fails; no copies),
`write_error_note`, `retry_failed`, `setup_logging` (safe to call twice).

### Results (end-to-end in a scratch folder, agent on real extracted items)
| scenario | outcome |
|---|---|
| m2 copied in before m1, + broken file | m1 synced first (date in name), broken → `failed/`, m2 next |
| m1 again under another name | skipped: same audio hash |
| Ollama down during m3's sync | 2 retries with rollback (0 changes) → `failed/`, tracker untouched |
| `--retry-failed` | m3 synced (1 LLM call); broken file failed again |
| (review) crash after sync, before recording | rerun undid 3 leftover changes before syncing again |

### Independent review (separate reviewer agent)
Found and fixed: a crash or Ctrl+C between the sync and `record_meeting` made the next run apply the meeting **twice**;
two recordings named `standup.m4a` shared one id, so a failed week 2 rolled back week 1; an impossible date in a name
crashed the whole run; on Windows, moving a file another program has open copied it into two folders. 6 new tests.

### 4. Key concepts I should understand
- **All or nothing across many small writes:** each tracker change commits on its own, so "the meeting" is not one
  database transaction. The change log makes it one anyway: undo everything this meeting did. Example: Ollama
  died in the middle of m3 → `undo_meeting("m3...")` → tracker exactly as before.
- **Idempotency needs a clean restart, not just a skip:** "already processed?" is only answered *after* the sync.
  A crash in between leaves changes without a record, so a rerun must first clear them (`start_clean()`).
- **Ctrl+C is an exception too** (`KeyboardInterrupt`, a `BaseException`): cleanup code must catch it as well.
- **Order is part of correctness:** tracker state depends on meeting order. Example: the older m1 must be synced
  before m2, whatever order the files arrive in. A late older meeting is logged as a warning.
- **Identity:** a file name isn't an identity (weekly `standup.m4a`); the audio hash is.

### 5. Files created or changed
- `02-meeting-action-agent/run.py`: the meeting pipeline.
- `02-meeting-action-agent/tracker.py`: `meetings` table, `meeting_done`, `record_meeting`, `undo_meeting`, helpers.
- `shared/pipeline.py`: reusable inbox helpers.
- `02-meeting-action-agent/tests/test_run.py`: 9 tests (dates, rollback, Ctrl+C, interrupted run, same name, ...).
- `.gitignore`: `02-meeting-action-agent/traces/`. `PROJECT_RULES.md`: status.

### 6. Try it yourself
```powershell
cd C:\Users\danuk\code\audio-to-action
.\.venv\Scripts\Activate.ps1
python -m pytest 02-meeting-action-agent\tests -q
copy 02-meeting-action-agent\testset\audio\m1_2026-09-07.wav 02-meeting-action-agent\inbox\
python 02-meeting-action-agent\run.py
python 02-meeting-action-agent\run.py
```
Expected: all tests pass; the first run logs transcribe / extract / `agent: N LLM calls` / `2026-09-07: 6 items,
6 tracker changes` (several minutes on CPU, the first time also ~1 min Whisper + ~4 min extraction); the second run
`0 processed` (inbox empty). Delete `02-meeting-action-agent\tracker.db` afterwards for a clean start.

### 7. What can go wrong
- **A late, older meeting** can overwrite newer news (e.g. reopen a finished task): warned, not prevented.
- **Two copies of run.py at once** could interleave changes: run one at a time.
- **The same audio re-encoded** has a new hash and is processed again.
- **Slow:** a new meeting is ~1 min Whisper + ~4 min extraction + 2-15 min agent on CPU.

### 8. Check my understanding
1. Why isn't "skip if already in the meetings table" enough to make reruns safe? What does `start_clean()` add?
2. Why must the rollback also catch `KeyboardInterrupt`?
3. Two recordings are both called `standup.m4a`. What goes wrong if the file name is the meeting id?

### 9. Next phase preview
Phase 6 evaluates Part 2 end to end and runs the experiment the reviews pointed to: the same input items synced by
the agent and by plain-code rules (no LLM), on perfect items and on real extracted items; plus extraction
precision/recall and the definition of done.

---

## Part 2, Phase 6: Evaluation: agent vs rules (2026-10-08)

### 1. What we built
A measuring setup for the whole meeting pipeline and the experiment Part 2 was really about: **is the agent worth it?**
We wrote a plain-code alternative for the tracker sync (`rules_sync.py`, no LLM: match each item to the most similar
open task, use the item's status) and ran both methods on exactly the same items: the answer key's perfect items and
the real extracted items. One report (`docs/part2-eval-results.md`) collects extraction scores, the four sync runs and
the conclusions.

### 2. Where it fits in the pipeline
```
 meeting audio ─► transcribe ─► extract ─► sync ─► tracker.db
                                   │        ├── agent (LLM, tools, guards)    ┐ same items, same order,
                                   │        └── rules (plain code, no LLM)    ┘ same scoring
                                   ▼                         ▼
             ┌──────────────────────────────────────────────────────────────────┐
             │ [Phase 6: EVALUATE] extraction P/R + tracker after every meeting │ <-- you are here
             └──────────────────────────────────────────────────────────────────┘
```

### 3. How it works, step by step
1. `rules_sync(conn, meeting, day, items)`: per item: repeat of an earlier item today (same status, similarity ≥ 0.5)
   → skip; done/cancelled → update the most similar open task if ≥ 0.2; open → update if ≥ 0.6; a cancelled item
   with no match → skip (a postponed idea); otherwise add. Same signature as the agent, so `run.py --sync rules` works.
2. `sync_testset.py --items gold|extracted --sync agent|rules`: empty tracker, five meetings in order, score after
   each (`score_tracker`), save a summary to `testset/sync_runs/<items>_<sync>.json`.
3. `evaluate.py`: extraction metrics from the cached x7 results, the sync table from the saved summaries, then the
   hand-written `docs/part2-eval-notes.md` (experiment, pre-registered rule, result, definition of done).

### Results
| sync | gold items | **extracted items (decides)** | LLM calls | time |
|---|---|---|---|---|
| agent | 11/14 | **7/14** | 51 | ~94 min |
| rules | **14/14** | 6/14 | 0 | < 1 s |

Pre-registered rule: rules become the default only if they do at least as well on extracted items → **agent stays**.
But: +1 task is within the agent's run-to-run noise (12/14 vs 11/14 on identical gold input), both methods were tuned
on this test set, and a perfect sync of the extracted items would reach only **9/14**: extraction is the bigger limit.
Definition of done: extraction and tracker ≥ 80 % **not met**; guardrails and idempotency **met**.

### Independent review (separate reviewer agent)
Reproduced every number, then corrected my conclusions: "the agent reached the 7/14 ceiling" was false (hand-linking
gives 9/14); "the agent skips items" wasn't what the traces show (it got stuck on refused updates and made one wrong
link); both methods were tuned on this data; a `--only m1` re-run had overwritten evidence of the full run. It also
caught that my decision rule was written after the (instant) rules results were known: now stated in the notes.

### 4. Key concepts I should understand
- **Always build the simple baseline:** without `rules_sync`, "11/14 on gold" would have looked fine. Next to 14/14 for
  about 20 lines of code, it's a warning.
- **Decide on the real pipeline, not the ideal input:** gold items test the method; extracted items test the product.
  Example: rules win on gold (14 vs 11), the agent wins on extracted (7 vs 6).
- **Cost is a metric:** 94 minutes and 51 LLM calls vs under a second. +1 task has to be worth that.
- **Noise and tuning limit what you can claim:** same input, same code (nearly), 12/14 one day and 11/14 the next.
  A 1-task difference on 14 tasks isn't evidence.
- **Find the real bottleneck:** even a perfect sync gets 9/14 from these extracted items. Improving the agent further
  matters less than improving extraction (or transcription).

### 5. Files created or changed
- `02-meeting-action-agent/rules_sync.py` (+ `tests/test_rules_sync.py`): the non-agent sync.
- `02-meeting-action-agent/sync_testset.py`: `--sync`, saved summaries in `testset/sync_runs/` (committed).
- `02-meeting-action-agent/evaluate.py` → `docs/part2-eval-results.md`; `docs/part2-eval-notes.md` (conclusions).
- `02-meeting-action-agent/run.py`: `--sync rules`. `agent.py`: not_work allowed for cancelled ideas with no similar task.
- `PROJECT_RULES.md`: status.

### 6. Try it yourself
```powershell
cd C:\Users\danuk\code\audio-to-action
.\.venv\Scripts\Activate.ps1
python 02-meeting-action-agent\sync_testset.py --items gold --sync rules
python 02-meeting-action-agent\evaluate.py
type docs\part2-eval-results.md
```
Expected: the rules run ends `tracker: 14/14 tasks right` in about a second; `evaluate.py` prints the extraction table
(84 % / 75 % task only) and the four sync rows; the report ends with the conclusions and the definition of done.

### 7. What can go wrong
- **Over-reading small differences:** 14 tasks, 28 mentions; a run-to-run swing of ±1 task is normal for the agent.
- **Overwriting evidence:** `sync_testset.py --only` reuses the same database and traces as a full run.
- **Stale numbers:** the saved agent runs predate the last agent fix; re-run (~95 min) before comparing again.
- **Tuning to the test:** guards and thresholds were chosen while looking at these meetings.

### 8. Check my understanding
1. Why does the decision use the extracted items and not the gold items?
2. The agent scored 7/14 and the rules 6/14. Why is that not enough to say "the agent is better"?
3. If a perfect sync only reaches 9/14, where would you spend the next week of work?

### 9. Next phase preview
Phase 7 polishes Part 2: README and handover updated with the meeting pipeline, the Windows file-move fix from Phase 5
applied to Part 1 too, corrected numbers in this log, and a final summary of Part 2.

---

## Part 2, Phase 7: Polish (2026-10-08)

### 1. What we built
Nothing new for the pipeline; everything around it made accurate and complete: Part 2's README rewritten from "the
plan" to "what was built and measured", the main README and the handover report (`docs/HANDOVER.md` §12) extended with
Part 2, the Windows file-move bug found in Part 2 fixed in Part 1 too, an overclaimed number in this log corrected,
and the Part 3 starting choices recorded in `PROJECT_RULES.md`.

### 2. Where it fits in the pipeline
```
 Part 1: voicemail ─► transcribe ─► analyze ─► route ─► deliver/store          (workflow)
 Part 2: meeting   ─► transcribe+hint ─► extract ─► sync (agent | rules) ─► tracker   (workflow + one agent)
 [Phase 7: docs + fixes across both]  <-- you are here
```

### 3. How it works, step by step
- `01-voicemail-triage/run.py` → `move_to()` now uses `path.rename()` (moves or fails, never copies), and a failed
  move to `failed/` leaves the file in the inbox instead of stopping the run (same fix as Part 2 Phase 5).
- `docs/HANDOVER.md` §12: Part 2's goal, decisions, files, data flow, every module's contract (tools, guards and
  thresholds of the agent, tracker tables, cache keys), test set, results, decision log, gotchas, next steps.
- `02-meeting-action-agent/README.md`, `README.md`, `docs/shared-for-part2.md`, `PROJECT_RULES.md`: current status.

### 4. Key concepts I should understand
- **A fix found in one part belongs in all parts:** the copy-instead-of-move bug existed in Part 1 too.
- **Docs drift unless they're rewritten at the end:** the Part 2 README described a plan with "mark duplicate" and
  "list_open_tasks" tools; the built agent has neither. The handover describes the code as it is.
- **Record decisions with their reasons:** the next person (or agent) needs "why rules aren't the default" more than
  the code itself.

### 5. Files created or changed
- `01-voicemail-triage/run.py`: rename-based move, safe failure path.
- `02-meeting-action-agent/README.md`: rewritten. `README.md`: Part 2 summary, tests, docs, status.
- `docs/HANDOVER.md`: §12 Part 2 + small updates. `docs/shared-for-part2.md`: "what actually happened" note.
- `docs/learning-log.md`: corrected the 7/14 → 9/14 claim (Phase 4 entry). `PROJECT_RULES.md`: status + Part 3 choices.

### 6. Try it yourself
```powershell
cd C:\Users\danuk\code\audio-to-action
.\.venv\Scripts\Activate.ps1
python -m pytest 01-voicemail-triage\tests 02-meeting-action-agent\tests -q
python 02-meeting-action-agent\evaluate.py
```
Expected: `81 passed`; the Part 2 report regenerated from saved runs in a few seconds (no LLM calls).

### 7. What can go wrong
- **Saved agent runs are from before the last agent fix:** re-run (~95 min) before quoting agent numbers again.
- **Docs and code drift again** if later changes skip the handover.

### 8. Check my understanding
1. Which parts of Part 2 are workflow, and why is only the sync allowed to be an agent?
2. If you had to ship Part 2 tomorrow on this laptop, would you use `--sync agent` or `--sync rules`? Why?
3. What would you fix first to reach the 80 % definition of done?

### 9. Next phase preview
Part 2 is complete. Part 3 (real-time phone receptionist) starts with a Phase 0 plan based on your early choices:
local simulation and "take a message" into the Part 1 pipeline.

---

## Part 2 summary
| Step | Code | Workflow or agent? | Why |
|---|---|---|---|
| Transcribe (+ name hint) | `shared/transcribe.py`, `context.py` | workflow | always the same conversion |
| Extract action items | `extract.py`, `shared/llm.py` | workflow step using an LLM | one meeting in, one validated form out; code checks owners, quotes, dates |
| Sync the tracker | `agent.py` (or `rules_sync.py`) | **agent** (or rules) | relating items to existing tasks needs lookups and judgement... and the experiment showed plain rules come close |
| Glue, rollback, idempotency | `run.py`, `tracker.py` | workflow | fixed order, all-or-nothing per meeting |

What Part 2 taught, in one line each:
- An agent must be **boxed in by code**: tools that only choose relationships, guards with helpful errors, step limits.
- Models **satisfy errors the cheapest way** (guess, delete, skip, rewrite): check that every "fix" is real.
- **Always build the simple baseline**: about 20 lines of rules scored 14/14 on perfect input where the agent scored 11/14.
- **Decide on the real pipeline**, and read the cost (51 calls, ~94 min) next to the gain (+1 task).
- **Independent review pays**: every Part 2 phase had a real bug or overclaim that the builder missed.

## Part 3, Phase 0: Plan, persona and FAQ (2026-10-08)

### 1. What we built
The groundwork for Part 3, the phone receptionist. We wrote down the plan and every decision, gave the receptionist a
name and a fixed set of sentences ("Holly", the automated assistant of Brightwater Plumbing & Heating), and wrote the
only facts she may state (a 23-entry FAQ). Nothing talks yet: this phase is data, rules and folders.

### 2. Where it fits in the pipeline
```
 Part 1: voicemail ─► transcribe ─► analyze ─► route ─► deliver/store
 Part 2: meeting   ─► transcribe+hint ─► extract ─► sync (agent | rules) ─► tracker
 Part 3: caller ─► listen ─► understand ─► decide ─► speak ─► ... ─► hand off to Part 1
 [Phase 0: persona.json + faq.json + plan + folders]  <-- you are here (nothing is wired together yet)
```

### 3. How it works, step by step
- `docs/PART3_PLAN.md`: the owner's brief, committed. `03-phone-receptionist/README.md`: what is built, the decisions taken
  without the owner, and the A-vs-B decision rule written before any model run.
- `persona.json`: Holly's 22 fixed sentences, the detail order (reason, name, number), the limits (12 turns, 2 re-asks, 2 silent
  turns), the Whisper hint (team and places, no customer names) and the voice ids that are taken. `persona.py` loads it and checks
  it: every line present, only known placeholders, no digits (the text is read aloud). `speak_number()` reads digits in groups.
- `faq.json`: the facts Holly may give, spoken form. `faq.py` loads and checks it. Three entries are real UK safety advice.
- Folders `calls/`, `logs/`, `testset/` with `.gitkeep`; `.gitignore` keeps call records, logs and test audio out of git.

### 4. Key concepts I should understand
- **Fixed sentences instead of generated ones:** a model that writes the reply can invent a price. A model that only *picks* which
  fixed sentence to say cannot. Example: "ninety-five pounds" exists in one place, `faq.json`.
- **A hint must not contain what you score:** the Whisper hint lists the team and places but no caller names; otherwise the name
  accuracy would measure the hint, not the transcription.
- **Write the decision rule before the experiment:** the A-vs-B rule (section 4 of the README) is fixed now, so the result cannot
  nudge the rule.
- **Autonomy needs a paper trail:** every choice the owner would normally make is in the "Decisions taken without the owner" table.

### 5. Files created or changed
- `docs/PART3_PLAN.md`, `03-phone-receptionist/README.md`: plan, decisions, rules.
- `03-phone-receptionist/persona.json`, `persona.py`, `faq.json`, `faq.py`, `tests/test_persona_data.py`: data, loaders, checks.
- `03-phone-receptionist/{calls,logs,testset}/.gitkeep`; `.gitignore`, `requirements.txt` (streamlit).
- `PROJECT_RULES.md` (Part 3 section), `README.md`, `docs/HANDOVER.md`: status lines.

### 6. Try it yourself
```powershell
python -m pytest 03-phone-receptionist\tests -q
python -c "import sys; sys.path.insert(0,'03-phone-receptionist'); from persona import *; p=load_persona(); print(p.say('greeting')); print(speak_number('01632960501'))"
```
Expected: the tests pass; Holly's greeting; `oh one six three two, nine six oh, five oh one`.

### 7. What can go wrong
- A persona line with a digit or a wrong placeholder: the loader refuses it at start-up, on purpose.
- The FAQ is fictional apart from the safety numbers: do not give the prices to anyone.
- The GPU check (`nvidia-smi`, `ollama ps`) could not be run in the cloud: do it first on the laptop.

### 8. Check my understanding
1. Why does Holly never let the model write her reply?
2. Why is the receptionist's voice id `null` in `persona.json`?
3. Why are customer names kept out of the Whisper hint?

### 9. Next phase preview
Phase 1 writes the synthetic callers: one "caller card" for each of the 18 Part 1 voicemails plus a few FAQ callers, with the
answer key, before any dialog code exists. A scripted simulator answers whatever Holly asks, from the card.

## Part 3, Phase 1: Synthetic callers (2026-10-08)

### 1. What we built
The answer key and the practice callers for the receptionist, before any dialog code exists. 24 "caller cards": the 18 Part 1
voicemails turned into phone calls, plus 6 callers who ask FAQ questions. A scripted caller (`simulate.py`) answers whatever the
receptionist asks, from its card, and can say it aloud in the card's voice with phone-line noise. A checker (`cards.py`) proves the
answer key matches Part 1's labels.

### 2. Where it fits in the pipeline
```
 Part 3: caller ─► listen ─► understand ─► decide ─► speak ─► ... ─► hand off to Part 1
 [Phase 1: the synthetic caller + answer key that every later phase is measured against]  <-- you are here
   card ─► SimulatedCaller.reply(question type) ─► text ─► Piper + noise + phone band ─► wav
```

### 3. How it works, step by step
- `testset/callers.json`: each card has facts (name, number, how to pronounce a hard name), quirks (refuses the number, spells the
  name, corrects the number, robocall, ...), the caller's opening words, an optional question and the expected outcome.
- `cards.load_cards()` reads the cards and **reads** the category/urgency/name/number labels of the 18 Part 1 scenarios from
  Part 1's `labels.json`. `check_cards()` finds every inconsistency (names, numbers, voices, FAQ ids, urgent vs category, ...).
- `asks.py`: the list of question types ("name", "number", "read-back", "anything else", ...). `SimulatedCaller.reply(asked)` answers by
  question type, never by reading the receptionist's sentence; the one exception is the read-back, where it checks the name and number.
- `spoken.py`: `digit_runs()` turns "oh seven seven double oh, nine hundred, one two three" or "01632 960 501" into digits;
  `letter_runs()` turns "S, I, O, B, H, A, N" into SIOBHAN. A correction ("four three, no sorry, three four nine") stays as separate runs.
- `simulate.render_turn()`: Piper voice (`shared/tts.py`), then Part 1's noise and phone-band filter, then a 16 kHz wav for Whisper.

### 4. Key concepts I should understand
- **Answer key first:** the cards and labels exist before the dialog, so a dialog bug can never move the goalposts.
- **Dev and score cards:** you may tune on 9 cards; 15 are only for the final score. Tuning on the scoring cards would measure memory.
- **One answer key, not two:** Part 1's labels are read, never copied, so the two parts cannot disagree.
- **A checker must be able to fail:** the tests break a card on purpose (wrong name, missing topic, taken voice) and expect a complaint.

### 5. Files created or changed
- `testset/callers.json`, `testset/README.md`: the 24 cards and their documentation.
- `cards.py`, `simulate.py`, `spoken.py`, `asks.py`: loader/checker, scripted caller, number and spelling parser, question types.
- `tests/test_cards.py`, `test_simulate.py`, `test_spoken.py`; `persona.json`: "spell your full name".

### 6. Try it yourself
```powershell
python 03-phone-receptionist\cards.py
python 03-phone-receptionist\simulate.py --preview c14
python -m pytest 03-phone-receptionist\tests -q
```
Expected: "24 cards: 9 dev ... 15 score" and "caller cards are consistent..."; card c14 spelling its name letter by letter; all tests pass.
On the laptop (needs the Piper voice): `python 03-phone-receptionist\simulate.py --audio c14` writes the opening turn as a wav.

### 7. What can go wrong
- The audio path (Piper voice, noise, ffmpeg) is tested here with a fake voice and a real ffmpeg; the real Piper voices were not run.
- "Oh" in ordinary speech reads as a zero, so always ask `digit_runs` for runs of at least 8 digits when looking for a phone number.
- The FAQ callers' names and numbers are invented; the 18 others are Part 1's.

### 8. Check my understanding
1. Why does the simulated caller answer by question type instead of reading the receptionist's sentence?
2. What would go wrong if the dialog were tuned on the `score` cards?
3. Why is the label for the Mum card "no number" even though the caller is asked for one twice?

### 9. Next phase preview
Phase 2 builds the audio loop: listen (Whisper) and speak (Piper) for one turn, a push-to-talk page in the browser, and the scripts that
measure how long each stage takes on the laptop. The measurements themselves cannot be made in the cloud.

## Part 3, Phase 2: Audio loop and speed budget (2026-10-08)

### 1. What we built
The audio plumbing of one phone turn: the caller's recording goes in, Whisper turns it into text (or decides nobody spoke), a reply
text comes back, Piper says it. Around that: a push-to-talk page for the browser, a script that times every stage on the laptop,
and a script that renders candidate voices so you can pick Holly's. **The speed numbers themselves are not measured**: the cloud
machine has no GPU, Whisper model, Ollama model or Piper voice. `docs/part3-speed.md` says so and lists the commands.

### 2. Where it fits in the pipeline
```
 caller ─► LISTEN ─► understand ─► decide ─► SPEAK ─► ... ─► hand off to Part 1
          audio_io.listen                    audio_io.speak
 [Phase 2: audio_io.py, audio_loop.process_turn, session.py, app.py, measure_speed.py, choose_voice.py]  <-- you are here
```

### 3. How it works, step by step
- `audio_io.listen()` transcribes one turn (hint = team and places, never customer names) and returns `Heard`. It marks the turn
  **ignored** if it holds under 0.3 s of speech, if Whisper thinks it is all silence, or if it is only a phrase Whisper invents on silence and Whisper is unsure about it (a confident "Thank you." is kept).
- `audio_io.speak()` synthesizes the reply with Piper and returns how long that took.
- `audio_loop.process_turn()` = listen, then `respond(text, heard)` (where the dialog plugs in), then speak; it times each stage.
- `session.AudioSession` keeps the turns of a call (files, timings) for the page; `scripted_responder` says fixed persona lines until Phase 3.
- `app.py`: Streamlit page with `st.audio_input`; all logic is in `session.py`. Tested headless with Streamlit's `AppTest`.
- `measure_speed.py` (laptop): times Whisper (base/small, CPU/GPU), the LLM (7b/3b) and Piper on the dev caller clips, then applies the
  rule written before the measurement and rewrites `docs/part3-speed.md`. `choose_voice.py` renders a dozen free voices.

### 4. Key concepts I should understand
- **A speed budget is a sum:** the caller waits for listen + understand + speak, so each stage is measured separately and the median is judged against 5 s.
- **Silence is not "no text":** Whisper always writes something; deciding whether anyone spoke is a separate, testable step.
- **Honest placeholders:** a report that says "NOT MEASURED YET" and how to measure it is better than a plausible-looking number.
- **Inject the models:** `listen(..., transcribe_fn=)` and `speak(..., synth_fn=)` let the tests run in milliseconds with fakes.

### 5. Files created or changed
- `audio_io.py`, `audio_loop.py`, `session.py`, `app.py`, `measure_speed.py`, `choose_voice.py` (new); `docs/part3-speed.md` (placeholder).
- `shared/evaluation.py`: `percentile()` and `median()` added. `requirements.txt` already lists streamlit.
- Tests: `test_audio_io.py`, `test_audio_loop.py`, `test_session.py`, `test_measure_speed.py`, `test_app.py`.

### 6. Try it yourself
```powershell
python -m pytest 03-phone-receptionist\tests -q
python 03-phone-receptionist\choose_voice.py
python 03-phone-receptionist\measure_speed.py --device cpu --whisper base small --llm qwen2.5:7b
python -m streamlit run 03-phone-receptionist\app.py
```
Expected on the laptop: voice samples in `testset\voice_samples\`; a table of medians in `docs\part3-speed.md`; the page greets you and
repeats fixed questions while showing what Whisper heard and the seconds per stage.

### 7. What can go wrong
- The 5 s target is unmeasured; on CPU the 7B model is expected to take 30-200 s per reply (from the plan), so the GPU fix comes first.
- The page needs a browser microphone; `st.audio_input` needs Streamlit 1.40+ (1.65 is installed).
- The list of invented phrases is a guess at Whisper's habits; check `Whisper wrote ...` captions on the page and extend it.

### 8. Check my understanding
1. Why is the first model call timed separately from the others?
2. Why does `listen()` return the raw Whisper text even when it ignores the turn?
3. What does the rule pick if `base` misses two names that `small` gets right?

### 9. Next phase preview
Phase 3 builds the dialog engine (version A): the per-turn form the LLM fills, the state machine that decides what to say, the FAQ matcher
and the call runner, tested against all 24 simulated callers.

## Part 3, Phase 3: The dialog engine, version A (2026-10-08)

### 1. What we built
The brain of the receptionist, as a state machine in plain code. Each turn: the caller's words are understood (a model fills a small
form, plain code checks every value against what was said), the call state is updated, the state machine chooses what to do, and
every reply is built from fixed sentences (`persona.json`) and approved answers (`faq.json`). It runs headless against all 24
simulated callers. **It has not been run with a real model**: the cloud machine has none, so every test uses the plain-code baseline.

### 2. Where it fits in the pipeline
```
 caller text ─► UNDERSTAND (turn.py: model fills CallerTurn; ground() keeps only what was said)
              ─► APPLY (dialog.apply_turn: slots, emergency, FAQ, yes/no)    <- shared by version A and the agent of Phase 5
              ─► DECIDE (dialog.decide_a: state machine -> actions)          <- the part version B will replace
              ─► RENDER (fixed sentences + FAQ answers) ─► reply
 [Phase 3: turn.py, rules_turn.py, dialog.py, faq.py, safety.py, call.py]  <-- you are here
```

### 3. How it works, step by step
- `turn.understand()`: builds the prompt (what was asked, what is already known, the FAQ topics), calls `shared/llm.structured_chat` for a
  `CallerTurn`, then `ground()`: a number must appear as digits in the speech, a name must be made of words (or spelled letters) that were said,
  a reason must share its words with the speech. If the model fails twice, `rules_turn.py` answers instead.
- `dialog.apply_turn()`: stores a detail only when this turn contained it; an existing value changes only on an explicit correction;
  spelled letters fix the heard name (`spoken.apply_spelling`); safety words (Part 1's patterns) or the model's flag raise the emergency.
- `dialog.decide_a()`: GREETING/COLLECTING ask the next missing detail (reason, name, [spelling], number); READ_BACK confirms or goes to
  CORRECTING; GOODBYE ("anything else?") ends or goes back to collecting; the turn limit and silence end the call.
- `faq.match_questions()`: only question-shaped sentences, whole-word keywords, the longest phrase wins, a generic price question
  looks at the sentence before it; unknown questions are passed on.
- `call.run_call()`: a simulated caller answers whatever the dialog is waiting for, in text or through the audio loop; the call is saved as JSON.

### 4. Key concepts I should understand
- **Ground every value:** the model's answer is a claim; code checks it against the words. A number never spoken cannot be stored.
- **Two layers for the same rule:** a model flag OR Part 1's safety words decide an emergency, so a model miss is caught by dumb reliable code.
- **Separate "what changed" from "what to say":** `apply_turn` is shared; only `decide` differs between the state machine and the agent.
- **Replies are never generated:** an invented price is impossible because no model writes a sentence; a test parses every reply into approved pieces.

### 5. Files created or changed
- New: `turn.py`, `rules_turn.py`, `dialog.py`, `safety.py`, `call.py`; `faq.py` (matching), `spoken.py` (`apply_spelling`).
- `faq.json`: `prices` is marked `fallback`; `booking_time` gained `how soon could someone ...` keywords (from the dev card f01); `callback_time` lost the loose keywords `call back` / `call me back` / `get back to me` / `ring me back` (they matched requests such as "can you give me a call back?"); `safety_water` lost the bare `leaking`. `persona.json`: four more fixed sentences (`what_else`, `goodbye_info`, `turn_limit_urgent`, `silence_end_urgent`).
- `shared/schemas.py`: `normalize_uk_number()` factored out of `Analysis.check_uk_number` (one rule for Part 1 and Part 3).
- Tests: `test_turn.py`, `test_faq_match.py`, `test_dialog.py`, `test_call.py`.

### 6. Try it yourself
```powershell
python -m pytest 03-phone-receptionist\tests -q
python 03-phone-receptionist\call.py --card c14 --understand rules
python 03-phone-receptionist\call.py --card dev --understand model     # needs Ollama + qwen2.5:7b
```
Expected: the tests pass; a transcript where Holly asks for the spelling and reads back "Siobhan Gallagher".

### 7. What can go wrong
- The prompt (`PROMPT_VERSION t1`) has never met a real model: tune it on the dev cards only, then score once.
- The rules baseline cannot judge urgency from context (it misses c04 and c06) or paraphrase a reason; the model is expected to.
- A caller who answers the read-back with a long unrelated story is read back again and eventually hits the turn limit.

**Review record (independent Opus reviewers):** round 1 FAIL (17 findings: grounding that never ran for a custom understanding step, an emergency lost to the spam check, premature hang-ups, a lost second request, tuning to score cards, ...), round 2 FAIL (11: junk stored as a name, naming the wrong field, false-positive number repairs, "yeah no", endless read-backs, lost late corrections, ...), round 3 FAIL (3: a robocall pattern hanging up on real callers, the phone number leaking into the reason, "I'm calling about ..." not taken as a reason). The round-3 findings were fixed afterwards and covered by new tests plus a seeded 300-call fuzz of the engine's invariants, **but not reviewed again** (limit: 3 rounds per phase). Open known limits: the rules baseline cannot judge urgency from context or paraphrase a reason, and "Do you fix gas leaks?" is treated as an emergency (safety first).

### 8. Check my understanding
1. Why is a number the model reports thrown away if the caller did not say it?
2. What is the difference between `apply_turn` and `decide_a`, and which one will the agent replace?
3. Why does Holly not ask a caller who only wanted the opening hours for a name?

### 9. Next phase preview
Phase 4 hands a finished call to the Part 1 pipeline: analyze the caller's side, route it, push if urgent, and store it in `voicemails.db`.

## Part 3, Phase 4: Hand-off to Part 1 (2026-10-08)

### 1. What we built
A finished call becomes a row in Part 1's `voicemails.db`, handled exactly like a voicemail: the caller's side is analysed, Part 1's routing
decides what happens, an urgent call is pushed (a dry run until NTFY is configured), and the row is saved. Calls are recognisable by
`source_file = call-<id>.json`. The receptionist's side of the conversation never goes to the analysis.

### 2. Where it fits in the pipeline
```
 call ─► listen ─► understand ─► decide ─► speak ─► ... ─► END OF CALL
                                     │ urgent? ─► LivePush (while the caller is still on the line)
                                     ▼
   handoff.hand_off: caller's words ─► Part 1 analyze (call-v1) ─► routing.route ─► push (if not already) ─► store.save
 [Phase 4: handoff.py + the call-v1 prompt in shared/analyze.py]  <-- you are here
```

### 3. How it works, step by step
- `call_transcript()` builds a Part 1 `Transcript` from the caller's answers (one segment per answer; Whisper's confidence is passed on, so unclear audio is flagged by Part 1's own rule).
- `analysis_for_call()`: robocalls, info-only and silent calls get an analysis from plain code. Others go to `analyze(..., prompt_version="call-v1")`; then `merge_model_analysis()` puts the dialog's checked name and number over the model's and keeps an urgent call urgent. If the model fails, plain code builds it and the row is flagged for review.
- `route()` (Part 1, unchanged) decides notify / inbox / archive and the review flags; `hand_off()` adds a reason line ("phone call, outcome ..., flagged urgent in turn 1").
- `LivePush` sends the minimal "Urgent call in progress" push the moment `run_call` flags the call, in a background thread so the caller never waits; `hand_off()` waits for it (15 s at most; after that the end push goes out too, so the phone may ring twice, never zero times) and otherwise does not push twice. Push first, save second (Part 1's at-least-once order).
- `save()` (Part 1) upserts by a hash of the call: handing the same call off again refreshes the row but never adds a row or a second push.

### 4. Key concepts I should understand
- **Reuse, don't copy:** routing, safety words, push text, retries and storage are Part 1's own functions; a call and a voicemail cannot disagree about what is urgent.
- **Check the model against what was checked:** the model sees the words; the dialog verified them with the caller. For name and number the dialog wins.
- **Push first, save second:** a crash can repeat a push (harmless) but cannot lose one (not harmless for a gas leak).
- **Idempotency is a hash:** same call, same row, no second push (a re-run refreshes the row but pushes only if the first run did not).

### 5. Files created or changed
- New: `handoff.py`, `tests/test_handoff.py`. Changed: `shared/analyze.py` (a new prompt version `call-v1` and the user-message label; v2/v3 untouched), `call.py` (`CallRecord.handoff`, `save_record(name=)`, logging in the command line).
- Review record: round 1 FAIL (6: a re-run could store an urgent call without ever pushing it, a message ending in silence was summarised as "said nothing", the info-only override wiped other review flags, a failed push lost the call record, `*.json` did not work on Windows, the live push could stall the call), round 2 FAIL (3 small: a test racing the push thread, a "never twice" claim, two stale sentences); all fixed.

### 6. Try it yourself
```powershell
python 03-phone-receptionist\call.py --card dev --understand rules --save
python 03-phone-receptionist\handoff.py 03-phone-receptionist\calls\*.json --analysis rules
python 01-voicemail-triage\store.py
```
Expected: one row per call; the gas call routed `notify_now` (the dry-run push text is printed), the robocall `archive`, the others `inbox`.

### 7. What can go wrong
- `call-v1` has only been run against a fake model here. A real model may need the prompt tuned (dev calls only).
- NTFY is a dry run until a real random topic is set in `.env`; nothing is sent to anyone.
- A model that judges an urgent-flagged call "not urgent" is overruled on purpose; a model that finds urgency the dialog missed is trusted.

### 8. Check my understanding
1. Why does the row's caller name come from the dialog and not from the model?
2. Why is an urgent call pushed in the middle of the call and not only at the end?
3. What stops the same call from creating two rows or two pushes?

### 9. Next phase preview
Phase 5 builds version B: the same receptionist, but the choice of the next action is made by a boxed tool-calling agent (with Part 2's guardrails) instead of the state machine, so the two can be compared on the same callers.

## Part 3, Phase 5: Dialog version B, the tool-calling agent (2026-10-08)

### 1. What we built
The same receptionist with a different brain for one step: instead of the state machine, a boxed tool-calling agent (Part 2's guardrails)
chooses the next action. It is built so the two versions can be compared fairly on the same callers in Phase 6. **It has never run with a
real model**: tests use scripted tool calls, a "perfect" fake agent that reproduces version A on all 24 callers, and a random agent.

### 2. Where it fits in the pipeline
```
 caller text ─► UNDERSTAND ─► APPLY (grounding, emergency, FAQ: shared) ─► DECIDE ─┬─ version A: dialog.decide_a (state machine)
                                                                                  └─ version B: agent_dialog.make_decide_b (tool loop)
                                                                          ─► RENDER (fixed sentences: shared)
 [Phase 5: agent_dialog.py]  <-- you are here
```

### 3. How it works, step by step
- `build_turn()` asks the state machine first; its decision is always one LEGAL option. A few more are legal where judgement can matter.
- The loop (max 4 LLM calls per turn): a fresh state message each step (what the caller said, what we have, questions to answer, the legal
  actions, feedback on the last calls) -> tool calls -> `run_tool()` validates each one and answers `ERROR: ...` with advice when it is illegal.
- `answer_faq(topic)` must be called for every question the caller asked before an action; `flag_urgent(why)` once; then exactly one of
  `ask(detail)`, `read_back()`, `take_message()`, `end_call(why)`.
- If no legal action comes out (model error, prose, step limit, same input twice), the state machine's decision is used: `fallback: true` in the trace.

### 4. Key concepts I should understand
- **Freedom is a budget:** the agent may only choose among actions the state allows, so its mistakes are bounded; the price is that it often has exactly one option.
- **Values never go through the model:** no tool takes a name, number or sentence, so nothing can be invented there.
- **The baseline stays underneath:** the fallback means a bad model day costs speed and a logged fallback, not a broken call.
- **A fair experiment shares everything but one step:** the equivalence test (an agent that takes the state machine's choice gives byte-identical calls) proves the harness adds no behaviour of its own.

### 5. Files created or changed
- New: `agent_dialog.py`, `tests/test_agent_dialog.py`. Changed: `dialog.py` (`Facts.caller_text`), `call.py` (`--decide a|b`, `CallRecord.decide`), `handoff.py` docstrings.

### 6. Try it yourself
```powershell
python -m pytest 03-phone-receptionist\tests\test_agent_dialog.py -q
python 03-phone-receptionist\call.py --card dev --understand model --decide b      # needs Ollama
```
Expected: the tests pass; on the laptop a transcript where "(agent fell back to the state machine: ...)" appears under a turn whenever the model failed.

### 7. What can go wrong
- `qwen2.5:7b` may call tools badly (the Part 2 agent guessed ids): expect fallbacks; Phase 6 counts them.
- Latency: the agent adds up to 4 LLM calls to a turn on top of understanding; the 5 s target will be hard on CPU.
- A model that always agrees with the state machine makes B equal to A at higher cost; that is a result, not a bug.

### 8. Check my understanding
1. Why does the agent always have the state machine's decision among its legal options?
2. Why is it safe that `flag_urgent` exists although urgency is also detected by code?
3. What does the equivalence test prove, and what does it not prove?

### Review record
Independent reviewer rounds: FAIL, FAIL, PASS. Round 1 forced: read-back after a name refusal must not skip the number; the urgent rules must hold after `flag_urgent`; `end_call` and the state message must match what is accepted. Round 2: the urgent rules must hold whoever flagged the call. Round 3: PASS.

### 9. Next phase preview
Phase 6 runs every caller card through both versions, scores them (slot accuracy, invented numbers, urgent flagged, FAQ answers, turns, latency) and applies the A-vs-B rule that was written before any run. The numbers need the laptop's models.

## Part 3, Phase 6: Evaluation harness (2026-10-08)

### 1. What we built
`evaluate.py`: it runs the caller cards through version A and version B, scores every call against the answer key, hands each call to Part 1
(in memory), prints the definition-of-done table and applies the A-vs-B rule that was written before any run. **No evaluation number exists
yet**: the cloud session has no models, so `docs/part3-eval-results.md` is an honest "NOT MEASURED" page with the laptop command. A run
without a model prints a "harness check only" banner and never writes that file.

### 2. Where it fits in the pipeline
```
 caller cards ─► run_call (A or B) ─► score_call (answer key) ─► summarize ─► dod_table / decide_ab ─► docs/part3-eval-results.md
                      └─► hand_off (in-memory Part 1 table, recorded pushes) ─┘
 [Phase 6: evaluate.py]  <-- you are here
```

### 3. How it works, step by step
- `run_set()` runs each card with a fresh decide function and a recording push sender (nothing is sent).
- `score_call()`: name/number classes (correct, wrong, missed, invented), invented number (digits the caller never said), urgent missed/false,
  FAQ topics, replies made only of approved sentences, outcome, time per reply, agent fallbacks, hand-off route and push.
- `decide_ab()` implements the four pre-registered rules literally; with one A run it cannot apply rule 4, so B cannot win.
- `--repeat-a 2` runs A twice: the difference between the two runs is the noise B's advantage must beat.

### 4. Key concepts I should understand
- **Pre-registration:** the rule and its constants (2 details, +3 s, 10 %) exist in the README and in code before any result, so the result cannot choose the rule.
- **Dev vs score cards:** prompts are tuned on 9 cards, scored on the other 15.
- **Noise:** a model at temperature 0 still varies between runs; a gain smaller than that variation is not a result.
- **Harness check vs measurement:** the rules baseline on dev cards proves the counting works; it says nothing about the real receptionist.

### 5. Files created or changed
- New: `evaluate.py`, `tests/test_evaluate.py`, `docs/part3-eval-notes.md`, `docs/part3-eval-results.md` (placeholder). Changed: `call.py` (`audio_channel_for`, shared by the CLI and the evaluation), README (decisions 44-46, commands).

### 6. Try it yourself
```powershell
python -m pytest 03-phone-receptionist\tests\test_evaluate.py -q
python 03-phone-receptionist\evaluate.py --split dev --understand rules --decide a --repeat-a 1     # harness check, no model
python 03-phone-receptionist\evaluate.py --split score --understand model --decide both --repeat-a 2 --write-docs   # the real evaluation (laptop)
```
Expected: the tests pass; the first command prints a "Harness check only" report; the last one rewrites `docs\part3-eval-results.md`.

### 7. What can go wrong
- The "approved sentence" check lets placeholders match anything, so it proves no sentence outside the approved set, not that a placeholder value is right (the invented-number and name checks cover values).
- 15 score cards is a small set: a difference of one or two calls is within noise; that is why rule 4 exists.
- Audio runs depend on Whisper's transcription of the caller voices; a low score can be the ear, not the dialog (the log keeps what was heard).

### 8. Check my understanding
1. Why can B not win when A was run only once?
2. Why is a run with `--understand rules` never written to the results file?
3. What does "invented number" mean here, and why is it a gate and not just a metric?

### 9. Next phase preview
Phase 7 wires the Streamlit receptionist page to the real dialog, polishes the README, writes HANDOVER section 13 (documents against code) and has a final reviewer check every claim.

## Part 3, Phase 7: Polish (2026-10-08)

### 1. What we built
The push-to-talk page now runs the real receptionist (version A or B, understanding by model or by the no-model rules), with a side panel of
what has been understood so far. README status lines, the root README and `docs/HANDOVER.md` section 13 (documents against code, what is not
done, run order) were brought in line with the code.

### 2. Where it fits in the pipeline
```
 browser microphone ─► AudioSession ─► Whisper ─► dialog_responder ─► dialog.next_reply (A or B) ─► Piper ─► browser
 [Phase 7: session.dialog_responder, app.py]  <-- you are here
```

### 3. How it works, step by step
- `session.dialog_responder()` owns one call state; an ignored turn (silence) reaches the dialog as an empty turn, so the silence rules apply.
- `app.py` builds the responder from the sidebar choices; the side panel reads `responder.state`; a finished call shows its outcome.
- The page does not hand the call to Part 1; `call.py --save` and `handoff.py` do.

### 4. Key concepts I should understand
- **Thin shell:** the page only wires; every rule is in tested modules.
- **Documents against code:** HANDOVER 13.2 maps each promise to the code and the test that keeps it true.
- **Honest status:** "built and tested" is not "measured"; the docs say which numbers do not exist yet.

### 5. Files created or changed
- Changed: `session.py`, `app.py`, `tests/test_session.py`, `tests/test_app.py`, README files, `docs/HANDOVER.md`. Removed: the unused `03-phone-receptionist/.gitkeep`.

### 6. Try it yourself
```powershell
python -m pytest 01-voicemail-triage\tests 02-meeting-action-agent\tests 03-phone-receptionist\tests -q
python -m streamlit run 03-phone-receptionist\app.py      # choose Understanding "rules" to try it without Ollama
```
Expected: all tests pass; the page greets, listens, answers, and the side panel fills in as you give your reason, name and number.

### 7. What can go wrong
- With "model" understanding and Ollama not running, the first turn raises an error: start Ollama or pick "rules".
- Whisper mishearing a number gives a wrong read-back: say "no, the number is wrong" and give it again.
- No chosen voice: replies are text only and the wait shown is too short.

### 8. Check my understanding
1. Why does an ignored recording go to the dialog as silence instead of being dropped?
2. Which document line would you check first if the code changed a limit?
3. Which numbers in this repo are still missing, and which command produces each?

### 9. Next phase preview
Part 3 is complete in code. Next: the laptop runs in README section 8, then the owner's review and merge.
