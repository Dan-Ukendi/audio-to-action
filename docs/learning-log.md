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
