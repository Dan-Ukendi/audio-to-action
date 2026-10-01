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
