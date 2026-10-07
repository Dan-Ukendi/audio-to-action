# audio-to-action

A learning project about using AI to handle phone calls and audio, fully local and free.

| Part | Folder | What | Style |
|---|---|---|---|
| 1 | `01-voicemail-triage/` | Voicemails → transcript → category, summary, extracted data → routed | Workflow |
| 2 | `02-meeting-action-agent/` | Meeting recordings → action items | Workflow + agent |
| 3 | `03-phone-receptionist/` | Real-time AI phone receptionist | Agent |

`shared/` holds code reused across parts (transcription, schemas, eval helpers).

## Stack
Python 3.13 · ffmpeg · faster-whisper · Ollama (`qwen2.5:7b`) · Pydantic · SQLite · ntfy

## Setup (Windows / PowerShell)
```powershell
py -V:3.13-64 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
ollama pull qwen2.5:7b
python scripts\check_setup.py
```

## Privacy
Voicemails contain personal data. Audio, transcripts, results and the database stay on this machine
and are git-ignored. Keep this repo outside synced folders (OneDrive, Dropbox).

## Status
- [x] Phase 0: Setup
- [x] Phase 1: Test set (18 synthetic English voicemails)
- [x] Phase 2: Transcription (`shared/transcribe.py`, results in `docs/transcription-comparison.md`)
- [x] Phase 3: Analysis (`shared/analyze.py`, Ollama structured output + Pydantic validation + 1 retry)
- [x] Phase 4: Routing (plain-code rules in `01-voicemail-triage/routing.py`, ntfy dry run, SQLite)
- [ ] Phase 5: Glue & reliability
- [ ] Phase 6: Evaluation
- [ ] Phase 7: Polish

_Architecture diagram and learnings will be added in Phase 7. See `docs/learning-log.md` for per-phase notes._
