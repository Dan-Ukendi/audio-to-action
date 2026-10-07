# What `shared/` gives Part 2 (meeting recordings → action items)

> Written at the end of Part 1, before Part 2 existed. What actually happened: `transcribe.py` was reused with a new
> `hint` (known names); the analyze pattern became `shared/llm.structured_chat()` (Part 1 uses it too); new shared
> modules `tts.py` and `pipeline.py`; chunking exists but our meetings fit one chunk; owners came from names in
> the words (no diarization). Current reference: `docs/HANDOVER.md` §12.

Part 1 built every module in `shared/` without voicemail-specific logic except where noted.
This is what Part 2 can import as-is, what it must replace, and what to watch out for.

| Module | Reuse in Part 2 | Change needed |
|---|---|---|
| `shared/transcribe.py` | **As-is.** `transcribe(path, cache_dir, model)` handles any format, caches by audio hash + model. | Meetings are long (30-60 min): at CPU speed (`small` ≈ 1.7× real time) a 1 h meeting takes ~1.7 h. Use the GPU (needs the NVIDIA driver fixed + `nvidia-cublas-cu12`/`nvidia-cudnn-cu12`) or accept overnight runs. Segments already carry timestamps, useful for "who said it when". No speaker labels (diarization): Whisper doesn't do that. |
| `shared/schemas.py` | `Segment`, `Transcript` as-is. | `Analysis`/`Result` are voicemail-specific: write `ActionItem` (task, owner, due date, source timestamp) and `MeetingResult` next to them, following the same pattern (validators = plain-code rules). |
| `shared/analyze.py` | The **pattern**: structured output via `format=<schema>`, temperature 0, validate, one retry with the error fed back ("fix, don't delete"), cache keyed by hash + models + prompt version, `PROMPTS` dict with versions. | The prompt and the target schema. A 1 h transcript (~10k words) is bigger than the 4096-token default context Ollama gave us: either raise `num_ctx` (more RAM/VRAM) or chunk the transcript (e.g. by 10-minute windows) and merge items. Generalising `analyze()` to take `(schema, prompt)` is the natural first refactor. |
| `shared/evaluation.py` | **As-is.** `same_value`, `same_first_word`, `null_aware` (correct/wrong/missed/invented), `confusion`, `pct`. | Action items are a *list* per meeting, so Part 2 also needs set matching (precision/recall of items), which is new. |
| `shared/retry.py` | **As-is.** `with_retries()` + `is_transient()` for Ollama/ntfy/any HTTP. | None. |
| `shared/notify.py` | **As-is.** `send_push()` with dry run. | Keep the privacy rule: counts only, never meeting content, on public ntfy.sh. |

## Lessons from Part 1 that carry over
- **Write labels before building** (Phase 1) and keep the test set small but targeted.
- **Prompt changes need the whole eval re-run**, and each experiment should change one thing (Phase 6).
- **The LLM fills forms; code decides.** Part 2 is billed as "workflow + agent": keep the agent part
  confined to where open-ended decisions are genuinely needed (e.g. deciding whom to follow up with),
  and keep transcription, extraction and storage as fixed workflow steps.
- **Whisper is confident when wrong** (Phase 4): don't use `avg_logprob` as an accuracy signal for names.
