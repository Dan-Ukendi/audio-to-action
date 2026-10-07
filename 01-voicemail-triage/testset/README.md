# Test set

The ruler we measure every pipeline change against. Built before any pipeline code.

| File | Committed? | What |
|---|---|---|
| `scripts.json` | yes | What each synthetic voicemail says + how it sounds (speaker, speed, noise, format) |
| `labels.json` | yes | The answer key: category, urgency, caller name, callback number, language |
| `generate.py` | yes | Rebuilds `audio/` from `scripts.json` with Piper TTS + ffmpeg |
| `audio/` | **no** | Generated audio (regenerate any time) |
| `voices/` | **no** | Downloaded Piper voice `en_GB-vctk-medium` (~73 MB) |
| `my_recordings/` | **no** | Your own real recordings + their labels (personal data) |

## Regenerate the synthetic audio
```powershell
.\.venv\Scripts\Activate.ps1
python 01-voicemail-triage\testset\generate.py            # all 18
python 01-voicemail-triage\testset\generate.py --only 05  # just one
```
Piper's speech is not bit-for-bit identical between runs (the model samples a little randomness),
but the words, speakers and noise (fixed seed) are.

## Add your own recordings
1. Put audio files in `my_recordings/` (any format ffmpeg reads).
2. Add `my_recordings/labels.json` with the same shape as `labels.json` (`{"items": [...]}`),
   following the conventions at the top of `labels.json`.
3. The eval script (Phase 6) will pick up both sets.

Only record people who agreed to it, and never commit this folder.

## Coverage (18 files, English)
| Category | Count | Notable traps |
|---|---|---|
| urgent | 6 | vulnerability without the word "urgent", vague message, heavy noise, error code next to a phone number |
| personal | 2 | relationship instead of a name, no number |
| sales | 2 | relevant trade offer, cold call with no person's name |
| spam | 2 | robocall, and one that says "urgent" three times |
| other | 6 | spelled name, unspellable name, rambling, no info at all, self-corrected number |
