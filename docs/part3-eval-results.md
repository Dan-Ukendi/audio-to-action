# Part 3 evaluation results

**NOT MEASURED YET.** The cloud session that built Part 3 has no GPU, no Ollama model, no Whisper model and no Piper voice, so no
evaluation number exists and none is written here. Runs without a model (`--understand rules`) only test the harness and are never
written to this file.

To produce the numbers, on the laptop (after the speed measurements and the voice choice, see README section 8):

```powershell
python 03-phone-receptionist\evaluate.py --split score --understand model --decide both --repeat-a 2 --write-docs
python 03-phone-receptionist\evaluate.py --split score --understand model --decide a --audio --write-docs   # adds the real speed (audio) run
```

The command rewrites this file with the counts per version, the definition-of-done table (PASS / FAIL / NOT MEASURED) and the
verdict of the pre-registered A-vs-B rule. The metrics and the rule are explained in `docs/part3-eval-notes.md`.
Tune prompts only on the `dev` cards (`--split dev`); the `score` cards are for this run.
