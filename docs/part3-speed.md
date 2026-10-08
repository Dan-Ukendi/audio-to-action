# Part 3 speed budget

**NOT MEASURED YET.** The cloud session that built Part 3 has no GPU, no Whisper model, no Ollama model and no Piper voice,
so no number is written here. To produce them, on the laptop:

```powershell
nvidia-smi
python 03-phone-receptionist\choose_voice.py             # pick Holly's voice first (persona.json piper_speaker)
python 03-phone-receptionist\measure_speed.py --device cpu  --whisper base small --llm qwen2.5:7b
python 03-phone-receptionist\measure_speed.py --device cuda --whisper base small --llm qwen2.5:7b qwen2.5:3b   # after the GPU is fixed
```

This file is then rewritten with the medians, p95 and the decision of the pre-registered rule (README section 4):
the smallest Whisper model within 1 name-hit of `small`; `qwen2.5:7b` if its understanding step takes <= 3 s, else `qwen2.5:3b`;
target = median turn (listen + understand + speak) of 5 s or less.
