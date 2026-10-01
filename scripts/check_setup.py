"""Phase 0 sanity check: are ffmpeg, faster-whisper and Ollama reachable?

Each check is independent and returns (ok, message), so one failure
doesn't hide the others. Nothing here downloads a model.
"""

import json
import os
import shutil
import subprocess
import sys
import urllib.request

from dotenv import load_dotenv

load_dotenv()  # read .env if present, else fall back to defaults below

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


def check_python() -> tuple[bool, str]:
    version = sys.version.split()[0]
    in_venv = sys.prefix != sys.base_prefix
    ok = sys.version_info[:2] == (3, 13) and in_venv
    hint = "" if ok else "  (expected Python 3.13 inside .venv)"
    return ok, f"Python {version}, venv={'yes' if in_venv else 'no'}{hint}"


def check_ffmpeg() -> tuple[bool, str]:
    path = shutil.which("ffmpeg")
    if not path:
        return False, "ffmpeg not on PATH (open a new terminal after installing)"
    first_line = subprocess.run(
        [path, "-version"], capture_output=True, text=True
    ).stdout.splitlines()[0]
    return True, first_line


def check_faster_whisper() -> tuple[bool, str]:
    try:
        import ctranslate2
        import faster_whisper
    except ImportError as e:
        return False, f"import failed: {e}"
    # CTranslate2 is the engine under faster-whisper; it decides CPU vs GPU.
    gpus = ctranslate2.get_cuda_device_count()
    device = f"{gpus} CUDA GPU(s) visible" if gpus else "CPU only (no CUDA GPU visible)"
    return True, (
        f"faster-whisper {faster_whisper.__version__}, "
        f"ctranslate2 {ctranslate2.__version__}, {device}"
    )


def check_ollama() -> tuple[bool, str]:
    # /api/tags lists the models that have been pulled locally.
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=5) as resp:
            models = [m["name"] for m in json.load(resp)["models"]]
    except OSError as e:
        return False, f"Ollama not reachable at {OLLAMA_HOST} ({e}). Is the Ollama app running?"
    if OLLAMA_MODEL not in models:
        return False, (
            f"Ollama is running, but '{OLLAMA_MODEL}' is not pulled yet "
            f"(run: ollama pull {OLLAMA_MODEL}). Local models: {models or 'none'}"
        )
    return True, f"Ollama at {OLLAMA_HOST}, model '{OLLAMA_MODEL}' available"


def main() -> int:
    checks = {
        "python": check_python,
        "ffmpeg": check_ffmpeg,
        "faster-whisper": check_faster_whisper,
        "ollama": check_ollama,
    }
    all_ok = True
    for name, check in checks.items():
        ok, msg = check()
        all_ok &= ok
        print(f"[{'OK' if ok else 'FAIL'}] {name:15} {msg}")
    print("\nAll good." if all_ok else "\nSome checks failed, see hints above.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
