"""The scripted synthetic caller: answers whatever the receptionist asks, from a caller card.

    caller = SimulatedCaller(card)
    text = caller.reply(asks.GREETING)                 # the opening turn
    text = caller.reply(asks.NAME)                     # "Could I take your name?" -> "It's Mark Thompson."
    text = caller.reply(asks.CONFIRM, heard="So that's ...")   # the caller checks the read-back
    render_turn(card, text, Path("turn_01.wav"))       # the caller's voice, with noise and a phone-line sound

It answers by QUESTION TYPE (asks.py), never by parsing the receptionist's sentences, so it keeps working
when her wording changes. The one exception is the read-back: an attentive caller notices a wrong name or
number and says so (at most twice), because that is what real callers do and it is what the correction flow
is for. All behaviour is deterministic: the same card and the same questions give the same words.

    python 03-phone-receptionist/simulate.py --preview c14      # print what a card says to every question
    python 03-phone-receptionist/simulate.py --audio c14        # render the opening turn (needs the Piper voice)
"""

import argparse
import re
import subprocess
import sys
import tempfile
import wave
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import asks  # noqa: E402
from cards import Card, load_cards  # noqa: E402
from spoken import digit_runs  # noqa: E402

MAX_REPAIRS = 2  # how many times an attentive caller corrects a wrong read-back
NO_ANSWER = "That's all, thank you."  # default answer to "anything else?"


# ---------------------------------------------------------------- saying things the way people say them

def number_words(digits: str) -> str:
    """'07700900123' -> 'oh seven seven double oh, nine hundred, one two three' (how a UK caller reads it out)."""
    words = {"0": "oh", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six",
             "7": "seven", "8": "eight", "9": "nine"}

    def group(part: str) -> str:
        if len(part) == 3 and part[0] != "0" and part[1:] == "00":
            return f"{words[part[0]]} hundred"
        out, i = [], 0
        while i < len(part):
            if part[i] == "0" and part[i:i + 2] == "00":
                out.append("double oh")
                i += 2
            else:
                out.append(words[part[i]])
                i += 1
        return " ".join(out)

    if len(digits) == 11:
        parts = [digits[:5], digits[5:8], digits[8:]]
    elif len(digits) == 10:
        parts = [digits[:4], digits[4:7], digits[7:]]
    else:
        parts = [digits]
    return ", ".join(group(p) for p in parts)


def wrong_number(digits: str) -> str:
    """The same number with its last three digits rotated: a plausible slip that is still a valid number."""
    for turn in (1, 2):
        tail = digits[-3:]
        rotated = tail[turn:] + tail[:turn]
        if rotated != tail:
            return digits[:-3] + rotated
    return digits[:-1] + str((int(digits[-1]) + 1) % 10)


def spell_out(name: str) -> str:
    """'Siobhan Gallagher' -> 'S, I, O, B, H, A, N, G, A, double L, A, G, H, E, R'."""
    letters = [ch.upper() for ch in name if ch.isalpha()]
    out, i = [], 0
    while i < len(letters):
        if i + 1 < len(letters) and letters[i] == letters[i + 1]:
            out.append(f"double {letters[i]}")
            i += 2
        else:
            out.append(letters[i])
            i += 1
    return ", ".join(out)


# ---------------------------------------------------------------- the caller

class SimulatedCaller:
    def __init__(self, card: Card):
        self.card = card
        self.counts: dict[str, int] = {}
        self.last = ""
        self.repairs = 0
        self.number_corrected = False
        self.pending_fix: list[str] = []  # what the last read-back got wrong: "name" and/or "number"

    # -- text helpers
    def fill(self, template: str) -> str:
        f = self.card.facts
        values = {"name_spoken": f.name_spoken or "", "number_words": number_words(f.number) if f.number else ""}
        return template.format(**values)

    def quirk(self, name: str) -> bool:
        return name in self.card.quirks

    def number_sentence(self) -> str:
        return f"My number is {number_words(self.card.facts.number)}."

    # -- the answers
    def reply(self, asked: str, heard: str | None = None) -> str:
        """What the caller says to this question ('' = silence). `heard` is the receptionist's read-back."""
        card, s = self.card, self.card.script
        if asked not in asks.ALL:
            raise ValueError(f"unknown question type {asked!r}")
        if asked == asks.NOTHING:
            return ""
        if self.quirk("robocall"):
            # A robocall plays its message once and never answers anything.
            text = self.fill(s.opening) if asked == asks.GREETING else ""
            self.last = text or self.last
            return text

        key = "number" if asked == asks.NUMBER_AGAIN else asked
        self.counts[key] = self.counts.get(key, 0) + 1
        n = self.counts[key]

        handlers = {
            asks.GREETING: lambda: self.fill(s.opening),
            asks.REASON: lambda: self.fill(s.reason or s.opening),  # asked again: say it again
            asks.NAME: lambda: self.name_answer(n),
            asks.SPELLING: self.spelling_answer,
            asks.NUMBER: lambda: self.number_answer(n),
            asks.NUMBER_AGAIN: lambda: self.number_answer(max(n, 2)),
            asks.REPEAT: lambda: self.last or self.fill(s.opening),
            asks.CONFIRM: lambda: self.confirm_answer(heard),
            asks.CORRECTION: self.correction_answer,
            asks.ANYTHING_ELSE: lambda: s.anything_else[n - 1] if n <= len(s.anything_else) else NO_ANSWER,
        }
        text = handlers[asked]()
        if self.quirk("hesitant") and text and asked not in (asks.GREETING, asks.CONFIRM, asks.REPEAT):
            text = "Um, yeah, " + text[0].lower() + text[1:]
        if asked != asks.REPEAT:
            self.last = text
        return text

    def name_answer(self, n: int) -> str:
        s, f = self.card.script, self.card.facts
        if f.name is None:
            if n >= 2 and s.name_again:
                return s.name_again
            return s.name or "I'd rather not say my name."
        if s.name and n == 1:
            return self.fill(s.name)
        return f"It's {f.name_spoken}."

    def spelling_answer(self) -> str:
        f = self.card.facts
        if f.name is None:
            return "I'd rather not."
        if self.quirk("spells_name"):
            return spell_out(f.name) + "."
        return f"It's {f.name_spoken}."

    def number_answer(self, n: int) -> str:
        s, f = self.card.script, self.card.facts
        if self.quirk("refuses_number") or self.quirk("withholds_number") or f.number is None:
            if n >= 2:
                return s.number_again or s.number or "As I said, I'd rather not give a number."
            return s.number or "I'd rather not give a number."
        if self.quirk("wrong_number_first") and not self.number_corrected:
            return f"My number is {number_words(wrong_number(f.number))}."
        if self.quirk("self_corrects_number") and n == 1:
            wrong = wrong_number(f.number)
            cut = len(f.number) - 3
            # "...nine hundred, four nine three, no sorry, three four nine. So that's <the whole number>."
            return (f"My number is {number_words(f.number[:cut])}, {number_words(wrong[cut:])}, no sorry, "
                    f"{number_words(f.number[cut:])}. So that's {number_words(f.number)}.")
        if s.number and n == 1:
            return self.fill(s.number)
        return self.number_sentence() if n == 1 else f"It's {number_words(f.number)}."

    def confirm_answer(self, heard: str | None) -> str:
        """The read-back: say yes, or notice what is wrong. Never invents a complaint the card cannot justify."""
        f = self.card.facts
        wrong = []
        if heard is not None and self.repairs < MAX_REPAIRS:
            read_words = set(re.findall(r"[a-z]+", heard.lower()))  # whole words: "Davey" is not "Dave"
            if f.name and not all(part in read_words for part in re.findall(r"[a-z]+", f.name.lower())):
                wrong.append("name")
            runs = digit_runs(heard, min_len=8)
            if f.number and f.number not in runs and not (self.quirk("wrong_number_first") and not self.number_corrected):
                wrong.append("number")
            elif not f.number and runs:
                self.repairs += 1
                self.pending_fix = (["name"] if "name" in wrong else []) + ["no_number"]
                said_name = f"my name is {f.name_spoken}" + (f", that's {spell_out(f.name)}" if self.quirk("spells_name") else "")
                return ("No, " + said_name + ", and I didn't give you a number.") if "name" in wrong \
                    else "No, I didn't give you a number."
        if self.quirk("wrong_number_first") and not self.number_corrected and f.number:
            self.number_corrected = True  # the card's own slip: objects once, whatever was read back
            wrong.append("number")
        wrong = sorted(set(wrong), key=["name", "number"].index)
        if not wrong:
            self.pending_fix = []
            return "Yes, that's right."
        if heard is not None:
            self.repairs += 1
        self.pending_fix = wrong
        parts = []
        if "name" in wrong:
            parts.append(f"my name is {f.name_spoken}" + (f", that's {spell_out(f.name)}" if self.quirk("spells_name") else ""))
        if "number" in wrong:
            parts.append(f"the number is {number_words(f.number)}")
        return "No, " + " and ".join(parts) + "."

    def correction_answer(self) -> str:
        f = self.card.facts
        fix = self.pending_fix or (["number"] if f.number else ["name"] if f.name else [])
        # (a caller with nothing wrong to point at just answers the most useful thing they know)
        parts = []
        if "name" in fix and f.name:
            parts.append(f"My name is {f.name_spoken}" + (f", that's {spell_out(f.name)}." if self.quirk("spells_name") else "."))
        if "no_number" in fix:
            parts.append("Please take that number off, I didn't give you one.")
        if "number" in fix and f.number:
            parts.append(f"The number is {number_words(f.number)}.")
        return " ".join(parts) or "No, that's fine, never mind."


# ---------------------------------------------------------------- audio

def silence_wav(path: Path, seconds: float = 1.0, rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))


def build_filter(noise: float, phone: bool, seed: int) -> tuple[list[str], str]:
    """The same noise + phone-band filter graph as Part 1's testset/generate.py (kept in step by a test).

    A copy rather than an import: that file imports the Piper package at load time and is a script, and a
    dozen lines are cheaper than a hidden dependency. Returns (extra ffmpeg inputs, filter_complex ending in [out]).
    """
    extra_inputs: list[str] = []
    chain = "[0:a]"
    graph = []
    if noise > 0:
        extra_inputs = ["-f", "lavfi", "-i", f"anoisesrc=color=pink:amplitude={noise}:seed={seed}:sample_rate=22050"]
        graph.append(f"{chain}[1:a]amix=inputs=2:duration=first:normalize=0[mix]")
        chain = "[mix]"
    if phone:
        graph.append(f"{chain}highpass=f=300,lowpass=f=3400,aresample=8000[out]")  # a phone line carries ~300-3400 Hz
    else:
        graph.append(f"{chain}anull[out]")
    return extra_inputs, ";".join(graph)


def degrade(clean_wav: Path, out_wav: Path, noise: float, phone: bool, seed: int, filter_builder=build_filter) -> None:
    """Clean speech -> what a phone call sounds like, as a 16 kHz mono wav (what Whisper reads)."""
    extra_inputs, graph = filter_builder(noise, phone, seed)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(clean_wav), *extra_inputs,
           "-filter_complex", graph, "-map", "[out]", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out_wav)]
    subprocess.run(cmd, check=True)


def piper_synth(text: str, speaker: int, speed: float, out_wav: Path) -> None:
    from shared.tts import load_voice, synthesize  # lazy: needs the Piper voice, which tests do not
    synthesize(load_voice(), text, speaker, out_wav, speed=speed)


def render_turn(card: Card, text: str, out_wav: Path, synth=piper_synth, filter_builder=build_filter) -> Path:
    """The caller's words as a wav in the caller's voice. Empty text = a second of silence (a caller who says nothing)."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if not text.strip():
        silence_wav(out_wav)
        return out_wav
    seed = zlib.crc32(card.id.encode("utf-8"))  # same card, same noise
    with tempfile.TemporaryDirectory() as tmp:
        clean = Path(tmp) / "clean.wav"
        synth(text, card.voice.speaker, card.voice.speed, clean)
        degrade(clean, out_wav, card.voice.noise, card.voice.phone, seed, filter_builder)
    return out_wav


# ---------------------------------------------------------------- command line

def preview(card: Card) -> None:
    caller = SimulatedCaller(card)
    print(f"{card.id} [{card.split}] quirks={card.quirks}\n  {card.about}")
    for asked in (asks.GREETING, asks.REASON, asks.NAME, asks.SPELLING, asks.NUMBER, asks.NUMBER_AGAIN, asks.CONFIRM,
                  asks.ANYTHING_ELSE):
        print(f"  {asked:14} -> {caller.reply(asked) or '(silence)'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preview", metavar="CARD", help="print what a card says to every question (id prefix, or 'all')")
    parser.add_argument("--audio", metavar="CARD", help="render the card's opening turn to testset/audio/<id>/")
    args = parser.parse_args()
    cards = load_cards()
    wanted = args.preview or args.audio
    if not wanted:
        parser.print_help()
        return 0
    chosen = [c for c in cards if wanted == "all" or c.id.startswith(wanted)]
    if not chosen:
        sys.exit(f"no card starts with {wanted!r}")
    for card in chosen:
        if args.preview:
            preview(card)
        else:
            out = render_turn(card, SimulatedCaller(card).reply(asks.GREETING), HERE / "testset" / "audio" / card.id / "00_greeting.wav")
            print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
