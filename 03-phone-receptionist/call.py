"""One call, end to end (also a command line: python 03-phone-receptionist/call.py --card c14 --understand rules): the receptionist and a caller talk until the call ends, and everything is saved as JSON.

    record = run_call(card, persona, faq, understand_fn)                    # text only: the caller's words go straight in
    record = run_call(card, persona, faq, understand_fn, channel=audio)     # full audio loop: voice -> Whisper -> reply -> voice
    save_record(record, Path("03-phone-receptionist/calls"))

Two ways to carry the words, one dialog:
  TextChannel   the simulated caller's text goes straight to the dialog. No audio, no models: what the tests and the
                quick evaluation runs use.
  AudioChannel  the caller's text is spoken by Piper in the card's voice (with noise and a phone-line sound), listened to
                by Whisper, and the reply is spoken back, exactly like the push-to-talk page. Laptop only.

The loop asks the simulated caller to answer the question the receptionist is waiting for (state.asking), so the conversation
is driven by the dialog's own state and never by parsing sentences. A call that has not ended after the turn limit plus a
margin is cut off and recorded as such: the loop itself can never run forever.
"""

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import asks  # noqa: E402
import dialog  # noqa: E402
from cards import Card  # noqa: E402
from faq import FaqEntry  # noqa: E402
from persona import Persona  # noqa: E402
from simulate import SimulatedCaller  # noqa: E402

SAFETY_MARGIN_TURNS = 3  # a call is cut off this many turns after the limit if the dialog somehow never ends


class TextChannel:
    """The caller's words reach the dialog as they are."""
    name = "text"

    def hear(self, card: Card, text: str, turn: int) -> tuple[str, object | None, dict]:
        return text, None, {}

    def speak(self, reply: str, turn: int) -> dict:
        return {}


class AudioChannel:
    """Full audio loop. All parts are injected, so tests use fakes and the laptop passes the real ones."""
    name = "audio"

    def __init__(self, workdir: Path, hint: str, speaker: int | None, render_fn, listen_fn, speak_fn, model: str | None = None):
        self.workdir, self.hint, self.speaker, self.model = workdir, hint, speaker, model
        self.render_fn, self.listen_fn, self.speak_fn = render_fn, listen_fn, speak_fn

    def hear(self, card: Card, text: str, turn: int) -> tuple[str, object | None, dict]:
        wav = self.render_fn(card, text, self.workdir / f"turn_{turn:02d}_caller.wav")
        heard = self.listen_fn(wav, hint=self.hint, model=self.model)
        return heard.text, heard, {"caller_audio": str(wav), "listen_s": heard.seconds}

    def speak(self, reply: str, turn: int) -> dict:
        if not reply.strip() or self.speaker is None:
            return {}
        wav = self.workdir / f"turn_{turn:02d}_receptionist.wav"
        seconds = self.speak_fn(reply, wav, self.speaker)
        return {"reply_audio": str(wav), "speak_s": seconds}


class CallRecord(BaseModel):
    """Saved JSON, one per call. Contains the caller's words: it lives in the git-ignored calls/ folder."""
    call_id: str
    card_id: str | None
    started: str
    channel: str
    understand: str                        # which understanding step produced the forms ("model" / "rules")
    prompt_version: str | None = None
    greeting: str
    turns: list[dict] = Field(default_factory=list)
    message: dict
    outcome: str | None
    urgent_flagged_at_turn: int | None
    cut_off: bool = False
    total_s: float
    final_state: dict


def run_call(card: Card, persona: Persona, faq: dict[str, FaqEntry], understand_fn, channel=None, decide_fn=dialog.decide_a,
             on_urgent=None, understand_label: str = "model", prompt_version: str | None = None) -> CallRecord:
    """Run one simulated call to its end. `on_urgent(state)` is called once, the turn the call is first flagged urgent
    (Phase 4 plugs the immediate push in there)."""
    channel = channel or TextChannel()
    caller = SimulatedCaller(card)
    greeting, state = dialog.start_call(persona)
    channel.speak(greeting, 0)
    reply, started = greeting, time.perf_counter()
    urgent_turn, cut_off = None, False

    while state.state != dialog.ENDED:
        if state.turns >= persona.max_turns + SAFETY_MARGIN_TURNS:
            cut_off = True
            break
        asked = state.asking
        said = caller.reply(asked, heard=reply if asked == asks.CONFIRM else None)
        text, heard, caller_io = channel.hear(card, said, state.turns + 1)
        reply, state = dialog.next_reply(state, text, persona, faq, understand_fn, decide_fn, heard=heard)
        entry = state.log[-1]
        entry.update(asked=asked, caller_said=said, **caller_io, **channel.speak(reply, state.turns))
        if state.urgent and urgent_turn is None:
            urgent_turn = state.turns
            if on_urgent:
                on_urgent(state)

    return CallRecord(
        call_id=f"{card.id}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}", card_id=card.id,
        started=datetime.now(timezone.utc).isoformat(timespec="seconds"), channel=channel.name, understand=understand_label,
        prompt_version=prompt_version, greeting=greeting, turns=state.log, message=dialog.message_of(state),
        outcome=state.outcome if not cut_off else "cut_off", urgent_flagged_at_turn=urgent_turn, cut_off=cut_off,
        total_s=round(time.perf_counter() - started, 3), final_state=state.model_dump(exclude={"log"}))


def save_record(record: CallRecord, folder: Path) -> Path:
    """calls/<id>.json, written through a temp name so a crash never leaves half a record."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{record.call_id}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def print_transcript(record: CallRecord, persona: Persona) -> None:
    print(f"\n=== {record.card_id}  [{record.channel}, understanding: {record.understand}]")
    print(f"HOLLY : {record.greeting}")
    for t in record.turns:
        print(f"CALLER: {t['caller_text'] or '(silence)'}")
        print(f"HOLLY : {t['reply'] or '(call ended)'}")
    m = record.message
    print(f"--- outcome={record.outcome} urgent={m['urgent']} name={m['name']!r} number={m['number']!r} reason={m['reason']!r} "
          f"faq={m['faq_answered']} passed-on={m['unanswered_questions']} turns={len(record.turns)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run simulated calls (a caller card talks to the receptionist).")
    parser.add_argument("--card", default="dev", help="a card id prefix (c14), 'dev', 'score' or 'all' (default: dev)")
    parser.add_argument("--understand", choices=["model", "rules"], default="model",
                        help="model = the LLM fills the per-turn form (needs Ollama); rules = plain-code baseline, no model")
    parser.add_argument("--audio", action="store_true", help="speak and listen for real (Piper + Whisper): laptop only")
    parser.add_argument("--save", action="store_true", help="save each call as JSON in 03-phone-receptionist/calls/ (git-ignored)")
    args = parser.parse_args(argv)

    from cards import load_cards
    from faq import load_faq
    from persona import load_persona
    from turn import PROMPT_VERSION, Understanding, understand

    persona, faq = load_persona(), load_faq()
    cards = load_cards()
    chosen = [c for c in cards if args.card in ("all", c.split) or c.id.startswith(args.card)]
    if not chosen:
        sys.exit(f"no card matches {args.card!r}")
    if args.understand == "rules":
        from rules_turn import rules_understand
        understand_fn, label, version = (lambda text, ctx: Understanding(rules_understand(text, ctx))), "rules", None
    else:
        understand_fn, label, version = understand, "model", PROMPT_VERSION

    for card in chosen:
        channel = None
        if args.audio:
            import tempfile
            from audio_io import listen, piper_synth, speak
            from persona import PersonaError, speaker_id
            from simulate import render_turn
            try:
                speaker = speaker_id(persona)
            except PersonaError as problem:
                print(f"(no receptionist voice yet: replies stay text. {problem})")
                speaker = None
            channel = AudioChannel(Path(tempfile.mkdtemp(prefix=f"{card.id}_")), persona.hint, speaker, render_turn, listen,
                                   lambda text, out, spk: speak(text, out, spk, synth_fn=piper_synth))
        record = run_call(card, persona, faq, understand_fn, channel=channel, understand_label=label, prompt_version=version)
        print_transcript(record, persona)
        if args.save:
            print("saved", save_record(record, HERE / "calls"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
