"""Push-to-talk page: press the microphone, speak, hear the receptionist answer.

    python -m streamlit run 03-phone-receptionist/app.py

A thin shell: every decision lives in session.py / the dialog. Needs a microphone in the browser, the Whisper model,
the Piper voice and a chosen receptionist voice (persona.json 'piper_speaker', or RECEPTIONIST_SPEAKER=20 for one run;
without a voice the page still shows what was heard, as text).

Until the dialog engine is wired in (Phase 3) the replies are a fixed sequence of persona lines: the page is for checking
what Whisper hears and how long each stage takes.
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from persona import PersonaError, load_persona, speaker_id  # noqa: E402
from session import AudioSession, scripted_responder  # noqa: E402

WHISPER_CHOICES = ["base", "small"]


def start_call(persona, model: str, speaker: int | None) -> AudioSession:
    session = AudioSession(scripted_responder(persona), Path(tempfile.mkdtemp(prefix="call_")), hint=persona.hint,
                           speaker=speaker, model=model)
    session.greet(persona.say("greeting"))
    st.session_state["play_turn"] = 0
    return session


def show_turns(session: AudioSession) -> None:
    for turn in session.turns:
        if turn.caller_text or turn.ignored:
            with st.chat_message("user"):
                st.write(turn.caller_text or f"(nothing heard: {turn.ignored_why})")
                if turn.ignored and turn.raw_text:
                    st.caption(f"Whisper wrote {turn.raw_text!r}, ignored")
        with st.chat_message("assistant"):
            st.write(turn.reply_text)
            if turn.timing:
                t = turn.timing
                st.caption(f"listen {t.listen_s:.1f} s · respond {t.respond_s:.1f} s · speak {t.speak_s:.1f} s · "
                           f"total {t.total_s:.1f} s")
            if turn.reply_wav:
                st.audio(str(turn.reply_wav), autoplay=st.session_state.get("play_turn") == turn.n)
    st.session_state["play_turn"] = None


def main() -> None:
    st.set_page_config(page_title="Receptionist", page_icon="☎")
    st.title("Receptionist: push-to-talk")
    persona = load_persona()
    model = st.sidebar.selectbox("Whisper model", WHISPER_CHOICES, index=1)
    try:
        speaker = speaker_id(persona)
    except PersonaError as problem:
        speaker = None
        st.warning(f"No receptionist voice yet, replies are shown as text only. {problem}")
    st.sidebar.caption("Prototype: replies are fixed persona lines until the dialog engine is connected.")

    if st.button("Start a call"):
        st.session_state["session"] = start_call(persona, model, speaker)
        st.session_state["mic"] = 0
    session: AudioSession | None = st.session_state.get("session")
    if session is None:
        st.info("Press 'Start a call', then hold the microphone button while you speak.")
        return

    show_turns(session)
    clip = st.audio_input("Your turn: press, speak, press again", key=f"mic_{st.session_state.get('mic', 0)}")
    if clip is not None:
        with st.spinner("Listening..."):
            turn = session.hear(clip.getvalue())
        st.session_state["play_turn"] = turn.n
        st.session_state["mic"] = st.session_state.get("mic", 0) + 1  # a new, empty microphone widget
        st.rerun()
    median = session.median_total_s()
    if median is not None:
        st.sidebar.metric("Median wait for an answer", f"{median:.1f} s", help="Target: 5 s or less")
        st.sidebar.caption("Prototype: no understanding step yet and, without a voice, no speaking step, so this understates the real wait.")


main()
