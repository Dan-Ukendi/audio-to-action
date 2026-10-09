"""Push-to-talk page: press the microphone, speak, hear the receptionist answer.

    python -m streamlit run 03-phone-receptionist/app.py

A thin shell: every decision lives in session.py / the dialog. Needs a microphone in the browser, the Whisper model,
the Piper voice and a chosen receptionist voice (persona.json 'piper_speaker', or RECEPTIONIST_SPEAKER=20 for one run;
without a voice the page still shows what was heard, as text).

The replies come from the real dialog (version A, the state machine, or B, the agent). The sidebar shows what the receptionist
has understood so far and the median wait for an answer. A finished call is not handed to Part 1 from this page: use
call.py / handoff.py for that.

The same page also runs as a tab of the local app (app/app.py): render(embedded=True) puts the settings in the tab
instead of the sidebar and leaves the page title and config to the host.
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from persona import PersonaError, load_persona, speaker_id  # noqa: E402
from faq import load_faq  # noqa: E402
from session import AudioSession, dialog_responder  # noqa: E402

WHISPER_CHOICES = ["base", "small"]


def make_understand(kind: str):
    """'model' = the LLM fills the per-turn form (Ollama); 'rules' = the plain-code baseline (no model, for trying the page)."""
    from turn import Understanding, understand
    if kind == "rules":
        from rules_turn import rules_understand
        return lambda text, ctx: Understanding(rules_understand(text, ctx))
    return understand


def make_decide(kind: str):
    if kind == "B (agent)":
        from agent_dialog import make_decide_b
        return make_decide_b()
    import dialog
    return dialog.decide_a


def start_call(persona, model: str, speaker: int | None, understanding: str, decision: str) -> AudioSession:
    responder = dialog_responder(persona, load_faq(), make_understand(understanding), make_decide(decision))
    session = AudioSession(responder, Path(tempfile.mkdtemp(prefix="call_")), hint=persona.hint, speaker=speaker, model=model)
    session.responder = responder
    session.greet(responder.greeting)
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


def side_panel(state, ui) -> None:
    """What the receptionist has understood so far (only what the caller said)."""
    ui.subheader("Understood so far")
    for detail in ("reason", "name", "number"):
        ui.write(f"**{detail}:** {state.value(detail) or '-'}")
    ui.write(f"**urgent:** {'yes' if state.urgent else 'no'}")


def render(embedded: bool = False) -> None:
    """The whole page. embedded=True: inside a tab of the local app (settings in an expander, no title/config)."""
    if embedded:
        st.subheader("Receptionist: push-to-talk")
        ui = st.expander("Settings and what Holly has understood", expanded=True)
    else:
        st.set_page_config(page_title="Receptionist", page_icon="☎")
        st.title("Receptionist: push-to-talk")
        ui = st.sidebar
    persona = load_persona()
    model = ui.selectbox("Whisper model", WHISPER_CHOICES, index=1)
    try:
        speaker = speaker_id(persona)
    except PersonaError as problem:
        speaker = None
        st.warning(f"No receptionist voice yet, replies are shown as text only. {problem}")
    understanding = ui.selectbox("Understanding", ["model", "rules"], help="model needs Ollama; rules is the no-model baseline")
    decision = ui.selectbox("Decision", ["A (state machine)", "B (agent)"])

    if st.button("Start a call"):
        st.session_state["call_session"] = start_call(persona, model, speaker, understanding, decision)
        st.session_state["mic"] = 0
    session: AudioSession | None = st.session_state.get("call_session")
    if session is None:
        st.info("Press 'Start a call', then hold the microphone button while you speak.")
        return

    show_turns(session)
    state = session.responder.state
    side_panel(state, ui)
    if state.state == "ENDED":
        st.success(f"The call has ended ({state.outcome}). Press 'Start a call' for a new one.")
        return
    clip = st.audio_input("Your turn: press, speak, press again", key=f"mic_{st.session_state.get('mic', 0)}")
    if clip is not None:
        with st.spinner("Listening..."):
            turn = session.hear(clip.getvalue())
        st.session_state["play_turn"] = turn.n
        st.session_state["mic"] = st.session_state.get("mic", 0) + 1  # a new, empty microphone widget
        st.rerun()
    median = session.median_total_s()
    if median is not None:
        ui.metric("Median wait for an answer", f"{median:.1f} s", help="Target: 5 s or less")
        if not session.speaker:
            ui.caption("No voice chosen: the speaking step is missing, so this understates the real wait.")


if __name__ == "__main__":  # `streamlit run` executes the script as __main__; the local app imports render() instead
    render()
