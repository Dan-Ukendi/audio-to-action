"""Local app: add audio, run the pipelines, look at the results. Everything stays on this computer.

    python -m streamlit run app/app.py        (from the repo root; opens http://localhost:8501)

Only layout lives here. Saving uploads, starting runs and reading results is in helpers.py; the real work is
done by the same run.py scripts you can run from the terminal.
"""

import sys
from datetime import date
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import helpers as h  # noqa: E402

st.set_page_config(page_title="Audio to action", page_icon="🎧", layout="wide")


# ---------------------------------------------------------------- shared pieces

@st.cache_data(ttl=30, show_spinner=False)
def environment_problems() -> list[str]:
    return h.environment_problems()  # cached: checking Ollama on every click would slow the page down


def sidebar() -> None:
    st.sidebar.title("🎧 Audio to action")
    st.sidebar.caption("Local pipelines for voicemails and meetings. Nothing leaves this computer.")
    problems = environment_problems()
    if problems:
        for p in problems:
            st.sidebar.error(p)
    else:
        st.sidebar.success("ffmpeg and Ollama are ready.")
    free = h.free_ram_gb()
    if free is not None and free < 6:
        st.sidebar.warning(f"Only {free:.1f} GB of memory is free. The pipeline needs about 6 GB "
                           "(Whisper ~1 GB + the language model ~5 GB): close other apps first, or runs may fail "
                           "with a memory error.")
    if st.sidebar.button("Check again"):
        environment_problems.clear()
        st.rerun()
    st.sidebar.info("On this laptop's CPU (GPU off) a voicemail takes ~1-3 min and a meeting ~5-20 min. "
                    "Runs continue in the background; you can keep using the page.")


def add_files(part: h.Part, label: str, meeting_day: date | None = None) -> None:
    """File picker + 'Add to inbox' button. The uploader key changes after a save, which empties it."""
    key = f"uploader_{part.key}_{st.session_state.get(f'uploads_{part.key}', 0)}"
    uploads = st.file_uploader(label, type=h.AUDIO_TYPES, accept_multiple_files=True, key=key)
    if uploads and st.button(f"Add {len(uploads)} file(s) to the inbox", key=f"add_{part.key}", type="primary"):
        for upload in uploads:
            name = h.with_meeting_date(upload.name, meeting_day) if meeting_day else upload.name
            saved = h.save_upload(upload.getvalue(), name, part.inbox)
            st.toast(f"Added {saved.name}")
        st.session_state[f"uploads_{part.key}"] = st.session_state.get(f"uploads_{part.key}", 0) + 1
        st.rerun()
    waiting = h.files_in(part.inbox)
    if waiting:
        st.caption("Waiting in the inbox: " + ", ".join(p.name for p in waiting))


def run_controls(part: h.Part, extra_args: list[str] | None = None) -> None:
    """Start / retry buttons and the live log of the latest run."""
    log = h.latest_run(part)
    running = log is not None and h.run_status(log) == "running"
    waiting, failed = h.files_in(part.inbox), h.files_in(part.failed)
    blocked = bool(environment_problems())

    col1, col2 = st.columns(2)
    if col1.button(f"▶ Process the inbox ({len(waiting)} file(s))", key=f"run_{part.key}", type="primary",
                   disabled=running or not waiting or blocked, width="stretch"):
        h.start_run(part, extra_args)
        st.rerun()
    if col2.button(f"↻ Retry failed ({len(failed)})", key=f"retry_{part.key}",
                   disabled=running or not failed or blocked, width="stretch"):
        h.start_run(part, ["--retry-failed", *(extra_args or [])])
        st.rerun()

    if log is None:
        return

    # While a run is busy, this part of the page refreshes itself every 2 seconds.
    @st.fragment(run_every=2 if running else None)
    def live_log() -> None:
        status = h.run_status(log)
        st.caption(("⏳ Running…" if status == "running" else "✅ Last run finished") + f"  ·  log: app/runs/{log.name}")
        st.code(h.read_log(log, 40) or "(starting…)", language=None)
        if running and status == "finished":
            st.rerun(scope="app")  # refresh the results below

    with st.expander("Pipeline log", expanded=running):
        live_log()


def show_failed(part: h.Part) -> None:
    notes = h.failed_notes(part)
    if notes:
        with st.expander(f"⚠️ {len(notes)} file(s) failed"):
            for n in notes:
                st.markdown(f"**{n['file']}**: failed at step `{n['step']}`")
                st.code(n["error"], language=None)


# ---------------------------------------------------------------- voicemails (Part 1)

ROUTE_LABEL = {"notify_now": "🚨 urgent (push)", "inbox": "📥 call back", "personal": "👤 personal", "archive": "🗄️ archived"}


def voicemails_tab() -> None:
    st.subheader("1 · Add voicemails")
    add_files(h.VOICEMAILS, "Drop voicemail audio files here")

    st.subheader("2 · Run the pipeline")
    st.caption("transcribe → analyze (LLM fills a form, checked by code) → route (plain rules) → push + save")
    run_controls(h.VOICEMAILS)
    show_failed(h.VOICEMAILS)

    st.subheader("3 · Results")
    rows = h.voicemails()
    if not rows:
        st.info("No voicemails processed yet.")
        return
    cols = st.columns(5)
    for col, (route, label) in zip(cols, ROUTE_LABEL.items()):
        col.metric(label, sum(r["route"] == route for r in rows))
    cols[4].metric("🔎 needs review", sum(r["review"] for r in rows))

    route_filter = st.segmented_control("Show", ["all", *ROUTE_LABEL.values(), "needs review"], default="all",
                                        key="vm_filter")
    shown = [r for r in rows if route_filter in (None, "all") or ROUTE_LABEL[r["route"]] == route_filter
             or (route_filter == "needs review" and r["review"])]
    table = [{"file": r["source_file"], "route": ROUTE_LABEL[r["route"]], "category": r["category"],
              "urgency": r["urgency"], "caller": r["caller_name"] or "-", "number": r["callback_number"] or "-",
              "review": "🔎" if r["review"] else "", "summary": r["summary"]} for r in shown]
    event = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun",
                         selection_mode="single-row", key="vm_table")
    if not event.selection.rows:
        st.caption("Click a row to see the details.")
        return
    voicemail_details(shown[event.selection.rows[0]])


def voicemail_details(r: dict) -> None:
    st.divider()
    st.markdown(f"#### {r['source_file']}")
    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"**{ROUTE_LABEL[r['route']]}** · category **{r['category']}** · urgency **{r['urgency']}**/3")
        st.markdown(f"**Summary:** {r['summary']}")
        st.markdown(f"**Caller:** {r['caller_name'] or 'not said'} · **Number:** {r['callback_number'] or 'not said'}")
        st.markdown("**Why this route:**\n" + "\n".join(f"- {reason}" for reason in r["reasons"]))
        st.markdown(f"**Push sent:** {r['notified_at'] or 'no (or dry run)'}")
    with right:
        audio = h.find_audio(h.VOICEMAILS, r["source_file"])
        if audio:
            st.audio(str(audio))
        st.caption(f"Transcribed with Whisper `{r['whisper_model']}`, analyzed with `{r['llm_model']}` "
                   f"(prompt {r['prompt_version']}), processed {r['processed_at']}")
    st.markdown("**Transcript**")
    st.text(r["transcript"])


# ---------------------------------------------------------------- meetings (Part 2)

STATUS_LABEL = {"open": "🟡 open", "done": "✅ done", "cancelled": "✖️ cancelled"}


def meetings_tab() -> None:
    st.subheader("1 · Add meeting recordings")
    day = st.date_input("Meeting date (used when the file name has no YYYY-MM-DD in it)", value=date.today(),
                        key="meeting_day")
    add_files(h.MEETINGS, "Drop meeting recordings here", meeting_day=day)
    st.caption("Meetings are processed oldest first: later meetings talk about tasks from earlier ones.")

    st.subheader("2 · Run the pipeline")
    mode = st.radio("How should the task tracker be updated?", ["agent", "rules"], horizontal=True, key="sync_mode",
                    captions=["LLM agent with tools and guards: better with messy wording, slow (minutes)",
                              "plain-code rules: instant, best when tasks are worded the same way"])
    st.caption("transcribe (+ known names) → extract action items (workflow) → sync the tracker (" + mode + ")")
    run_controls(h.MEETINGS, ["--sync", mode])
    show_failed(h.MEETINGS)

    st.subheader("3 · Results")
    tasks, meetings = h.tracker_tasks(), h.meetings()
    if not meetings:
        st.info("No meetings processed yet.")
        return
    tracker_view, meeting_view = st.tabs(["📋 Task tracker", "🗓️ Meetings"])
    with tracker_view:
        counts = {s: sum(t["status"] == s for t in tasks) for s in STATUS_LABEL}
        cols = st.columns(3)
        for col, (status, label) in zip(cols, STATUS_LABEL.items()):
            col.metric(label, counts[status])
        st.dataframe([{"#": t["id"], "status": STATUS_LABEL[t["status"]], "task": t["task"], "owner": t["owner"] or "?",
                       "due": t["due"] or "-", "added in": t["created_in"], "last change": t["updated_in"]}
                      for t in tasks], hide_index=True, width="stretch")
    with meeting_view:
        choice = st.selectbox("Meeting", [m["meeting"] for m in meetings],
                              format_func=lambda m: next(f"{x['meeting_date']} · {x['source_file']} · {x['items']} items · "
                                                         f"{x['sync_mode']}" for x in meetings if x["meeting"] == m))
        meeting_details(next(m for m in meetings if m["meeting"] == choice))


def meeting_details(m: dict) -> None:
    if m["review"]:
        st.warning("Left for a human to check: " + "; ".join(m["review"]))
    items = h.meeting_items(m["audio_sha256"])
    st.markdown("**Action items extracted from this meeting**")
    st.dataframe([{"status": STATUS_LABEL.get(i["status"], i["status"]), "task": i["task"], "owner": i["owner"] or "?",
                   "due": i["due"] or "-", "said": i["evidence"]} for i in items], hide_index=True, width="stretch")
    st.markdown("**What changed in the tracker, and why**")
    st.dataframe([{"task #": c["task_id"], "change": c["what"], "reason": c["reason"]} for c in h.changes(m["meeting"])],
                 hide_index=True, width="stretch")
    trace = h.agent_trace(m["meeting"])
    if m["sync_mode"] == "agent" and trace:
        with st.expander(f"🤖 Agent steps ({trace['llm_calls']} LLM calls, {trace['seconds']:.0f} s)"):
            for step in trace["trace"]:
                if step.get("tool"):
                    st.markdown(f"**step {step['step']}** · `{step['tool']}` {step.get('args', '')}")
                    st.caption(step.get("result", ""))
                else:
                    st.caption(f"step {step['step']}: {step.get('error') or step.get('said') or step.get('stopped', '')}")
    audio = h.find_audio(h.MEETINGS, m["source_file"])
    if audio:
        st.audio(str(audio))


# ---------------------------------------------------------------- how it works

def help_tab() -> None:
    st.markdown("""
**Voicemails** (Part 1, a fixed workflow): each file is transcribed by Whisper, the local LLM fills in a form
(category, urgency, caller, number, summary) that code checks, and plain `if` rules decide the route:
urgent → push notification, customer → call-back list, personal, or archive (sales / spam). A safety-word net
never archives "smell of gas", "burst", "sparking" and similar.

**Meetings** (Part 2, workflow + one agent): each recording is transcribed with the team's and customers' names
as a hint, the LLM extracts action items (task, owner, due date, status, quote), and the task tracker is updated
either by a bounded LLM agent or by plain rules. Every tracker change is stored with its reason.

**Re-adding the same audio does nothing**: files are recognised by their content, not their name.
If something fails, the file goes to *failed* with the reason; fix it and press **Retry failed**.

Command-line equivalents: `python 01-voicemail-triage\\run.py`, `python 02-meeting-action-agent\\run.py --sync agent`.
Details: `README.md`, `docs/learning-log.md`.
""")


# ---------------------------------------------------------------- page

sidebar()
voicemail_tab, meeting_tab, how_tab = st.tabs(["📞 Voicemails", "🗓️ Meetings", "❓ How it works"])
with voicemail_tab:
    voicemails_tab()
with meeting_tab:
    meetings_tab()
with how_tab:
    help_tab()
