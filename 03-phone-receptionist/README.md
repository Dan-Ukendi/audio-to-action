# Part 3: phone receptionist

A caller "phones" Brightwater Plumbing & Heating and talks to **Holly**, an automated receptionist. Holly greets
the caller (and says she is automated and that the call is recorded), answers simple questions from a fixed FAQ,
takes a message (reason, name, callback number), flags emergencies at once, and hands the message to the Part 1
pipeline. Everything is local and free; calls are simulated (a browser push-to-talk page, or a scripted synthetic caller).

The full brief from the owner is `docs/PART3_PLAN.md`. This file records what was built from it, the decisions
that were taken, and what still has to be run on the laptop.

Status: see the phase table (section 5). **Numbers that need the laptop's GPU, models or microphone are not in this file
because they have not been measured**; section 8 lists every command that produces them.

---

## 1. The problem in one picture

```
 caller turn (push-to-talk recording, or a synthetic caller's audio / text)
    │
    ▼
 1. LISTEN      ffmpeg → 16 kHz → Whisper (hint: team, places; never customer names) → text      shared/transcribe.py
                turns with under 0.3 s of speech, or only a doubtful invented phrase, are ignored (Whisper invents "Thank you." on silence)
 2. UNDERSTAND  LLM fills a per-turn form (CallerTurn): details given, correction, question, emergency, end    turn.py
                plain-code rules check it: UK number rule, the words must really have been said (grounding)
 3. DECIDE      version A: a state machine in plain code (dialog.py)      version B: a boxed tool-calling agent
                missing detail → ask · question → FAQ answer · details complete → read back · confirmed → goodbye
                safety words at ANY turn → urgent fast path (safety advice + immediate push)
 4. SPEAK       a fixed sentence (persona.json) or an approved answer (faq.json) → Piper → wav       shared/tts.py
    ⋮ repeat until the call ends or a limit is reached
 5. HAND OFF    call record (JSON) → Part 1 analyze → route → push if urgent → voicemails.db             handoff.py
```

The principle that shapes everything: **the dialog engine is plain Python with no screen and no microphone**
(`next_reply(state, caller_text) -> (reply_text, new_state)`). The synthetic-caller tests run headless, and the
browser page is a thin shell around it. **A model never writes what Holly says**: it only helps *understand* the caller.
Every sentence comes from `persona.json` or `faq.json`, so an invented price or opening hour is impossible by construction.

## 2. Decisions

### Owner's decisions (from the plan)
| Topic | Decision |
|---|---|
| How calls reach Holly | Local simulation, no phone number. Live: push-to-talk in the browser (Streamlit `st.audio_input`). Tests: scripted synthetic caller. |
| Holly's job | Take a message (name, callback number, reason, urgency) and answer simple questions from `faq.json`. |
| Order | State machine first (version A); a tool-calling agent (version B) later as a measured experiment. |
| Speed | GPU being fixed by the owner; everything is re-measured in Phase 2 on the laptop. |

### Decisions taken without the owner
The run that built this was fully autonomous (the owner reviews at the end). Wherever the plan said "the owner's call" the
plan's default, or the most conservative option, was used. **Change any of these and re-run the affected phase.**

| # | Topic | Decision taken | Why / how to change |
|---|---|---|---|
| 1 | Persona and wording | Holly, "the automated assistant of Brightwater Plumbing & Heating". Greeting says she is automated and the call is recorded. All fixed sentences are in `persona.json`. | The plan says the wording is confirmed in Phase 0; the owner's brief gave name and role, the sentences are mine. Edit `persona.json` (a test checks every line is present and speakable). |
| 2 | FAQ content | 23 invented entries in `faq.json` for the fictional business (Kelmbridge, Unit 4 Mill Lane Trading Estate, 01632 960000 from Ofcom's fictional range, `.example` e-mail and website, villages Wexley Green / Harrowfield / Thornsby Cross / Overmere). Prices, times and policies are exactly the list in the owner's run instructions (copied into section 2.1 below so they can be checked). "Work we do not do" = electrical work, roofing, drain unblocking, gardening (mine). Only the three safety entries carry real UK advice (National Gas Emergency 0800 111 999, 999, stopcock and electrics). | Every price is a fiction; the safety numbers are real. Edit `faq.json`; tests check the facts against the brief. |
| 3 | Draft definition of done | Kept exactly as drafted in the plan (section 8 of `docs/PART3_PLAN.md`; section 6 below). | Nothing to confirm without the owner. |
| 4 | GPU check (`nvidia-smi`, `ollama ps`) | Not possible in the cloud session. Left as the first laptop command (section 8). | |
| 5 | Git author e-mail | `dan.ukendi1@gmail.com`, as in the existing history and the run instructions. The plan text as pasted had `danukendi1@gmail.com`, which looks like a typo, so the committed plan uses the history's spelling. | If `danukendi1` really was meant, say so: every commit would have to be re-authored. |
| 6 | Plan mentions an `app/` folder (and `start_app.bat`) | Neither is in the git repository (they may exist untracked on the laptop). The browser page will be built inside `03-phone-receptionist/` (`app.py`, with all logic in `session.py`; both written in Phases 2 and 7). | Move the page into `app/` if that folder exists on the laptop. |
| 7 | Plan's `context.py` | Not created under that name: Part 2 already has a `context.py` and the flat module style would clash in one test run. The Whisper hint is built in `persona.py`. **The hint never contains customer names**: the caller's name is a scored field, and hinting the test callers' names would inflate name accuracy (the Part 2 "hint is only proper nouns" lesson, taken one step further). The team's first names *are* hinted, and three Part 1 callers share them (Priya Shah, Tom Bradley, Jamie): their first names are therefore slightly easier; surnames are never hinted. | |
| 8 | Receptionist voice | `piper_speaker: null` in `persona.json`, chosen on the laptop in Phase 2 (`choose_voice.py`, written in Phase 2, renders the candidates). Must differ from Part 2 (7, 0, 9, 11), Part 1 callers (3, 5, 11, 15, 25, 27, 33, 40, 45, 50, 55, 60, 64, 72, 80, 90) and the Part 3 FAQ callers (18, 22, 29, 38, 47, 58); the persona loader refuses all of those. `RECEPTIONIST_SPEAKER` overrides it for one run. | Taste decision; there is no ear in the cloud. |
| 9 | Spelling | When the caller has given a first and last name, Holly asks once for the *full name* to be spelled (unless they already spelled it). A single first name is not spelled. | The two hardest Part 1 names are first names (Siobhan, Wojciech), so spelling only the surname would not help. Costs one turn of the 12. The spelled letters are matched back to the heard name in code (Phase 3). |
| 10 | Silence rule | "Silent twice" = **two empty turns in a row** (a turn with speech resets the count). | Friendlier; the 12-turn limit bounds the call anyway. |
| 11 | Calls in `voicemails.db` | Part 1's table has no "source" column and Part 1's `save()` uses positional columns, so it is **not changed**. A call is stored with `source_file = "call-<id>.json"`; `source_file LIKE 'call-%'` finds them. | Conservative: no schema migration of a database the owner already uses. |
| 12 | Part 1 analysis of a call (confirmed in Phase 4) | `analyze(..., cache_dir=None, prompt_version="call-v1")` on the **caller's side only** (one answer per line). `call-v1` = Part 1's v2 prompt + three sentences saying it is a phone call; added to `shared/analyze.py` as a NEW prompt version (v2/v3 untouched) and its user-message label says "Phone call" instead of "Voicemail transcript". The dialog's own verified name and number replace the model's for those two fields; a call flagged urgent during the call stays urgent. | No cache: its key has no hint fingerprint (HANDOVER 9). |
| 13 | Test cards: dev and score split | Each caller card has `split: dev` or `score`. Prompts and rules may be tuned on `dev` cards only; `evaluate.py` scores `score` cards by default. | Plan lesson: never tune on the cards you score with. |
| 14 | Streamlit version | `streamlit>=1.40` in `requirements.txt` (`st.audio_input` appeared in 1.39). The laptop has 1.65. | |
| 15 | Which callers, and what they do | The 18 Part 1 scenarios become calls (same speakers, same labels). The 6 FAQ callers, their questions, all scripted openings and every quirk are mine; see `testset/README.md`. 9 dev cards (c03, c07, c11, c13, c14, c18, f01, f03, f04) cover the core behaviours once (emergency, refused number, robocall, wrong-then-corrected number, spelled name, in-sentence correction, a question, info-only, an unanswerable question); the other 15 are for scoring only. Behaviours only the scoring run sees: a caller who withholds the name or number, an all-in-one opening, a rambling opening, two questions in one turn and a new request after "anything else?". | The plan asked for "4-6 FAQ callers"; replace any card, `cards.py` re-checks the key. |
| 16 | Callers who only ask questions | If a caller's first turn is only FAQ questions (no request), Holly answers and asks "anything else?" instead of pressing for a message; a new request after that goes back to collecting details. An unanswerable question is passed on in the message. | The plan does not say when a message is required. Such a call ends `info_only`. |
| 17 | The simulated caller checks the read-back | It notices a wrong name or number in Holly's read-back and corrects it (at most twice), like a real caller. Without this the correction flow would never run on the cards. | Deterministic. It checks the name (whole words) and the number; a caller who gave no name cannot object to an invented one (Holly's sentence would have to be parsed). |
| 18 | The page before the dialog exists | In Phase 2 `app.py` replied with a fixed sequence of persona lines. **Since Phase 7 it runs the real dialog** (sidebar: understanding model/rules, decision A/B, what was understood so far). Without a chosen voice it shows replies as text. A finished call is not handed to Part 1 from the page (use `call.py` / `handoff.py`). | Plan: "push-to-talk page". |
| 19 | When a turn counts as silence | Under 0.3 s of speech; or every segment above 0.6 no-speech probability; or the text is just "thank you / thanks (for watching) / you / bye / please subscribe" **and Whisper doubts it**: under 1.5 s of speech with confidence below -0.8, or (any length) a segment with no-speech probability above 0.3. A confident "Thank you." and a bare "Yes." / "Okay." are real answers and are kept. | The plan says "ignore turns under ~0.5 s"; 0.3 s keeps a one-word answer. The phrase list and the doubt rule are mine. All thresholds are constants at the top of `audio_io.py`. |
| 20 | One extra field in the per-turn form | `is_automated` (robocall, recorded message, scam) is added to the plan's `CallerTurn` sketch, because "robocall -> short goodbye" needs a detector. Code also recognises "press one", "automated message", "recorded message" and "legal proceedings" without the model. | Spam is decided before urgency, so a robocall that says "urgent" three times is never flagged. |
| 21 | Question, request or small talk | The FAQ answers question-shaped sentences only. "Can you give me a call back?" is a request (never an unknown question); "Hello?" and "Pardon?" are small talk; an information question with no FAQ entry is passed on in the message ("I'll pass your question on"). | Keywords first, the model may only pick among the entries (decision 25). |
| 22 | When the model fails | After two invalid answers, an unreachable Ollama or an Ollama server error (HTTP 500 "runner terminated", 404 "model not found") the plain-code baseline (`rules_turn.py`) understands the turn, the record says `fallback: true`, and the call goes on. | A call must never end because a model had a bad moment. |
| 23 | What is stored as the reason | The model's short phrase if its content words are in the speech; otherwise the caller's own first sentence. Never a phrase the caller did not say. | Reads less neatly in the read-back, but is never invented. |
| 24 | Refusals and re-asks | After a refused name or number: one more try, then recorded as "not given". Any other missing detail: asked up to 3 times (first ask + 2 re-asks), then dropped. | The plan's "ask again once" and "max 2 re-asks per detail". |
| 25 | The model's FAQ pick | Used only when no keyword matches; if keywords and model disagree, the keywords win (their facts come from the caller's words). | |
| 26 | A topic answered once is not answered again in the same call | The caller is probably repeating themselves. | |
| 27 | Tests use the rules baseline as the understanding step | Headless call tests (all 24 cards) check invariants (a call always ends, nothing is invented, emergencies fire, every FAQ answer is a real entry) and make exact assertions on **dev cards only**. They make **no accuracy claim**: accuracy needs the model (Phase 6). | |
| 28 | Rules and keywords were generalised, not fitted to scoring cards | A review found phrases copied from score cards (c04, c12, c17, f05). They were removed: `final notice` (robocall), `on your system`, `he'll know`, `just say it's me` (refusals) and the payment keywords `pay by card` / `can i pay` / `pay by`. A caller who says "can I pay by card?" is now answered only if the model picks the `payment` entry, which Phase 6 will measure honestly. Keywords added from the **dev** card f01 (`how soon could someone`, ...) stay. | If you want those phrases back, add them knowing they were seen in the scoring cards. |
| 29 | A second request after the message is complete | It is **added** to the message (`extra_requests`, at most 3), never lost, and never replaces the first reason. An emergency that comes up late is flagged and advised at once. | |
| 30 | A robocall is not hung up on if safety words are in it | "final notice ... I can smell gas" is a person, whatever the pattern or the model says. | Safety beats spam. |
| 31 | Partial number correction | "No, it's 349" / "it ends in three four nine" replaces the matching END of the stored number; every digit comes from the stored number or this turn. The digits after "not" are ignored. | The plan's own example. |
| 32 | More fixed sentences | Four added to `persona.json`: `what_else` (a plain "yes" to "anything else?"), `goodbye_info` (no message was taken), `turn_limit_urgent` and `silence_end_urgent` (an urgent call always ends with the callback promise and the 999 line). | |
| 33 | Naming the wrong detail | "No, the number is wrong" (or "Number." after "what should I change?") makes Holly ask for exactly that detail; the answer then changes only that detail. A bare "no" asks "what should I change?" up to twice. | |
| 34 | Read-backs are capped | The message is read back at most 1 + 2 times; a caller who keeps saying no, or says goodbye at the read-back, gets a goodbye and the message stays unconfirmed (`caller_ended`). A "yes" that contains a late "no" ("Yeah no, that's wrong") is a no. | Max 2 re-asks per detail applies to read-backs too. |
| 35 | Late corrections | After "anything else?" a corrected or late name/number ("actually my number is ...") is stored and the message is read back again; a plain "yes" asks "what else?" once. | |
| 36 | Calls that need no model | Robocalls, info-only calls and silent calls get their analysis from plain code (exact, instant). Only calls that carry a message use the model. An info-only call is stored, but is not flagged "missing name/number" (Part 1's rule would, wrongly, for a caller who only asked the opening hours). | Plan: "every completed call lands in `voicemails.db`": info-only and spam calls do too. |
| 37 | The push for an urgent call | Sent the moment the call is flagged urgent (`LivePush`, minimal text), in a **background thread with one attempt**: a slow ntfy server (up to 10 s) must never make the caller hear silence in an emergency. It never raises. Hand-off waits for it; if it failed or is still running, hand-off sends Part 1's usual push with Part 1's retries. A later hand-off pushes only if the earlier one did not (judged from the stored row: `notified_at`, route `notify_now`, or the safety-net reason). The one exception is the safe direction: if the background push is still running after 15 s, the end push goes out as well and the phone may ring twice, never zero times. Dry run until NTFY is configured (`NTFY_DRY_RUN=1`). | |
| 38 | When the model fails at hand-off | Plain code builds the analysis from the dialog's message and the row is flagged for review (`attempts = 2`, Part 1's "analysis needed a retry" rule). A bug (not a model/network error) is not swallowed. | |
| 39 | Re-handing a call off | One row per call (hash of the call id + the caller's words). A second hand-off **refreshes** the row (analysis, route, processed time) but never adds a row or a second push; if the first run did not push and the second finds urgency (for example `--analysis rules`, then `--analysis model`), the second pushes. The record is saved before the hand-off, so a failed push loses nothing: re-run `handoff.py`. | Like Part 1's "re-running on the same file does nothing", except that new information is kept. |
| 40 | What the agent (version B) is allowed to decide | Only `decide`: which legal action comes next. The state machine's own decision is always one legal option; three more are allowed where judgement can matter: ask what to change instead of repeating an unclear read-back, let go a caller whose "anything else?" answer is unclear, stop asking and read back after a refusal. Everything else (understanding, grounding, `apply_turn`, emergencies, spam, silence, limits, every sentence) is shared with version A. | The plan's tools `ask / answer_faq / read_back / flag_urgent / take_message / end_call`; `ask` also covers the follow-ups (`correction`, `anything_else`, `what_else`, `repeat`). |
| 41 | No tool takes a value | The agent cannot name, number or reason anything: those come from `apply_turn`, grounded in the caller's words. A test checks no tool has such a field. | "Values copied by code" from Part 2. |
| 42 | Fallback | If the agent fails (model error, prose instead of tools, step limit 4, no progress, only illegal choices) the state machine's own decision is used and the turn is logged `fallback: true` with the reason. The evaluation counts fallbacks (pre-registered rule: at most 10 % of turns). | A call never depends on the model behaving. |
| 43 | `flag_urgent` | The agent may flag a call urgent that the safety words missed (once; refused if already flagged). The acknowledgement and the urgent goodbye follow even if the agent then fails. The code-side safety floor (acknowledgement, advice, "I'll pass your question on") is said whatever the agent does. | |

| 44 | Which calls are scored | The `score` split by default (the 15 cards nobody tuned on); `--split dev` for tuning. Results are only written to the docs after a run with a real model. | The plan's rule: never tune on the cards used for scoring. |
| 45 | What counts as a correct detail | Name or number "correct" in the null-aware sense: a caller who gives none must end with none (numbers compared as digits). The A-vs-B quality margin counts name-correct + number-correct over all scored cards. | Decision 13 and the pre-registered rule. |
| 46 | Hand-off inside the evaluation | Every evaluated call is handed to Part 1 in an in-memory table with a recording sender: nothing touches `voicemails.db` or the phone. The push check compares with the card's Part 1 label (urgent = pushed, nothing else pushed). | Definition of done: urgent calls pushed, completed calls land in the table. |

(More rows are added below as later phases take decisions.)

### 2.1 The FAQ facts the owner specified (for checking `faq.json`)
Opening hours Mon-Fri 8:00-17:30, Sat 9:00-13:00, emergency 24/7 · emergency call-out within 2 h, 95 pounds out of hours
including the first hour, then 65 pounds an hour · area: Kelmbridge + 15 miles · boiler service 85 · landlord gas certificate 70 ·
new boilers: Worcester Bosch, Vaillant, free survey · bathrooms: free quote, 25 % deposit · 55 pounds an hour in office hours,
minimum 1 hour · free quotes · booking time 3-5 working days · payment by card or bank transfer · 12-month labour guarantee ·
Gas Safe registered and 5 million pounds insurance · cancellation: 24 h notice, otherwise 30 pounds · priority for vulnerable
customers · contact: 01632 960000 (Ofcom's fictional range), e-mail and website on the `.example` domain · callback within 2 h
in office hours · office: Unit 4, Mill Lane Trading Estate, Kelmbridge. Safety entries: gas smell and carbon monoxide
(National Gas Emergency 0800 111 999, 999 if someone is ill), water leak (stopcock, electrics).

## 3. Dialog rules (version A, plain code)
- One missing detail at a time, in the order reason → name → number (or whatever the caller already gave).
- **Never invent:** a detail is stored only if the caller said it in this call (the words must appear in the transcript; the
  number must pass the UK number rule). "You've got my number" → ask once more, then record "no number given".
- **Read-back** with digits in groups ("oh one six three two, nine six oh, five oh one"); "No, it's ..." re-opens only that field.
- **Questions** are answered only from `faq.json`; an unknown question → "I'll pass your question on", and it is added to the message.
- **Emergency fast path** at any turn: safety words (Part 1's `SAFETY_PATTERNS`) or the model's `emergency` flag → acknowledgement
  and the matching safety advice (gas / carbon monoxide / water), an immediate push with minimal text, then still collect name and number.
- **Limits:** 12 turns, 2 re-asks per detail, 2 silent turns in a row. Robocall or spam → a short goodbye, still logged.

## 4. Pre-registered decision rules (written before any model run)

**A vs B (Phase 6).** Same understanding step, same cards (`score` split), temperature 0. Version B replaces version A as the
default only if **all** of these hold; otherwise A stays (ties go to A: simpler, cheaper, testable):
1. Safety gates: B has 0 invented numbers, 0 missed urgent callers, and 0 spoken sentences that are not from `persona.json`/`faq.json`.
2. Quality: B gets at least **2 more** correct details (name correct + number correct, counted over all score cards) than A.
3. Cost: B's median reply latency is at most A's + 3 s, and B's fallbacks to the state machine are at most 10 % of its turns.
4. Noise: A is run twice; if A's two runs differ by at least B's margin in rule 2, the result is "inconclusive" and A stays.

**Speed defaults (Phase 2).** Per turn = listen + understand + speak. Pick the *smallest* Whisper model (`base` before `small`)
whose name hit-rate on the `dev` cards is within 1 miss of `small`; pick `qwen2.5:7b` if its median understand time on the GPU is
<= 3 s, else `qwen2.5:3b` (asking before the ~1.9 GB download). The target is a median turn <= 5 s.

## 5. Phases

| # | Phase | Status | Where |
|---|---|---|---|
| 0 | Plan, persona, FAQ, folders | built; the owner has not yet confirmed persona/FAQ wording, and the laptop GPU check is pending (decisions 1-4) | this file, `persona.json`, `faq.json`, `persona.py`, `faq.py` |
| 1 | Synthetic callers (cards + labels + simulator) | built | `testset/callers.json`, `cards.py`, `simulate.py`, `spoken.py`, `asks.py`, `testset/README.md` |
| 2 | Audio loop + speed budget | built; **the speeds are not measured** (laptop only), see `docs/part3-speed.md` | `audio_io.py`, `audio_loop.py`, `session.py`, `app.py`, `measure_speed.py`, `choose_voice.py` |
| 3 | Dialog version A | built (tested with the rules baseline; **the model prompt is untested on a real model**) | `turn.py`, `rules_turn.py`, `dialog.py`, `faq.py`, `safety.py`, `call.py`, `spoken.py` |
| 4 | Hand-off to Part 1 | built (analysis tested with a fake model; **the call prompt `call-v1` has not met a real model**) | `handoff.py`, `shared/analyze.py` (`call-v1`) |
| 5 | Dialog version B (agent) | built (tested with scripted tool calls; **never run with a real model**) | `agent_dialog.py` |
| 6 | Evaluation | harness built and tested without a model; **no evaluation number exists** (laptop only), see `docs/part3-eval-results.md` | `evaluate.py`, `docs/part3-eval-notes.md` |
| 7 | Polish | built: page wired to the real dialog, docs, HANDOVER section 13 | `app.py`, `session.py`, `docs/HANDOVER.md` §13 |

## 6. Definition of done (draft, kept as in the plan)
- Synthetic callers: callback number exactly right >= 80 %, name >= 80 %, **0 invented numbers**.
- **Every urgent caller flagged during the call** (0 missed) and a push sent (or dry-run logged).
- FAQ questions answered correctly from `faq.json` >= 80 %; 0 answers that are not in the FAQ.
- Every completed call lands in `voicemails.db` through the Part 1 pipeline.
- Median receptionist reply <= 5 s on the laptop (measured in Phase 2 / 6).
- The owner can explain why the dialog is a state machine (or not) and what the agent version costs.

## 7. Reuse map
| Need | Reused | Note |
|---|---|---|
| Speech to text, name hint | `shared/transcribe.py` | short turns, no cache |
| Fill a form with the LLM | `shared/llm.structured_chat` | temperature 0, one retry |
| UK number rule | `shared/schemas.normalize_uk_number` | factored out of `Analysis.check_uk_number` in Phase 3 (one rule, two users) |
| Voice | `shared/tts.py` | Holly's speaker id differs from every Part 1/2 voice |
| Safety words, routing, push, storage | Part 1 `routing.py`, `deliver.py`, `store.py`, `shared/notify.py` | minimal push text, dry run by default |
| Analysis of the call | `shared/analyze.analyze` | see decision 12 |
| Agent guardrails (version B) | Part 2 `agent.py` patterns | state message, validated arguments with advice, no-progress stop, trace |

## 8. What has to be run on the laptop
Nothing below could be run in the cloud session (no GPU, Ollama model, Whisper model, Piper voice or microphone).
The commands are filled in as each phase is built; the complete ordered list will be in the final report and in `docs/HANDOVER.md` section 13 (both written at the end of Phase 7).

```powershell
cd C:\Users\danuk\code\audio-to-action
.\.venv\Scripts\Activate.ps1
git fetch origin part3-receptionist; git checkout part3-receptionist
pip install -r requirements.txt                     # installs streamlit too
nvidia-smi                                          # the RTX 5050 must be listed without "Code 43"
ollama ps                                           # while a model runs: PROCESSOR should say 100% GPU
python -m pytest 01-voicemail-triage\tests 02-meeting-action-agent\tests 03-phone-receptionist\tests -q

# Phase 5: the agent (needs Ollama; a failing agent falls back to the state machine and says so)
python 03-phone-receptionist\call.py --card dev --understand model --decide b

# Phase 4: hand a saved call to Part 1 (needs Ollama for --analysis model)
python 03-phone-receptionist\handoff.py 03-phone-receptionist\calls\*.json --analysis rules   # no model; writes 01-voicemail-triage\voicemails.db
python 01-voicemail-triage\store.py                                                       # Part 1's summary of the table (calls are the rows named call-*.json)

# Phase 3: simulated calls (the rules run anywhere; 'model' needs Ollama with qwen2.5:7b)
python 03-phone-receptionist\call.py --card dev --understand rules      # no model: a quick look at the dialog
python 03-phone-receptionist\call.py --card dev --understand model      # the LLM fills the per-turn form (use dev cards only while tuning)
python 03-phone-receptionist\call.py --card c14 --understand model --audio --save   # full audio loop, saves calls\*.json

# Phase 2: voice and speed (the numbers in docs\part3-speed.md exist only after this)
python 03-phone-receptionist\choose_voice.py       # listen to testset\voice_samples\*.wav, set piper_speaker in persona.json
python 03-phone-receptionist\measure_speed.py --device cpu  --whisper base small --llm qwen2.5:7b
python 03-phone-receptionist\measure_speed.py --device cuda --whisper base small --llm qwen2.5:7b qwen2.5:3b   # after the GPU fix; ask before pulling qwen2.5:3b (~1.9 GB)
python -m streamlit run 03-phone-receptionist\app.py   # push-to-talk page (the real dialog; pick rules or model in the sidebar)

# Phase 6: the evaluation (needs Ollama; the audio run also Whisper + Piper). Rewrites docs\part3-eval-results.md
python 03-phone-receptionist\evaluate.py --split score --understand model --decide both --repeat-a 2 --write-docs
python 03-phone-receptionist\evaluate.py --split score --understand model --decide a --audio --write-docs     # the real speed (median reply <= 5 s)
```
Later phases add their own commands here.

## 9. Lessons from Parts 1-2 applied from day one
- The answer key (caller cards and labels) is written **before** the dialog code, and prompts are tuned on `dev` cards only.
- Models satisfy a rule the cheapest way: every validator checks that the "fix" is real, and error messages say what to do instead.
- Number handling and dialog flow are code; the model only understands each turn.
- The simple baseline (state machine) comes first; the agent is judged by a rule written before the runs (section 4).
- Counts, not just percentages: the card set is small.

## 10. Risks
| Risk | Mitigation |
|---|---|
| Latency too high for a call | GPU fix, short prompts, `qwen2.5:3b` and Whisper `base` fallbacks (section 4) |
| Whisper hallucinations on silence | ignore turns under ~0.5 s of speech; never act on an empty turn |
| Echo and feedback | push-to-talk, headphones for demos |
| FAQ drift | answers are said word for word from `faq.json` |
| Privacy | `calls/`, `logs/`, test audio and databases are git-ignored; Holly announces the recording |
