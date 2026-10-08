# Part 3 report: the phone receptionist (for the next agent)

Branch work, all phases 0-7, written autonomously. Plan: `docs/PART3_PLAN.md`. Details: `03-phone-receptionist/README.md` (decisions table, rules,
laptop commands), `docs/learning-log.md` (one recap per phase), `docs/HANDOVER.md` section 13 (documents against code).
Test suite: `python -m pytest 01-voicemail-triage/tests 02-meeting-action-agent/tests 03-phone-receptionist/tests -q` -> 473 passed, 1 skipped.

## 1. Honest status
Built and tested in a cloud container with **no GPU, no Whisper, no Ollama model, no Piper voice, no microphone**. Everything that needs those
was written and tested with fakes. Therefore:
- NO speed number and NO evaluation number exists. `docs/part3-speed.md` and `docs/part3-eval-results.md` are "NOT MEASURED" placeholders.
- The prompts (`t1` understanding in `turn.py`, `call-v1` analysis in `shared/analyze.py`, the agent prompt in `agent_dialog.py`) have never met a real model.
- The A-vs-B decision has not been made; the rule was written before any run (README section 4).
- Owner has not confirmed Holly's wording / FAQ facts; the laptop GPU check (Code 43) is pending.

## 2. What the system is
"Holly" answers calls for the fictional Brightwater Plumbing & Heating: listen -> understand -> decide -> speak -> hand off.
Per caller turn (`dialog.next_reply`): UNDERSTAND (LLM fills a `CallerTurn` form; plain code grounds it: a number must be digits the caller said,
a name must be words the caller said or a whole spelled run, a reason must be made of said words) -> APPLY (`apply_turn`, shared, enforces "never
invent", emergencies, FAQ, spam) -> DECIDE (version A `decide_a` state machine, or version B `agent_dialog.make_decide_b`) -> RENDER (fixed
sentences from `persona.json` / `faq.json` only; models never write replies). States: GREETING, COLLECTING, READ_BACK, CORRECTING, GOODBYE, ENDED.
Limits: 12 turns, 2 re-asks per detail, 2 silent turns. Emergencies (Part 1 `SAFETY_PATTERNS` or model flag) get an acknowledgement, safety advice
(gas/CO/water) once, and an urgent goodbye with a callback promise and "nine nine nine". Robocalls get a short goodbye (only on first contact, never
with safety words). Questions not in the FAQ are "passed on", never answered.

## 3. Phases (all in `03-phone-receptionist/` unless noted)
| Phase | Files | Notes |
|---|---|---|
| 0 | `persona.json/.py`, `faq.json/.py`, folders, `.gitignore`, rules | 23 FAQ entries; call audio/records git-ignored |
| 1 | `testset/callers.json`, `cards.py`, `simulate.py`, `asks.py`, `spoken.py`; `shared/schemas.normalize_uk_number` | 24 caller cards (18 from Part 1 voicemails + 6 FAQ callers), answer key written before dialog code; 9 `dev` cards (tune here), 15 `score` cards (never tune) |
| 2 | `audio_io.py`, `audio_loop.py`, `session.py`, `app.py`, `measure_speed.py`, `choose_voice.py` | speeds NOT measured |
| 3 | `turn.py`, `rules_turn.py` (no-model baseline/fallback), `dialog.py`, `faq.py`, `safety.py`, `call.py` | version A |
| 4 | `handoff.py`, `shared/analyze.py` (`call-v1`) | caller-side transcript -> Part 1 analyze/route/push/store; live urgent push in a background thread; rows `call-<id>.json`; dialog's verified name/number override the model's |
| 5 | `agent_dialog.py` | tools ask/answer_faq/read_back/flag_urgent/take_message/end_call; no tool takes a value; legality from the state; MAX_STEPS 4; no-progress stop; fallback to A (`fallback: true` in trace); safety floor in code |
| 6 | `evaluate.py`, `docs/part3-eval-notes.md`, `docs/part3-eval-results.md` | metrics + DoD table + `decide_ab()` |
| 7 | `app.py`/`session.py` wired to the real dialog, READMEs, `docs/HANDOVER.md` section 13 | page does not hand calls to Part 1 (use `call.py --save`, `handoff.py`) |

## 4. The A-vs-B rule (pre-registered, README section 4, coded in `evaluate.decide_ab`)
B replaces A only if ALL: (1) 0 invented numbers, 0 missed urgent, 0 non-approved sentences; (2) B has >= 2 more correct details (name + number) than A;
(3) B median latency <= A + 3 s and B fallbacks <= 10 % of turns; (4) A run twice and its two runs differ by less than B's margin. "inconclusive, A stays" =
only rule 4 failed. One A run => B cannot win. Constants are checked against the README by a test.
Speed rule: smallest Whisper model within 1 name-hit of `small`; `qwen2.5:7b` if median understand <= 3 s else `qwen2.5:3b` (ask before downloading); target median turn <= 5 s.

## 5. Evaluation safeguards (do not weaken)
`--understand rules` runs only test the harness: banner "Harness check only", never written to docs. A model run where the model silently failed (understand
fallback, or hand-off analysis "none (model failed)"; note "none (rules)" for spam/no-message calls is normal) is marked INVALID and never written.
`--write-docs` only for `--split score` with `--decide both` (or an `--audio` run with a chosen voice), with hand-off on. The audio run writes `docs/part3-eval-results-audio.md`.
Hand-off in the evaluation uses an in-memory table and a recording sender: nothing touches `voicemails.db` or a phone.

## 6. Process followed
One branch (`part3-receptionist`), phases committed in order; after each phase a separate Opus reviewer agent (read-only, max 3 rounds) checked it.
Verdicts: P0 FAIL,PASS; P1 FAIL,FAIL,PASS; P2 FAIL,FAIL,PASS; P3 FAIL,FAIL,FAIL (last fixes not re-reviewed; recorded in the learning log); P4 FAIL,FAIL,PASS;
P5 FAIL,FAIL,PASS; P6/P7 FAIL,FAIL,PASS. A force-push was denied, so Phases 3-7 have follow-up fix commits (bends "one commit per phase").
Every commit: author and committer Dan-Ukendi <dan.ukendi1@gmail.com>, no trailers, no tool mentions.
Decisions the plan left to the owner were taken with defaults and listed in README section 2 ("Decisions taken without the owner", rows 1-46).

## 7. What to do next (on the laptop, in this order; exact commands in `03-phone-receptionist/README.md` section 8)
1. Pull, install (`pip install -r requirements.txt`), `nvidia-smi`, `ollama ps`, run the tests.
2. `choose_voice.py` -> set `piper_speaker`; `measure_speed.py --device cpu` and `--device cuda` (ask before pulling `qwen2.5:3b`, ~1.9 GB). Rewrite `docs/part3-speed.md`.
3. `call.py --card dev --understand model` (then `--decide b`, then `--audio --save` for one card). Tune prompts ONLY on dev cards.
4. `handoff.py 03-phone-receptionist/calls/*.json`, then `01-voicemail-triage/store.py`.
5. `evaluate.py --split score --understand model --decide both --repeat-a 2 --write-docs`, then `--decide a --audio --write-docs`.
6. Fill the DoD table from the results; the owner confirms Holly's persona/FAQ wording; then review the branch.

## 8. Known open items
- Phase 3 third-review fixes were not re-reviewed (round limit).
- An `--audio` evaluation run with no voice chosen still times the audio channel (never written to docs).
- The approved-sentence check lets placeholders match any text (values are checked separately by the invented-number and name checks).
- Rows of the decisions table 1-46 are the owner's to accept or change.
