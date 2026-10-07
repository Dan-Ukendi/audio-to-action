## Conclusions (hand-written in `docs/part2-eval-notes.md`; the rest of this file is generated)

### The experiment: agent vs plain-code rules for the tracker sync (one variable)
- **Question:** does the agent earn its cost (LLM calls, minutes per meeting, complexity), or would plain rules do?
- **Single variable:** the sync method. The input items are identical for both (the answer key's items = "gold",
  or the real cached extraction x7 = "extracted"), the tracker starts empty, the meeting order is the same.
- **Decision rule, written BEFORE the agent runs** (evening of 2026-10-07, after the two instant rules runs were already known: gold 14/14, extracted 6/14): the default sync becomes
  `rules` if, on the **extracted** items (the real pipeline), rules get at least as many tracker tasks right after
  the series as the agent **and** do not leave more spurious rows. Otherwise the default stays `agent`.
  Gold results are reported but don't decide: they measure the method on perfect input, not the real pipeline.
- **Same scoring:** both methods are scored by the same `score_tracker()` in `sync_testset.py`. One asymmetry: the
  agent may leave items for a human ("left for review" column); those count as not done, the rules never defer.
- **Tuned on this test set (no held-out data):** the agent's guards were iterated on the gold items in Phase 4, and
  the rules' thresholds (`LINK` 0.2, `SAME` 0.6, `REPEAT` 0.5) were proposed by the Phase 4 reviewer after looking
  at the same meetings. So gold/rules 14/14 is a fit to this data, not an estimate for new meetings.
- **Code versions:** both agent rows come from `agent.py` as committed in Phase 5 (before the not-work change
  below); the rules rows from the current `rules_sync.py` (re-run by the review: identical results). The later
  meeting-1-only re-run overwrote `traces_extracted/m1…trace.json` and `tracker_extracted_agent.db`, so for the 7/14
  run only `sync_runs/extracted_agent.json` and the m2-m5 traces remain.

### Result
| sync | gold items (perfect input) | **extracted items (real pipeline)** | LLM calls (both runs) | time (both runs) |
|---|---|---|---|---|
| agent | 11/14 | **7/14**, 2 duplicated, 1 spurious | 51 | ~94 min |
| rules | **14/14** | 6/14, 4 duplicated, 1 spurious | 0 | < 1 s |

**Decision (by the rule above): the default stays `agent`.** On the extracted items it got one more task right with
the same number of spurious rows. Its edge is where wording drifts between meetings: it linked
"Update the insurance" to "Renew the van insurance" (word overlap 0.25, below the rules' 0.6 bar for open items),
and "Fix Wood's van brake light" to the brake-light task (2 duplicated tasks vs 4).

**How far from perfect?** Linking these extracted items by hand (as a person would, e.g. "Gallagher court invoice"
→ the quote task) gives **9/14**: 5 tasks are lost to extraction whatever the sync does (T4, T6, T7, T8, T10).
The agent lost 2 more: T12 (it kept retrying a refused update and was stopped, item left for review) and T14
(duplicate). The rules lost 3: T5 and T14 (duplicates) and T11 ("Do Ellis gas check" was skipped as a repeat of
"Book Ellis gas check", word overlap 0.75). (Phase 4's "at most 7/14" is what you get when only links the scorer's
match rules accept are allowed; it misses T1 and T14.)

**But read the cost next to the gain:** +1 task for ~49 minutes and 28 LLM calls on this CPU, versus instant and free.
And +1 is within the agent's own run-to-run noise: on the same gold items the Phase 4 agent got 12/14 (its later
review changes were meant to keep decisions the same) and 11/14 here. On perfect input the rules are better (14/14 vs 11/14, but see "tuned" above): the agent twice got
stuck retrying an update the guards refused (T9, T11 left for review) and once linked new work to the wrong task
("Lead the Gallagher bathroom fit" → the bathroom-quote task, which broke T1 and T12). `--sync rules` is a reasonable
choice when speed matters; this test set is too small to show which method handles messy wording better.

**Found by the experiment, fixed afterwards (not re-measured on the full series):** in the extracted run, meeting 1 used
all 16 steps because a guard written on gold data ("done/cancelled news is never 'not work'") forbade skipping
"Discuss the new website" [cancelled], a postponed idea. Now that skip is allowed when no tracker task resembles the
item. Meeting 1 re-run: 2 LLM calls instead of 16, same 4/5.

### Definition of done (Part 2)
| Criterion | Status |
|---|---|
| Action items: precision and recall ≥ 80 % (task + owner) | **Not met:** 64 % / 57 % (task only: 84 % / 75 %) |
| Tracker state after the series ≥ 80 % of tasks right | **Not met** on the real pipeline: 7/14 (50 %); gold 11/14 (agent), 14/14 (rules) |
| Agent never exceeds its step limit or calls a tool outside the allowlist; every change has a logged reason | **Met:** the limit was reached, never exceeded: extracted m1 used 16 of 16 calls (1 item to review); 3 other meetings were stopped early by the no-progress guard (1 item each to review). The traces show only allowlisted tools; every change in `tracker_gold_agent.db` and every OK step in the extracted traces has a non-empty reason (often boilerplate, e.g. "Task is now done as per meeting notes" on an update that kept the task open) |
| Re-running on the same meeting does nothing | **Met:** skipped by audio hash (Phase 5); an interrupted run is rolled back first |
| The user can explain workflow vs agent and the agent's cost | For the user (`docs/learning-log.md`) |

### Where the errors come from
| source | effect |
|---|---|
| Extraction misses (recall 75 %), wrong status/due and garbled task text ("buff from Sweet") | 5 of 14 tasks lost whatever the sync does: perfect sync of the extracted items = 9/14 |
| Owners only knowable from the voice (T5, T7, T8; no diarization by design) | owner missing until a later meeting names it (T5 fixed in m3); lowers the per-meeting scores |
| Agent retries an update the guards refused until the no-progress stop | items left for review: T12 (extracted), T9 and T11 (gold) |
| Agent links new work to a wrong task that shares a few words (similarity ≥ 0.2) | gold: "Lead the bathroom fit" → quote task (T1 and T12 wrong) |
| Rules: fixed word-overlap thresholds | reworded tasks duplicated (T4, T5, T7, T14 on extracted); different work said alike merged (T11) |

### Next experiments (one variable each)
1. Whisper `large-v3-turbo` + hint transcripts for extraction (Phase 2: fewer misheard task words).
2. A hybrid sync: rules first, the agent only for items no rule matches confidently (keeps most of the speed).
3. Speaker diarization, only to measure how many of the voice-only owners it would recover.
4. More (and real, consented) meetings: 28 mentions is too few to separate small differences from noise.

