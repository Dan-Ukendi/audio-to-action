# Part 2: meeting-action-agent

Meeting recordings → action items → a task tracker that stays in sync across meetings.
Part 1 was a pure workflow. Part 2 adds **one** agent, in the one place where it might earn its keep, and then
measures whether it does.

## The problem
The business from Part 1 (Brightwater Plumbing & Heating: Sam, the owner, Priya in the office, Tom and Jamie the
plumbers) holds a short weekly meeting. Each meeting creates new tasks and also talks about *old* ones:

> "Priya, did the Gallagher quote go out?" "Yes, sent it Tuesday." → close an existing task
> "Tom, have you booked the Ellis check? … Jamie, could you book it instead?" → change the owner
> "Someone should look at the van." → a task without an owner (flag it, don't invent one)
> "Let's leave the website for now." → not a task at all

Turning one meeting into a list of items is a fixed job: a **workflow**. Deciding how those items relate to what's
already in the tracker (update, new, repeat, not work) needs lookups and judgement: the **agent**, or plain rules.

## How it works
```
 inbox/<meeting>.m4a        (date from YYYY-MM-DD in the file name, else the file's date; oldest meeting first)
        │  run.py
        ▼
 0. hash in tracker.meetings? ── yes ──► processed/ (nothing else)
 1. TRANSCRIBE  shared/transcribe.py: Whisper small + hint with known names (context.py), cached
 2. EXTRACT     extract.py (workflow): LLM fills MeetingItems = jobs_mentioned + items (evidence, task, owner,
                due_text, status); validators: owner on the team, quote in the transcript; ground_owner() drops
                owners not named nearby; dates.py turns "by Wednesday" into a date in code; cached
 3. SYNC        --sync agent (default): agent.py, tool loop on qwen2.5:7b, bounded by code guards
                --sync rules:           rules_sync.py, plain code, no LLM
                either way: tracker.py (tasks + change log with reasons); a failed sync is rolled back
 4. RECORD      tracker.meetings + move to processed/   (any error: failed/ + .error.txt)
```

The agent (`agent.py`) only decides **relationships**: its tools take an item letter and a task number, and code
copies owner, date and status from the item. Each step it gets a fresh **state message** (items and what's been done
to them, the tracker's tasks, feedback on its last calls), not a growing chat. Guards in plain code refuse: unknown
tools, bad arguments (with advice), handling an item twice, linking unrelated work (word overlap < 0.2), changing one
task twice in a meeting, near-duplicate adds (unless confirmed), fake "repeats", skipping real done/cancelled news as
"not work", finishing early, blank reasons. A step limit, a no-progress stop and a review list bound it further.

## Using it
```powershell
.\.venv\Scripts\Activate.ps1
copy my_meeting_2026-10-12.m4a 02-meeting-action-agent\inbox\
python 02-meeting-action-agent\run.py                 # or --watch, --retry-failed, --sync rules
python -m pytest 02-meeting-action-agent\tests -q     # 70 tests, no LLM
python 02-meeting-action-agent\evaluate.py            # report from saved runs -> docs\part2-eval-results.md
```
Look inside: `tracker.db` (tables `tasks`, `changes` with the reason for every change, `meetings`), agent traces in
`traces/<meeting>.trace.json` (every tool call, argument, result), logs in `logs/run.log`.
Edit `context.py` (team, customers, places) for a real business: it feeds both the Whisper hint and the owner checks.

## Results (5 synthetic meetings, 14 tasks, CPU only)
| step | result |
|---|---|
| Transcription, small + hint | 5.3 % word errors, 98 % of names right (71 % without the hint) |
| Extraction (prompt x7) | task only: precision 84 %, recall 75 %; task + owner: 64 % / 57 %; due 95 %, status 95 % |
| Sync on perfect items | rules 14/14, agent 11/14 |
| Sync on real extracted items | agent 7/14, rules 6/14 (a perfect sync would reach 9/14) |
| Cost of the agent | 51 LLM calls and ~94 min for the two runs; rules: under a second |

The agent stays the default by a rule fixed before the run, but its +1 is within run-to-run noise. The real bottleneck
is extraction (and owners only knowable from the voice). Full details and the definition of done:
[docs/part2-eval-results.md](../docs/part2-eval-results.md). The story of every design change: the Part 2 entries in
[docs/learning-log.md](../docs/learning-log.md).

## Decisions
| Decision | Choice | Why |
|---|---|---|
| Test data | 5 short synthetic meetings (~30 s), a weekly series | reproducible, no personal data, fast on CPU; a series is needed to test sync |
| Who said what | no speaker diarization; owners from names in the words | free, no PyTorch / HF token; 3 owners are voice-only by design and measured |
| Whisper | `small` + known-names hint | 71 % → 98 % names for ~15 % more time (turbo + hint: best, 2× slower) |
| Dates | LLM copies the words, `dates.py` computes the date | calendar arithmetic is what code never gets wrong |
| Agent's job | tracker sync only, relationships only | everything else is fixed work; values copied by code can't be invented |
| Agent input | fresh state each step, not chat history | the history version replayed failing calls and guessed ids |
| Default sync | agent (pre-registered rule) | +1 task on the real pipeline; `--sync rules` is the fast alternative |

## Folders
`inbox/ processed/ failed/ transcripts/ results/` (contents git-ignored), `traces/`, `logs/`, `tracker.db` (ignored),
`testset/` (scripts, labels, generator, saved sync runs; audio and caches ignored), `tests/`.
