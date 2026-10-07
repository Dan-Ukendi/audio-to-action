# Part 2: meeting-action-agent

Meeting recordings → action items → a task tracker that stays in sync across meetings.
Part 1 was a pure workflow. Part 2 adds **one** agent, in the one place where it earns its keep.

## The problem
The business from Part 1 (Brightwater Plumbing & Heating: Sam, the owner, plus a small team) holds a
short weekly meeting. Each meeting produces new tasks, but it also talks about *old* ones:

> "Did Priya send the Gallagher quote?" "Yes, done Tuesday." → close an existing task
> "Tom, can you take over the supplier call from Jamie?" → change an owner
> "Let's push the boiler order to next Friday." → change a due date
> "Someone should look at the van." → a task without an owner (flag it, don't invent one)

Turning one meeting into a list of items is a fixed job, so it stays a workflow. Deciding how those items
relate to what's already in the tracker (new, update, done, duplicate) needs lookups and judgement about
existing state, and that's the agent's job.

## Planned architecture
```
 meeting audio ─► 1. TRANSCRIBE   shared/transcribe.py (reused as-is), timestamps per segment
                        │
                        ▼
                  2. EXTRACT      workflow: LLM fills a MeetingResult schema (action items: task, owner,
                        │         due, quote + timestamp); validated; 1 retry; chunked for long meetings
                        ▼
                  items JSON
                        │
                        ▼
                  3. SYNC         AGENT: tool-calling loop over the tracker (SQLite)
                        │           tools: list_open_tasks, search_tasks, add_task, update_task,
                        │                  close_task, mark_duplicate, finish
                        │           guardrails: step limit, allowlisted tools only, Pydantic-checked
                        │                       arguments, every change stored with its reason
                        ▼
                  tracker.db + trace JSON (every step the agent took)
```

## Design decisions (Phase 0)
| Decision | Choice | Why |
|---|---|---|
| Test data | Short synthetic meetings (2-4 min, 2-4 Piper voices), a weekly *series* | reproducible, no personal data, fast enough on CPU; a series is needed to test sync |
| Who said what | No speaker diarization; owners from names in speech | free, no PyTorch/HF token; "I'll do it" from an unknown speaker is a measured weak spot |
| Agent's job | Task tracker sync only | needs state lookups and judgement; everything else is fixed workflow |
| Agent safety | step limit, tool allowlist, validated args, change log with reasons, trace | an agent's path isn't fixed, so it has to be bounded and inspectable |

## Phases
0. Plan & setup ✔
1. Test set: meeting series + answer key (items per meeting, tracker state after each)
2. Transcription of longer audio
3. Extraction workflow (generalized analyze pattern, chunk + merge)
4. The agent (tracker DB, tools, loop, trace)
5. Glue & reliability (`run.py`)
6. Evaluation (+ experiment: plain-code sync rules vs agent)
7. Polish

## Folders
`inbox/ processed/ failed/ transcripts/ results/` work like in Part 1 (contents git-ignored).
`testset/` will hold the meeting scripts, labels and generator (audio git-ignored).
