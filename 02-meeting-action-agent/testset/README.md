# Part 2 test set: a meeting series

Five weekly team meetings (Mondays 7 Sep - 5 Oct 2026) at Brightwater Plumbing & Heating, the fictional
business from Part 1. Four voices: Sam (owner, chairs), Priya (office), Tom and Jamie (plumbers).
Each meeting is ~30 s: short on purpose so the CPU-only pipeline stays fast, but dense with traps.

| File | Committed? | What |
|---|---|---|
| `scripts.json` | yes | Who says what, turn by turn; voice per person; noise; output format |
| `labels.json` | yes | Answer key: 14 tasks, what each meeting says about them (mentions) |
| `answer_key.py` | yes | Loads the labels, checks them, derives the expected tracker state after each meeting |
| `generate.py` | yes | Rebuilds `audio/` with Piper (one voice per person) + ffmpeg |
| `audio/` | **no** | `<id>.<format>` + `<id>.turns.json` (who spoke when: ground truth we don't give the pipeline) |

```powershell
python 02-meeting-action-agent\testset\generate.py
```

## What the series tests
| Trap | Where |
|---|---|
| Task with no owner, later assigned | T4 van brake light (m1 → m2) |
| Explicit non-task ("let's leave the website for now") | m1, then it becomes a real task in m5 (T14) |
| Owner only knowable from the voice ("I'll handle that") | T5 (m1), T7 (m2), T8 (m3): `owner_from_text: false` |
| Task closed in a later meeting | T1, T2, T3, T4, T5, T6, T7, T8, T10, T11 |
| One task finishes, a related new one starts | T2 order valve → T6 fit valve; T7 book check → T11 do check |
| New task that looks like an old one (not a duplicate) | T8 revised Gallagher quote vs T1 original quote |
| Owner change | T7 Tom → Jamie (m3) |
| Re-mention with no change (must not create a duplicate) | T5 insurance (m3) |
| Task cancelled | T9 Mill Lane follow-up (m4) |
| Task with no due date | T9 ("at some point") |
| Relative dates | "by Wednesday", "tomorrow", "this afternoon", "end of the month", "Friday" |
| Task assigned by someone other than the chair | T13 van tax (Priya → Jamie) |

Expected tracker after the series: 14 tasks, 10 done, 1 cancelled, 3 open (T12, T13, T14).
