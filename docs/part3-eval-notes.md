# Part 3 evaluation: what is measured and how

No result is written in this file. The numbers go to `docs/part3-eval-results.md`, and only after a run with a real model on the laptop.

## The command

```powershell
python 03-phone-receptionist\evaluate.py --split score --understand model --decide both --repeat-a 2 --write-docs
python 03-phone-receptionist\evaluate.py --split score --understand model --decide a --audio --write-docs
```

`--write-docs` is refused for a run without a model, an invalid run (the model failed and plain code filled in: the report says so), or anything but the score cards. The audio run writes its own file, `docs/part3-eval-results-audio.md`, so it never overwrites the A-vs-B report. With one A run (`--repeat-a 1`) rule 4 cannot be applied and B cannot win.

`--split dev` is for tuning prompts and rules; `--split score` is for the evaluation (the 15 cards nobody tuned on). `--understand rules` uses the
plain-code baseline: it only checks that the harness works and never writes the results file. Each run saves per-call details as JSON in
`03-phone-receptionist/logs/` (git-ignored).

## Metrics, per call, against the answer key in `testset/callers.json`

| Metric | Definition |
|---|---|
| name / number | `null_aware`: correct, wrong, missed (said but not found), invented (not said but recorded). Numbers are compared as digits. A caller who gives no number must end with no number. |
| invented number | the recorded number is not made of digits the caller said, in order. Must be 0. |
| urgent | flagged during the call or not: missed (should be flagged, was not) and false (the opposite). |
| FAQ | the expected topics were answered (a gas, carbon monoxide or water emergency counts as the matching safety topic), nothing else was answered (an unexpected answer makes the card wrong), and an unknown question was passed on. |
| sentences | every reply is made only of `persona.json` lines and `faq.json` answers (placeholders may be anything). |
| outcome | the call ended as the card expects (completed, spam, info only). |
| time per reply | understand + decide in a text run; plus listening and speaking in an audio run. Median and p95. |
| fallbacks | version B only: turns where the agent failed and the state machine decided. |
| hand-off | the call goes through `handoff.py` into an in-memory copy of Part 1's table; category, route and push are compared with the card's Part 1 label (an urgent call is pushed, nothing else is). Pushes are recorded, never sent. |

## The rule for A or B (written before any run; README section 4)

B replaces A only if all hold: (1) 0 invented numbers, 0 missed urgent, 0 non-approved sentences; (2) at least 2 more correct details
(name + number) than A; (3) median latency at most A + 3 s and fallbacks at most 10 % of turns; (4) A's two runs differ by less than B's
margin. If rule 4 fails the answer is "inconclusive, A stays". With one A run the rule cannot be applied and B cannot win.
