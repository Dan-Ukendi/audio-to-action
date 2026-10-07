## Conclusions (hand-written in `docs/eval-notes.md`; the rest of this file is generated)

### Definition of done (Part 1)
| Criterion | Status |
|---|---|
| ≥ 90 % category accuracy on the test set | **Not met: 15/18 (83 %).** Misses: 08 friend → other, 12 scam → urgent, 17 empty message → personal |
| No urgent voicemail routed to the archive | **Met:** 0/6 urgent missed, 0 archived (and the safety-word net would catch a miss) |
| Re-running on the same file does nothing | **Met:** skipped by audio hash in SQLite (tested in Phase 5) |
| The user can explain why each step is a workflow step, not an agent | For the user (see `docs/learning-log.md`) |

### Experiment: prompt v2 → v3 (one variable)
- **Change:** one line of the prompt. v2's example `"this is Rachel from Acme" -> "Rachel"` became
  `"this is Rachel Moore from Acme" -> "Rachel Moore". Keep the full name when one is given.`
  Everything else fixed: same `small` transcripts, same model, temperature 0.
- **Hypothesis:** v2's example taught the model to drop surnames (Helen, Laura, Brenda); v3 restores them.
- **Adoption rule, set before the run:** adopt v3 only if urgent false negatives stay 0, no urgent is archived,
  category accuracy does not drop, **and** exact name accuracy improves.
- **Result:** surnames came back (13 Laura Jenkins, 16 Brenda Walsh, 14 Shivvon Gallagher, 06 Helen Carver),
  but two unrelated answers broke (07 Mum → null, 17 "Yak" invented from a mis-transcription) and urgency
  accuracy fell 11 → 10. Exact names: 10/18 in both.
- **Decision: keep v2** (rule not met). `ANALYSIS_PROMPT=v2` stays the default; v3 stays in `PROMPTS` for reference.
- **Lesson:** a one-line prompt change moved answers on files it had nothing to do with. With 18 files, a ±2
  difference is within that noise, so small prompt tweaks can't be judged on this set alone. The next
  improvements need more (held-out) data, not more tweaking.

### Where the remaining errors come from
| Source | Errors (v2) | Fixable by |
|---|---|---|
| Transcription (Whisper `small`) | names 03 Priyashar, 06 Carver, 15 Vojcik, 18 Ocifer; numbers 15 (digit dropped), 18 (940 for 900); 10 "Colleen" | `large-v3-turbo` (fixed 03, 15-number, 18-number in Phase 2), or GPU to make it affordable |
| LLM judgement | 12 scam → urgent; 08, 17 category; 2-vs-1 urgency on 6 files | an explicit "automated/robocall is never urgent" rule (test on held-out data), larger LLM |
| Label boundary | 08 (friend's football: personal vs other), urgency 1 vs 2 | sharpen the label conventions, then re-label consistently |

### Next experiments (one at a time, each against a held-out set)
1. Whisper `large-v3-turbo` transcripts with prompt v2 (expected: names/numbers up, category unchanged).
2. Prompt rule "messages that are automated or ask you to press a key are spam, never urgent" (target: file 12).
3. Real recordings in `testset/my_recordings/` (with consent) to check how optimistic the synthetic scores are.
