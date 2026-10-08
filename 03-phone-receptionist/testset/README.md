# Part 3 test set: synthetic callers

`callers.json` holds 24 **caller cards**: the 18 Part 1 voicemail scenarios turned into phone calls, plus 6 FAQ callers.
It is the input side *and* the answer key, written before any dialog code so the dialog cannot influence what "correct" means.

```
python 03-phone-receptionist/cards.py                  # checks the answer key (also run by the tests)
python 03-phone-receptionist/simulate.py --preview c14 # what card c14 says to every question the receptionist can ask
python 03-phone-receptionist/simulate.py --audio c14   # laptop only: renders the opening turn with Piper into testset/audio/
```

## What a card is
| Part | Meaning |
|---|---|
| `part1_id` | the Part 1 voicemail it re-uses; its category, urgency, name and number are **read** from `01-voicemail-triage/testset/labels.json` (one answer key, never a copy) |
| `labels` | FAQ cards only: the same four labels, written here |
| `facts` | what is true about the caller (`name`, `name_spoken` = how the voice must pronounce a hard name, `number`, `reason_keywords`) |
| `quirks` | how the caller behaves: `all_upfront`, `hesitant`, `refuses_number`, `withholds_name`, `withholds_number`, `self_corrects_number`, `wrong_number_first`, `spells_name`, `robocall`, `rambles`, `emergency`, `question_only` |
| `script` | what the caller says. `opening` is the first turn; the other answers have defaults in `simulate.py` |
| `question` | a question the caller asks and the FAQ entry that must answer it (`topic: null` = no entry exists) |
| `expect` | `urgent_flag` (flagged urgent *during the call*), `outcome` (`completed`, `spam`, `info_only`), `faq_topics`, `faq_unknown`, `safety` (gas / co / water: the advice the emergency fast path must give) |
| `split` | `dev` = may be looked at and tuned on; `score` = only `evaluate.py` scores these. **Never tune on score cards.** |

## The cards
| Card | Part 1 scenario | Split | Tests |
|---|---|---|---|
| c01 | burst pipe | score | everything given in the first turn; nothing may be asked twice |
| c02 | no heating, elderly | score | urgent by vulnerability |
| c03 | gas smell | dev | safety emergency (must get the gas safety advice), hesitant speech, asks what to do |
| c04 | vague leak | score | first name only, "you've got my number" must give no number |
| c05 | noisy water heater | score | heavy noise, sparking near electrics, "forty-two Mill Lane" is not a number |
| c06 | landlord deadline | score | urgent by deadline; "F twenty-eight" is not a number |
| c07 | Mum | dev | personal call, no number |
| c08 | friend, five-a-side | score | personal call with number |
| c09 | boiler supplier | score | trade sales call |
| c10 | SEO cold call | score | declines to give a name |
| c11 | tax robocall | dev | robocall: short goodbye |
| c12 | fake "urgent" listing | score | robocall that says "urgent": must NOT be flagged |
| c13 | appointment confirmation | dev | name given up front; wrong number first, corrected at the read-back |
| c14 | quote, hard name | dev | spells the name when asked |
| c15 | Polish name | score | spells the name when asked |
| c16 | rambling tap | score | number buried in a long opening; asks the price |
| c17 | no information | score | no name, no number, vague request |
| c18 | number corrected | dev | corrects the number inside one sentence |
| f01 | radiator, how soon? | dev | question in the first turn |
| f02 | boiler service price | score | question in the first turn |
| f03 | opening hours only | dev | no message: `info_only` |
| f04 | solar panels | dev | question faq.json cannot answer |
| f05 | area + payment | score | two questions in one turn, then a real request after "anything else?" |
| f06 | cancellation policy | score | question plus a message |

Voices: the 18 Part 1 cards keep Part 1's speaker ids; the 6 FAQ callers use 18, 22, 29, 38, 47 and 58, which differ from Parts 1 and 2.
`persona.json` lists them as taken, so the receptionist's own voice can never be one of them (the persona loader refuses it;
`choose_voice.py`, written in Phase 2, will skip them). Audio is generated on demand (`simulate.render_turn`) into
`testset/audio/`, which is git-ignored. `--audio` needs the Piper voice: if it is not in `01-voicemail-triage/testset/voices/` or
`models/piper/`, `shared/tts.py` downloads it (~77 MB), so run it only where that is acceptable.

`expect.safety` (gas / co / water) names the safety advice the emergency fast path must give: c03 gas, c01 and c05 water.
Splits: 9 dev, 15 score. Behaviours seen only by the scoring run: withheld name/number (c10, c17), the all-in-one opening (c01), the
rambling opening (c16), two questions in one turn and a new request after "anything else?" (f05).

Scoring note: card c13 (`wrong_number_first`) objects at its first read-back **whatever Holly read back**: that one correction is the
card's own slip, so it must not be counted as a receptionist mistake.
