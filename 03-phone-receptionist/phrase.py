"""Natural wording: the model rewords the fixed sentences, code checks every word it may not change.

    phrase_fn = make_phraser()                                   # pass to dialog.next_reply(..., phrase_fn=phrase_fn)
    text, trace = phrase_fn(parts, state, caller_text, persona)

The dialog still decides WHAT Holly does (ask for the number, read back, say goodbye) and still builds the fixed sentence
for it. This step only changes HOW she says it: warmer, shorter, reacting to what the caller just said. The model never
chooses the flow and never sees a fact it may not use:

  - Safety lines (urgent acknowledgement, safety advice, the urgent goodbye with "nine nine nine", silence / turn-limit
    endings) are NOT sent to the model, not even as context; code inserts them word for word. The model also never sees
    what Holly said earlier, only what the caller said.
  - Every reworded sentence is checked by check_sentence() below. Any problem (a digit changed, a name or a place that
    nobody said, a promise, a price, a time, a negation, a question that disappeared) is shown to the model once; if the
    second answer is also rejected, or the model is down or too slow, the FIXED sentence is used. So a call can never be
    worse than version A without phrasing.
  - Information lines (FAQ answers, goodbyes) may only use words that were already in the line, plus a short list of
    friendly filler words: they cannot grow new facts. Questions may also use the caller's own words in a short reaction
    ("Oh dear, a smell of gas"), but never a promise, advice or negation word, even if the caller said it first.

The checks are plain code on purpose: "is this natural?" is a taste question, but "did she invent a number?" is not.
Known limit: a list of banned words cannot catch every possible promise; the model is also told the rules, and the owner
rates the wording by ear (README 3.1).
"""

import re
import sys
from pathlib import Path

from pydantic import BaseModel, ValidationInfo, model_validator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from shared.llm import LLMFormError, structured_chat  # noqa: E402

SPOKEN_FORBIDDEN = re.compile(r"[\d£$€%&/@#*+=<>_\[\]{}|\\~^]")  # the voice reads these oddly or skips them
NUMBER_WORDS = set("zero oh one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
                   "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand "
                   "double triple".split())
# Words that make a promise, an instruction, a price, a time, a negation or an urgency claim. Allowed ONLY if the fixed
# sentence itself uses them; the caller saying them first does not make them safe for Holly to repeat.
RISK_WORDS = set("""guarantee guaranteed promise promised refund free discount cheap cheaper today tomorrow tonight
monday tuesday wednesday thursday friday saturday sunday soon shortly quickly asap immediately within definitely certainly
warranty insured safe dangerous turn switch shut
will shall someone somebody anyone anybody everyone round away straight leave stay wait avoid evacuate open close
urgent urgently emergency marked flagged priority
not never no cannot without""".split())
# Ordinary job words: Holly may echo them when the caller said them ("Lovely, a bathroom quote"), but not introduce them.
JOB_WORDS = set("""morning afternoon evening minutes minute hours hour pounds pound price cost quote appointment book booked
engineer plumber send dispatch visit arrive arrives""".split())
# Friendly filler an information line may add. Anything else new in such a line is a new fact.
FILLER = set("""a an the i you we your our my me us it its is are was be been am do does did have has had to for on in at by with
about from as if or and but so then also just now well yes of course please could would can may might what where when who how
that this there here thank thanks lovely great sorry oh dear right okay ok sure perfect wonderful understood
absolutely got noted brilliant happy let know tell say said again help team""".split())
# Everyday words a reworded QUESTION may use. Anything else new in a sentence must come from the caller's own words.
ASK_WORDS = set("""take reach give share provide spell repeat hear catch missed miss once more slowly clearly bit little moment
number name phone mobile landline call back ring contact best way details full letter first last wonderful happy glad
whats thats youre ive ill speaking talking pleasure able assist message""".split())
OPENERS = FILLER | set("any anything something thats im ive id ill".split())
OTHER_ASKS = {"name", "who", "number", "spell"}  # a reworded question must not start asking for one of these unless it did before
MAX_WORDS_FACTOR = 2.0
MIN_WORDS, EXTRA_WORDS = 3, 25
MODEL_OPTIONS = {"temperature": 0.7, "num_predict": 300}  # a little variety; the checks keep it safe
MODEL_TIMEOUT_S = 20  # per model call: a reply that takes longer is not worth waiting for, the fixed sentence is used
STOP = set("""that this with your have been will would could should about what when then them they their there from into
only just also some more than very much here where which while""".split())


def normal(text: str) -> str:
    """Lower case, letters and apostrophes only, single spaces, "'s" dropped ("number's" -> "number"): for phrase checks."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"'s\b", "", text)
    return " ".join(re.sub(r"[^a-z' ]", " ", text).split())


def words(text: str) -> list[str]:
    return normal(text).split()


def number_runs(text: str, strict: bool = False) -> list[tuple[str, ...]]:
    """Runs of consecutive number words ('oh one six three two'). A lone 'one' or 'oh' is ignored: it is a pronoun
    or an interjection ('which one?', 'Oh dear'), not part of a number. strict=True (information lines) counts every
    number word except a lone 'oh', so "half an hour" cannot lose its 'one'."""
    runs, current = [], []
    for w in words(text) + [""]:
        if w in NUMBER_WORDS:
            current.append(w)
            continue
        ignore = len(current) == 1 and (current[0] == "oh" or (current[0] == "one" and not strict))
        if current and not ignore:
            runs.append(tuple(current))
        current = []
    return runs


def sentences_of(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def unknown_names(new: str, allowed: set[str]) -> set[str]:
    """Capitalised words that nobody gave Holly: places, names, days. Every capitalised word counts, including the first
    of a sentence, unless that first word is an ordinary opener ('Lovely', 'Could', 'Thanks')."""
    found = set()
    for sentence in sentences_of(new):
        for i, m in enumerate(re.finditer(r"[A-Za-z][A-Za-z']*", sentence)):
            word = m.group()
            if not word[0].isupper() or word == "I" or word.startswith("I'"):
                continue
            key = normal(word)
            if key and key not in allowed and not (i == 0 and key in OPENERS):
                found.add(key)
    return found


def contains_exact(new: str, item: str, at_end_of_phrase: bool = False) -> bool:
    """`item` appears in `new` as whole words, not extended ('Gallagher-smith'); a reason must also end its phrase."""
    text = new.lower().replace("’", "'")
    wanted = re.escape(item.lower().replace("’", "'").strip())
    tail = r"(?=\s*[,.?!]|\s*$)" if at_end_of_phrase else r"(?![\w'-])"
    return re.search(r"(?<![\w'-])" + wanted + tail, text) is not None


def content_words(text: str) -> set[str]:
    """Words that carry information (4+ letters, not filler), with a trailing 's' removed."""
    return {w.rstrip("s") for w in words(text) if len(w) >= 4 and w not in STOP}


def question_part(text: str) -> str:
    return " ".join(s for s in sentences_of(text) if s.endswith("?"))


def check_sentence(new: str, line, said: str) -> list[str]:
    """Everything wrong with `new` as a rewording of `line` (a dialog.Part); an empty list = acceptable.
    `said` = what the caller said this turn (words the caller used may come back in a short reaction)."""
    draft, keep, ordered_numbers = line.text, line.keep, line.ordered_numbers
    problems = []
    new = new.strip()
    n_words, draft_words = len(new.split()), len(draft.split())
    if not new:
        return ["the sentence is empty"]
    if SPOKEN_FORBIDDEN.search(new):
        problems.append("it contains a digit or symbol: write everything in words, it is read aloud")
    if n_words < MIN_WORDS:
        problems.append("it is too short to be a sentence")
    if n_words > max(draft_words * MAX_WORDS_FACTOR, draft_words + EXTRA_WORDS):
        problems.append("it is much longer than the original: keep it short")
    if ("?" in draft) != ("?" in new):
        problems.append("the question was lost or added: it must contain a question mark exactly when the original does")
    for item in keep:
        if normal(item) and not contains_exact(new, item, at_end_of_phrase=item in line.keep_end):
            problems.append(f"it must contain '{item}' exactly, with nothing added to it")
    asked = question_part(new) if "?" in draft else new  # a question line is judged by its question, a statement as a whole
    if line.mention and not {w.rstrip("s") for w in line.mention} & {w.rstrip("s") for w in words(asked)}:
        problems.append("it no longer asks for the same thing: it must still mention " + " or ".join(f"'{w}'" for w in line.mention))
    if (line.mention or line.kind == "repeat_request") and "?" in draft:
        extra = (OTHER_ASKS - set(line.mention)) & set(words(asked)) - set(words(draft))
        if extra:
            problems.append("it asks for something else (" + ", ".join(sorted(extra)) + "): ask only what the original asks")
    info = line.coverage > 0
    draft_set, said_set = set(words(draft)), set(words(said))
    if info:
        needed, have = content_words(draft), content_words(new)
        if needed and len(needed & have) / len(needed) < line.coverage:
            problems.append("it dropped information from the original (" + ", ".join(sorted(needed - have)) + "): keep every fact")
    if info:
        novel = sorted(w for w in set(words(new)) - draft_set - FILLER if w not in NUMBER_WORDS)
        if novel:
            problems.append("it adds words that are not in the original (" + ", ".join(novel) + "): an information line must not grow new facts")
    else:  # a question or a short statement: new words must be friendly filler, ordinary asking words, or the caller's own words
        novel = sorted(w for w in set(words(new)) - draft_set - FILLER - ASK_WORDS - said_set if w not in NUMBER_WORDS and len(w) >= 3)
        if novel:
            problems.append("it adds words that neither you nor the caller said (" + ", ".join(novel) + "): do not react to things the caller did not say")
    draft_runs, new_runs = number_runs(draft, strict=info), number_runs(new, strict=info)
    if (new_runs != draft_runs) if (ordered_numbers or info) else (sorted(new_runs) != sorted(draft_runs)):
        problems.append("the numbers changed: say every number exactly as in the original, in the same order, nothing added or dropped")
    for name in sorted(unknown_names(new, draft_set | said_set)):
        problems.append(f"'{name}' is not in the original or in what the caller said: do not add names or places")
    risky = {w for w in words(new) if w in RISK_WORDS or w.endswith("n't")} - draft_set
    risky |= {w for w in words(new) if w in JOB_WORDS} - draft_set - said_set
    for risk in sorted(risky):
        problems.append(f"'{risk}' adds a promise, instruction, negation or fact that was not given to you: remove it")
    return problems


class Phrased(BaseModel):
    """The model's answer: one reworded sentence per line it was given, in the same order. Code checks each one."""
    sentences: list[str]

    @model_validator(mode="after")
    def every_sentence_is_acceptable(self, info: ValidationInfo) -> "Phrased":
        context = info.context or {}
        lines = context.get("lines", [])
        if len(self.sentences) != len(lines):
            raise ValueError(f"give exactly {len(lines)} sentence(s), one per numbered line, in order")
        problems = []
        for i, (new, line) in enumerate(zip(self.sentences, lines), start=1):
            problems += [f"line {i}: {p}" for p in check_sentence(new, line, context.get("said", ""))]
        if problems:
            raise ValueError("; ".join(problems))
        return self


def recent_turns(state, count: int = 3) -> str:
    """What the caller said in the last few turns (never Holly's own earlier lines: they may be safety advice)."""
    said = [entry["caller_text"] for entry in state.log[-count:] if entry.get("caller_text")]
    return "\n".join(f"Caller: {text}" for text in said) or "(this is the first turn)"


def used_openers(state, count: int = 3) -> list[str]:
    """The first word of Holly's last few replies ("Lovely"), so the next one can start differently. Not her sentences."""
    firsts = [entry["reply"].split()[0].strip(",.!?").lower() for entry in state.log[-count:] if entry.get("reply")]
    return sorted(set(firsts))


def build_messages(parts: list, state, said: str, persona) -> tuple[list[dict], list]:
    """The prompt, and the rewritable parts in the order the model must answer them. Fixed parts are not in it."""
    lines = [p for p in parts if p.rewrite]
    numbered = []
    for i, p in enumerate(lines, start=1):
        must = f"\n   (keep exactly: {', '.join(repr(k) for k in p.keep)})" if p.keep else ""
        numbered.append(f"{i}. {p.text}{must}")
    system = "\n".join([
        f"You are {persona.receptionist}, the automated phone assistant of {persona.business}. {persona.style}",
        "Your job is only to reword lines so they sound natural on the phone. You do not decide what to say: each line "
        "below already says the right thing.",
        "Rules:",
        "- Keep each line's meaning and its question. If it asks something, your version asks the same thing.",
        "- Never add facts, names, places, numbers, prices, times, advice or promises. Do not say what the team will do.",
        "- Never use a negative (not, never, no) unless the line does.",
        "- Copy names, reasons and every number word exactly as they are in the line.",
        "- Write numbers in words, no digits or symbols. Spoken English only.",
        "- You may start the first line with a very short reaction to what the caller just said, using only their words.",
        "- Do not greet again or say goodbye unless the line does.",
        "- Do not copy a line word for word: say it the way a friendly person would on the phone, in your own words.",
        "- If the caller sounds worried, begin with a brief kind reaction. Never react to something the caller did not say.",
        "Examples of the style (other situations): 'Could I take your name, please?' -> 'Of course, and who am I speaking to?'; "
        "'And what is the best number to call you back on?' -> 'Lovely. What is the best number to reach you on?'; "
        "'Is there anything else I can help you with?' -> 'Is there anything else you need while I have you?'",
        'Answer as JSON: {"sentences": [one string per numbered line, in the same order]}.',
    ])
    openers = used_openers(state)
    avoid = ("\nDo not start with these words, you used them recently: " + ", ".join(openers) + ".") if openers else ""
    user = (f"What the caller said so far:\n{recent_turns(state)}\n\nThe caller just said: {said!r}\n\n"
            "Lines to reword (answer with the same number of sentences):\n" + "\n".join(numbered) + avoid)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], lines


def make_phraser(chat=structured_chat, llm: str | None = None, timeout: float = MODEL_TIMEOUT_S):
    """The function next_reply calls. `chat` can be replaced by a fake in tests."""

    def phrase(parts: list, state, said: str, persona) -> tuple[str, dict]:
        messages, lines = build_messages(parts, state, said, persona)
        fixed_text = " ".join(p.text for p in parts)
        trace = {"used": "fixed", "attempts": 0, "problems": None, "fixed": [p.text for p in lines]}
        try:
            reply = chat(Phrased, messages, llm=llm, context={"lines": lines, "said": said}, options=MODEL_OPTIONS,
                         fix_hint="Fix those lines. Keep the same meaning, add nothing, and return the corrected JSON.",
                         timeout=timeout)
        except LLMFormError as problem:  # invalid twice: the fixed sentences are the safe answer
            trace.update(attempts=2, problems=str(problem).splitlines()[1:3])
            return fixed_text, trace
        except Exception as problem:  # the model is down, too slow, anything: never break the call
            trace["problems"] = [f"{type(problem).__name__}: {problem}"]
            return fixed_text, trace
        reworded = iter(reply.value.sentences)
        out = [next(reworded).strip() if p.rewrite else p.text for p in parts]
        trace.update(used="model", attempts=reply.attempts, reworded=list(reply.value.sentences),
                     problems=reply.rejected_because)
        return " ".join(out), trace

    return phrase
