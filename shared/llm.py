"""Ask the local LLM to fill in a Pydantic form: the pattern shared by Part 1 (voicemails) and Part 2 (meetings).

    reply = structured_chat(MySchema, messages, context={...})
    reply.value     # a validated MySchema
    reply.attempts  # 1, or 2 if the retry was needed

1. Ollama chat with format=<the schema's JSON schema>: the reply MUST have that shape.
2. Pydantic validates it, including our plain-code rules (validators). `context` is passed to the
   validators, for rules that need outside facts (e.g. who is on the team, what the transcript says).
3. If validation fails: show the model its answer and the errors, ask once more. Still invalid -> LLMFormError.
"""

import os
import time
from dataclasses import dataclass

import ollama
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv()

DEFAULT_FIX_HINT = ("Fix the values using the source text; do not replace a value that was actually "
                    "said with null. Return the corrected JSON.")


class LLMFormError(Exception):
    """The model gave an invalid answer twice."""


@dataclass
class StructuredReply:
    value: BaseModel
    attempts: int  # 1 = valid first time, 2 = needed the retry
    rejected_reply: str | None  # the first answer, if it was rejected
    rejected_because: str | None
    seconds: float


def settings() -> dict:
    return {
        "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        "model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
    }


def short_errors(error: ValidationError) -> str:
    """Turn Pydantic's error list into a few readable lines for the model (and for us)."""
    lines = []
    for e in error.errors():
        where = ".".join(str(part) for part in e["loc"]) or "answer"
        lines.append(f"- {where}: {e['msg']}")
    return "\n".join(lines)


def structured_chat(schema: type[BaseModel], messages: list[dict], llm: str | None = None,
                    context: dict | None = None, fix_hint: str = DEFAULT_FIX_HINT,
                    options: dict | None = None) -> StructuredReply:
    """Get a validated `schema` from the LLM, with exactly one retry on invalid output."""
    cfg = settings()
    client = ollama.Client(host=cfg["host"])
    messages = list(messages)  # we append to it; don't change the caller's list
    started = time.perf_counter()
    replies, rejected_because = [], None

    for attempt in (1, 2):  # first try + exactly one retry
        response = client.chat(
            model=llm or cfg["model"],
            messages=messages,
            format=schema.model_json_schema(),
            options={"temperature": 0, **(options or {})},  # temperature 0: repeatable runs
        )
        raw = response.message.content
        replies.append(raw)
        try:
            value = schema.model_validate_json(raw, context=context)
            return StructuredReply(value, attempt, replies[0] if attempt == 2 else None, rejected_because,
                                   round(time.perf_counter() - started, 2))
        except ValidationError as error:
            if attempt == 2:
                raise LLMFormError(f"invalid answer twice.\n{short_errors(error)}\nReplies: {replies}") from error
            rejected_because = short_errors(error)
            # Show the model its own answer and what was wrong with it. Without this feedback a
            # temperature-0 model would just repeat the same answer.
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"Your answer broke these rules:\n{rejected_because}\n{fix_hint}"},
            ]
