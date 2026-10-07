"""The sync AGENT: relate one meeting's extracted items to the task tracker, using tools in a loop.

    run = sync_meeting(conn, meeting_id, meeting_date, items, trace_dir=...)

This is the only place in the project where the model decides what happens next. So it is boxed in:
  - TOOLS: search_tasks, add_task, update_task, skip_item, finish. Nothing else exists for it.
  - It decides RELATIONSHIPS only. Tools take an item id; code copies owner / due / status from that
    item, so the model can't retype a date or invent an owner (and update never wipes a known value).
  - Every argument is checked by a Pydantic model; bad calls get an error message back, not a crash.
  - Each item gets exactly one action; finish is refused while items are unhandled.
  - add_task refuses an item that looks like an open task (duplicate) unless the model confirms it's new.
  - update_task refuses an item and a task with too few meaningful words in common (blocks guessed task ids).
  - A task can be changed by only one item per meeting; a "repeat" must mostly match the item it repeats;
    a done/cancelled item about a tracked task can't be skipped as "not work".
  - A step limit (= LLM calls) caps time and cost; items left at the limit go to a review list.
  - A failed LLM reply (e.g. the model looping) costs one step, not the run; the next input is changed.
  - Every change is stored with the agent's (non-empty) reason (tracker.changes) and every step in a JSON trace.

State, not history: each step the model gets a fresh message with the CURRENT state (items and what
was done to them, the relevant tracker tasks, feedback on its last calls), not the whole chat so far.
A first version that kept the chat history replayed its old failing calls every step and finally
"fixed" the errors by guessing task ids (see docs/learning-log.md, Part 2 Phase 4).
"""

import json
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import ollama
from pydantic import AfterValidator, BaseModel, Field, ValidationError

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import context  # noqa: E402
import tracker  # noqa: E402
from shared.llm import settings  # noqa: E402
from shared.schemas import ExtractedItem  # noqa: E402

SYSTEM_PROMPT = f"""\
You keep the task tracker of {context.BUSINESS} up to date after each weekly team meeting.
You get the action items extracted from today's meeting (ids A, B, C, ...) and the tracker's tasks (#1, #2, ...).
Item letters and task numbers are unrelated: item C is NOT task #3.
For EACH item choose exactly one action:
- update_task(task_id, item_id): the item is about a task already in the tracker (the same piece of work,
  even if worded differently): it is now done, cancelled, reassigned, re-dated, or just mentioned again.
- add_task(item_id): the item is new work that is not in the tracker. A follow-up job is NEW work
  (e.g. "fit the valve" after "order the valve" is a new task, not an update).
- skip_item(item_id, kind): kind "not_work" for a postponed idea or small talk, or kind "repeat_of_item"
  (with other_item) when it repeats ANOTHER ITEM of today's meeting. Never skip an item because it is
  like a tracker task: that is update_task (same work) or add_task (new work).
If the tracker has no task about the same work, the item is new: use add_task.
Use search_tasks when an item might match a task that is not in the list, or when unsure.
Always give a short reason. The owner, due date and status are copied from the item automatically.
When every item is handled, call finish.
"""


# ---------------------------------------------------------------- tool arguments (validated)

def not_blank(text: str) -> str:
    """Every tracker change must say why: an empty or blank reason is refused like any other bad argument.
    (An AfterValidator, so the JSON schema the model sees is unchanged.)"""
    if not text.strip():
        raise ValueError("reason must not be empty: say briefly why")
    return text


Reason = Annotated[str, AfterValidator(not_blank)]


class SearchArgs(BaseModel):
    query: str = Field(description="Words to look for in task descriptions (all statuses).")


class AddArgs(BaseModel):
    item_id: str = Field(description="The item to add as a new task, e.g. 'B'.")
    reason: Reason
    confirm_new: bool = Field(default=False, description="Set true only if a similar open task exists but this "
                                                         "really is different work.")


class UpdateArgs(BaseModel):
    # Required on purpose: the schema is part of the prompt. When it was optional, the model left it out and
    # wrote "task #1" in its reason instead. A missing id is answered with advice (see bad_arguments()).
    task_id: int = Field(description="The id number of the existing tracker task this item is about, e.g. 3.")
    item_id: str
    reason: Reason


class SkipArgs(BaseModel):
    item_id: str
    kind: Literal["not_work", "repeat_of_item"]
    other_item: str | None = Field(default=None, description="For repeat_of_item: the item it repeats, e.g. 'A'.")
    reason: Reason


class FinishArgs(BaseModel):
    summary: str = Field(description="One sentence: what changed in the tracker.")


TOOLS: dict[str, tuple[type[BaseModel], str]] = {
    "search_tasks": (SearchArgs, "Search all tracker tasks (open, done, cancelled) by words."),
    "add_task": (AddArgs, "Add an item as a NEW task."),
    "update_task": (UpdateArgs, "Apply an item to an EXISTING task (status, owner, due come from the item)."),
    "skip_item": (SkipArgs, "Handle an item without changing the tracker (not real work, or a repeat)."),
    "finish": (FinishArgs, "Call when every item has been handled."),
}


def tool_schemas() -> list[dict]:
    """The tool list in the format Ollama expects; parameters come straight from the Pydantic models."""
    return [{"type": "function", "function": {"name": name, "description": text, "parameters": args.model_json_schema()}}
            for name, (args, text) in TOOLS.items()]


# ---------------------------------------------------------------- one run

@dataclass
class SyncRun:
    meeting: str
    items: dict[str, ExtractedItem]
    handled: dict[str, str] = field(default_factory=dict)  # item id -> "add #3" / "update #1" / "skip"
    changed_tasks: dict[int, str] = field(default_factory=dict)  # task id -> the item that added/updated it
    trace: list[dict] = field(default_factory=list)
    finished: bool = False
    llm_calls: int = 0
    seconds: float = 0.0

    def unhandled(self) -> list[str]:
        return [i for i in self.items if i not in self.handled]


def item_letter(n: int) -> str:
    """A, B, ..., Z, AA, AB, ...: item ids that can't be mistaken for task numbers."""
    return chr(65 + n) if n < 26 else chr(64 + n // 26) + chr(65 + n % 26)


def show_item(item_id: str, item: ExtractedItem) -> str:
    due = item.due.isoformat() if item.due else "none"
    return (f"{item_id} [{item.status}] {item.task} | owner: {item.owner or 'unknown'} | due: {due} "
            f"| said: \"{item.evidence}\"")


def show_task(task: dict) -> str:
    return f"#{task['id']} [{task['status']}] {task['task']} | owner: {task['owner'] or 'unknown'} | due: {task['due'] or 'none'}"


# Word overlap (Jaccard) between an item and a task. Below MIN_SIMILARITY they can't be the same work
# ("Fix the van's brake light" vs "Renew the van insurance" share only "van": 0.14). At DUPLICATE_SIMILARITY
# or more, adding the item as a new task is probably a duplicate.
MIN_SIMILARITY = 0.2
REPEAT_SIMILARITY = 0.5  # two items of one meeting are "the same" only if most words match
DUPLICATE_SIMILARITY = 0.6


def similar_open_task(conn: sqlite3.Connection, task_text: str) -> dict | None:
    """An open task that looks like the same work as task_text: probably a duplicate."""
    return next((t for t in tracker.all_tasks(conn, status="open")
                 if tracker.similarity(t["task"], task_text) >= DUPLICATE_SIMILARITY), None)


def bad_arguments(name: str, raw_args: dict, error: ValidationError) -> str:
    """Turn a validation error into advice the model can act on (a bare type error just gets repeated)."""
    if name == "update_task" and any(e["loc"] == ("task_id",) for e in error.errors()):
        item_id = raw_args.get("item_id", "the item")
        return (f"ERROR: update_task needs task_id, the number of an existing tracker task (e.g. task_id=3). "
                f"If {item_id} is new work, use add_task instead.")
    first = error.errors()[0]
    return f"ERROR: bad arguments for {name}: {first['msg']} ({first['loc']})."


def run_tool(run: SyncRun, conn: sqlite3.Connection, name: str, raw_args: dict) -> str:
    """Execute one tool call. Every problem becomes an 'ERROR: ...' message for the model, never a crash."""
    if name not in TOOLS:
        return f"ERROR: unknown tool '{name}'. Available: {', '.join(TOOLS)}."
    try:
        args = TOOLS[name][0].model_validate(raw_args)
    except ValidationError as error:
        return bad_arguments(name, raw_args, error)

    if name == "search_tasks":
        found = tracker.search_tasks(conn, args.query)
        return "\n".join(show_task(t) for t in found) or "No matching tasks."
    if name == "finish":
        return finish(run)
    # The three item tools share two rules: the item must exist, and it gets exactly one action.
    if args.item_id not in run.items:
        return f"ERROR: there is no item '{args.item_id}'. Items: {', '.join(run.items)}."
    if args.item_id in run.handled:
        return f"ERROR: item {args.item_id} was already handled ({run.handled[args.item_id]}). One action per item."
    item_tools = {"add_task": add_item, "update_task": update_item, "skip_item": skip_item}
    return item_tools[name](run, conn, args)


def finish(run: SyncRun) -> str:
    if run.unhandled():
        return f"ERROR: not finished: items {', '.join(run.unhandled())} are not handled yet."
    run.finished = True
    return "OK, finished."


def add_item(run: SyncRun, conn: sqlite3.Connection, args: AddArgs) -> str:
    """add_task: the item becomes a new task, unless it looks like an open task (then it must be confirmed)."""
    item = run.items[args.item_id]
    twin = similar_open_task(conn, item.task)
    if twin and not args.confirm_new:
        return (f"ERROR: open task #{twin['id']} ('{twin['task']}') looks like the same work as {args.item_id}. "
                f"If it is, use update_task(task_id={twin['id']}, item_id={args.item_id}). If it really is "
                f"different work, call add_task again with confirm_new=true.")
    task = tracker.add_task(conn, item.task, item.owner, iso(item.due), item.status, run.meeting, args.reason)
    record(run, args.item_id, "add", task["id"])
    return f"OK, added {show_task(task)}"


def update_item(run: SyncRun, conn: sqlite3.Connection, args: UpdateArgs) -> str:
    """update_task: the item is news about an existing task. Three checks stop guessed or double links."""
    item = run.items[args.item_id]
    existing = tracker.get_task(conn, args.task_id)
    if existing is None:
        return (f"ERROR: there is no task #{args.task_id} in the tracker. If {args.item_id} is new work, "
                f"use add_task; otherwise use the id of the matching task from the list.")
    changed_by = run.changed_tasks.get(existing["id"])
    if changed_by:
        return (f"ERROR: task #{existing['id']} was already changed by item {changed_by} in this meeting. "
                f"If {args.item_id} says the same thing, skip_item it as repeat_of_item {changed_by}; "
                f"if it is different work, use add_task.")
    if tracker.similarity(existing["task"], item.task) < MIN_SIMILARITY:
        return (f"ERROR: task #{existing['id']} ('{existing['task']}') and {args.item_id} ('{item.task}') have too "
                f"little in common to be the same work. Pick the right task, or add_task if it is new.")
    task = tracker.update_task(conn, args.task_id, item.owner, iso(item.due), item.status, run.meeting, args.reason)
    record(run, args.item_id, "update", task["id"])
    return f"OK, now {show_task(task)}"


def skip_item(run: SyncRun, conn: sqlite3.Connection, args: SkipArgs) -> str:
    """skip_item: no tracker change. A repeat must mostly match another real item of today (not a tracker
    task: that's update_task); done/cancelled news about a tracked task is never 'not_work'."""
    item = run.items[args.item_id]
    if args.kind == "repeat_of_item":
        other = run.items.get(args.other_item) if args.other_item != args.item_id else None
        if other is None:
            return (f"ERROR: repeat_of_item needs other_item = another item of today ({', '.join(run.items)}). "
                    f"If {args.item_id} is about a tracker task, use update_task.")
        if tracker.similarity(item.task, other.task) < REPEAT_SIMILARITY:
            return (f"ERROR: {args.item_id} ('{item.task}') and {args.other_item} ('{other.task}') "
                    f"are different work, so it isn't a repeat. Use update_task or add_task for {args.item_id}.")
        run.handled[args.item_id] = f"skip (= {args.other_item})"
    else:
        # Done/cancelled news about a task we track is real work, not "not_work"... but a cancelled idea that
        # never became a task ("let's leave the website for now") is fine to skip. Only refuse when some
        # tracker task resembles it (Phase 6: refusing always made the agent loop to its step limit).
        related = any(tracker.similarity(t["task"], item.task) >= MIN_SIMILARITY for t in tracker.all_tasks(conn))
        if item.status != "open" and related:
            return (f"ERROR: {args.item_id} says work is {item.status}, which is news about real work, not 'not_work'. "
                    f"Use update_task on the task it is about (or add_task if the tracker doesn't have it).")
        run.handled[args.item_id] = "skip"
    return f"OK, {args.item_id} skipped."


def iso(day: date | None) -> str | None:
    return day.isoformat() if day else None


def record(run: SyncRun, item_id: str, action: str, task_id: int) -> None:
    """Remember what an item did, and which item changed which task (a task may change once per meeting)."""
    run.handled[item_id] = f"{action} #{task_id}"
    run.changed_tasks[task_id] = item_id


def state_message(run: SyncRun, conn: sqlite3.Connection, meeting_date: date, feedback: list[str]) -> str:
    """Everything the model needs for its next step, rebuilt from scratch every step."""
    def item_line(item_id: str) -> str:
        done = run.handled.get(item_id)
        return f"{show_item(item_id, run.items[item_id])}  ->  " + (f"HANDLED ({done})" if done else "TO DO")

    all_tasks = tracker.all_tasks(conn)
    tasks = [t for t in all_tasks if t["status"] == "open" or t["id"] in run.changed_tasks]
    # "Empty" only when it really is: with only done/cancelled tasks an item may still be about one of them.
    nothing = ("(no open tasks: use search_tasks to find done or cancelled ones)" if all_tasks
               else "(the tracker is empty: every item is new work)")
    lines = [f"Meeting {run.meeting} on {meeting_date:%A %Y-%m-%d}.", "", "Items from today's meeting:"]
    lines += [item_line(i) for i in run.items]
    lines += ["", "Tracker tasks (open, plus any you changed today):"]
    lines += [show_task(t) for t in tasks] or [nothing]
    if feedback:
        lines += ["", "Results of your last tool calls:"] + feedback
    todo = run.unhandled()
    lines += ["", f"Still to do: {', '.join(todo)}." if todo else "All items are handled: call finish."]
    return "\n".join(lines)


def sync_meeting(conn: sqlite3.Connection, meeting: str, meeting_date: date, items: list[ExtractedItem],
                 trace_dir: str | Path | None = None, llm: str | None = None, max_steps: int | None = None) -> SyncRun:
    # Letters for items, numbers for tasks: with "I5" and "#5" the model linked them because the numbers matched.
    run = SyncRun(meeting, {item_letter(n): item for n, item in enumerate(items)})
    max_steps = max_steps or 2 * len(items) + 4  # enough for one action per item plus a few corrections
    client = ollama.Client(host=settings()["host"])
    feedback: list[str] = []
    run.finished = not run.items  # a meeting without items needs no LLM call

    last_state = None
    for step in range(1, max_steps + 1):
        if run.finished:
            break
        state = state_message(run, conn, meeting_date, feedback)
        if state == last_state:
            # Nothing changed since the last step, and at temperature 0 the same input gives the same reply:
            # every further call would repeat the same refused calls. Stop; the items go to the review list.
            run.trace.append({"step": step, "tool": None, "stopped": "no progress: same input as the last step"})
            break
        last_state = state
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": state}]
        started = time.perf_counter()
        try:
            # num_predict caps the reply: a model stuck repeating itself stops after ~1000 tokens, not minutes.
            response = client.chat(model=llm or settings()["model"], messages=messages, tools=tool_schemas(),
                                   options={"temperature": 0, "num_predict": 1024})
        except ollama.ResponseError as error:
            # e.g. "token repeat limit reached": the model looped. At temperature 0 the same input loops
            # again, so the next step's input must be different: the feedback asks for a shorter reply.
            run.llm_calls += 1
            run.seconds += time.perf_counter() - started
            run.trace.append({"step": step, "tool": None, "error": str(error)})
            feedback = ["Your last reply broke off (it kept repeating itself). This time make at most 3 tool calls."]
            continue
        run.llm_calls += 1
        run.seconds += time.perf_counter() - started
        calls = response.message.tool_calls or []
        feedback = []
        if not calls:  # the model answered in prose: steer it back to the tools
            run.trace.append({"step": step, "tool": None, "said": response.message.content})
            feedback = ["You answered in text. Use the tools."]
            continue
        for call in calls:
            args = dict(call.function.arguments)
            result = run_tool(run, conn, call.function.name, args)
            run.trace.append({"step": step, "tool": call.function.name, "args": args, "result": result})
            feedback.append(f"{call.function.name}({json.dumps(args)}) -> {result}")
            if run.finished:
                break  # calls after a successful finish are ignored

    if trace_dir:
        write_trace(run, Path(trace_dir), max_steps)
    return run


def write_trace(run: SyncRun, trace_dir: Path, max_steps: int) -> None:
    """Every step of the run as JSON; items left unhandled at the step limit are listed for human review."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / f"{run.meeting}.trace.json").write_text(json.dumps({
        "meeting": run.meeting, "finished": run.finished, "llm_calls": run.llm_calls, "max_steps": max_steps,
        "seconds": round(run.seconds, 1), "handled": run.handled, "unhandled_for_review": run.unhandled(),
        "items": {i: item.model_dump(mode="json") for i, item in run.items.items()}, "trace": run.trace,
    }, indent=2), encoding="utf-8")
