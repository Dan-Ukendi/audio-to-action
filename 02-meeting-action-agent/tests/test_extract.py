"""Chunking and merging, without the LLM (instant)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 02-meeting-action-agent

from extract import chunk_segments, format_transcript, ground_owner, merge_items, same_task  # noqa: E402
from shared.schemas import ActionItem, Segment  # noqa: E402


def seg(start: float, words: int) -> Segment:
    return Segment(start=start, end=start + 1, text=" ".join(["word"] * words), avg_logprob=0, no_speech_prob=0)


def item(task: str, status: str = "open", owner=None, due_text=None) -> ActionItem:
    return ActionItem(evidence="e", task=task, owner=owner, due_text=due_text, status=status)


def test_short_meeting_is_one_chunk():
    segments = [seg(i, 10) for i in range(20)]  # 200 words
    assert chunk_segments(segments, max_words=1200) == [segments]


def test_long_meeting_splits_with_one_segment_overlap():
    segments = [seg(i, 10) for i in range(10)]  # 100 words
    chunks = chunk_segments(segments, max_words=40)
    assert all(sum(len(s.text.split()) for s in c) <= 40 for c in chunks)
    assert all(a[-1] is b[0] for a, b in zip(chunks, chunks[1:]))  # overlap
    # Drop each later chunk's first (overlap) segment: what's left is every segment once, in order.
    assert chunks[0] + [s for c in chunks[1:] for s in c[1:]] == segments


def test_format_transcript_shows_minutes_and_seconds():
    assert format_transcript([Segment(start=75.4, end=80, text="Hi.", avg_logprob=0, no_speech_prob=0)]) == "[01:15] Hi."


def test_same_task_tolerates_rewording_but_not_other_tasks():
    assert same_task("Send Siobhan Gallagher the quote", "Send the quote to Siobhan Gallagher")
    assert not same_task("Order the Gallagher bathroom suite", "Send Siobhan Gallagher the quote")


def test_merge_later_status_wins_and_fills_gaps():
    first = [item("Fit the valve at Mill Lane", owner="Tom", due_text="Friday")]
    second = [item("Fit the valve at Mill Lane", status="done")]
    merged = merge_items([first, second])
    assert len(merged) == 1
    assert (merged[0].status, merged[0].owner, merged[0].due_text) == ("done", "Tom", "Friday")


def test_merge_never_joins_two_items_of_the_same_chunk():
    # "ordered it, I'll fit it Friday": two items whose words overlap a lot (same_task says yes)
    one_chunk = [item("Order the Gallagher bathroom suite", status="done"),
                 item("Fit the Gallagher bathroom suite", due_text="Monday the 12th")]
    assert same_task(one_chunk[0].task, one_chunk[1].task)
    assert [i.task for i in merge_items([one_chunk])] == [i.task for i in one_chunk]


def test_merge_exact_repeat_in_one_chunk():
    # The model sometimes writes one item per line: "Jamie, the Ellis gas check." / "Booked for Friday."
    one_chunk = [item("Book Ellis Gas Check", owner="Jamie", due_text="the 2nd of October"),
                 item("Book Ellis gas check", status="done"),
                 item("Do Ellis Gas Check", owner="Tom")]
    merged = merge_items([one_chunk])
    assert [(i.task, i.status, i.owner) for i in merged] == [("Book Ellis Gas Check", "done", "Jamie"),
                                                             ("Do Ellis Gas Check", "open", "Tom")]


def test_merge_repeated_item_in_overlap_doesnt_swallow_its_neighbour():
    first = [item("Order the Gallagher bathroom suite", status="done")]
    second = [item("Order the Gallagher bathroom suite", status="done"),  # the overlap segment, seen again
              item("Fit the Gallagher bathroom suite")]
    merged = merge_items([first, second])
    assert [(i.task, i.status) for i in merged] == [("Order the Gallagher bathroom suite", "done"),
                                                     ("Fit the Gallagher bathroom suite", "open")]


def lines(*texts: str) -> list[Segment]:
    return [Segment(start=i, end=i + 1, text=t, avg_logprob=0, no_speech_prob=0) for i, t in enumerate(texts)]


def test_owner_kept_when_named_just_before():
    segments = lines("Jamie, can you take that one?", "Yes, I'll do it tomorrow.")
    owned = ActionItem(evidence="Yes, I'll do it tomorrow.", task="Fix van light", owner="Jamie",
                       due_text="tomorrow", status="open")
    assert ground_owner(owned, segments) == "Jamie"


def test_owner_dropped_when_nobody_is_named():
    segments = lines("We need to book the gas check for Margaret Ellis.", "I'll book that in.", "Thanks everyone.")
    guessed = ActionItem(evidence="I'll book that in.", task="Book gas check", owner="Sam",
                         due_text=None, status="open")
    assert ground_owner(guessed, segments) is None


def test_owner_kept_when_named_with_possessive():
    segments = lines("Right, Tom's doing the valve on Friday.")
    owned = ActionItem(evidence="Tom's doing the valve on Friday.", task="Fit the valve", owner="Tom",
                       due_text="on Friday", status="open")
    assert ground_owner(owned, segments) == "Tom"
