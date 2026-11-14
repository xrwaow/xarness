"""Unit tests for the pure transcript reducer (no terminal required)."""

from __future__ import annotations

import unittest

from xarness.conversation import SUMMARY_PREFIX, Timing, ToolCall, Usage
from xarness.conversation import AssistantMessage, ToolMessage, TurnStats, user_message
from xarness.protocol import (
    ContentDelta,
    ProcessingStarted,
    ReasoningDelta,
    Snapshot,
    Status,
    StreamError,
    ToolCallArgumentsDelta,
    ToolCallArgumentsDone,
    ToolCallStarted,
    ToolFinished,
    ToolUsageChanged,
    RoundComplete,
    MessageAdded,
    TurnEnded,
)
from xarness.tui import transcript as tr


def _snap(messages):
    return Snapshot(status=Status(state="idle"), messages=messages)


def _user(text, mid="u"):
    return user_message(text, id=mid)


def _stamped(start: str, end: str, **stats) -> Timing:
    """The timing a finished turn's last answer carries: the host's numbers."""
    figures = {"total": 5.0, "busy": 4.0, "idle": 1.0}
    figures.update(stats)
    return Timing(start=start, end=end, stats=TurnStats(**figures))


class TranscriptTest(unittest.TestCase):
    def test_load_rebuilds_messages_tools_and_timing(self):
        user = user_message("hi", id="u",
                       timing=Timing(send="2025-01-01T00:00:00+00:00"))
        call = ToolCall(call_id="c1", name="read_file", arguments_json="{}")
        assistant = AssistantMessage(content="", tool_calls=[call],
            timing=_stamped(
                "2025-01-01T00:00:00+00:00",
                "2025-01-01T00:00:01+00:00",
            ),
        )
        result = ToolMessage(tool_call_id="c1", content="body", status="error")
        t = tr.Transcript()
        items = t.load(_snap([user, assistant, result]))
        self.assertIsInstance(items[0], tr.UserItem)
        self.assertIsInstance(items[1], tr.ToolItem)
        self.assertEqual(items[1].status, tr.ToolStatus.CALL_FAILED)
        self.assertEqual(items[1].output, "body")
        self.assertIsInstance(items[2], tr.TurnSummaryItem)
        self.assertIsNotNone(items[2].total)

    def test_load_renders_hidden_epochs_before_the_current_one(self):
        """Compaction hides old epochs from the model, not from the human:
        a resumed session must still show the pre-handoff transcript."""
        old_user = _user("before", mid="u1")
        old_answer = AssistantMessage(content="old answer", id="a1")
        summary = user_message(
            f"{SUMMARY_PREFIX}\n\nthe handoff", id="s1", is_summary=True,
        )
        new_answer = AssistantMessage(content="after", id="a2")
        snap = Snapshot(
            status=Status(state="idle"),
            messages=[old_user, old_answer, summary, new_answer],
        )
        items = tr.Transcript().load(snap)
        self.assertEqual(
            [type(item).__name__ for item in items],
            ["UserItem", "AssistantItem", "DividerItem", "AssistantItem"],
        )
        self.assertEqual(items[0].text, "before")
        self.assertIn("the handoff", items[2].text)
        self.assertEqual(items[3].chunks, ["after"])

    def test_history_tool_results_resolve_across_the_epoch_boundary(self):
        call = ToolCall(call_id="c1", name="read_file", arguments_json="{}")
        assistant = AssistantMessage(content="", tool_calls=[call], id="a1")
        result = ToolMessage(tool_call_id="c1", content="body", id="t1")
        snap = Snapshot(
            status=Status(state="idle"),
            messages=[_user("hi"), assistant, result],
        )
        items = tr.Transcript().load(snap)
        tool = next(item for item in items if isinstance(item, tr.ToolItem))
        self.assertEqual(tool.status, tr.ToolStatus.CALL_SUCCEEDED)
        self.assertEqual(tool.output, "body")

    def test_partial_round_renders_without_replay(self):
        from xarness.protocol import PartialRound, PartialToolCall

        assistant = AssistantMessage(content="",
            tool_calls=[ToolCall(call_id="c1", name="read_file",
                                 arguments_json="{")],
        )
        snap = _snap([_user("hi"), assistant])
        snap.partial_round = PartialRound(
            reasoning="still thinking", content="partial",
            tool_calls=[PartialToolCall(
                call_id="c1", name="read_file", arguments="{\"path\": \"a\"}",
                content="body",
            )],
        )
        items = tr.Transcript().load(snap)
        thinking = next(i for i in items if isinstance(i, tr.ThinkingItem))
        self.assertEqual(thinking.reasoning, "still thinking")
        self.assertTrue(thinking.done)
        live = [i for i in items if isinstance(i, tr.AssistantItem)]
        self.assertEqual(live[-1].live, "partial")
        tools = [i for i in items if isinstance(i, tr.ToolItem)]
        self.assertEqual(len(tools), 1)  # updated in place, not duplicated
        self.assertEqual(tools[0].args, '{"path": "a"}')
        self.assertEqual(tools[0].output, "body")
        self.assertIs(tools[0].status, tr.ToolStatus.MAKING_CALL)

    def test_partial_reasoning_only_is_still_streaming(self):
        from xarness.protocol import PartialRound

        snap = _snap([_user("hi")])
        snap.partial_round = PartialRound(reasoning="thinking")
        items = tr.Transcript().load(snap)
        thinking = next(i for i in items if isinstance(i, tr.ThinkingItem))
        self.assertFalse(thinking.done)

    def test_live_and_replay_match(self):
        events = [
            ReasoningDelta(text="think", t=0.0),
            ContentDelta(text="hello", t=0.5),
            RoundComplete(t=1.0),
        ]
        live = tr.Transcript()
        for event in events:
            live.apply(event)
        replay = tr.Transcript()
        assistant = AssistantMessage(content="hello", reasoning="think",
                            timing=Timing(reasoning_seconds=0.5))
        items = replay.load(_snap([assistant]))
        self.assertEqual(live.thinking.reasoning, "think")
        self.assertEqual(live.thinking.duration, 0.5)
        self.assertEqual(live.assistant.chunks, ["hello"])
        self.assertEqual(items[0].reasoning, "think")
        self.assertEqual(items[1].chunks, ["hello"])

    def test_cot_duration_is_stamped_when_the_block_ends(self):
        """The pill gets its number at the first answer token, not at
        RoundComplete: the answer stream must never read "Thought for …"
        (and the number must not grow with the answer's length)."""
        t = tr.Transcript()
        t.apply(ProcessingStarted(t=0.0))
        t.apply(ReasoningDelta(text="think", t=1.0))
        t.apply(ReasoningDelta(text=" more", t=3.0))
        changed = t.apply(ContentDelta(text="answer", t=4.5))
        self.assertIn(t.thinking, changed)  # repainted, not left stale
        self.assertEqual(t.thinking.summary_text(), "Thought for 3.5s")
        t.apply(ContentDelta(text=" answer", t=9.0))
        t.apply(RoundComplete(t=20.0))
        self.assertEqual(t.thinking.duration, 3.5)

    def test_tool_call_ends_the_cot_block_not_the_round(self):
        """Tool-argument streaming is call time: the pill must stop at the
        first tool call, not at RoundComplete (what it used to do)."""
        t = tr.Transcript()
        t.apply(ProcessingStarted(t=0.0))
        t.apply(ReasoningDelta(text="think", t=1.0))
        t.apply(ToolCallStarted(call_id="c", name="run_bash", t=2.0))
        self.assertEqual(t.thinking.summary_text(), "Thought for 1.0s")
        t.apply(ToolCallArgumentsDelta(call_id="c", text='{"command":"ls"}',
                                       t=8.0))
        t.apply(RoundComplete(has_tool_calls=True, t=9.0))
        self.assertEqual(t.thinking.summary_text(), "Thought for 1.0s")

    def test_reasoning_only_round_falls_back_to_round_complete(self):
        t = tr.Transcript()
        t.apply(ProcessingStarted(t=0.0))
        t.apply(ReasoningDelta(text="think", t=1.0))
        t.apply(RoundComplete(t=6.0))
        self.assertEqual(t.thinking.summary_text(), "Thought for 5.0s")

    def test_tool_lifecycle_and_cancel(self):
        t = tr.Transcript()
        self.assertEqual(t.apply(MessageAdded(message=user_message("u"))), t.items)
        t.apply(ToolCallStarted(call_id="c", name="run_bash"))
        t.apply(ToolCallArgumentsDelta(call_id="c", text='{"command":'))
        t.apply(ToolCallArgumentsDone(call_id="c", name="run_bash",
                                      arguments_json='{"command": "ls"}'))
        tool = t.tool_blocks["c"]
        self.assertEqual(tool.args, '{"command": "ls"}')
        t.apply(ToolFinished(call_id="c", status='ok', content="ok"))
        self.assertEqual(tool.status, tr.ToolStatus.CALL_SUCCEEDED)
        t.apply(StreamError(message="boom"))
        self.assertEqual(t.cancel(), [])
        # Events after cancel are ignored.
        self.assertEqual(t.apply(ContentDelta(text="late")), [])

    def test_subagent_usage_is_live_then_final_and_reloads(self):
        t = tr.Transcript()
        t.apply(ToolCallStarted(call_id="c", name="subagent"))
        t.apply(ToolUsageChanged(call_id="c", usage=Usage(input_tokens=7, output_tokens=3)))
        tool = t.tool_blocks["c"]
        self.assertEqual((tool.usage.input_tokens, tool.usage.output_tokens),
                         (7, 3))
        t.apply(ToolFinished(call_id="c", status="ok", content="done"))
        self.assertEqual((tool.usage.input_tokens, tool.usage.output_tokens),
                         (7, 3))
        # A reload reads the local-only usage stamped on the tool message.
        loaded = tr.Transcript().load(Snapshot(
            status=Status(state="idle"),
            messages=[
                AssistantMessage(tool_calls=[
                    ToolCall(call_id="c", name="subagent", arguments_json="{}")]),
                ToolMessage(tool_call_id="c", content="done",
                            usage=Usage(input_tokens=8, output_tokens=4)),
            ],
        ))
        reloaded = next(i for i in loaded if isinstance(i, tr.ToolItem)).usage
        self.assertEqual((reloaded.input_tokens, reloaded.output_tokens), (8, 4))

    def test_failed_call_keeps_both_output_and_reason(self):
        """A timed-out run_bash returns what it printed *and* the reason;
        the transcript must show both, not drop the reason."""
        t = tr.Transcript()
        t.apply(MessageAdded(message=user_message("u")))
        t.apply(ToolCallStarted(call_id="c", name="run_bash"))
        t.apply(ToolFinished(
            call_id="c", status="error",
            content="partial\ncommand timed out",
        ))
        tool = t.tool_blocks["c"]
        self.assertEqual(tool.status, tr.ToolStatus.CALL_FAILED)
        self.assertIn("partial", tool.output)
        self.assertIn("command timed out", tool.output)

    def test_cancel_settles_in_flight_tool_calls(self):
        """An interrupt kills a running call: it must settle as failed and
        be reported, or the app never stops its Running shimmer."""
        t = tr.Transcript()
        t.apply(ProcessingStarted())
        t.apply(ReasoningDelta(text="think"))
        t.apply(ToolCallStarted(call_id="c", name="run_bash"))
        t.apply(ToolCallArgumentsDelta(call_id="c", text='{"command":'))
        changed = t.cancel()
        tool = t.tool_blocks["c"]
        self.assertEqual(tool.status, tr.ToolStatus.CALL_FAILED)
        self.assertEqual(tool.output, "interrupted")
        self.assertIn(tool, changed)
        # Settling twice is a no-op; a late result is ignored after cancel.
        self.assertEqual(t.cancel(), [])
        t.apply(ToolFinished(call_id="c", status="error", content="it broke"))
        self.assertEqual(tool.status, tr.ToolStatus.CALL_FAILED)

    def test_processing_started_opens_a_fresh_round(self):
        """Every round (post-tool included) is opened by ProcessingStarted:
        the previous round's settled entries are never appended to."""
        t = tr.Transcript()
        t.apply(ProcessingStarted(t=0.0))
        t.apply(ReasoningDelta(text="one", t=0.0))
        t.apply(ContentDelta(text="answer one", t=0.5))
        first_thinking, first_answer = t.thinking, t.assistant
        t.apply(RoundComplete(has_tool_calls=True, t=1.0))
        self.assertTrue(first_thinking.done)
        self.assertTrue(first_answer.finalized)

        t.apply(ProcessingStarted(t=2.0))
        self.assertIsNone(t.thinking)
        self.assertIsNone(t.assistant)
        t.apply(ReasoningDelta(text="two", t=2.0))
        t.apply(ContentDelta(text="answer two", t=2.5))
        self.assertIsNot(t.thinking, first_thinking)
        self.assertIsNot(t.assistant, first_answer)
        self.assertEqual(first_answer.chunks, ["answer one"])
        self.assertEqual(t.thinking.reasoning, "two")

    def test_settle_is_idempotent_and_survives_late_tool_results(self):
        t = tr.Transcript()
        t.apply(ProcessingStarted())
        t.apply(ToolCallStarted(call_id="c", name="read_file"))
        t.apply(RoundComplete(has_tool_calls=True))
        self.assertEqual(t.settle(None), [])  # nothing live to settle twice
        # The tool result lands after the round closed and still resolves.
        t.apply(ToolFinished(call_id="c", status='ok', content="ok"))
        self.assertEqual(t.tool_blocks["c"].status, tr.ToolStatus.CALL_SUCCEEDED)

    def test_processing_started_reopens_after_cancel(self):
        t = tr.Transcript()
        t.apply(ProcessingStarted())
        t.apply(ContentDelta(text="partial"))
        t.cancel()
        self.assertEqual(t.apply(ContentDelta(text="late")), [])
        t.apply(ProcessingStarted())
        self.assertFalse(t.cancelled)
        t.apply(ContentDelta(text="next"))
        self.assertEqual(t.assistant.live, "next")

    def test_turn_footer_sums_usage_across_rounds(self):
        """The footer's token counts are the turn's whole spend: every
        round, not just the final answer's. The host sums them when it stamps
        the turn's end, so the client just renders them."""
        from xarness.conversation import Conversation, Usage

        opening = user_message(
            "hi", id="u1", timing=Timing(send="2025-01-01T00:00:00+00:00"))
        messages = [
            opening,
            AssistantMessage(
                content="a", id="a1", usage=Usage(
                    input_tokens=1_234, output_tokens=56),
            ),
            AssistantMessage(
                content="b", id="a2", usage=Usage(
                    input_tokens=2_000, output_tokens=400),
            ),
        ]
        conversation = Conversation(messages=messages)
        conversation.begin_turn(opening, before_tree=None)
        self.assertIsNotNone(conversation.stamp_turn_end(5.0, 4.0))
        items = tr.Transcript().load(_snap(conversation.messages))
        footer = next(i for i in items if isinstance(i, tr.TurnSummaryItem))
        assert footer.usage is not None
        self.assertEqual(
            (footer.usage.input_tokens, footer.usage.output_tokens),
            (3_234, 456),
        )

    def test_turn_without_host_stamp_has_no_footer(self):
        """The footer is the host's verdict that the turn is over. A round
        that finished its tool call but was never stamped (the turn is still
        running — e.g. a session resumed while it works headless) must not
        claim an end, however answerable the history looks."""
        user = user_message(
            "hi", id="u", timing=Timing(send="2025-01-01T00:00:00+00:00"))
        call = ToolCall(call_id="c1", name="read_file", arguments_json="{}")
        assistant = AssistantMessage(content="", tool_calls=[call])
        result = ToolMessage(tool_call_id="c1", content="body", status="ok")
        items = tr.Transcript().load(_snap([user, assistant, result]))
        self.assertFalse(
            [i for i in items if isinstance(i, tr.TurnSummaryItem)])

    def test_cancelled_turn_reports_how_long_it_ran(self):
        """A turn cut short has ended too: it gets a footer with the wall
        clock it spent and no busy/idle split (its rounds were not measured
        to an end, so a split would lie)."""
        user = user_message(
            "hi", id="u", timing=Timing(send="2025-01-01T00:00:00+00:00"))
        done = AssistantMessage(
            content="",
            tool_calls=[ToolCall(call_id="c1", name="read_file",
                                 arguments_json="{}")],
        )
        result = ToolMessage(tool_call_id="c1", content="body", status="ok")
        partial = AssistantMessage(
            content="half", interrupted=True,
            timing=Timing(stats=TurnStats(total=62.0)),
        )
        items = tr.Transcript().load(_snap([user, done, result, partial]))
        footers = [i for i in items if isinstance(i, tr.TurnSummaryItem)]
        self.assertEqual(len(footers), 1)
        self.assertEqual(footers[0].total, 62.0)
        self.assertIsNone(footers[0].busy)
        self.assertIsNone(footers[0].idle)

    def test_live_and_replay_footer_match(self):
        """A live turn appends exactly the footer a reload renders: the host
        puts the same figures on ``TurnEnded`` as it stamps on the last
        answer."""
        from xarness.conversation import Conversation, Usage

        usage = Usage(input_tokens=1_234, output_tokens=56)
        live = tr.Transcript()
        live.apply(MessageAdded(message=_user("hi")))
        live.apply(ContentDelta(text="hello"))
        live.apply(RoundComplete(usage=usage))
        changed = live.apply(TurnEnded(
            outcome="finished", stats=TurnStats(
                total=5.0, busy=4.0, idle=1.0, usage=usage),
        ))
        footers = [i for i in changed if isinstance(i, tr.TurnSummaryItem)]
        self.assertEqual(len(footers), 1)
        self.assertEqual(footers[0].total, 5.0)
        self.assertEqual(footers[0].busy, 4.0)
        self.assertEqual(footers[0].idle, 1.0)
        self.assertEqual(footers[0].usage, usage)
        # Settling a second time never duplicates the footer.
        self.assertEqual(
            live.apply(TurnEnded(outcome="finished", stats=TurnStats(
                total=5.0, busy=4.0, idle=1.0, usage=usage))), [])

        # The same turn as the host persists it replays to the same four
        # numbers: the footer reads the stamped stats, it derives nothing.
        opening = user_message("hi", id="u")
        answer = AssistantMessage(content="hello", usage=usage)
        conversation = Conversation(messages=[opening, answer])
        conversation.begin_turn(opening, before_tree=None)
        self.assertIsNotNone(conversation.stamp_turn_end(5.0, 4.0))
        replayed = [
            i for i in tr.Transcript().load(_snap(conversation.messages))
            if isinstance(i, tr.TurnSummaryItem)
        ]
        self.assertEqual(len(replayed), 1)
        self.assertEqual(
            (replayed[0].total, replayed[0].busy, replayed[0].idle,
             replayed[0].usage),
            (footers[0].total, footers[0].busy, footers[0].idle,
             footers[0].usage),
        )

    def test_finish_keeps_first_duration(self):
        item = tr.ThinkingItem(parts=["x"])
        item.finish(None)  # e.g. settled from a snapshot mid-round
        self.assertEqual(item.summary_text(), "Thought for …")
        item.finish(2.0)   # ...then stamped by a later event
        self.assertEqual(item.summary_text(), "Thought for 2.0s")
        item.finish(9.0)
        self.assertEqual(item.summary_text(), "Thought for 2.0s")


if __name__ == "__main__":
    unittest.main()
