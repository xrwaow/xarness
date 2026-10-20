"""Unit tests for the pure transcript reducer (no terminal required)."""

from __future__ import annotations

import unittest

from xarness.conversation import SUMMARY_PREFIX, Message, Timing, ToolCall
from xarness.events import (
    ContentDelta, ReasoningDelta, StreamError, ToolCallArgumentsDelta,
    ToolCallArgumentsDone, ToolCallStarted, TurnComplete,
)
from xarness.protocol import Snapshot, StatusPayload, ToolFinished, UserMessageAdded
from xarness.tui import transcript as tr


def _snap(messages):
    return Snapshot(status=StatusPayload(state="idle"), messages=messages)


def _user(text, mid="u"):
    return Message(role="user", content=text, id=mid)


class TranscriptTest(unittest.TestCase):
    def test_load_rebuilds_messages_tools_and_timing(self):
        user = Message(role="user", content="hi", id="u",
                       timing=Timing(send="2025-01-01T00:00:00+00:00"))
        call = ToolCall(call_id="c1", name="read_file", arguments_json="{}")
        assistant = Message(
            role="assistant", content="", tool_calls=[call],
            timing=Timing(
                start="2025-01-01T00:00:00+00:00",
                end="2025-01-01T00:00:01+00:00",
            ),
        )
        result = Message(role="tool", content="body", tool_call_id="c1",
                         ok=False, header="h")
        t = tr.Transcript()
        items = t.load(_snap([user, assistant, result]))
        self.assertIsInstance(items[0], tr.UserItem)
        self.assertIsInstance(items[1], tr.ToolItem)
        self.assertEqual(items[1].status, tr.ToolStatus.CALL_FAILED)
        self.assertEqual(items[1].output, "body")
        self.assertIsInstance(items[2], tr.LineItem)

    def test_load_renders_hidden_epochs_before_the_current_one(self):
        """Compaction hides old epochs from the model, not from the human:
        a resumed session must still show the pre-handoff transcript."""
        old_user = _user("before", mid="u1")
        old_answer = Message(role="assistant", content="old answer", id="a1")
        summary = Message(
            role="user", id="s1", kind="summary",
            content=f"{SUMMARY_PREFIX}\n\nthe handoff",
        )
        new_answer = Message(role="assistant", content="after", id="a2")
        snap = Snapshot(
            status=StatusPayload(state="idle"),
            history=[old_user, old_answer, summary],
            messages=[new_answer],
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
        assistant = Message(role="assistant", content="", tool_calls=[call], id="a1")
        result = Message(role="tool", content="body", tool_call_id="c1",
                         ok=True, id="t1")
        snap = Snapshot(
            status=StatusPayload(state="idle"),
            history=[_user("hi"), assistant],
            messages=[result],
        )
        items = tr.Transcript().load(snap)
        tool = next(item for item in items if isinstance(item, tr.ToolItem))
        self.assertEqual(tool.status, tr.ToolStatus.CALL_SUCCEEDED)
        self.assertEqual(tool.output, "body")

    def test_partial_round_renders_without_replay(self):
        from xarness.protocol import PartialRound, PartialToolCall

        assistant = Message(
            role="assistant", content="",
            tool_calls=[ToolCall(call_id="c1", name="read_file",
                                 arguments_json="{")],
        )
        snap = _snap([_user("hi"), assistant])
        snap.partial_round = PartialRound(
            reasoning="still thinking", content="partial",
            tool_calls=[PartialToolCall(
                call_id="c1", name="read_file", arguments="{\"path\": \"a\"}",
                output="body",
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
            TurnComplete(t=1.0),
        ]
        live = tr.Transcript()
        for event in events:
            live.apply(event)
        replay = tr.Transcript()
        assistant = Message(role="assistant", content="hello", reasoning="think",
                            timing=Timing(reasoning_seconds=1.0))
        items = replay.load(_snap([assistant]))
        self.assertEqual(live.thinking.reasoning, "think")
        self.assertEqual(live.thinking.duration, 1.0)
        self.assertEqual(live.assistant.chunks, ["hello"])
        self.assertEqual(items[0].reasoning, "think")
        self.assertEqual(items[1].chunks, ["hello"])

    def test_tool_lifecycle_and_cancel(self):
        t = tr.Transcript()
        self.assertEqual(t.apply(UserMessageAdded(text="u")), t.items)
        t.apply(ToolCallStarted(call_id="c", name="run_bash"))
        t.apply(ToolCallArgumentsDelta(call_id="c", text='{"command":'))
        t.apply(ToolCallArgumentsDone(call_id="c", name="run_bash",
                                      arguments_json='{"command": "ls"}'))
        tool = t.tool_blocks["c"]
        self.assertEqual(tool.args, '{"command": "ls"}')
        t.apply(ToolFinished(call_id="c", ok=True, output="ok", header="+1"))
        self.assertEqual(tool.status, tr.ToolStatus.CALL_SUCCEEDED)
        t.apply(StreamError(message="boom"))
        self.assertEqual(t.cancel(), [])
        # Events after cancel are ignored.
        self.assertEqual(t.apply(ContentDelta(text="late")), [])

    def test_finish_keeps_first_duration(self):
        item = tr.ThinkingItem(parts=["x"])
        item.finish(None)
        self.assertEqual(item.summary_text(), "Thought for …")
        item.finish(2.0)
        self.assertEqual(item.summary_text(), "Thought for 2.0s")
        item.finish(9.0)
        self.assertEqual(item.summary_text(), "Thought for 2.0s")


if __name__ == "__main__":
    unittest.main()
