"""Context compaction, independent of the controller/UI.

A compaction replaces the visible conversation with a single handoff summary
message, advancing the epoch. The old messages stay in the append-only list
(and reappear if the compaction is undone); only ``Conversation.epoch``
changes.

Summarization is one tool-free round through the same stream client, sent as
a continuation of the live conversation: the exact system prompt and history
the model just saw, plus one handoff instruction as a final user message.
That keeps the whole prefix cacheable at the provider; only the instruction is
new. The "no tools" notice lives in that message rather than in the system
prompt (which would change the cached prefix).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .client import StreamClient
from .conversation import SUMMARY_PREFIX, Conversation, Message
from .prompts import COMPACTION_PROMPT
from .protocol import ContentDelta, StreamError, TurnComplete, Usage

# Rough chars-per-token for the fallback estimate when the provider sends no
# usage data. Deliberately conservative; results are labeled approximate.
_CHARS_PER_TOKEN = 4


@dataclass(slots=True)
class CompactionResult:
    """What one compaction did: the handoff text, its cost, and token counts."""

    summary: str
    usage: Usage | None
    before_tokens: int
    after_tokens: int


def message_tokens(message: Message) -> int:
    """Rough context contribution of one message, in tokens.

    Assistant messages carry their round's real usage; its output tokens are
    what the message added to the context (input tokens would count the whole
    prompt again). Everything else falls back to the chars-per-token
    heuristic.
    """
    if message.usage is not None:
        return message.usage.output_tokens
    return len(message.content or "") // _CHARS_PER_TOKEN


async def summarize(
    client: StreamClient, wire_messages: Sequence[dict[str, Any]],
) -> tuple[str, Usage | None]:
    """One-off handoff round through the same client, no tools.

    ``wire_messages`` is the live conversation's wire history with the
    handoff instruction appended, so the provider can serve every cached
    prefix token. Returns the handoff text and the round's usage, so its cost
    can be folded into the session totals instead of being silently dropped.
    """
    parts: list[str] = []
    usage: Usage | None = None
    async for event in client.stream(wire_messages):
        if isinstance(event, ContentDelta):
            parts.append(event.text)
        elif isinstance(event, StreamError):
            raise RuntimeError(event.message)
        elif isinstance(event, TurnComplete):
            usage = event.usage
            break
    summary = "".join(parts).strip()
    if not summary:
        raise RuntimeError("summarization returned no content")
    return summary, usage


async def compact(
    conversation: Conversation, client: StreamClient,
    summarizer: Callable[
        [Sequence[dict[str, Any]]], Awaitable[tuple[str, Usage | None]]
    ] | None = None,
    *,
    system: str | None = None,
    keep_reasoning: bool = False,
    image_loader: Callable[[str], bytes | None] | None = None,
) -> CompactionResult | None:
    """Summarize the visible history and advance the epoch.

    The summarizer request reuses the conversation's wire history (same
    system prompt, reasoning setting and images as a normal round) and appends
    the handoff instruction as a final user message. Tools are omitted for
    this round.

    Returns None when there is nothing to compact yet. The caller is
    responsible for folding ``result.usage`` into its session totals and
    recording ``before_tokens`` / ``after_tokens`` for the notice.
    """
    compactable = conversation.visible_messages()
    if not compactable:
        return None
    before_tokens = sum(message_tokens(m) for m in compactable)
    wire_messages = conversation.to_wire(
        keep_reasoning=keep_reasoning, system=system,
        image_loader=image_loader,
    )
    wire_messages.append({"role": "user", "content": COMPACTION_PROMPT})
    if summarizer is None:
        summary, usage = await summarize(client, wire_messages)
    else:
        summary, usage = await summarizer(wire_messages)
    summary_message = Message(
        role="user", content=f"{SUMMARY_PREFIX}\n\n{summary}",
        kind="summary", usage=usage,
    )
    conversation.epoch += 1
    conversation.add(summary_message)
    return CompactionResult(
        summary=summary, usage=usage, before_tokens=before_tokens,
        after_tokens=message_tokens(summary_message),
    )
