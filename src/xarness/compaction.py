"""Context compaction, independent of the controller/UI.

A compaction appends one summary user message to the append-only history;
the visible slice (everything after the last summary) shrinks to that
summary alone. Undoing a compaction pops the summary, which brings the old
messages back — there is no epoch counter to keep in step.

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
from .conversation import (
    SUMMARY_PREFIX,
    Conversation,
    Message,
    Usage,
    user_message,
)
from .images import ImageRef
from .prompts import COMPACTION_PROMPT
from .protocol import ContentDelta, RoundComplete, StreamError

# Rough chars-per-token for the fallback estimate when the provider sends no
# usage data. Deliberately conservative; results are labeled approximate.
CHARS_PER_TOKEN = 4

@dataclass(slots=True)
class CompactionResult:
    """What one compaction did: the handoff text, its cost, and token counts."""

    summary: str
    usage: Usage | None
    before_tokens: int
    after_tokens: int


def context_tokens(messages: Sequence[Message]) -> int:
    """Rough context size of ``messages``, in tokens.

    The last real usage (the provider counted the whole prompt it saw) is
    authoritative; messages appended after that round — tool results,
    steering — are estimated from their characters."""
    total = 0
    estimated_from = len(messages)
    for index in range(len(messages) - 1, -1, -1):
        usage = getattr(messages[index], "usage", None)
        if usage is not None:
            total += usage.input_tokens + usage.output_tokens
            estimated_from = index
            break
    return total + sum(
        len(messages[i].content or "") // CHARS_PER_TOKEN
        for i in range(estimated_from + 1, len(messages))
    )


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
        elif isinstance(event, RoundComplete):
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
    image_loader: Callable[[ImageRef], str | None] | None = None,
) -> CompactionResult | None:
    """Summarize the visible history and shrink the visible slice to the
    summary message.

    The summarizer request reuses the conversation's wire history (same
    system prompt, reasoning setting and images as a normal round) and appends
    the handoff instruction as a final user message. Tools are omitted for
    this round.

    Returns None when there is nothing to compact yet. The summary message
    carries no usage of its own — the summarization round's usage is folded
    into the session totals by the caller instead, so "the last
    usage-bearing visible message" is None right after a compaction and the
    auto-compactor cannot loop on it."""
    visible = conversation.visible_messages()
    if not visible:
        return None
    before_tokens = context_tokens(visible)
    wire_messages = conversation.to_wire(
        keep_reasoning=keep_reasoning, system=system,
        image_loader=image_loader,
    )
    wire_messages.append({"role": "user", "content": COMPACTION_PROMPT})
    if summarizer is None:
        summary, usage = await summarize(client, wire_messages)
    else:
        summary, usage = await summarizer(wire_messages)
    conversation.add(user_message(
        f"{SUMMARY_PREFIX}\n\n{summary}", is_summary=True,
    ))
    return CompactionResult(
        summary=summary, usage=usage, before_tokens=before_tokens,
        after_tokens=len(summary) // CHARS_PER_TOKEN,
    )
