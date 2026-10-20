"""Context compaction, independent of the controller/UI.

A compaction replaces the visible conversation with a single handoff summary
message, advancing the epoch. The old messages stay in the append-only list
(and reappear if the compaction is undone); only ``Conversation.epoch``
changes. Summarization is one tool-free round through the same stream client.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .client import StreamClient
from .conversation import SUMMARY_PREFIX, Conversation, Message
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


def render_for_summary(message: Message) -> str:
    """One message as plain text for the summarizer prompt."""
    role = message.role
    parts = [f"{role}: {message.content}"] if message.content else [f"{role}:"]
    for call in message.tool_calls or []:
        parts.append(f"  {role} called {call.name}({call.arguments_json})")
    if role == "tool" and message.tool_call_id:
        parts.append(f"  (result for call {message.tool_call_id})")
    return "\n".join(parts)


async def summarize(
    client: StreamClient, transcript: str,
) -> tuple[str, Usage | None]:
    """One-off handoff round through the same client, no tools.

    Returns the handoff text and the round's usage, so its cost can be folded
    into the session totals instead of being silently dropped.
    """
    prompt = (
        "You are handing this conversation off to a fresh context. Write "
        "the handoff the next assistant instance needs to continue the "
        "work seamlessly. Cover, in this order:\n"
        "1. The task: what the user is trying to accomplish, in their "
        "terms.\n"
        "2. What's known so far: key decisions, file paths, relevant "
        "commands and their results, constraints discovered.\n"
        "3. What's next: the current state of any work in progress and "
        "the immediate next steps.\n"
        "Be concise — a few short paragraphs at most. Reply with the "
        "handoff only.\n\n"
        "--- conversation ---\n"
        f"{transcript}"
    )
    parts: list[str] = []
    usage: Usage | None = None
    async for event in client.stream([{"role": "user", "content": prompt}]):
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
    summarizer: Callable[[str], Awaitable[tuple[str, Usage | None]]] | None = None,
) -> CompactionResult | None:
    """Summarize the visible history and advance the epoch.

    Returns None when there is nothing to compact yet. The caller is
    responsible for folding ``result.usage`` into its session totals and
    recording ``before_tokens`` / ``after_tokens`` for the notice.
    """
    compactable = conversation.visible_messages()
    if not compactable:
        return None
    before_tokens = sum(message_tokens(m) for m in compactable)
    transcript = "\n\n".join(render_for_summary(m) for m in compactable)
    if summarizer is None:
        summary, usage = await summarize(client, transcript)
    else:
        summary, usage = await summarizer(transcript)
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
