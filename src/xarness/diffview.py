"""The /diff payload: gitwork data shaped for clients.

Split out of session.py: it is a pure view over a :class:`~xarness.gitwork.GitInfo`
block and turn checkpoints, shared by hosted and saved sessions (the saved
path has no live session at all).
"""

from __future__ import annotations

from . import protocol
from .errors import SessionError
from .gitwork import (
    GitInfo,
    GitWorktreeError,
    TurnCheckpoint,
    attributed_diff_stat,
    git_diff,
    snapshot_state,
)


async def build_diff_info(
    info: GitInfo, turns: list[TurnCheckpoint], path: str | None,
    open_before: str | None = None,
) -> protocol.DiffInfo:
    """The /diff payload, from a git block and turn checkpoints — shared by
    hosted and saved sessions (the saved path has no live open turn).

    Git failures raise :class:`SessionError` instead of reporting
    ``tracking=False``: a transient ``index.lock`` (concurrent snapshots)
    must not make clients believe tracking disappeared."""
    try:
        # A per-file click-through needs both the stat rows and the text;
        # snapshot the workspace once and share the sha between them.
        current = await snapshot_state(info) if path is not None else None
        stat = await attributed_diff_stat(info, turns, open_before=open_before,
                                          current=current)
    except GitWorktreeError as exc:
        raise SessionError(f"diff failed: {exc}") from exc
    files = [
        protocol.DiffFileInfo(
            path=f.path, additions=f.additions, deletions=f.deletions,
            is_new=f.is_new, source=f.source or "agent",
        )
        for f in stat.files
        if path is None or f.path == path
    ]
    result = protocol.DiffInfo(
        files=files,
        additions=sum(f.additions for f in files),
        deletions=sum(f.deletions for f in files),
        tracking=True,
    )
    if path is not None:
        try:
            result.text = await git_diff(info, path, current=current)
        except GitWorktreeError as exc:
            raise SessionError(f"diff failed: {exc}") from exc
    return result
