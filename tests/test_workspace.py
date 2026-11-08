"""Workspace tracking: events, per-viewer labels, auto groups, accept/reject."""

import asyncio
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xarness import gitwork, session_store
from xarness.gitwork import TurnCheckpoint
from xarness.workspace import WorkspaceState, open_workspace


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, f"git {' '.join(args)}: {result.stderr}"
    return result.stdout


async def open_state(path: Path, **kwargs) -> WorkspaceState:
    kwargs.setdefault("persist", False)
    state = await open_workspace(path, **kwargs)
    assert state is not None
    return state


class WorkspaceTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)

    def make_repo(self) -> Path:
        repo = self.base / "repo"
        repo.mkdir()
        _git(repo, "init", "-q", "-b", "main", ".")
        _git(repo, "config", "user.email", "t@t")
        _git(repo, "config", "user.name", "t")
        (repo / "app.py").write_text("line1\nline2\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "init")
        return repo

    async def kinds(self, state: WorkspaceState, viewer=None):
        summary = await state.diff(viewer)
        return {row.path: row.kind for row in summary.files}

    # ------------------------------------------------------------------
    # labels

    async def test_explicit_edit_is_mine_for_its_author_and_agent_for_others(
        self,
    ) -> None:
        repo = self.make_repo()
        state = await open_state(repo)

        (repo / "agent.txt").write_text("agent\n")
        state.record(
            session="A", tool="edit_file", turn="t1",
            paths=("agent.txt",), auto=(),
        )
        (repo / "manual.txt").write_text("manual\n")

        assert await self.kinds(state, "A") == {
            "agent.txt": "mine", "manual.txt": "drift",
        }
        assert await self.kinds(state, "B") == {
            "agent.txt": "agent", "manual.txt": "drift",
        }

    async def test_two_explicit_editors_are_both_mine(self) -> None:
        repo = self.make_repo()
        state = await open_state(repo)
        (repo / "both.txt").write_text("x\n")
        state.record(
            session="A", tool="edit_file", turn="t1",
            paths=("both.txt",), auto=(),
        )
        state.record(
            session="B", tool="write_file", turn="t2",
            paths=("both.txt",), auto=(),
        )
        assert (await self.kinds(state, "A"))["both.txt"] == "mine"
        assert (await self.kinds(state, "B"))["both.txt"] == "mine"
        assert (await self.kinds(state, "C"))["both.txt"] == "agent"

    async def test_a_viewer_set_counts_its_subagents_work(self) -> None:
        """A parent's /sessions totals include what its subagent wrote,
        even though the diff row itself reads ``agent`` for the parent."""
        repo = self.make_repo()
        state = await open_state(repo)
        (repo / "child.txt").write_text("child\n")
        state.record(
            session="child", tool="edit_file", turn="t1",
            paths=("child.txt",), auto=(),
        )

        assert (await self.kinds(state, "parent"))["child.txt"] == "agent"
        assert (await self.kinds(state, {"parent", "child"}))[
            "child.txt"] == "mine"
        assert (await self.kinds(state, {"other"}))[
            "child.txt"] == "agent"

    async def test_explicit_write_wins_over_a_later_bash_touch(self) -> None:
        repo = self.make_repo()
        state = await open_state(repo)

        (repo / "shared.txt").write_text("agent\n")
        state.record(
            session="A", tool="edit_file", turn="t1",
            paths=("shared.txt",), auto=(),
        )
        (repo / "shared.txt").write_text("agent\nbash\n")
        state.record(
            session="B", tool="run_bash", turn="t2",
            paths=("shared.txt",), auto=(),
        )

        assert (await self.kinds(state, "A"))["shared.txt"] == "mine"
        assert (await self.kinds(state, "B"))["shared.txt"] == "agent"

    async def test_bash_paths_are_agent_with_no_candidate_sets(self) -> None:
        repo = self.make_repo()
        state = await open_state(repo)
        (repo / "one.txt").write_text("one\n")
        (repo / "two.txt").write_text("two\n")
        state.record(
            session="A", tool="run_bash", turn="t1",
            paths=("one.txt", "two.txt"), auto=(),
        )
        assert await self.kinds(state, "A") == {
            "one.txt": "agent", "two.txt": "agent",
        }
        summary = await state.diff("A")
        assert not any("candidate" in repr(row) for row in summary.files)

    async def test_bash_event_excludes_earlier_drift(self) -> None:
        """The bash event carries only the paths its before→after diff saw;
        an earlier user edit stays drift."""
        repo = self.make_repo()
        state = await open_state(repo)
        (repo / "user.txt").write_text("user\n")  # outside the harness
        (repo / "bash.txt").write_text("bash\n")
        state.record(
            session="A", tool="run_bash", turn="t1",
            paths=("bash.txt",), auto=(),
        )
        assert await self.kinds(state, "A") == {
            "user.txt": "drift", "bash.txt": "agent",
        }

    # ------------------------------------------------------------------
    # auto groups

    async def test_build_output_is_one_auto_group(self) -> None:
        repo = self.make_repo()
        state = await open_state(repo)
        build = repo / "build"
        build.mkdir()
        for i in range(20):
            (build / f"out{i}.o").write_text("x\n")
        state.record(
            session="A", tool="run_bash", turn="t1",
            paths=(), auto=("build",),
        )
        (repo / "node_modules").mkdir()
        (repo / "node_modules" / "pkg.js").write_text("x\n")

        summary = await state.diff("A")
        groups = {group.prefix: group for group in summary.groups}
        assert groups["build/"].kind == "auto"
        assert groups["build/"].files == 20
        assert groups["node_modules/"].kind == "drift"
        assert summary.ignored_files == 21
        assert summary.files == []

        # A second build creates no new group and no extra event.
        state.record(
            session="A", tool="run_bash", turn="t1",
            paths=(), auto=(),
        )
        summary = await state.diff("A")
        assert len(summary.groups) == 2

    async def test_explicit_edit_to_an_ignored_path_is_a_pending_row(self) -> None:
        """An ignored path never enters a tree, so it has no stat row; the
        event alone makes it that agent's pending edit — even when its
        directory was already ignored before the baseline (no fresh group to
        hang the row off)."""
        repo = self.make_repo()
        (repo / "dist").mkdir()
        (repo / "dist" / "bundle.js").write_text("built\n")
        state = await open_state(repo)
        assert state.auto == ("dist",)
        assert await self.kinds(state, "A") == {}

        (repo / "dist" / "bundle.js").write_text("agent\n")
        state.record(
            session="A", tool="edit_file", turn="t1",
            paths=("dist/bundle.js",), auto=(),
        )
        summary = await state.diff("A")
        assert {row.path: row.kind for row in summary.files} == {
            "dist/bundle.js": "mine",
        }
        assert summary.groups == []
        assert await self.kinds(state, "B") == {"dist/bundle.js": "agent"}

        # Accept: the agent's own write stops being pending — no drift row.
        await state.accept()
        summary = await state.diff("A")
        assert (summary.files, summary.groups) == ([], [])
        assert await self.kinds(state, "B") == {}
        assert (repo / "dist" / "bundle.js").read_text() == "agent\n"
        assert state.events == []

    async def test_reject_lists_an_ignored_explicit_write_as_unrestorable(
        self,
    ) -> None:
        """No baseline copy exists for an ignored file an agent edited, so
        reject leaves it alone and names it."""
        repo = self.make_repo()
        state = await open_state(repo)
        (repo / "dist").mkdir()
        (repo / "dist" / "bundle.js").write_text("agent\n")
        state.record(
            session="A", tool="write_file", turn="t1",
            paths=("dist/bundle.js",), auto=(),
        )

        assert await state.reject() == ("dist/bundle.js",)
        assert (repo / "dist" / "bundle.js").read_text() == "agent\n"
        assert state.events == []

    # ------------------------------------------------------------------
    # accept / reject

    async def make_turn(
        self, state: WorkspaceState, session: str, turn: str,
        edits: dict[str, str | None],
    ) -> TurnCheckpoint:
        before = await state.refresh()
        for path, content in edits.items():
            target = state.root / path
            if content is None:
                target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
        after = await state.refresh()
        state.record(
            session=session, tool="edit_file", turn=turn,
            paths=tuple(edits), auto=(),
        )
        return TurnCheckpoint(
            before_tree=before, after_tree=after, epoch=state.epoch,
        )

    async def test_accept_splices_only_non_drift_paths(self) -> None:
        repo = self.make_repo()
        state = await open_state(repo)
        baseline = state.baseline_tree

        (repo / "agent.txt").write_text("agent\n")
        (repo / "drift.txt").write_text("user\n")
        state.record(
            session="A", tool="edit_file", turn="t1",
            paths=("agent.txt",), auto=(),
        )
        await state.accept()

        # Both files are still pending, but only the agent's path advanced.
        summary = await state.diff("A")
        assert {r.path: r.kind for r in summary.files} == {"drift.txt": "drift"}
        assert state.baseline_tree != baseline
        assert await gitwork.tree_entry(state, state.baseline_tree, "agent.txt")
        assert await gitwork.tree_entry(state, state.baseline_tree, "drift.txt") is None
        assert state.events == []
        assert state.epoch == 1

    async def test_accept_records_a_deletion(self) -> None:
        repo = self.make_repo()
        state = await open_state(repo)
        (repo / "app.py").unlink()
        state.record(
            session="A", tool="edit_file", turn="t1",
            paths=("app.py",), auto=(),
        )
        await state.accept()
        assert await gitwork.tree_entry(state, state.baseline_tree, "app.py") is None

    async def test_reject_restores_drift_and_removes_event_groups(self) -> None:
        repo = self.make_repo()
        state = await open_state(repo)
        # Pre-existing ignored output must survive reject.
        (repo / "node_modules").mkdir()
        (repo / "node_modules" / "keep.js").write_text("keep\n")

        (repo / "build").mkdir()
        (repo / "build" / "out.o").write_text("x\n")
        state.record(
            session="A", tool="run_bash", turn="t1",
            paths=(), auto=("build",),
        )
        (repo / "app.py").write_text("drift\n")

        await state.reject()

        assert (repo / "app.py").read_text() == "line1\nline2\n"
        assert (repo / "node_modules" / "keep.js").exists()
        assert not (repo / "build").exists()
        assert state.events == []

    async def test_reject_lists_ignored_files_it_cannot_restore(self) -> None:
        repo = self.make_repo()
        state = await open_state(repo)
        (repo / "dist").mkdir()
        (repo / "dist" / "bundle.js").write_text("agent built\n")
        state.record(
            session="A", tool="edit_file", turn="t1",
            paths=("dist/bundle.js",), auto=(),
        )
        unrestorable = await state.reject()
        assert unrestorable == ("dist/bundle.js",)
        assert (repo / "dist" / "bundle.js").exists()

    # ------------------------------------------------------------------
    # undo

    async def test_undo_restores_only_the_turn_paths(self) -> None:
        repo = self.make_repo()
        state = await open_state(repo)
        turn = await self.make_turn(
            state, "A", "t1", {"app.py": "agent\n", "new.txt": "new\n"},
        )
        (repo / "unrelated.txt").write_text("user\n")

        assert await state.undo("A", [("t1", turn)]) == []

        assert (repo / "app.py").read_text() == "line1\nline2\n"
        assert not (repo / "new.txt").exists()
        assert (repo / "unrelated.txt").read_text() == "user\n"
        assert state.events == []

    async def test_undo_refuses_when_a_later_change_exists(self) -> None:
        repo = self.make_repo()
        state = await open_state(repo)
        turn = await self.make_turn(state, "A", "t1", {"app.py": "agent\n"})
        (repo / "app.py").write_text("manual\n")

        conflicts = await state.undo("A", [("t1", turn)])

        assert conflicts == ["app.py"]
        assert (repo / "app.py").read_text() == "manual\n"  # untouched
        assert len(state.events) == 1  # not dropped on refusal

    async def test_undo_reverts_a_range_to_the_oldest_before_tree(self) -> None:
        repo = self.make_repo()
        state = await open_state(repo)
        first = await self.make_turn(state, "A", "t1", {"one.txt": "one\n"})
        second = await self.make_turn(state, "A", "t2", {"two.txt": "two\n"})

        assert await state.undo("A", [("t1", first), ("t2", second)]) == []
        assert not (repo / "one.txt").exists()
        assert not (repo / "two.txt").exists()

    async def test_undo_leaves_other_sessions_events_alone(self) -> None:
        repo = self.make_repo()
        state = await open_state(repo)
        turn = await self.make_turn(state, "A", "t1", {"one.txt": "one\n"})
        (repo / "two.txt").write_text("two\n")
        state.record(
            session="B", tool="edit_file", turn="tb",
            paths=("two.txt",), auto=(),
        )

        await state.undo("A", [("t1", turn)])

        assert [(e.session, e.turn) for e in state.events] == [("B", "tb")]
        assert (repo / "two.txt").read_text() == "two\n"

    # ------------------------------------------------------------------
    # locking / persistence

    async def test_concurrent_refresh_and_accept_do_not_race_the_index(self) -> None:
        repo = self.make_repo()
        state = await open_state(repo)
        (repo / "a.txt").write_text("a\n")
        await asyncio.gather(state.refresh(), state.accept(), state.refresh())
        assert state.baseline_tree

    async def test_two_concurrent_accepts_serialize(self) -> None:
        """There is no no-op short-circuit: the second /accept runs after the
        first (a second epoch bump), and neither ever splices twice."""
        repo = self.make_repo()
        state = await open_state(repo)
        (repo / "a.txt").write_text("a\n")
        state.record(
            session="A", tool="edit_file", turn="t1",
            paths=("a.txt",), auto=(),
        )
        first, second = await asyncio.gather(state.accept(), state.accept())
        assert first == second == state.baseline_tree
        assert state.epoch == 2
        assert state.events == []
        assert await self.kinds(state, "A") == {}
        assert (repo / "a.txt").read_text() == "a\n"

    async def test_no_session_workspace_never_opens_the_database(self) -> None:
        repo = self.make_repo()
        with mock.patch.object(
            session_store, "SESSIONS_DIR", self.base / "db",
        ):
            state = await open_state(repo)  # persist=False
            (repo / "a.txt").write_text("a\n")
            state.record(
                session="A", tool="edit_file", turn="t1",
                paths=("a.txt",), auto=(),
            )
            await state.accept()
        assert not (self.base / "db" / "sessions.db").exists()

    async def test_enable_persistence_flushes_state_and_events(self) -> None:
        """A saved session joining a --no-session workspace flushes the
        in-memory state and its events in one transaction, so a reopened
        workspace gets both (a row without its events would turn pending
        edits into authorless drift)."""
        repo = self.make_repo()
        db_dir = self.base / "db"
        with mock.patch.object(session_store, "SESSIONS_DIR", db_dir):
            state = await open_state(repo)  # persist=False
            (repo / "a.txt").write_text("a\n")
            state.record(
                session="A", tool="edit_file", turn="t1",
                paths=("a.txt",), auto=(),
            )
            assert not (db_dir / "sessions.db").exists()

            state.enable_persistence()
            state.enable_persistence()  # already persistent: a no-op
            reopened = await open_workspace(repo)  # persist=True
        assert reopened is not None
        assert reopened.baseline_tree == state.baseline_tree
        assert reopened.epoch == state.epoch
        assert [(e.session, e.tool, e.paths) for e in reopened.events] == [
            ("A", "edit_file", ("a.txt",)),
        ]
        assert reopened.next_seq == 2


if __name__ == "__main__":
    unittest.main()
