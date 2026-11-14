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

    async def write_path(
        self, state: WorkspaceState, session: str, turn: str,
        path: str, content: str | None,
    ) -> tuple[tuple[str, str | None, str | None], ...]:
        """Do what Session._record_event does for an explicit write: change
        the file, store both contents as blobs, record the provenance."""
        target = state.root / path
        before = target.read_bytes() if target.exists() else None
        if content is None:
            target.unlink()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        after = target.read_bytes() if target.exists() else None
        writes: tuple[tuple[str, str | None, str | None], ...] = ()
        if before != after:
            writes = ((path, *[
                None if data is None else await gitwork.write_blob(state, data)
                for data in (before, after)
            ]),)
        state.record(
            session=session, tool="edit_file", turn=turn,
            paths=(path,), auto=(), writes=writes,
        )
        return writes

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

    # ------------------------------------------------------------------
    # per-author attribution

    async def test_a_shared_file_is_one_row_per_author(self) -> None:
        """Two agents edited app.py: each author's delta is a row of its own."""
        repo = self.make_repo()
        state = await open_state(repo)
        await self.write_path(state, "A", "t1", "app.py", "line1\nA\n")
        await self.write_path(state, "B", "t2", "app.py", "line1\nA\nB\n")

        rows = (await state.diff("A", scope="mine")).files
        assert [
            (row.path, row.author, row.kind, row.additions, row.deletions)
            for row in rows
        ] == [
            ("app.py", "A", "mine", 1, 1),
            ("app.py", "B", "agent", 1, 0),
        ]

        # A third session has no delta of its own: both rows are another
        # agent's work, each with its own counts.
        rows = (await state.diff("C", scope="mine")).files
        assert [(row.author, row.kind) for row in rows] == [
            ("A", "agent"), ("B", "agent"),
        ]

        # Either way the workspace totals are the file's pending change.
        info = await state.diff("A", scope="mine")
        assert (info.additions, info.deletions) == (2, 1)
        assert info.files[0].authors == ("A", "B")

    async def test_all_scope_reports_the_files_whole_pending_change(self) -> None:
        """Scoped to every session, a shared file is one combined row, credited
        to its last writer."""
        repo = self.make_repo()
        state = await open_state(repo)
        await self.write_path(state, "A", "t1", "app.py", "line1\nA\n")
        await self.write_path(state, "B", "t2", "app.py", "line1\nA\nB\n")

        info = await state.diff("A")
        assert [
            (row.path, row.author, row.kind, row.additions, row.deletions)
            for row in info.files
        ] == [("app.py", None, "agent", 2, 1)]
        assert info.files[0].authors == ("A", "B")
        # The badge follows the last write, not the viewer: B wrote the file
        # as it now stands, so A sees an ``agent`` row and B its own.
        assert [row.kind for row in (await state.diff("C")).files] == ["agent"]
        assert [row.kind for row in (await state.diff("B")).files] == ["mine"]

    async def test_a_new_sessions_edit_is_not_counted_as_the_first_ones(
        self,
    ) -> None:
        """A session writes a one-line file, a new one edits it: two rows."""
        repo = self.make_repo()
        state = await open_state(repo)
        await self.write_path(state, "s1", "t1", "test.py", "meow\n")
        await self.write_path(state, "s2", "t2", "test.py", "meow1\n")

        def rows(rows):
            return [
                (row.author, row.kind, row.additions, row.deletions, row.is_new)
                for row in rows
            ]

        first = (await state.diff("s1", scope="mine")).files
        assert rows(first) == [
            ("s1", "mine", 1, 0, True),
            ("s2", "agent", 1, 1, False),
        ]
        second = (await state.diff("s2", scope="mine")).files
        assert rows(second) == [
            ("s1", "agent", 1, 0, True),
            ("s2", "mine", 1, 1, False),
        ]
        # The combined row is the file against the baseline — a new one-line
        # path — and credits the session that wrote it last: for s1 that is
        # s2's work, for s2 its own.
        assert rows((await state.diff("s1")).files) == [
            (None, "agent", 1, 0, True),
        ]
        assert rows((await state.diff("s2")).files) == [
            (None, "mine", 1, 0, True),
        ]
        whole = await state.file_diff("test.py")
        assert "-meow" not in whole and "+meow1" in whole
        first_text = await state.file_diff("test.py", author="s1")
        assert "+meow\n" in first_text and "meow1" not in first_text
        second_text = await state.file_diff("test.py", author="s2")
        assert "-meow\n+meow1" in second_text

    async def test_a_shared_files_diff_text_is_the_row_it_opened(self) -> None:
        repo = self.make_repo()
        state = await open_state(repo)
        await self.write_path(state, "A", "t1", "app.py", "line1\nA\n")
        await self.write_path(state, "B", "t2", "app.py", "line1\nA\nB\n")

        a_text = await state.file_diff("app.py", author="A")
        assert a_text.startswith("diff --git a/app.py b/app.py")
        assert "+A" in a_text and "+B" not in a_text
        b_text = await state.file_diff("app.py", author="B")
        assert "+B" in b_text and "+A" not in b_text
        # No author: the whole pending change, which is what the combined
        # row counts.
        whole = await state.file_diff("app.py")
        assert "+A" in whole and "+B" in whole

    async def test_a_created_file_is_one_line_of_provenance(self) -> None:
        repo = self.make_repo()
        state = await open_state(repo)
        await self.write_path(state, "A", "t1", "new.txt", "one\ntwo\n")
        summary = await state.diff("A")
        assert [
            (row.path, row.kind, row.additions, row.deletions, row.is_new)
            for row in summary.files
        ] == [("new.txt", "mine", 2, 0, True)]
        assert [row.path for row in await self._rows(state, "B")] == ["new.txt"]

    async def _rows(self, state: WorkspaceState, viewer):
        return (await state.diff(viewer)).files

    async def test_a_viewer_set_gets_a_row_per_member(self) -> None:
        """A parent and its subagent both wrote the file: both deltas are
        ``mine`` to the set, so each member gets its own row."""
        repo = self.make_repo()
        state = await open_state(repo)
        await self.write_path(state, "parent", "t1", "app.py", "P\n")
        await self.write_path(state, "child", "t2", "app.py", "P\nC\n")

        rows = (await state.diff({"parent", "child"}, scope="mine")).files
        assert [
            (r.author, r.kind, r.additions, r.deletions) for r in rows
        ] == [
            ("parent", "mine", 1, 2),
            ("child", "mine", 1, 0),
        ]
        assert rows[0].authors == ("parent", "child")

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

    async def test_a_path_request_returns_only_that_row(self) -> None:
        """`/diff <path>` asks for one row: no other files, no groups, no
        ignored-file counting."""
        repo = self.make_repo()
        state = await open_state(repo)
        (repo / "a.txt").write_text("a\n")
        (repo / "b.txt").write_text("b\n")
        (repo / "build").mkdir()
        (repo / "build" / "out.o").write_text("obj\n")

        whole = await state.diff("A")
        assert [row.path for row in whole.files] == ["a.txt", "b.txt"]
        assert [group.prefix for group in whole.groups] == ["build/"]

        one = await state.diff("A", path="a.txt")
        assert [row.path for row in one.files] == ["a.txt"]
        assert (one.groups, one.ignored_files) == ([], 0)
        # A group prefix has neither a stat row nor a group of its own; its
        # diff is file_diff's "added" patch.
        assert (await state.diff("A", path="build/")).files == []

    async def test_an_ignored_group_diffs_as_files_added_from_nothing(
        self,
    ) -> None:
        """An ignored path never enters a tree, so it has no baseline copy to
        diff against: the only view of its contents is "added", one patch per
        file under the group."""
        repo = self.make_repo()
        (repo / ".gitignore").write_text("*.log\nbuild/\n")
        state = await open_state(repo)
        build = repo / "build"
        build.mkdir()
        (build / "out.o").write_text("obj\n")
        (build / "link").symlink_to("/etc/hostname")
        (repo / "notes.log").write_text("hello\n")
        assert {g.prefix for g in (await state.diff("A")).groups} == {
            "build/", "notes.log",
        }

        group = await state.file_diff("build/")
        assert group.startswith("diff --git a/build/ b/build/\n")
        assert "+++ b/build/out.o" in group and "+obj" in group
        # A symlink is skipped: reading one would print its target's bytes.
        assert "hostname" not in group

        single = await state.file_diff("notes.log")
        assert single.startswith("diff --git a/notes.log b/notes.log\n")
        assert "--- /dev/null" in single and "+hello" in single

        # A tracked path keeps the baseline diff, ignored siblings or not.
        (repo / "app.py").write_text("line1\nline2\nline3\n")
        assert "+line3" in await state.file_diff("app.py")

    async def test_explicit_edit_to_an_ignored_path_is_a_pending_row(self) -> None:
        """An ignored path never enters a tree, so it has no stat row: the
        event's own write is its pending row."""
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

    async def test_a_shared_ignored_paths_row_counts_the_logs_span(self) -> None:
        """An ignored file has no baseline copy, so its combined row counts the
        log's span, and the diff it opens says the same."""
        repo = self.make_repo()
        (repo / "dist").mkdir()
        (repo / "dist" / "bundle.js").write_text("old\n")
        state = await open_state(repo)
        assert state.auto == ("dist",)
        await self.write_path(state, "A", "t1", "dist/bundle.js", "A\n")
        await self.write_path(state, "B", "t2", "dist/bundle.js", "A\nB\n")

        rows = (await state.diff("A")).files
        assert [
            (row.author, row.kind, row.additions, row.deletions) for row in rows
        ] == [(None, "agent", 2, 1)]
        text = await state.file_diff("dist/bundle.js")
        assert "-old" in text and "+A" in text and "+B" in text
        # Each author's row opens exactly its own span.
        assert "+B" not in await state.file_diff(
            "dist/bundle.js", author="A",
        )

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

    async def test_accept_absorbs_drift(self) -> None:
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

        # The current worktree is the baseline: nothing is pending, and the
        # drift path advanced along with the agent's own write.
        summary = await state.diff("A")
        assert summary.files == []
        assert state.baseline_tree != baseline
        assert await gitwork.tree_entry(state, state.baseline_tree, "agent.txt")
        assert await gitwork.tree_entry(state, state.baseline_tree, "drift.txt")
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

    async def test_reject_removes_a_binary_a_command_created(self) -> None:
        """A content-detected binary is an auto group of its own; reject
        removes it and the directory the deletion left empty."""
        repo = self.make_repo()
        state = await open_state(repo)
        (repo / "out").mkdir()
        (repo / "out" / "app.bin").write_bytes(b"\0built")
        state.record(
            session="A", tool="run_bash", turn="t1",
            paths=(), auto=("out/app.bin",),
        )

        await state.reject()

        assert not (repo / "out").exists()
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

    async def test_undo_merges_a_later_edit_elsewhere_in_the_file(self) -> None:
        """Another session's edit elsewhere in the file no longer blocks the
        undo: the turn's own hunk is reversed and theirs kept."""
        repo = self.make_repo()
        state = await open_state(repo)
        (repo / "big.txt").write_text(
            "one\ntwo\nthree\nfour\nfive\nsix\nseven\neight\nnine\n"
        )
        turn = await self.make_turn(
            state, "A", "t1", {
                "big.txt":
                    "one\nAGENT\nthree\nfour\nfive\nsix\nseven\neight\nnine\n",
            },
        )
        (repo / "big.txt").write_text(
            "one\nAGENT\nthree\nfour\nfive\nsix\nseven\nOTHER\nnine\n"
        )

        assert await state.undo("A", [("t1", turn)]) == []

        assert (repo / "big.txt").read_text() == (
            "one\ntwo\nthree\nfour\nfive\nsix\nseven\nOTHER\nnine\n"
        )

    async def test_undo_refuses_when_the_later_edit_overlaps(self) -> None:
        repo = self.make_repo()
        state = await open_state(repo)
        turn = await self.make_turn(
            state, "A", "t1", {"app.py": "agent1\nline2\n"},
        )
        (repo / "app.py").write_text("manual\nline2\n")

        assert await state.undo("A", [("t1", turn)]) == ["app.py"]

        assert (repo / "app.py").read_text() == "manual\nline2\n"  # untouched

    async def test_undo_reverts_two_turns_on_the_same_path(self) -> None:
        """Reversing a newer turn first leaves content (not a tree name) for
        the older one to compare against."""
        repo = self.make_repo()
        state = await open_state(repo)
        first = await self.make_turn(state, "A", "t1", {"one.txt": "one\n"})
        second = await self.make_turn(state, "A", "t2", {"one.txt": "two\n"})

        assert await state.undo(
            "A", [("t1", first), ("t2", second)],
        ) == []

        assert not (repo / "one.txt").exists()

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

    async def test_two_live_hosts_share_one_event_log(self) -> None:
        """Two processes on one workspace: each sees the other's writes, and
        neither overwrites the other's event."""
        repo = self.make_repo()
        db_dir = self.base / "db"
        with mock.patch.object(session_store, "SESSIONS_DIR", db_dir):
            first = await open_workspace(repo)
            second = await open_workspace(repo)
            assert first is not None and second is not None
            await self.write_path(first, "s1", "t1", "test.py", "meow\n")
            await self.write_path(second, "s2", "t2", "test.py", "meow1\n")

            # Neither host's copy is authoritative: both re-read the log.
            for state in (first, second):
                rows = (await state.diff("s1", scope="mine")).files
                assert [(row.author, row.kind, row.additions, row.deletions)
                        for row in rows] == [
                    ("s1", "mine", 1, 0),
                    ("s2", "agent", 1, 1),
                ]
            reopened = await open_workspace(repo)
        assert reopened is not None
        assert [e.session for e in reopened.events] == ["s1", "s2"]

    async def test_event_provenance_survives_a_reopen(self) -> None:
        """A reopened workspace keeps the blob shas, so per-author deltas
        still work after a restart (the objects are in the same store)."""
        repo = self.make_repo()
        db_dir = self.base / "db"
        with mock.patch.object(session_store, "SESSIONS_DIR", db_dir):
            state = await open_workspace(repo)
            assert state is not None
            writes = await self.write_path(state, "A", "t1", "app.py", "new\n")
            assert writes
            reopened = await open_workspace(repo)
            assert reopened is not None
            assert reopened.events[0].writes == writes
            rows = (await reopened.diff("A")).files
        assert [(row.path, row.kind) for row in rows] == [("app.py", "mine")]

    async def test_enable_persistence_flushes_state_and_events(self) -> None:
        """A saved session joining a --no-session workspace flushes the
        in-memory state and its events in one transaction."""
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


def test_collapse_ignored_drops_descendants_of_a_kept_parent() -> None:
    """A sibling sorting below ``/`` must not hide a parent's coverage."""
    from xarness.workspace import _collapse_ignored

    assert _collapse_ignored({"a/b", "a/b/c", "a/b!x", "a/b0", "z"}) == (
        "a/b", "a/b!x", "a/b0", "z",
    )


if __name__ == "__main__":
    unittest.main()
