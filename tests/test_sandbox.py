"""Tests for SandboxConfig subtree scoping: bwrap argv binds and the tool
path validation. bwrap itself is faked (shutil.which patched) — these tests
only exercise config/argv construction, not real sandboxes."""

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xarness.sandbox import SandboxConfig, SandboxResult, SandboxSession, SandboxUnavailable


class SandboxSubtreeTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name) / "wt"
        (self.workspace / "f").mkdir(parents=True)
        (self.workspace / "f" / "notes.txt").write_text("x\n")
        (self.workspace / "app.py").write_text("root file\n")
        patcher = mock.patch("shutil.which", return_value="/usr/bin/bwrap")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_subtree_scopes_binds_chdir_and_paths(self):
        config = SandboxConfig(workspace=self.workspace, subtree="f")

        argv = config.build_argv(["/bin/sh"])
        # Locate the binds by their /workspace targets (system ro-binds come
        # first in the argv, so --ro-bind alone is not unique).
        ws = argv.index("/workspace")
        assert argv[ws - 2:ws] == ["--ro-bind", str(self.workspace)]
        sub = argv.index("/workspace/f")
        assert argv[sub - 2:sub] == ["--bind", str(self.workspace / "f")]
        assert argv[argv.index("--chdir") + 1] == "/workspace/f"
        assert config.ref_path("docs") == "/workspace/f/.refs/docs"

        # Tool paths anchor at the subtree: the agent's workspace IS the
        # --workspace dir, so any non-escaping relative path is valid and
        # maps inside the read-write subtree. Traversal and absolute paths
        # are always rejected.
        assert config.tool_root == "/workspace/f"
        assert config.tool_path("notes.txt") == "/workspace/f/notes.txt"
        assert config.tool_path("./notes.txt") == "/workspace/f/notes.txt"
        assert config.tool_path(".") == "/workspace/f/."
        assert config.validate_relpath("notes.txt") is None
        assert config.validate_relpath("./notes.txt") is None
        assert config.validate_relpath("sub/dir/x.py") is None
        assert config.validate_relpath(".refs/docs") is None
        assert config.validate_relpath("../etc") is not None
        assert config.validate_relpath("/etc") is not None
        assert config.validate_relpath("../etc", mode="read") is not None
        assert config.validate_relpath("/etc", mode="read") is not None

    def test_missing_subtree_dir_is_unavailable(self):
        with self.assertRaises(SandboxUnavailable):
            SandboxConfig(workspace=self.workspace, subtree="nope")

    def test_no_subtree_keeps_flat_bind(self):
        config = SandboxConfig(workspace=self.workspace)

        argv = config.build_argv(["/bin/sh"])
        bind = argv.index("--bind")
        assert argv[bind + 1:bind + 3] == [str(self.workspace), "/workspace"]
        assert argv[argv.index("--chdir") + 1] == "/workspace"
        assert config.ref_path("docs") == "/workspace/.refs/docs"
        assert config.tool_root == "/workspace"
        assert config.tool_path("app.py") == "/workspace/app.py"
        assert config.validate_relpath("app.py") is None

class SandboxSessionProtocolTest(unittest.TestCase):
    """SandboxSession.run's marker protocol against a real persistent shell
    (bwrap bypassed via build_argv — the protocol, not the sandbox, is under
    test).

    Regression: output without a trailing newline (echo -n, cat of a file
    that lacks one) glued the done-marker onto the last line, so the read
    loop never recognized it and the call spun until the timeout even though
    the command had finished instantly; commands that read stdin (cat, wc)
    swallowed the marker line outright."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name) / "wt"
        self.workspace.mkdir()
        which = mock.patch("shutil.which", return_value="/usr/bin/bwrap")
        which.start()
        self.addCleanup(which.stop)
        argv = mock.patch.object(
            SandboxConfig, "build_argv", lambda self, command: list(command)
        )
        argv.start()
        self.addCleanup(argv.stop)

    def _run_all(self, *commands: str, timeout: float = 5.0) -> list[SandboxResult]:
        config = SandboxConfig(workspace=self.workspace, timeout_seconds=timeout)
        session = SandboxSession(config)

        async def drive() -> list[SandboxResult]:
            results: list[SandboxResult] = []
            try:
                for command in commands:
                    results.append(await session.run(command))
            finally:
                await session.close()
            return results

        return asyncio.run(drive())

    def test_plain_command_streams_and_reports_exit_code(self):
        (result,) = self._run_all("echo start; (exit 2)")
        assert result.exit_code == 2
        assert result.stdout == "start\n"
        assert not result.timed_out

    def test_output_without_trailing_newline_completes(self):
        (result,) = self._run_all("printf 'partial'")
        assert result.exit_code == 0
        assert result.stdout == "partial"
        assert not result.timed_out

    def test_glued_marker_keeps_output_and_exit_code(self):
        (result,) = self._run_all("printf 'partial'; (exit 3)")
        assert result.exit_code == 3
        assert result.stdout == "partial"

    def test_stdin_consumer_reads_eof_instead_of_hanging(self):
        (cat,) = self._run_all("cat")
        assert cat.exit_code == 0
        assert cat.stdout == ""
        (wc,) = self._run_all("wc")
        assert wc.exit_code == 0
        assert wc.stdout.split() == ["0", "0", "0"]

    def test_background_job_cannot_steal_stdin(self):
        first, second = self._run_all("cat & echo bg", "echo after")
        assert first.exit_code == 0
        assert second.exit_code == 0
        assert second.stdout == "after\n"

    def test_shell_exit_then_next_command_restarts(self):
        first, second = self._run_all("echo bye; exit 0", "echo back")
        assert first.stdout == "bye\n"
        # The shell died; the next call must still run (fresh shell), not
        # crash writing into the dead shell's pipe.
        assert second.exit_code == 0
        assert second.stdout == "back\n"

    def test_long_single_line_completes(self):
        (result,) = self._run_all("head -c 200000 /dev/zero | tr '\\0' 'x'")
        assert result.exit_code == 0
        assert result.stdout == "x" * 200000


if __name__ == "__main__":
    unittest.main()
