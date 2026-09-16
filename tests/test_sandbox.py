"""Tests for SandboxConfig subtree scoping: bwrap argv binds and the tool
path validation. bwrap itself is faked (shutil.which patched) — these tests
only exercise config/argv construction, not real sandboxes."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xarness.sandbox import SandboxConfig, SandboxUnavailable


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

        # Tool paths: inside the subtree allowed, outside rejected, the
        # read-only .refs/ binds allowed, traversal always rejected.
        assert config.validate_relpath("f/notes.txt") is None
        assert config.validate_relpath("./f/notes.txt") is None
        assert config.validate_relpath("f") is None
        assert config.validate_relpath("app.py") is not None
        assert config.validate_relpath(".refs/docs") is None
        assert config.validate_relpath("../etc") is not None
        assert config.validate_relpath("/etc") is not None

        # Read mode: the whole workspace is bound read-only, so paths outside
        # the subtree are readable — only traversal/absolute paths are wrong.
        assert config.validate_relpath("app.py", mode="read") is None
        assert config.validate_relpath("f/notes.txt", mode="read") is None
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
        assert config.validate_relpath("app.py") is None


if __name__ == "__main__":
    unittest.main()
