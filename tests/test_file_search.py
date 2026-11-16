"""Tests for file search: @-mention candidates, ranking, binary filtering,
and the glob/grep engine behind the read-only search tools."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from xarness import file_search


class FileSearchTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def git(self, *args: str) -> None:
        subprocess.run(
            ["git", *args], cwd=self.root, check=True, capture_output=True,
        )

    async def test_content_binaries_are_not_suggested(self) -> None:
        """Git's excludes no longer hide media types, so the search pipeline
        probes its ranked matches and drops the binaries itself."""
        self.git("init", "-q", "-b", "main", ".")
        (self.root / "notes.txt").write_text("hello\n")
        (self.root / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 8)
        (self.root / "blob").write_bytes(b"\0\0")

        found = await file_search.search_files(self.root, "")

        assert found == ["notes.txt"]

    async def test_query_ranks_text_files_only(self) -> None:
        self.git("init", "-q", "-b", "main", ".")
        (self.root / "a").mkdir()
        (self.root / "a" / "sheep.txt").write_text("baa\n")
        (self.root / "sheep.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 8)

        found = await file_search.search_files(self.root, "sheep")

        assert found == ["a/sheep.txt"]


class GlobGrepTest(unittest.IsolatedAsyncioTestCase):
    """The glob/grep engine; tool wiring and confinement are in test_tools."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "app.py").write_text("class App:\n    pass\n")
        (self.root / "notes.txt").write_text("plain notes\n")
        pkg = self.root / "pkg"
        pkg.mkdir()
        (pkg / "controller.py").write_text("class Controller:\n    pass\n")
        (self.root / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 16)
        # Fixtures for the grep tests; the candidate index is cached for a
        # short TTL, so all files must exist before the first search.
        (self.root / "many.txt").write_text("hit\n" * 5)
        (self.root / "needle.txt").write_text("needle\n")
        (self.root / "blob.bin").write_bytes(b"\0\0needle\0\0")

    async def glob(self, pattern: str, base: str = "."):
        return await file_search.glob_files(self.root, base, pattern)

    def git(self, *args: str) -> None:
        subprocess.run(
            ["git", *args], cwd=self.root, check=True, capture_output=True,
        )

    async def grep(self, pattern: str, base: str = ".", **kwargs):
        return await file_search.grep_files(self.root, base, pattern, **kwargs)

    async def test_glob_bare_name_matches_at_any_depth(self) -> None:
        found, total = await self.glob("*.py")
        assert found == ["app.py", "pkg/controller.py"]
        assert total == 2

    async def test_glob_double_star_and_subdirectory(self) -> None:
        assert (await self.glob("**/*.py"))[0] == ["app.py", "pkg/controller.py"]
        assert (await self.glob("*.py", base="pkg"))[0] == ["pkg/controller.py"]

    async def test_glob_honours_gitignore(self) -> None:
        (self.root / ".gitignore").write_text("pkg/\n")
        self.git("init", "-q", "-b", "main", ".")
        assert (await self.glob("*.py"))[0] == ["app.py"]

    async def test_grep_returns_matches_and_filters_files(self) -> None:
        found, _ = await self.grep(r"class\s+\w+")
        assert found == [
            "app.py:1: class App:",
            "pkg/controller.py:1: class Controller:",
        ]
        assert (await self.grep("class", base="pkg", file_glob="*.py"))[0] == [
            "pkg/controller.py:1: class Controller:",
        ]

    async def test_grep_case_sensitivity_and_caps(self) -> None:
        assert (await self.grep("app"))[0] == []
        assert len((await self.grep("app", case_sensitive=False))[0]) == 1
        found, truncated = await self.grep("hit", limit=3)
        assert len(found) == 3 and truncated

    async def test_grep_skips_binaries(self) -> None:
        assert (await self.grep("needle"))[0] == ["needle.txt:1: needle"]

    def test_glob_translation(self) -> None:
        matches = file_search._glob_regex
        assert matches("*.py").fullmatch("a.py")
        assert not matches("*.py").fullmatch("a/b.py")
        assert matches("a/**/b.py").fullmatch("a/b.py")
        assert matches("a/**/b.py").fullmatch("a/x/y/b.py")
        assert matches("pkg/[ab].py").fullmatch("pkg/a.py")
        assert not matches("pkg/[ab].py").fullmatch("pkg/c.py")
