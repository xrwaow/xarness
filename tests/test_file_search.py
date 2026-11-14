"""Tests for @-mention file search: candidates, ranking, binary filtering."""

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
