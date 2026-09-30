"""Bubblewrap-based sandbox for filesystem/shell tool execution.

read_file/edit_file use one-shot bwrap invocations (run_in_sandbox).
run_bash uses a persistent shell (SandboxSession) so cwd, env vars, and
background jobs survive across multiple calls within one chat session.
Tools that need the network (web search) run outside the sandbox entirely,
in the harness process — no tool the model can reach ever gets a raw socket
into the sandbox itself.
"""

from __future__ import annotations

import asyncio
import atexit
import os
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Callable

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


class SandboxUnavailable(Exception):
    """Raised when bwrap isn't installed or usable on this host."""


_SHADOW: tuple[str, str] | None = None


def _shadow_placeholders() -> tuple[str, str]:
    """Host paths of an empty file and an empty directory, bind-mounted over
    hidden paths to make them vanish (empty and read-only: writes hit the
    placeholder, never the underlying worktree). One set is reused for the
    process lifetime and cleaned up at exit."""
    global _SHADOW
    if _SHADOW is None:
        root = tempfile.mkdtemp(prefix="xarness-shadow-")
        file_path = os.path.join(root, "f")
        dir_path = os.path.join(root, "d")
        open(file_path, "w").close()
        os.mkdir(dir_path)
        atexit.register(shutil.rmtree, root, True)
        _SHADOW = (file_path, dir_path)
    return _SHADOW


def _git_ignored_paths(workspace: Path) -> list[str]:
    """Workspace-relative paths the repo's ignore rules flag, directories
    collapsed to single entries (``--directory``). Empty when git can't
    answer (no repo, git failed) — the default walk still covers its own
    noise."""
    try:
        res = subprocess.run(
            ["git", "-C", str(workspace), "ls-files", "--others", "--ignored",
             "--exclude-standard", "--directory"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if res.returncode != 0:
        return []
    return [line.rstrip("/") for line in res.stdout.splitlines() if line]


def _hidden_paths(workspace: Path) -> list[str]:
    """Workspace-relative paths to shadow out of the container: exactly what
    the repo's ignore rules flag (``.gitignore`` + ``.git/info/exclude``),
    nothing invented here. ".git" stays visible (read-only). Children of an
    already-hidden directory are dropped — shadowing the parent suffices.
    Empty when git can't answer (no repo, failed)."""
    hidden: list[str] = []
    seen_dirs: list[str] = []

    def buried(rel: str) -> bool:
        return any(rel == d or rel.startswith(d + "/") for d in seen_dirs)

    for rel in _git_ignored_paths(workspace):
        if rel == ".git" or rel.startswith(".git/") or buried(rel):
            continue
        if (workspace / rel).is_dir():
            seen_dirs.append(rel)
        hidden.append(rel)
    return hidden


@dataclass(slots=True)
class SandboxConfig:
    """Paths and policy for one sandboxed session."""

    workspace: Path
    # Workspace-root-relative subdirectory the session is scoped to ("" = the
    # whole workspace). When set, the whole workspace is bound read-only (so
    # git and the surrounding project stay visible) and only this subtree is
    # re-bound read-write on top; the shell starts inside it.
    subtree: str = ""
    external_refs: dict[str, Path] = field(default_factory=dict)
    allow_network: bool = False
    timeout_seconds: float = 60.0
    # Host path of the repo's .git dir. Bound into the sandbox at its real
    # path so git reads (status/diff/log) work inside the sandbox. Bound
    # read-only: the agent never commits — the harness checkpoints on the
    # host (gitwork), so only it touches the index and object store.
    git_dir: Path | None = None
    # Workspace-relative paths (files and directories) hidden from the
    # container: exactly what the repo's ignore rules flag. Computed once at
    # construction; each path is shadowed with an empty placeholder in
    # build_argv, so ignored content is not just filtered from tool output —
    # it does not exist as far as every tool (bash included) can tell.
    # ".git" is excluded: it stays visible, read-only.
    hidden_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if shutil.which("bwrap") is None:
            raise SandboxUnavailable(
                "bwrap (bubblewrap) not found on PATH; install it "
                "(e.g. `sudo dnf install bubblewrap` on Fedora) "
                "before using read_file/edit_file/run_bash tools"
            )
        self.workspace = self.workspace.resolve()
        if not self.workspace.is_dir():
            raise SandboxUnavailable(f"workspace directory does not exist: {self.workspace}")
        self.subtree = self.subtree.strip("/")
        if self.subtree:
            sub = Path(self.subtree)
            if sub.is_absolute() or ".." in sub.parts:
                raise SandboxUnavailable(f"invalid workspace subtree: {self.subtree}")
            if not (self.workspace / self.subtree).is_dir():
                raise SandboxUnavailable(
                    f"workspace subtree directory does not exist: {self.subtree}"
                )
        if self.git_dir is not None:
            self.git_dir = self.git_dir.resolve()
        self.hidden_paths = tuple(
            p for p in _hidden_paths(self.workspace)
            if not self.subtree or not (self.subtree == p or self.subtree.startswith(p + "/"))
        )

    def ref_path(self, alias: str) -> str:
        return f"{self.tool_root}/.refs/{alias}"

    @property
    def tool_root(self) -> str:
        """Sandbox path the agent's workspace is anchored at: the scoped
        subtree when one is set (the --workspace dir inside a bigger repo),
        else the whole mounted workspace."""
        return f"/workspace/{self.subtree}" if self.subtree else "/workspace"

    def tool_path(self, path: str) -> str:
        """Anchor a validated agent-workspace-relative path at its sandbox
        mount point. Commands must not rely on the process cwd: with a
        subtree session bwrap chdirs into the subtree, and tool paths are
        relative to it (see validate_relpath)."""
        norm = "." if path in ("", ".") else (path[2:] if path.startswith("./") else path)
        return f"{self.tool_root}/{norm}"

    def validate_relpath(self, path: str, mode: Literal["read", "write"] = "write") -> str | None:
        """Defense-in-depth path check; bwrap's mount namespace is the real
        enforcement — this just gives a clean error instead of a bwrap
        failure.

        Paths are relative to the agent's workspace (the scoped subtree when
        one is set, else the whole mounted workspace — see tool_path).
        Absolute paths and traversal are rejected in every mode, so a path
        can never point outside the workspace's mount."""
        p = Path(path)
        if p.is_absolute():
            return f"path must be relative to the workspace, got absolute path '{path}'"
        if ".." in p.parts:
            return f"path must not contain '..', got '{path}'"
        return None

    def _system_ro_binds(self) -> list[str]:
        args: list[str] = []
        for path in ("/usr", "/etc/resolv.conf", "/etc/ssl"):
            p = Path(path)
            if p.exists():
                args += ["--ro-bind", str(p), str(p)]
        for link, target in (
            ("/bin", "usr/bin"),
            ("/lib", "usr/lib"),
            ("/lib64", "usr/lib64"),
            ("/sbin", "usr/sbin"),
        ):
            p = Path(link)
            if p.is_symlink():
                args += ["--symlink", target, link]
            elif p.is_dir():
                args += ["--ro-bind", link, link]
        return args

    def build_argv(self, command: list[str]) -> list[str]:
        argv = ["bwrap"]
        argv += self._system_ro_binds()
        argv += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
        chdir = "/workspace"
        if self.subtree:
            # Whole workspace read-only — git keeps working (the .git file at
            # the worktree root stays reachable) and the agent can read the
            # surrounding project — with the session's subtree re-bound
            # read-write on top (later binds shadow earlier ones). Hidden
            # paths are shadowed before that re-bind, so the subtree itself
            # is never hidden from the agent; everything else ignores apply
            # to the surrounding repo too.
            argv += ["--ro-bind", str(self.workspace), "/workspace"]
        else:
            argv += ["--bind", str(self.workspace), "/workspace"]
        if self.subtree:
            argv += [
                "--bind", str(self.workspace / self.subtree), f"/workspace/{self.subtree}",
            ]
            chdir = f"/workspace/{self.subtree}"
        # Shadow hidden paths last: the subtree re-bind above restores the
        # agent's own tree wholesale, so shadows must land after it to stick
        # (ancestors of the subtree were filtered out at construction —
        # shadowing one would bury the re-bind beneath it).
        empty_file, empty_dir = _shadow_placeholders()
        for rel in self.hidden_paths:
            host = self.workspace / rel
            is_dir = host.is_dir() and not host.is_symlink()
            argv += ["--bind", empty_dir if is_dir else empty_file, f"/workspace/{rel}"]
        if self.git_dir is not None:
            # Same path as on the host, so git inside the sandbox resolves
            # it unchanged. Read-only: the harness does all git writes.
            argv += ["--ro-bind", str(self.git_dir), str(self.git_dir)]
        for alias, host_path in self.external_refs.items():
            argv += ["--ro-bind", str(host_path.resolve()), self.ref_path(alias)]
        argv += ["--chdir", chdir, "--unshare-all"]
        if self.allow_network:
            argv += ["--share-net"]
        argv += ["--die-with-parent", "--", *command]
        return argv


@dataclass(slots=True)
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


async def run_in_sandbox(
    config: SandboxConfig,
    command: list[str],
    input_bytes: bytes | None = None,
) -> SandboxResult:
    """One-shot: spawn, run, tear down. Used by read_file/edit_file."""
    argv = config.build_argv(command)
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE if input_bytes is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input_bytes), timeout=config.timeout_seconds
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return SandboxResult(exit_code=-1, stdout="", stderr="timed out", timed_out=True)
    return SandboxResult(
        exit_code=proc.returncode or 0,
        stdout=stdout.decode(errors="replace"),
        stderr=stderr.decode(errors="replace"),
    )


class SandboxSession:
    """One persistent shell running inside one bwrap sandbox for a whole chat.

    cwd, env vars, and background jobs survive across multiple run_bash
    calls within the same conversation. Calls are serialized against each
    other via a lock — one command completes before the next starts.
    """

    def __init__(self, config: SandboxConfig) -> None:
        self._config = config
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            return
        argv = self._config.build_argv(["/bin/sh"])
        self._proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

    async def run(self, command: str, on_output: Callable[[str], None] | None = None) -> SandboxResult:
        """Run one command in the persistent shell.

        ``on_output``, when given, is called with each output line as it
        arrives (so a TUI can show live output while the command runs)."""
        async with self._lock:
            await self._ensure_started()
            assert self._proc is not None and self._proc.stdin and self._proc.stdout
            marker = f"__xarness_done_{uuid.uuid4().hex}__"
            self._proc.stdin.write(f"{command}\necho {marker} $?\n".encode())
            await self._proc.stdin.drain()

            output: list[str] = []

            async def _read_until_marker() -> int:
                while True:
                    line = await self._proc.stdout.readline()
                    if not line:
                        return -1
                    text = line.decode(errors="replace")
                    if text.startswith(marker):
                        return int(text[len(marker):].strip() or "-1")
                    output.append(text)
                    if on_output is not None:
                        on_output(text)

            try:
                exit_code = await asyncio.wait_for(
                    _read_until_marker(), timeout=self._config.timeout_seconds
                )
            except asyncio.TimeoutError:
                await self.close()
                return SandboxResult(exit_code=-1, stdout="".join(output), stderr="", timed_out=True)

            return SandboxResult(exit_code=exit_code, stdout="".join(output), stderr="")

    async def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            await asyncio.wait_for(self._proc.wait(), timeout=5)
        except (asyncio.TimeoutError, ProcessLookupError):
            self._proc.kill()
            await self._proc.wait()
        finally:
            self._proc = None
