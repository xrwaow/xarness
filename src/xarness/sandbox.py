"""Bubblewrap-based sandbox for filesystem/shell tool execution.

Every filesystem-touching or shell-executing tool runs inside a bwrap
sandbox scoped to the workspace directory, with no network access by
default. Tools that need the network (web search) run outside the sandbox,
in this process — see tools.py's ``web_search`` handler — so no tool the
model can reach ever gets a raw socket into the sandbox itself.
"""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass, field
from pathlib import Path


class SandboxUnavailable(Exception):
    """Raised when bwrap isn't installed or usable on this host."""


@dataclass(slots=True)
class SandboxConfig:
    """Paths and policy for one sandboxed session."""

    workspace: Path
    # alias -> host path, mounted read-only at /workspace/.refs/<alias>
    external_refs: dict[str, Path] = field(default_factory=dict)
    allow_network: bool = False
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if shutil.which("bwrap") is None:
            raise SandboxUnavailable(
                "bwrap (bubblewrap) not found on PATH; install it "
                "(e.g. `sudo dnf install bubblewrap` on Fedora) "
                "before using read_file/write_file/run_bash tools"
            )
        self.workspace = self.workspace.resolve()
        if not self.workspace.is_dir():
            raise SandboxUnavailable(f"workspace directory does not exist: {self.workspace}")

    def ref_path(self, alias: str) -> str:
        """Path the sandboxed process sees for a given external reference."""
        return f"/workspace/.refs/{alias}"

    def _system_ro_binds(self) -> list[str]:
        args: list[str] = []
        for path in ("/usr", "/etc/resolv.conf", "/etc/ssl"):
            p = Path(path)
            if p.exists():
                args += ["--ro-bind", str(p), str(p)]
        # Merged-/usr distros (Fedora, Arch, modern Debian/Ubuntu) have
        # /bin, /lib, /lib64, /sbin as symlinks into /usr at the real root.
        # Recreate them inside the sandbox; fall back to a real bind if a
        # given distro still uses separate top-level directories.
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
        argv += ["--bind", str(self.workspace), "/workspace"]
        for alias, host_path in self.external_refs.items():
            argv += ["--ro-bind", str(host_path.resolve()), self.ref_path(alias)]
        argv += ["--chdir", "/workspace", "--unshare-all"]
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


def validate_relpath(path: str) -> str | None:
    """Return an error message if path escapes the workspace, else None.

    This is defense-in-depth, not the real enforcement — bwrap's mount
    namespace is what actually stops escapes. This just gives the model
    (and you, debugging) a clean error instead of a bwrap-level failure.
    """
    p = Path(path)
    if p.is_absolute():
        return f"path must be relative to the workspace, got absolute path '{path}'"
    if ".." in p.parts:
        return f"path must not contain '..', got '{path}'"
    return None
