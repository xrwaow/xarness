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
import functools
import hashlib
import os
import posixpath
import shutil
import signal
import tempfile
import uuid
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import RefSpec, SandboxSettings
    from .gitwork import GitInfo

# Runtime import: RefSpec validation and the sandbox must agree on the
# canonical refs path.
from .config import REFS_ANCHOR
from .gitwork import git_shim_script

# Per-command default when neither the sandbox profile's default_timeout nor
# a per-command timeout applies (one-shot runs keep the same default).
DEFAULT_TIMEOUT = 120.0


def has_ref_mounts(sandbox: "SandboxConfig") -> bool:
    """True when the canonical refs area is actually mounted.

    Refs only exist inside the container; unsandboxed sessions have no refs
    area at all, so neither the tools nor the prompt should advertise one.
    """
    return sandbox.use_sandbox and any(
        ref.mount.startswith(REFS_ANCHOR + "/")
        for ref in sandbox.external_refs.values()
    )


def is_refs_path(path: str) -> bool:
    """True for an absolute path inside the canonical refs area."""
    if not Path(path).is_absolute():
        return False
    norm = posixpath.normpath("/" + path.lstrip("/"))
    return norm == REFS_ANCHOR or norm.startswith(REFS_ANCHOR + "/")


def ref_infos(external: dict[str, "ContainerRef"]) -> list:
    """Client-facing ``protocol.RefInfo`` rows for a resolved ref map.

    One helper for every caller (session status, workspace refs): the rows
    are always sorted by host path, so a listing never reshuffles.
    """
    from .protocol import RefInfo  # local: keep the sandbox import-light

    return [
        RefInfo(path=str(ref.host), mount=ref.mount)
        for _, ref in sorted(external.items(), key=lambda item: str(item[1].host))
    ]


def bind_ref(
    workspace: Path,
    refs: dict[str, ContainerRef],
    host: str | Path,
    *,
    alias: str | None = None,
    mount: str | None = None,
    read_only: bool = True,
) -> ContainerRef:
    """Bind a host path into the container (appended to ``refs``); returns
    the (possibly disambiguated) entry. ``mount`` defaults to
    ``/tmp/refs/<alias>`` with the alias from the path's basename; duplicate
    basenames prepend parent segments until unique."""
    host_path = Path(host).expanduser()
    host_path = host_path if host_path.is_absolute() else workspace / host_path
    host_path = host_path.resolve()
    if not host_path.exists():
        raise FileNotFoundError(f"no such path: {host_path}")
    if alias is None:
        alias = host_path.name
        for depth in range(2, len(host_path.parts) + 1):
            existing = refs.get(alias)
            if existing is None or existing.host == host_path:
                break
            alias = "-".join(host_path.parts[-depth:])
    if mount is None:
        mount = f"{REFS_ANCHOR}/{alias}"
    mount = mount.strip()
    if mount.startswith("/"):
        # Absolute mounts are only the canonical refs area (see REFS_ANCHOR);
        # a ref must never shadow /usr, /etc, the workspace or the caches.
        mount = posixpath.normpath(mount)
        if mount == REFS_ANCHOR or not mount.startswith(REFS_ANCHOR + "/"):
            raise ValueError(
                f"absolute ref mount must be below {REFS_ANCHOR}/: {mount!r}"
            )
    else:
        mount = mount.strip("/")
        if not mount or ".." in Path(mount).parts:
            raise ValueError(f"invalid ref mount point: {mount!r}")
        if mount in (".", ".git") or mount.startswith(".git/"):
            # '.' would mount over the workspace root; '.git' would shadow
            # the harness's read-only git bind (the git guard assumes that
            # mount stays ro).
            raise ValueError(f"ref mount point {mount!r} is not allowed")
    ref = ContainerRef(host=host_path, mount=mount, read_only=read_only)
    refs[alias] = ref
    return ref


# Subdirectories of the persistent sandbox cache, created on demand.
_CACHE_SUBDIRS = ("pip", "cargo", "npm", "venvs")

# Container path of the external-references area (REFS_ANCHOR, imported from
# config). It is a tmpfs nested under the container-only /tmp tmpfs (mounted
# at the top of build_argv, before the workspace bind), so bwrap creates its
# mountpoint — and every per-alias mountpoint below it — inside that tmpfs.
# Nothing, not even an empty anchor directory, ever appears in the host
# worktree. (An earlier design mounted the anchor at <workspace>/.refs, which
# forced a transient .refs directory into the real worktree on every run and
# failed outright in plan mode, where the workspace bind is read-only.)

# Container path the agent's workspace is mounted at (the only one tools and
# argv building ever name).
_WORKSPACE = "/workspace"
# Container path the external (no-.git) snapshot store is bound at, when a
# session tracks a workspace that has no repo of its own. GIT_DIR/GIT_WORK
# _TREE are set to point here so the agent's read-only git reads work.
EXTERNAL_GIT_ANCHOR = "/.xarness-snapshot-store"

# Subdirectory of the sandbox cache holding the read-only git shim, bound at
# <CACHE_ANCHOR>/<GIT_SHIM_DIR> and prepended to the container's PATH when
# SandboxConfig.git_shim is set (see gitwork.git_shim_script).
GIT_SHIM_DIR = "git-shim"


def _write_git_shim(directory: Path) -> Path:
    """Materialize the git shim script into the sandbox cache (idempotent:
    rewritten only when the generated text changes)."""
    directory.mkdir(parents=True, exist_ok=True)
    shim = directory / "git"
    script = git_shim_script()
    if not shim.exists() or shim.read_text(encoding="utf-8") != script:
        shim.write_text(script, encoding="utf-8")
        shim.chmod(0o755)
    return shim

# Container anchor for shared toolchain state (caches, uv-managed pythons,
# rustup). Everything the sandbox shares with the host is bound *under this
# neutral path*, never at its real host path: a bind whose destination is
# e.g. /home/<user>/... makes bwrap recreate that directory tree inside the
# container, which leaks the host's username and layout and gives the agent
# paths that only look like host paths. With everything anchored here the
# container has no /home at all: nothing the agent can name — or list —
# reveals the host layout. (One residual channel: /proc/self/mountinfo
# shows bind sources; masking it is not worth the fragility. The
# workspace's own files — .venv/pyvenv.cfg, .git/config — may contain host
# paths too; that is the user's data.)
CACHE_ANCHOR = "/cache"

# Container path of the sandbox's own cargo target directory. Cargo records
# absolute source paths in its build fingerprints, and the host sees the
# project at its real location (e.g. ~/code/... with ~/.cargo) while the
# sandbox sees /workspace and /cache/cargo. Sharing ./target makes every
# registry crate report "Dirty: the path to the source changed" whenever a
# build switches between host and sandbox, forcing a full rebuild. A
# sandbox-only target dir keeps the two build graphs (and their absolute
# paths) disjoint; host builds keep using ./target.
SANDBOX_CARGO_TARGET_DIR = f"{_WORKSPACE}/target-sandbox"

# Bind mountpoints that have to exist in the workspace on the host (a
# custom ref's target, the project venv at .venv). bwrap can create a
# missing mountpoint itself, but only inside a writable mount: in plan mode
# the workspace is read-only and the container would refuse to start, so the
# host creates the directory up front. Directories the sandbox created are
# empty and removed at exit, so the worktree looks untouched (git ignores
# empty dirs anyway, but ls doesn't). The refs tmpfs anchor is not here: it
# lives under the container-only /tmp tmpfs and never touches the host at
# all.
_MOUNT_DIRS: set[Path] = set()
_CLEANUP_REGISTERED = False


def _ensure_mount_dir(path: Path) -> bool:
    """Make ``path`` a directory bwrap can bind onto; False if impossible.

    A pre-existing directory is used as-is and never cleaned up; one the
    sandbox creates is empty and removed at exit. A file (or a dangling
    symlink) is refused — binding over it would replace something of the
    user's, and the caller can just skip that mount.
    """
    global _CLEANUP_REGISTERED
    if path.is_dir():
        return True
    if path.exists() or path.is_symlink():
        return False
    try:
        path.mkdir(parents=True)
    except OSError:
        return False
    _MOUNT_DIRS.add(path)
    if not _CLEANUP_REGISTERED:
        atexit.register(_cleanup_mount_dirs)
        _CLEANUP_REGISTERED = True
    return True


def _cleanup_mount_dirs() -> None:
    for p in list(_MOUNT_DIRS):
        try:
            if p.is_dir():
                p.rmdir()  # raises if it has (or gained) content
            elif p.is_file() and p.stat().st_size == 0:
                p.unlink()
        except OSError:
            pass


# Workspace markers meaning "this project has a venv the agent should reach at
# the conventional path" (see SandboxConfig.uses_project_venv). The nested
# .venv/pyvenv.cfg entry covers a bare venv next to no project file; it is
# never created here, so it cannot feed back into the decision.
_PROJECT_VENV_MARKERS = (
    "pyproject.toml", "uv.lock", "setup.py", "requirements.txt", "Pipfile",
    ".venv/pyvenv.cfg",
)


@functools.cache
def _etc_skeleton() -> Path:
    """Host directory bound over /etc (read-only). Pre-creates every
    mountpoint bwrap will bind into /etc (it cannot create mountpoints
    inside a ro mount) and adds the minimal synthetic files the toolchain
    needs — hosts (localhost), passwd/group (getpwuid; placeholder name,
    the container's real uid/gid) — so nothing from the host's identity is
    needed or leaked. The result: an /etc that carries no host identity and
    is not writable."""
    root = Path(tempfile.mkdtemp(prefix="xarness-etc-"))
    for sub in ("ssl", "pki", "alternatives"):
        (root / sub).mkdir()
    (root / "resolv.conf").touch()
    uid, gid = os.getuid(), os.getgid()
    (root / "passwd").write_text(
        f"sandbox:x:{uid}:{gid}:sandbox:/tmp:/bin/sh\n"
    )
    (root / "group").write_text(f"sandbox:x:{gid}:\n")
    (root / "hosts").write_text(
        "127.0.0.1 localhost\n::1 localhost\n"
    )
    atexit.register(shutil.rmtree, root, True)
    return root


def cleanup_mount_dirs_now() -> None:
    """Best-effort removal of the empty mountpoint dirs a session created
    (called when a sandbox session closes; leftovers retry at exit)."""
    _cleanup_mount_dirs()


def sandbox_cache_root() -> Path:
    """Host directory backing the sandbox's persistent toolchain state
    (uv-managed pythons, pip cache, cargo, npm, project venvs). Overridable
    with XARNESS_SANDBOX_CACHE; defaults to ``~/.sandbox-cache``. Created
    (with its subdirs) on first use. The uv cache itself is the host's own
    ``~/.cache/uv``, bound separately."""
    override = os.environ.get("XARNESS_SANDBOX_CACHE")
    root = Path(override).expanduser() if override else Path.home() / ".sandbox-cache"
    for sub in _CACHE_SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


class SandboxUnavailable(Exception):
    """Raised when bwrap isn't installed or usable on this host (the
    session falls back to filesystem/bash tools being disabled)."""


@functools.cache
def _shadow_placeholders() -> tuple[str, str]:
    """Host paths of an empty file and an empty directory, bind-mounted
    (read-only) over hidden paths to make them vanish: reads see emptiness,
    writes fail. One set is reused for the process lifetime and cleaned up
    at exit."""
    root = tempfile.mkdtemp(prefix="xarness-shadow-")
    file_path = os.path.join(root, "f")
    dir_path = os.path.join(root, "d")
    open(file_path, "w").close()
    os.mkdir(dir_path)
    atexit.register(shutil.rmtree, root, True)
    return file_path, dir_path


def _read_pyvenv_cfg(path: Path) -> dict[str, str]:
    """Parse the ``key = value`` lines of a venv's pyvenv.cfg; {} if absent."""
    try:
        text = path.read_text()
    except OSError:
        return {}
    cfg: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and value.strip():
            cfg[key.strip()] = value.strip()
    return cfg


def _host_venv_python(workspace: Path, python_dir: Path) -> str | None:
    """Python version the host venv was built with, if it can be pinned.

    Only uv-managed interpreters are bound into the container (read-only, at
    ``/cache/uv-python``), so a pin is returned only for a venv built on one:
    a system-python venv would name an interpreter that does not exist in
    here, turning the pin into "no interpreter found". uv writes
    ``version_info``; CPython's own venv writes ``version``.
    """
    cfg = _read_pyvenv_cfg(workspace / ".venv" / "pyvenv.cfg")
    version = cfg.get("version_info") or cfg.get("version")
    home = cfg.get("home")
    if not (version and home) or not python_dir.is_dir():
        return None
    try:
        Path(home).resolve().relative_to(python_dir.resolve())
    except (OSError, ValueError):
        return None
    return version


@dataclass(slots=True)
class ContainerRef:
    """One external path bound into the container.

    ``mount`` is the canonical container path (``/tmp/refs/<alias>``, a
    purely container-side tmpfs, nothing is ever created in the worktree) or
    any workspace-relative path (e.g. ``.venv``, to drop a host-built venv
    into the project)."""

    host: Path
    mount: str
    read_only: bool = True


@dataclass(slots=True)
class SandboxConfig:
    """Paths and policy for one sandboxed session."""

    workspace: Path
    # Read-only workspace (plan mode): the whole workspace is bound
    # read-only. Flipped by build_registry per
    # mode; a persistent SandboxSession restarts its shell when this changes.
    read_only: bool = False
    external_refs: dict[str, ContainerRef] = field(default_factory=dict)
    allow_network: bool = False
    # Expose the host's GPU device nodes (/dev/dri, /dev/kfd, /dev/nvidia*)
    # and the sysfs/proc bits drivers enumerate through, so CUDA/ROCm work
    # inside the container. Off by default; driver userland ships in
    # /usr/lib(64), which is already bound read-only.
    gpu_access: bool = False
    # False = tools run as a user-level stateful bash on the host (an
    # unsandboxed mode): no bwrap container, no mount isolation. Documented
    # in the config; the TUI shows a visible marker when it is off.
    # NOTE: with use_sandbox=False the mount-isolation settings are
    # IGNORED — filter_list (no mounts to exclude), allow_network (nothing
    # blocks network on the host) and gpu_access (all devices are already
    # reachable) have no effect. Plan mode therefore refuses run_bash in
    # this mode (see tools.build_registry): there is no read-only mount to
    # enforce it with.
    use_sandbox: bool = True
    # Install the read-only git shim (gitwork.git_shim_script) first on
    # PATH: blocked subcommands fail in place at execution time instead of
    # the whole command string being vetoed before it runs. Set by
    # from_settings when git tracking is active and allow_git_write is off;
    # unsandboxed sessions prepend the same shim to the host PATH (see
    # SandboxSession._ensure_started).
    # Part of mount_key, so enabling it restarts the persistent shell.
    git_shim: bool = False
    # Workspace-root-relative paths excluded from the container's mounts
    # (a mount exclusion, not a gitignore). Part of mount_key, so changing
    # it restarts the persistent shell.
    filter_list: list[str] = field(default_factory=list)
    # Host path of the repo's .git dir. Bound into the sandbox at its real
    # path so git reads (status/diff/log) work inside the sandbox. Bound
    # read-only: the agent never commits — the harness checkpoints on the
    # host (gitwork), so only it touches the index and object store.
    git_dir: Path | None = None
    # External snapshot-store mode (tracked with no .git in the workspace):
    # git_dir lives outside the workspace by design; bind it at
    # EXTERNAL_GIT_ANCHOR and point GIT_DIR/GIT_WORK_TREE at it. (A scoped
    # subtree of a real repo also has its git_dir outside the workspace,
    # but there it stays unbound — useless to the agent.)
    git_external: bool = False
    _argv_prefix_cache: tuple[tuple, list[str]] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.use_sandbox and shutil.which("bwrap") is None:
            raise SandboxUnavailable(
                "bwrap (bubblewrap) not found on PATH; install it "
                "(e.g. `sudo dnf install bubblewrap` on Fedora) "
                "before using read_file/edit_file/run_bash tools"
            )
        self.workspace = self.workspace.resolve()
        if not self.workspace.is_dir():
            raise SandboxUnavailable(f"workspace directory does not exist: {self.workspace}")
        if self.git_dir is not None:
            self.git_dir = self.git_dir.resolve()

    @property
    def mount_key(self) -> tuple:
        """Identity of everything that changes the container's mounts. A
        persistent SandboxSession compares this before each command and
        restarts its shell when it differs, so runtime changes (mode flips,
        ref edits) take effect without losing the shell more often than
        necessary."""
        home = Path.home()
        return (
            self.read_only,
            self.allow_network,
            self.gpu_access,
            self.use_sandbox,
            self.git_shim,
            tuple(sorted(self.filter_list)),
            self.uses_project_venv(),
            # Inputs of _cache_binds_and_env's mounts and env, which can
            # appear after session start (a first `uv lock`, a first
            # `cargo install`).
            (self.workspace / "uv.lock").is_file(),
            (home / ".cargo" / "bin").is_dir(),
            (home / ".local" / "bin").is_dir(),
            (home / ".rustup").is_dir(),
            (home / ".local" / "share" / "uv" / "python").is_dir(),
            _host_venv_python(self.workspace, home / ".local" / "share" / "uv" / "python"),
            tuple(sorted(
                (alias, str(ref.host), ref.mount, ref.read_only)
                for alias, ref in self.external_refs.items()
            )),
        )

    def uses_project_venv(self) -> bool:
        """True when the workspace is a Python project whose venv the agent
        should reach at ``.venv`` (see ``_cache_binds_and_env``).

        Deliberately decided from marker files only: the mountpoint itself
        may be created by the sandbox, and a decision that flipped once that
        directory existed would change ``mount_key`` and restart the shell
        for nothing. ``.venv`` in ``filter_list`` means the user wants it
        hidden, so it is not exposed.
        """
        if ".venv" in {rel.strip("/") for rel in self.filter_list}:
            return False
        return any((self.workspace / marker).exists()
                   for marker in _PROJECT_VENV_MARKERS)

    # ------------------------------------------------------------------
    # Config construction

    @classmethod
    def from_settings(
        cls,
        workspace: Path,
        settings: "SandboxSettings",
        git_info: "GitInfo | None" = None,
        extra_refs: "Iterable[str | RefSpec] | None" = None,
    ) -> "SandboxConfig":
        """Build a config from a config.SandboxSettings profile.

        ``extra_refs`` are the workspace refs merged on top of the
        profile's own ``auto_include_refs`` (the session owns the merge;
        settings stay profile-only)."""
        return cls(
            workspace=workspace,
            allow_network=settings.allow_network,
            gpu_access=settings.gpu_access,
            use_sandbox=settings.use_sandbox,
            filter_list=list(settings.filter_list),
            external_refs=cls.resolve_auto_refs(
                workspace,
                [*settings.auto_include_refs, *(extra_refs or [])],
            ),
            git_dir=git_info.git_dir if git_info is not None else None,
            git_external=git_info.external if git_info is not None else False,
            git_shim=(git_info is not None and not settings.allow_git_write),
        )

    @staticmethod
    def resolve_auto_refs(
        workspace: Path,
        specs: Iterable["str | RefSpec"],
        external: dict[str, ContainerRef] | None = None,
    ) -> dict[str, ContainerRef]:
        """Turn ``auto_include_refs`` entries (plain path strings or
        ``RefSpec`` objects with ``path``/``mount``/``read_only``) into the
        alias → ContainerRef map bound into the container.

        Entries may be absolute (or ``~``-expanded) host paths or
        workspace-relative paths. The alias is the path's basename,
        disambiguated with parent segments when two paths share one (so
        ``docs/spec.md`` and ``~/notes/spec.md`` become ``spec.md`` and
        ``notes-spec.md``). Missing paths are skipped — refs are
        conveniences, not hard requirements."""
        resolved: dict[str, ContainerRef] = dict(external or {})
        for spec in specs:
            if isinstance(spec, str):
                path, mount, ro = spec, None, True
            else:
                path, mount, ro = spec.path, spec.mount, spec.read_only
            try:
                bind_ref(workspace, resolved, path, mount=mount, read_only=ro)
            except FileNotFoundError:
                continue
        return resolved

    @property
    def tool_root(self) -> str:
        """Path the agent's workspace is anchored at: the container mount,
        or the real directory in unsandboxed mode. Every use of this for
        argv building is inside a container-only branch."""
        return _WORKSPACE if self.use_sandbox else str(self.workspace)

    def tool_path(self, path: str) -> str:
        """Anchor a validated agent-workspace-relative path at its mount point.
        Absolute container paths (e.g. /tmp/refs/<alias>) are returned as-is.
        Commands must not rely on the process cwd.

        Unsandboxed mode (use_sandbox=False) has no container mount: paths
        resolve to the real workspace on the host."""
        norm = posixpath.normpath(path) if path else "."
        if posixpath.isabs(norm):
            return norm
        if not self.use_sandbox:
            return str(self.workspace / norm) if norm != "." else str(self.workspace)
        return f"{_WORKSPACE}/{norm}"

    def validate_relpath(self, path: str) -> str | None:
        """Defense-in-depth path check; bwrap's mount namespace is the real
        enforcement — this just gives a clean error instead of a bwrap
        failure.

        Paths are relative to the agent's workspace (see tool_path).
        Absolute paths and traversal are rejected, so a path can never
        point outside the workspace's mount (the read-only mount in plan
        mode is what enforces write restrictions)."""
        p = Path(path)
        if p.is_absolute():
            return f"path must be relative to the workspace, got absolute path '{path}'"
        if ".." in p.parts:
            return f"path must not contain '..', got '{path}'"
        return None

    def _system_ro_binds(self) -> list[str]:
        args: list[str] = []
        # /etc first: a read-only skeleton with synthetic identity files and
        # pre-created mountpoints (bwrap can't create mountpoints inside a
        # ro mount), then the real DNS/TLS/trust files bound into it. The
        # host's identity files (/etc/passwd, hostname, machine-id, …) don't
        # exist in here — the container carries no trace of the host's user
        # or name — and nothing under /etc is writable.
        args += ["--ro-bind", str(_etc_skeleton()), "/etc"]
        resolv = Path("/etc/resolv.conf")
        if resolv.exists():
            args += ["--ro-bind", str(resolv), "/etc/resolv.conf"]
        for path in ("/usr", "/etc/ssl", "/etc/pki", "/etc/alternatives"):
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

    def _cache_binds_and_env(self) -> list[str]:
        """Shared toolchain state, bound under the neutral CACHE_ANCHOR (see
        the CACHE_ANCHOR note for why never at a host path). Sources are the
        host's real directories, so caches stay warm across sessions.

        Writable: the sandbox cache root (pip/cargo/npm caches, project
        venvs) and the host's own ``~/.cache/uv``. Read-only: uv-managed
        interpreters, rustup and the toolchain bins on PATH. Anything a tool
        installs therefore lands in the writable caches."""
        root = sandbox_cache_root()
        anchor = CACHE_ANCHOR
        argv = ["--bind", str(root), anchor]
        for var, sub in (
            ("PIP_CACHE_DIR", "pip"),
            ("CARGO_HOME", "cargo"),
            ("npm_config_cache", "npm"),
        ):
            argv += ["--setenv", var, f"{anchor}/{sub}"]
        # Never share the host's ./target: cargo fingerprints embed absolute
        # source paths, which differ between the host and the /workspace,
        # /cache/cargo view inside the container (every registry crate would
        # be 'Dirty: the path to the source changed' on each switch). The
        # workspace is bound read-write, so this directory is created on the
        # first build and persists alongside the user's sources.
        argv += ["--setenv", "CARGO_TARGET_DIR", SANDBOX_CARGO_TARGET_DIR]
        argv += ["--setenv", "UV_LINK_MODE", "copy"]
        # The shell must not inherit a HOME that names a host path (see
        # CACHE_ANCHOR); /tmp is the container's own writable tmpfs.
        argv += ["--setenv", "HOME", "/tmp"]
        # uv's cache is the host's own (~/.cache/uv), bound read-write.
        host_uv_cache = Path.home() / ".cache" / "uv"
        host_uv_cache.mkdir(parents=True, exist_ok=True)
        argv += ["--bind", str(host_uv_cache), f"{anchor}/uv"]
        argv += ["--setenv", "UV_CACHE_DIR", f"{anchor}/uv"]
        # uv's project env lives outside the workspace, keyed by the
        # workspace path: the project's own .venv has absolute paths baked
        # in for its real host location, which never matches the /workspace
        # mount inside the sandbox — without this uv would try to rebuild
        # .venv in the user's worktree, corrupting it. VIRTUAL_ENV is
        # dropped for the same mismatch reason.
        venv = root / "venvs" / hashlib.sha1(
            str(self.workspace).encode()
        ).hexdigest()[:12]
        venv.mkdir(parents=True, exist_ok=True)
        argv += [
            "--setenv", "UV_PROJECT_ENVIRONMENT", f"{anchor}/venvs/{venv.name}",
            "--unsetenv", "VIRTUAL_ENV",
        ]
        # Interpreter + discovery: match the host project venv's Python so
        # the wheels already in the host cache fit (e.g. host on 3.13 while
        # the sandbox's default `python3` is 3.14 — different wheel tags).
        # uv-managed interpreters are the host's (~/.local/share/uv/python),
        # read-only: discovery works, installs of new interpreters would
        # need write access and fail — the writable UV_CACHE_DIR and
        # UV_PROJECT_ENVIRONMENT cover everything uv must write.
        host_python_dir = Path.home() / ".local" / "share" / "uv" / "python"
        if host_python_dir.is_dir():
            argv += ["--ro-bind", str(host_python_dir), f"{anchor}/uv-python"]
            argv += ["--setenv", "UV_PYTHON_INSTALL_DIR", f"{anchor}/uv-python"]
        # Pin the interpreter to the host venv's so the wheels already in the
        # host cache fit (e.g. host on 3.13 while the sandbox's default
        # python3 is 3.14 — different wheel tags).
        if pinned := _host_venv_python(self.workspace, host_python_dir):
            argv += ["--setenv", "UV_PYTHON", pinned]
        # The venv the agent reaches is the project env, at the path it
        # expects: bind the uv-managed env over the workspace's .venv. The
        # workspace's own .venv belongs to the host — its interpreter symlink
        # and its console-script shebangs name host paths that do not exist
        # in here, so leaving it in place reads as "python is missing" (exit
        # 127) and invites the agent to "repair" the user's worktree. Bind
        # read-write even in plan mode: the state lives in the cache, and uv
        # must be able to sync there to run a test. Ref mounts are emitted
        # after this, so a deliberately dropped-in host venv still wins.
        if (self.uses_project_venv()
                and _ensure_mount_dir(self.workspace / ".venv")):
            argv += ["--bind", str(venv), f"{self.tool_root}/.venv"]
        # No network: never reach for the index — cached index data and
        # wheels must be enough (otherwise every uv call stalls on DNS
        # retries before failing).
        if not self.allow_network:
            argv += ["--setenv", "UV_OFFLINE", "1"]
        # Toolchain homes, read-only, at container paths. RUSTUP_HOME is
        # redirected so rustup shims find the toolchains; the shims
        # themselves and other user-installed tools (uv, pip) come from the
        # host bin dirs, bound read-only and prepended to PATH — the
        # inherited PATH names host paths that do not exist in here.
        home = Path.home()
        host_rustup = home / ".rustup"
        if host_rustup.is_dir():
            argv += ["--ro-bind", str(host_rustup), f"{anchor}/rustup"]
            argv += ["--setenv", "RUSTUP_HOME", f"{anchor}/rustup"]
        path_prepend: list[str] = []
        cargo_bin = home / ".cargo" / "bin"
        if cargo_bin.is_dir():
            argv += ["--ro-bind", str(cargo_bin), f"{anchor}/cargo-bin"]
            path_prepend.append(f"{anchor}/cargo-bin")
        local_bin = home / ".local" / "bin"
        if local_bin.is_dir():
            argv += ["--ro-bind", str(local_bin), f"{anchor}/local-bin"]
            path_prepend.append(f"{anchor}/local-bin")
        if self.git_shim:
            # First on PATH, ahead of every real bin dir: the shim is the
            # `git` the shell resolves. The cache root is already bound at
            # the anchor, so only the PATH entry is needed here.
            path_prepend.append(f"{anchor}/{GIT_SHIM_DIR}")
            _write_git_shim(sandbox_cache_root() / GIT_SHIM_DIR)
        # PATH is always set: --clearenv left none, and the inherited one
        # names host paths that do not exist in here. /usr is bound
        # wholesale, so its entries are real; the toolchain bins come from
        # the host, read-only, at container paths.
        argv += ["--setenv", "PATH", ":".join([*path_prepend, "/usr/local/bin", "/usr/bin", "/bin"])]
        return argv

    def _gpu_binds(self) -> list[str]:
        """--dev-bind the host's GPU device nodes plus the driver sysfs the
        CUDA/NVML userspace walks. Only called with ``gpu_access`` on."""
        argv: list[str] = []
        gpu_devs = [
            "/dev/dri", "/dev/kfd",
            "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools",
            "/dev/nvidia-modeset",
            # Capability device nodes (nvidia-cap<N>): as a non-root user
            # NVML is forced onto the caps path (/proc/driver/nvidia/
            # capabilities → /dev/nvidia-caps/nvidia-cap<N>) and reports
            # "GPU access blocked by the operating system" without them,
            # even when the plain device nodes open fine.
            "/dev/nvidia-caps",
        ]
        gpu_devs += sorted(str(p) for p in Path("/dev").glob("nvidia[0-9]*"))
        for dev in gpu_devs:
            p = Path(dev)
            if p.is_dir():
                if any(p.iterdir()):
                    argv += ["--dev-bind", dev, dev]
            elif p.exists():
                argv += ["--dev-bind", dev, dev]
        # Driver enumeration paths (nvidia management, PCI/DRM topology
        # walks by torch and friends). Read-only; missing ones skipped.
        for sys_path in (
            "/proc/driver/nvidia",
            "/sys/bus/pci", "/sys/class/drm", "/sys/devices",
        ):
            if Path(sys_path).exists():
                argv += ["--ro-bind", sys_path, sys_path]
        # CUDA (cuInit) hard-requires the kernel modules' sysfs state
        # files: without /sys/module/nvidia/initstate it aborts with
        # CUDA_ERROR_OS (304); without nvidia_uvm's, with UNKNOWN (999)
        # — even though every device ioctl succeeds. Bind every nvidia*
        # module dir the parent exposes.
        for mod in sorted(Path("/sys/module").glob("nvidia*")):
            if mod.is_dir():
                argv += ["--ro-bind", str(mod), str(mod)]
        if _caps_unusable():
            # NVML prefers the capability path (caps files →
            # /dev/nvidia-caps/nvidia-cap<N>) and fails hard when those
            # nodes are missing/unreadable. Shadow the caps directory so
            # NVML falls back to direct device access, like on hosts
            # without exposed caps — direct ioctls are unaffected.
            _, empty_dir = _shadow_placeholders()
            argv += ["--bind", empty_dir, "/proc/driver/nvidia/capabilities"]
        return argv

    def _filter_binds(self) -> list[str]:
        """Shadow filter_list paths so they are simply not visible in the
        container (a mount exclusion, not a gitignore)."""
        argv: list[str] = []
        for rel in self.filter_list:
            # Shadowed with an existing host placeholder (empty file over
            # files, empty dir over directories), so no mountpoint is ever
            # created inside the workspace bind — i.e. in the user's
            # worktree. With tracking on, .git is always mounted and cannot
            # be filtered. Paths that do not exist in the workspace are
            # skipped: bwrap needs the mountpoint to exist.
            rel_path = self.workspace / rel.strip("/")
            if not rel_path.exists():
                continue
            empty_file, empty_dir = _shadow_placeholders()
            target = f"{_WORKSPACE}/{rel.strip('/')}"
            if rel_path.is_dir():
                argv += ["--ro-bind", empty_dir, target]
            else:
                argv += ["--ro-bind", empty_file, target]
        return argv

    def _ref_binds(self) -> list[str]:
        """The external-references mounts (part of build_argv; see the
        REFS_ANCHOR note there for why the workspace bind must come first)."""
        argv: list[str] = []
        refs_anchor = False
        for ref in self.external_refs.values():
            if ref.mount.startswith("/"):
                # Canonical refs area, a tmpfs nested under the container-only
                # /tmp tmpfs (mounted at the top of build_argv — that ordering
                # is what makes the anchor virtual: bwrap creates the
                # mountpoint inside the tmpfs, so the host worktree is never
                # touched, even transiently).
                if not refs_anchor:
                    argv += ["--tmpfs", REFS_ANCHOR]
                    refs_anchor = True
                target = ref.mount
            else:
                target = f"{self.tool_root}/{ref.mount}"
                if not _ensure_mount_dir(self.workspace / ref.mount):
                    continue  # no mountpoint possible: refs are conveniences
            # Plan mode forces every ref read-only, like the rest of the
            # workspace.
            argv += ["--ro-bind" if (ref.read_only or self.read_only) else "--bind",
                     str(ref.host), target]
        return argv

    def build_argv(self, command: list[str]) -> list[str]:
        if not self.use_sandbox:
            # Unsandboxed mode: a user-level stateful shell on the host —
            # no bwrap, no mount isolation. The persistent shell starts in
            # the workspace (see SandboxSession._ensure_started).
            return list(command)
        proxies = tuple(
            os.environ.get(var)
            for var in ("http_proxy", "https_proxy", "no_proxy",
                        "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY")
        )
        cache_key = (self.mount_key, proxies)
        cached = self._argv_prefix_cache
        if cached is not None and cached[0] == cache_key:
            return [*cached[1], "--", *command]
        argv = ["bwrap"]
        # Clear the host environment before anything else: everything the
        # toolchain needs is set explicitly below (see
        # _cache_binds_and_env), and the inherited env would leak identity
        # and secrets (USER, LOGNAME, XDG_RUNTIME_DIR, SSH_AUTH_SOCK,
        # DBUS_*, API keys). Proxies are re-added for network mode at the
        # bottom; --clearenv must precede every --setenv (bwrap processes
        # options in order).
        argv += ["--clearenv"]
        argv += self._system_ro_binds()
        argv += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
        if self.gpu_access:
            # GPU compute: --dev above created an empty /dev, so bind the
            # host's render/compute nodes back in (dev-bind: devices need
            # more than read access). NVIDIA userland lives in /usr/lib(64),
            # already bound read-only by _system_ro_binds.
            argv += self._gpu_binds()
        chdir = _WORKSPACE
        argv += ["--ro-bind" if self.read_only else "--bind",
                 str(self.workspace), _WORKSPACE]
        argv += self._filter_binds()
        uv_lock = self.workspace / "uv.lock"
        if uv_lock.is_file():
            # Read-only: uv run may read the lock to build the project env
            # (UV_PROJECT_ENVIRONMENT, outside the workspace) but must not
            # rewrite the user's lockfile from inside the sandbox.
            argv += ["--ro-bind", str(uv_lock), f"{_WORKSPACE}/uv.lock"]
        if self.git_dir is not None:
            # Bound read-only over the workspace bind (later binds shadow
            # earlier ones), so git reads work in write mode too while the
            # agent never touches the index or object store — the harness
            # checkpoints on the host (gitwork). Never at its host path: see
            # the CACHE_ANCHOR note.
            if self.git_external:
                # External snapshot store (tracked-with-no-.git mode): bind
                # at a fixed anchor and point git at it explicitly, so the
                # agent's read-only git status/diff see the tracked state.
                argv += ["--ro-bind", str(self.git_dir), EXTERNAL_GIT_ANCHOR]
                argv += ["--setenv", "GIT_DIR", EXTERNAL_GIT_ANCHOR]
                argv += ["--setenv", "GIT_WORK_TREE", _WORKSPACE]
            else:
                try:
                    rel_git = self.git_dir.relative_to(self.workspace)
                except ValueError:
                    pass  # scoped subtree of a bigger repo: unbindable, useless
                else:
                    argv += ["--ro-bind", str(self.git_dir),
                             f"{self.tool_root}/{rel_git}"]
        argv += self._cache_binds_and_env()
        argv += self._ref_binds()
        argv += ["--chdir", chdir, "--unshare-all", "--new-session"]
        # Fresh hostname: --unshare-all unshares the UTS ns but the host's
        # hostname would still be inherited.
        argv += ["--hostname", "sandbox"]
        argv += ["--setenv", "LANG", "C.UTF-8"]
        # Python block-buffers stdout when it is not a tty, so a long
        # pytest run would stream nothing until it exits. --clearenv dropped
        # any inherited value, so it has to be set in here.
        argv += ["--setenv", "PYTHONUNBUFFERED", "1"]
        if self.allow_network:
            for var in ("http_proxy", "https_proxy", "no_proxy",
                        "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
                if var in os.environ:
                    argv += ["--setenv", var, os.environ[var]]
            argv += ["--share-net"]
        argv += ["--die-with-parent"]
        self._argv_prefix_cache = (cache_key, list(argv))
        return [*argv, "--", *command]


@dataclass(slots=True)
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


# How much of a command's output SandboxSession keeps: the first MiB and the
# last MiB, with the middle replaced by a marker.
_CAPTURE_HEAD = 1 << 20
_CAPTURE_TAIL = 1 << 20
_ELIDED = "\n[... {chars} chars of output elided ...]\n"


def _caps_unusable() -> bool:
    """True when the driver exposes MIG capability files but the matching
    /dev/nvidia-caps/nvidia-cap<N> nodes are absent OR not readable by this
    user (e.g. nvidia-cap1 is cr-------- owned by another uid) — non-root
    NVML then refuses GPU access outright with "GPU access blocked by the
    operating system" before ever touching the real device."""
    if not Path("/proc/driver/nvidia/capabilities/mig/config").exists():
        return False
    nodes = list(Path("/dev/nvidia-caps").glob("nvidia-cap*"))
    if not nodes:
        return True
    return not all(os.access(p, os.R_OK) for p in nodes)


async def run_in_sandbox(
    config: SandboxConfig,
    command: list[str],
    input_bytes: bytes | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> SandboxResult:
    """One-shot: spawn, run, tear down. Used by read_file/edit_file."""
    argv = config.build_argv(command)
    proc = await asyncio.create_subprocess_exec(
        *argv,
        # DEVNULL, not an inherited stdin: a one-shot read must never
        # consume the harness's own stdin.
        stdin=asyncio.subprocess.PIPE if input_bytes is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input_bytes), timeout=timeout
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

    def __init__(
        self, config: SandboxConfig,
        default_timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._config = config
        # Lives on the session, not on SandboxConfig: it comes from the
        # sandbox profile's default_timeout, and the config would otherwise
        # duplicate it.
        self.default_timeout = default_timeout
        self._mount_key: tuple | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    @property
    def config(self) -> SandboxConfig:
        return self._config

    def adopt(
        self, config: "SandboxConfig",
        default_timeout: float | None = None,
    ) -> None:
        """Swap in a new config for the same shell: the next ``run`` sees
        the changed ``mount_key`` and restarts itself under the new mounts
        (a settings/refs/mode change needs no new process, just a new
        argv)."""
        self._config = config
        if default_timeout is not None:
            self.default_timeout = default_timeout

    async def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            # Any change to the container's mounts (plan/write flip, ref
            # changes) alters the mount
            # argv; the running shell was started under the old mounts, so
            # restart it to pick up the new view.
            key = self._config.mount_key
            if self._mount_key == key:
                return
            await self.close()
        self._mount_key = self._config.mount_key
        argv = self._config.build_argv(["/bin/sh"])
        # Container mode sets PYTHONUNBUFFERED via --setenv (--clearenv drops
        # an inherited value); unsandboxed mode has no bwrap, so hand it to
        # the shell's own environment.
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        if not self._config.use_sandbox and self._config.git_shim:
            # Unsandboxed mode has no container PATH to rewrite from the
            # inside, so prepend the same read-only git shim here: both modes
            # then enforce one policy, at execution time.
            shim_dir = sandbox_cache_root() / GIT_SHIM_DIR
            _write_git_shim(shim_dir)
            env["PATH"] = f"{shim_dir}:{env.get('PATH', '')}"
        self._proc = await asyncio.create_subprocess_exec(
            *argv,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            # Unsandboxed mode: the host shell starts in the workspace.
            cwd=str(self._config.workspace) if not self._config.use_sandbox else None,
            # readline's default 64 KiB line limit raises ValueError on
            # longer lines (minified bundles), which surfaces as a confusing
            # tool error — allow chunky single lines instead.
            limit=8 * 1024 * 1024,
            # Own process group, so an interrupt can kill the command and
            # every child it spawned, not just the shell.
            start_new_session=True,
        )

    async def run(
        self,
        command: str,
        on_output: Callable[[str], None] | None = None,
        timeout: float | None = None,
    ) -> SandboxResult:
        """Run one command in the persistent shell.

        ``on_output``, when given, is called with each output line as it
        arrives (so a TUI can show live output while the command runs).
        ``timeout`` overrides the session's default timeout for this one
        command (None = default)."""
        async with self._lock:
            await self._ensure_started()
            assert self._proc is not None and self._proc.stdin and self._proc.stdout
            marker = f"__xarness_done_{uuid.uuid4().hex}__"
            # The command runs in a group with stdin from /dev/null: a command
            # that reads stdin (cat, wc, a REPL) — or a job it backgrounds —
            # would otherwise swallow the marker line below (or a later
            # command's lines) and stall the read loop until the timeout.
            payload = f"{{ {command}\n}} </dev/null\necho {marker} $?\n".encode()
            try:
                self._proc.stdin.write(payload)
                await self._proc.stdin.drain()
            except (ConnectionResetError, BrokenPipeError):
                # The shell died between the liveness check in
                # _ensure_started and this write (e.g. the previous command
                # ran `exit`) — asyncio may not have reaped it yet, so the
                # check passed. Restart once and resend; a fresh shell has
                # no pending state to lose.
                await self.close()
                await self._ensure_started()
                assert self._proc is not None and self._proc.stdin is not None
                self._proc.stdin.write(payload)
                await self._proc.stdin.drain()

            # What is kept of the output: a head and a tail with the middle
            # elided, so a runaway command (`yes`, a huge log) cannot grow
            # this until the timeout. on_output still sees every line.
            head: list[str] = []
            tail: deque[str] = deque()
            head_size = tail_size = dropped = 0

            def keep(text: str) -> None:
                nonlocal head_size, tail_size, dropped
                if on_output is not None:
                    on_output(text)
                if head_size + len(text) <= _CAPTURE_HEAD:
                    head.append(text)
                    head_size += len(text)
                    return
                tail.append(text)
                tail_size += len(text)
                while tail and tail_size > _CAPTURE_TAIL:
                    dropped += len(tail.popleft())

            def captured() -> str:
                parts = ["".join(head)]
                if dropped:
                    parts.append(_ELIDED.format(chars=dropped))
                parts.extend(tail)
                return "".join(parts)

            async def _read_until_marker() -> int:
                while True:
                    line = await self._proc.stdout.readline()
                    if not line:
                        return -1
                    text = line.decode(errors="replace")
                    # The marker can be glued to a final output line that
                    # has no trailing newline (echo -n, cat of a file that
                    # lacks one) — match it anywhere in the line, keeping
                    # whatever precedes it as output.
                    idx = text.find(marker)
                    if idx != -1:
                        before, status = text[:idx], text[idx + len(marker):].strip()
                        if before:
                            keep(before)
                        try:
                            return int(status or "-1")
                        except ValueError:
                            return -1
                    keep(text)

            try:
                exit_code = await asyncio.wait_for(
                    _read_until_marker(),
                    timeout=timeout if timeout is not None else self.default_timeout,
                )
            except asyncio.CancelledError:
                # Stop the command too, not just stop waiting for it.
                await self.close(force=True)
                raise
            except asyncio.TimeoutError:
                # Keep what the command printed so far: for a hung test run
                # that output is the most useful thing the caller can see.
                partial = captured()
                await self.close(force=True)
                return SandboxResult(exit_code=-1, stdout=partial, stderr="", timed_out=True)
            return SandboxResult(exit_code=exit_code, stdout=captured(), stderr="")

    async def close(self, *, force: bool = False) -> None:
        """Close the persistent shell (idempotent). ``force`` kills any
        running command (and its children) immediately instead of waiting
        for the shell to exit cleanly."""
        if self._proc is None:
            cleanup_mount_dirs_now()
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            await asyncio.wait_for(self._proc.wait(), timeout=0 if force else 5)
        except (asyncio.TimeoutError, ProcessLookupError):
            try:
                os.killpg(self._proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await self._proc.wait()
        finally:
            self._proc = None
            # bwrap has torn down its tmpfs/bind mounts by now; drop the
            # empty mountpoint dirs it created in the worktree.
            cleanup_mount_dirs_now()


@dataclass
class SandboxRuntime:
    """A session's resolved sandbox: profile identity, config and shell.

    The four values travel together everywhere (construction, mode switches,
    status, persistence, shutdown), so keeping them in one object removes the
    repeated ``if sandbox is not None`` and keeps the mode/read-only mapping in
    one place."""

    config: SandboxConfig | None = None
    shell: SandboxSession | None = None
    name: str | None = None
    settings: "SandboxSettings | None" = None

    @classmethod
    def make(
        cls, workspace: Path, settings: "SandboxSettings",
        git_info: "GitInfo | None", *, name: str | None = None,
        extra_refs: "Iterable[str | RefSpec] | None" = None,
    ) -> "SandboxRuntime":
        config = SandboxConfig.from_settings(
            workspace, settings, git_info, extra_refs=extra_refs,
        )
        return cls(
            config=config,
            shell=SandboxSession(config, default_timeout=settings.default_timeout),
            name=name, settings=settings,
        )

    @property
    def use_sandbox(self) -> bool:
        return self.config.use_sandbox if self.config is not None else True

    @property
    def allow_git_write(self) -> bool:
        return self.settings.allow_git_write if self.settings is not None else False

    def set_mode(self, mode: str) -> None:
        if self.config is not None:
            self.config.read_only = mode != "write"

    async def close(self) -> None:
        if self.shell is not None:
            await self.shell.close()
            self.shell = None
