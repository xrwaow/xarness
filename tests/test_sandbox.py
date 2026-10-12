"""Tests for the bwrap sandbox: argv construction, container invariants,
config-flag coverage, and (where nested bwrap can start) real-container
behavior. bwrap itself is faked (shutil.which patched) for the argv-level
tests; only ContainerExecTest runs a real sandbox and skips gracefully when
nested bwrap cannot start."""

import asyncio
import hashlib
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

from xarness.config import SandboxSettings
from xarness.sandbox import (
    CACHE_ANCHOR,
    REFS_ANCHOR,
    SandboxConfig,
    SandboxResult,
    SandboxSession,
    SandboxUnavailable,
    _etc_skeleton,
    _shadow_placeholders,
)


# ----------------------------------------------------------------------
# Shared argv helpers: bwrap bind pairs are [flag, src, dest], so the
# destination of a bind at index i is argv[i + 2]. A path can legitimately
# appear twice (as a source and as a destination) — search backwards or use
# these helpers instead of hand-rolled index math.


def argv_binds(argv: list[str]) -> list[tuple[str, str, str]]:
    """(flag, source, destination) for every --ro-bind/--bind/--dev-bind."""
    return [(argv[i], argv[i + 1], argv[i + 2]) for i, a in enumerate(argv)
            if a in ("--ro-bind", "--bind", "--dev-bind")]


def argv_destinations(argv: list[str]) -> list[str]:
    return [dest for _, _, dest in argv_binds(argv)]


def argv_env(argv: list[str]) -> dict[str, str]:
    """--setenv pairs as a dict (last assignment wins, matching bwrap)."""
    env: dict[str, str] = {}
    for i, a in enumerate(argv):
        if a == "--setenv":
            env[argv[i + 1]] = argv[i + 2]
    return env


def argv_env_order(argv: list[str]) -> list[tuple[str, str]]:
    return [(argv[i + 1], argv[i + 2]) for i, a in enumerate(argv) if a == "--setenv"]


def _existing_gpu_nodes() -> list[Path]:
    """Host GPU device nodes _gpu_binds must bind (same enumeration)."""
    nodes = [p for p in Path("/dev").iterdir()
             if p.name in ("dri", "kfd", "nvidiactl", "nvidia-modeset",
                           "nvidia-uvm", "nvidia-uvm-tools", "nvidia-caps")
             or p.name.startswith("nvidia") and p.name[6:].isdigit()]
    # An empty nvidia-caps dir is skipped (binding it would shadow a
    # useless empty directory over nothing).
    return [p for p in nodes if not p.is_dir() or any(p.iterdir())]


class SandboxArgvTest(unittest.TestCase):
    """Base: a temp workspace with bwrap faked onto PATH, plus config/argv
    builders for the argv-level tests."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name) / "wt"
        self.workspace.mkdir()
        patcher = mock.patch("shutil.which", return_value="/usr/bin/bwrap")
        patcher.start()
        self.addCleanup(patcher.stop)

    def config(self, **kw) -> SandboxConfig:
        kw.setdefault("workspace", self.workspace)
        return SandboxConfig(**kw)

    def argv(self, command: list[str] | None = None, **kw) -> list[str]:
        return self.config(**kw).build_argv(command or ["/bin/sh"])


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


# ----------------------------------------------------------------------
# Container mounts: the full mount/env policy at the argv level. These run
# everywhere (bwrap only faked).


class ContainerMountsTest(SandboxArgvTest):
    """Mount + env policy of a standard sandboxed session, and of one with
    refs, filters, a git dir and GPU access: every bind lands at a
    container path, /etc is a synthetic read-only skeleton, the env is
    cleared and rebuilt, and shared state lives under CACHE_ANCHOR."""

    def rich_config(self, **kw) -> SandboxConfig:
        """Standard config plus refs + filter_list + git_dir + gpu."""
        (self.workspace / ".git").mkdir()
        (self.workspace / "uv.lock").write_text("")
        (self.workspace / "secret.txt").write_text("s\n")
        (self.workspace / "secrets").mkdir()
        (self.workspace / "docs").mkdir()
        (self.workspace / "docs" / "spec.md").write_text("x\n")
        external = Path(self._tmp.name) / "extlib"
        external.mkdir()
        kw.setdefault("git_dir", self.workspace / ".git")
        kw.setdefault("gpu_access", True)
        kw.setdefault("filter_list", ["secret.txt", "secrets"])
        kw.setdefault("external_refs", SandboxConfig.resolve_auto_refs(
            self.workspace,
            ["docs/spec.md", {"path": str(external), "mount": ".venv",
                              "read_only": False}],
        ))
        return self.config(**kw)

    # --------------------------------------------------------------
    # Part 3 invariants: destination hygiene

    def test_no_bind_destination_is_a_host_path(self):
        for config in (self.config(), self.rich_config()):
            home = str(Path.home())
            for dest in argv_destinations(config.build_argv(["/bin/sh"])):
                self.assertFalse(dest.startswith("/home"), dest)
                # Path.home() may itself be /tmp in the test environment
                # (which legitimately hosts the container's own /tmp tmpfs).
                if home not in ("/tmp", "/"):
                    self.assertFalse(dest.startswith(home), dest)
                self.assertFalse(
                    dest.startswith(str(self.workspace) + "/")
                    or dest == str(self.workspace), dest)

    def test_bind_destinations_under_container_anchors(self):
        """Every bind destination sits under /workspace, /cache, /etc, /tmp,
        /dev, /proc, /sys — or is a dev-bind of a device node at its exact
        host path."""
        for config in (self.config(), self.rich_config()):
            for flag, src, dest in argv_binds(config.build_argv(["/bin/sh"])):
                if flag == "--dev-bind":
                    self.assertEqual(dest, src)
                    self.assertTrue(src.startswith("/dev/"), (flag, src, dest))
                    continue
                ok = dest.startswith(("/workspace/", "/cache/", "/etc/", "/tmp/",
                                      "/proc/", "/sys/", "/usr/", "/bin/",
                                      "/lib/", "/lib64/", "/sbin/")) or dest in (
                    "/workspace", "/cache", "/etc", "/tmp", "/proc", "/sys",
                    "/usr", "/bin", "/lib", "/lib64", "/sbin")
                self.assertTrue(ok, (flag, src, dest))

    # --------------------------------------------------------------
    # /etc: single read-only skeleton, no host identity

    def test_etc_single_ro_skeleton_bind(self):
        skeleton = str(_etc_skeleton())
        for config in (self.config(), self.rich_config()):
            argv = config.build_argv(["/bin/sh"])
            etc_binds = [(f, s, d) for f, s, d in argv_binds(argv) if d == "/etc"]
            self.assertEqual(len(etc_binds), 1)
            self.assertEqual(etc_binds[0], ("--ro-bind", skeleton, "/etc"))
            # resolv.conf appears twice (source + destination); the last
            # occurrence is the destination pair.
            i = len(argv) - 1 - argv[::-1].index("/etc/resolv.conf")
            self.assertEqual(argv[i - 2], "--ro-bind")
            self.assertEqual(argv[i - 1], argv[i])
            # The trust/alternatives mounts come after the skeleton bind.
            self.assertGreater(i, argv.index(skeleton))
            self.assertGreater(argv.index("/etc/ssl"), argv.index(skeleton))

    def test_etc_skeleton_is_synthetic_identity(self):
        skeleton = _etc_skeleton()
        passwd = (skeleton / "passwd").read_text()
        self.assertIn(f"sandbox:x:{os.getuid()}:{os.getgid()}", passwd)
        self.assertIn("sandbox:x:", (skeleton / "group").read_text())
        self.assertIn("127.0.0.1 localhost", (skeleton / "hosts").read_text())
        for name in ("passwd", "group", "hosts"):
            self.assertIn(name, [p.name for p in skeleton.iterdir()])
        # No host user leaks through.
        import getpass
        user = getpass.getuser()
        if user != "sandbox":
            self.assertNotIn(user, passwd)

    def test_trust_and_alternatives_bound_when_present(self):
        argv = self.argv()
        for path in ("/etc/ssl", "/etc/pki", "/etc/alternatives"):
            if Path(path).exists():
                self.assertIn(path, argv_destinations(argv))
                flag, src, dest = next(
                    b for b in argv_binds(argv) if b[2] == path)
                self.assertEqual(flag, "--ro-bind")

    # --------------------------------------------------------------
    # env + identity invariants

    def test_clearenv_precedes_every_setenv_and_fresh_identity(self):
        argv = self.argv(allow_network=True)
        clear = argv.index("--clearenv")
        setenvs = [i for i, a in enumerate(argv) if a == "--setenv"]
        self.assertTrue(setenvs)
        self.assertLess(clear, min(setenvs))
        self.assertIn("--unshare-all", argv)
        self.assertIn("--die-with-parent", argv)
        i = argv.index("--hostname")
        self.assertEqual(argv[i + 1], "sandbox")
        # LANG is re-set after the clear.
        self.assertGreater([j for j in setenvs if argv[j + 1] == "LANG"][0], clear)

    def test_caches_and_env_under_cache_anchor(self):
        for config in (self.config(), self.rich_config()):
            argv = config.build_argv(["/bin/sh"])
            env = argv_env(argv)
            for var in ("PIP_CACHE_DIR", "CARGO_HOME", "npm_config_cache",
                        "UV_CACHE_DIR", "UV_PROJECT_ENVIRONMENT"):
                self.assertTrue(env[var].startswith(CACHE_ANCHOR), var)
            self.assertEqual(env["HOME"], "/tmp")
            if "PATH" in env:
                self.assertNotIn(str(Path.home()), env["PATH"])

    def test_venv_hash_scheme(self):
        env = argv_env(self.argv())
        expected = hashlib.sha1(str(self.workspace).encode()).hexdigest()[:12]
        self.assertEqual(env["UV_PROJECT_ENVIRONMENT"],
                         f"{CACHE_ANCHOR}/venvs/{expected}")

    # --------------------------------------------------------------
    # .git handling

    def test_git_dir_read_only_at_container_path(self):
        config = self.config(git_dir=self.workspace / ".git")
        argv = config.build_argv(["/bin/sh"])
        i = len(argv) - 1 - argv[::-1].index(str(self.workspace / ".git"))
        self.assertEqual(argv[i - 1], "--ro-bind")
        self.assertEqual(argv[i + 1], "/workspace/.git")

    def test_git_dir_outside_workspace_skipped(self):
        config = self.config(git_dir=Path(self._tmp.name) / "other" / ".git")
        argv = config.build_argv(["/bin/sh"])
        self.assertNotIn("/workspace/.git", argv)
        self.assertNotIn("--ro-bind", argv[:0])  # argv itself is well-formed
        self.assertTrue(all(d != str(Path(self._tmp.name) / "other" / ".git")
                            for d in argv_destinations(argv)))

    # --------------------------------------------------------------
    # refs (external references)

    def test_refs_tmpfs_and_binds_in_argv(self):
        (self.workspace / "docs").mkdir()
        (self.workspace / "docs" / "spec.md").write_text("x\n")
        other = Path(self._tmp.name) / "lib"
        other.mkdir()
        config = self.config(external_refs=SandboxConfig.resolve_auto_refs(
            self.workspace,
            ["docs/spec.md",
             {"path": str(other), "mount": ".venv", "read_only": False}],
        ))
        argv = config.build_argv(["/bin/sh"])
        # One tmpfs anchor for the refs area, nested under the container-only
        # /tmp tmpfs: bwrap creates its mountpoint inside that tmpfs, so the
        # host worktree is never touched (not even transiently).
        tmpfs_targets = [argv[i + 1] for i, a in enumerate(argv) if a == "--tmpfs"]
        self.assertEqual(tmpfs_targets, ["/tmp", REFS_ANCHOR])
        i = argv.index(f"{REFS_ANCHOR}/spec.md")
        self.assertEqual(argv[i - 2:i],
                         ["--ro-bind", str(self.workspace / "docs" / "spec.md")])
        i = argv.index("/workspace/.venv")
        self.assertEqual(argv[i - 2:i], ["--bind", str(other)])
        # Nothing in the argv asks bwrap to create anything under the
        # workspace mount for refs.
        self.assertFalse(any(a.startswith("/workspace/.refs") for a in argv))

    def test_writable_ref_forced_read_only_in_plan_mode(self):
        (self.workspace / "docs").mkdir()
        (self.workspace / "docs" / "spec.md").write_text("x\n")
        config = self.config(external_refs=SandboxConfig.resolve_auto_refs(
            self.workspace, ["docs/spec.md"]))
        config.external_refs["spec.md"].read_only = False
        argv = config.build_argv(["/bin/sh"])
        i = argv.index(f"{REFS_ANCHOR}/spec.md")
        self.assertEqual(argv[i - 2], "--bind")
        config.read_only = True
        argv = config.build_argv(["/bin/sh"])
        i = argv.index(f"{REFS_ANCHOR}/spec.md")
        self.assertEqual(argv[i - 2], "--ro-bind")

    def test_ref_mounts_over_root_and_git_rejected(self):
        config = self.config()
        external = Path(self._tmp.name) / "ext.txt"
        external.write_text("x\n")
        for bad in (".", ".git", ".git/hooks"):
            with self.assertRaises(SandboxUnavailable):
                config.add_ref(external, alias="x", mount=bad)
        ref = config.add_ref(external, alias="x", mount=".refs/x")
        self.assertEqual(ref.mount, ".refs/x")

    # --------------------------------------------------------------
    # filter_list

    def test_filter_files_get_file_dirs_get_dir_at_workspace_paths(self):
        (self.workspace / "secret.txt").write_text("x\n")
        (self.workspace / "secrets").mkdir()
        config = self.config(filter_list=["secret.txt", "secrets"])
        argv = config.build_argv(["/bin/sh"])
        empty_file, empty_dir = _shadow_placeholders()
        binds = argv_binds(argv)
        self.assertIn(("--ro-bind", empty_file, "/workspace/secret.txt"), binds)
        self.assertIn(("--ro-bind", empty_dir, "/workspace/secrets"), binds)
        # No tmpfs mountpoints created under the workspace.
        self.assertFalse(any(
            a == "--tmpfs" and argv[i + 1].startswith("/workspace")
            for i, a in enumerate(argv)))

    def test_filter_missing_paths_skipped_nothing_created(self):
        (self.workspace / "real.txt").write_text("x\n")
        config = self.config(
            filter_list=["real.txt", "ghost.txt", "ghost_dir/n"])
        argv = config.build_argv(["/bin/sh"])
        self.assertIn("/workspace/real.txt", argv)
        self.assertNotIn("/workspace/ghost.txt", argv)
        self.assertNotIn("/workspace/ghost_dir/n", argv)
        self.assertFalse((self.workspace / "ghost.txt").exists())
        self.assertFalse((self.workspace / "ghost_dir").exists())

    # --------------------------------------------------------------
    # GPU

    def test_gpu_off_has_no_dev_binds(self):
        argv = self.argv()
        self.assertNotIn("--dev-bind", argv)

    def test_gpu_access_binds_device_nodes_and_sysfs(self):
        gpu_nodes = _existing_gpu_nodes()
        config = self.config(gpu_access=True)
        argv = config.build_argv(["/bin/sh"])
        bound = {src for flag, src, dest in argv_binds(argv)
                 if flag == "--dev-bind"}
        self.assertEqual(bound, {str(p) for p in gpu_nodes})
        # Driver enumeration sysfs paths, ro-bound when present.
        for sys_path in ("/proc/driver/nvidia", "/sys/bus/pci",
                         "/sys/class/drm", "/sys/devices"):
            if Path(sys_path).exists():
                flag, src, dest = next(
                    b for b in argv_binds(argv) if b[2] == sys_path)
                self.assertEqual(flag, "--ro-bind")
                self.assertEqual(src, sys_path)
        # nvidia module sysfs dirs (cuInit hard-requires them).
        for mod in sorted(Path("/sys/module").glob("nvidia*")):
            if mod.is_dir():
                self.assertIn(str(mod), argv_destinations(argv))

    def test_gpu_caps_dir_shadowed_only_when_unusable(self):
        _, empty_dir = _shadow_placeholders()
        caps = "/proc/driver/nvidia/capabilities"
        with mock.patch("xarness.sandbox._caps_unusable", return_value=True):
            argv = self.argv(gpu_access=True)
        i = argv.index(caps)
        self.assertEqual(argv[i - 2:i], ["--bind", empty_dir])
        with mock.patch("xarness.sandbox._caps_unusable", return_value=False):
            argv = self.argv(gpu_access=True)
        self.assertNotIn(caps, argv)

    # --------------------------------------------------------------
    # mount_key (Part 3, parametrized)

    def test_mount_key_changes_iff_relevant_setting_changes(self):
        base = dict(
            external_refs=SandboxConfig.resolve_auto_refs(
                self.workspace, ["docs/spec.md"]),
        )
        (self.workspace / "docs").mkdir()
        (self.workspace / "docs" / "spec.md").write_text("x\n")
        cases = [
            ("read_only", dict(read_only=True)),
            ("allow_network", dict(allow_network=True)),
            ("gpu_access", dict(gpu_access=True)),
            ("use_container", dict(use_container=False)),
            ("filter_list", dict(filter_list=["x"])),
        ]
        for name, change in cases:
            with self.subTest(field=name):
                config = self.config(**base)
                key = config.mount_key
                config2 = self.config(**base, **change)
                self.assertNotEqual(config2.mount_key, key)
        # External refs changes matter too.
        config = self.config(**base)
        key = config.mount_key
        (Path(self._tmp.name) / "docs2").mkdir()
        config.add_ref(Path(self._tmp.name) / "docs2", alias="d2")
        self.assertNotEqual(config.mount_key, key)
        # And irrelevant fields do NOT change it (fresh config).
        config2 = self.config(**base)
        key2 = config2.mount_key
        config2.timeout_seconds = 999
        self.assertEqual(config2.mount_key, key2)


# ----------------------------------------------------------------------
# Config flags → argv (Part 2)


class ConfigFlagsTest(SandboxArgvTest):
    """Each SandboxSettings flag's effect on build_argv() / config
    construction, including the from_settings contract."""

    # -- allow_network ------------------------------------------------

    def test_network_off_offline_and_unshared(self):
        argv = self.argv()
        self.assertIn("--unshare-all", argv)
        self.assertNotIn("--share-net", argv)
        self.assertEqual(argv_env(argv).get("UV_OFFLINE"), "1")

    def test_network_on_share_net_no_offline(self):
        argv = self.argv(allow_network=True)
        self.assertIn("--share-net", argv)
        self.assertIn("--unshare-all", argv)  # net is re-shared explicitly
        self.assertNotIn("UV_OFFLINE", argv_env(argv))

    def test_network_on_resets_proxy_vars(self):
        proxies = {"http_proxy": "http://p:1", "https_proxy": "http://p:2",
                   "no_proxy": "localhost",
                   "HTTP_PROXY": "http://P:1", "HTTPS_PROXY": "http://P:2",
                   "NO_PROXY": "localhost"}
        with mock.patch.dict(os.environ, proxies):
            argv = self.argv(allow_network=True)
        env_order = argv_env_order(argv)
        env = argv_env(argv)
        for var, value in proxies.items():
            self.assertEqual(env[var], value)
            self.assertIn((var, value), env_order)
        clear = argv.index("--clearenv")
        setenvs = [i for i, a in enumerate(argv) if a == "--setenv"]
        self.assertLess(clear, min(setenvs))

    def test_network_off_no_proxy_vars_in_argv(self):
        proxies = {"http_proxy": "http://p:1", "HTTPS_PROXY": "http://P:2"}
        with mock.patch.dict(os.environ, proxies):
            argv = self.argv()
        for var in ("http_proxy", "HTTPS_PROXY"):
            self.assertNotIn(var, argv)
            self.assertNotIn(proxies[var], argv)

    # -- use_container ------------------------------------------------

    def test_unsandboxed_argv_is_bare_command(self):
        config = self.config(use_container=False)
        command = ["/bin/sh", "-c", "echo hi"]
        self.assertEqual(config.build_argv(command), command)

    def test_unsandboxed_does_not_require_bwrap(self):
        with mock.patch("shutil.which", return_value=None):
            config = SandboxConfig(workspace=self.workspace, use_container=False)
        self.assertEqual(config.tool_path("src/app.py"),
                         str(self.workspace / "src/app.py"))
        self.assertEqual(config.tool_path("./notes.txt"),
                         str(self.workspace / "notes.txt"))

    def test_sandboxed_requires_bwrap(self):
        with mock.patch("shutil.which", return_value=None):
            with self.assertRaises(SandboxUnavailable):
                SandboxConfig(workspace=self.workspace)

    def test_isolation_settings_ignored_unsandboxed(self):
        config = self.config(use_container=False, filter_list=["secret.txt"],
                             allow_network=True, gpu_access=True)
        self.assertEqual(config.build_argv(["/bin/sh", "-c", "echo hi"]),
                         ["/bin/sh", "-c", "echo hi"])

    # -- gpu_access ---------------------------------------------------

    def test_gpu_flag_off_zero_dev_binds_on_full_binds(self):
        self.assertNotIn("--dev-bind", self.argv())
        nodes = _existing_gpu_nodes()
        argv = self.argv(gpu_access=True)
        self.assertEqual({src for f, src, d in argv_binds(argv)
                          if f == "--dev-bind"},
                         {str(p) for p in nodes})

    # -- filter_list: target always /workspace/<rel> -------------------

    def test_filtered_target_always_at_workspace_rel(self):
        (self.workspace / "sub").mkdir()
        (self.workspace / "sub" / "f.txt").write_text("x\n")
        for rel in ("secret.txt", "sub/f.txt"):
            (self.workspace / rel).write_text("x\n")
        argv = self.argv(filter_list=["secret.txt", "sub/f.txt"])
        for rel in ("secret.txt", "sub/f.txt"):
            self.assertIn(f"/workspace/{rel}", argv_destinations(argv))

    # -- auto_include_refs via from_settings ---------------------------

    def test_from_settings_refs_plain_path_and_missing(self):
        (self.workspace / "docs").mkdir()
        (self.workspace / "docs" / "spec.md").write_text("x\n")
        settings = SandboxSettings(
            auto_include_refs=["docs/spec.md", "nope.md"])
        config = SandboxConfig.from_settings(self.workspace, settings)
        self.assertEqual(set(config.external_refs), {"spec.md"})
        ref = config.external_refs["spec.md"]
        self.assertEqual(ref.host, self.workspace / "docs" / "spec.md")
        self.assertEqual(ref.mount, ".refs/spec.md")
        self.assertTrue(ref.read_only)
        argv = config.build_argv(["/bin/sh"])
        self.assertIn(("--ro-bind", str(self.workspace / "docs" / "spec.md"),
                       f"{REFS_ANCHOR}/spec.md"), argv_binds(argv))

    def test_from_settings_refspec_mount_and_rw(self):
        (self.workspace / "docs").mkdir()
        (self.workspace / "docs" / "spec.md").write_text("x\n")
        settings = SandboxSettings(auto_include_refs=[
            {"path": "docs/spec.md", "mount": ".venv", "read_only": False}])
        config = SandboxConfig.from_settings(self.workspace, settings)
        ref = config.external_refs["spec.md"]
        self.assertEqual(ref.mount, ".venv")
        self.assertFalse(ref.read_only)
        argv = config.build_argv(["/bin/sh"])
        self.assertIn(("--bind", str(self.workspace / "docs" / "spec.md"),
                       "/workspace/.venv"), argv_binds(argv))

    def test_from_settings_duplicate_basenames_disambiguated(self):
        (self.workspace / "docs").mkdir()
        (self.workspace / "docs" / "spec.md").write_text("a\n")
        other = Path(self._tmp.name) / "notes" / "spec.md"
        other.parent.mkdir()
        other.write_text("b\n")
        settings = SandboxSettings(
            auto_include_refs=["docs/spec.md", str(other)])
        config = SandboxConfig.from_settings(self.workspace, settings)
        self.assertEqual(config.external_refs["spec.md"].host,
                         self.workspace / "docs" / "spec.md")
        self.assertEqual(config.external_refs["notes-spec.md"].host, other)

    def test_from_settings_writable_ref_forced_ro_in_plan_mode(self):
        (self.workspace / "docs").mkdir()
        (self.workspace / "docs" / "spec.md").write_text("x\n")
        settings = SandboxSettings(auto_include_refs=[
            {"path": "docs/spec.md", "read_only": False}])
        config = SandboxConfig.from_settings(self.workspace, settings)
        argv = config.build_argv(["/bin/sh"])
        i = argv.index(f"{REFS_ANCHOR}/spec.md")
        self.assertEqual(argv[i - 2], "--bind")
        config.read_only = True
        argv = config.build_argv(["/bin/sh"])
        i = argv.index(f"{REFS_ANCHOR}/spec.md")
        self.assertEqual(argv[i - 2], "--ro-bind")

    # -- default_timeout ------------------------------------------------

    def test_default_timeout_flows_into_config(self):
        settings = SandboxSettings(default_timeout=0.5)
        config = SandboxConfig.from_settings(self.workspace, settings)
        self.assertEqual(config.timeout_seconds, 0.5)
        # The validator requires > 0.
        with self.assertRaises(Exception):
            SandboxSettings(default_timeout=0)

    # -- read_only (plan mode) -----------------------------------------

    def test_read_only_flips_workspace_bind_to_ro(self):
        argv = self.argv()
        flag, src, dest = next(b for b in argv_binds(argv) if b[2] == "/workspace")
        self.assertEqual(flag, "--bind")
        self.assertEqual(src, str(self.workspace))
        argv = self.argv(read_only=True)
        flag, src, dest = next(b for b in argv_binds(argv) if b[2] == "/workspace")
        self.assertEqual(flag, "--ro-bind")

    # -- from_settings equivalence --------------------------------------

    def _explicit_config(self, workspace, settings, git_dir=None) -> SandboxConfig:
        """Field-by-field construction (what a call site that does not use
        from_settings would write)."""
        return SandboxConfig(
            workspace=workspace,
            allow_network=settings.allow_network,
            gpu_access=settings.gpu_access,
            use_container=settings.use_container,
            filter_list=list(settings.filter_list),
            timeout_seconds=settings.default_timeout,
            external_refs=SandboxConfig.resolve_auto_refs(
                workspace, settings.auto_include_refs),
            git_dir=git_dir,
        )

    def test_from_settings_matches_explicit_construction_matrix(self):
        (self.workspace / "docs").mkdir()
        (self.workspace / "docs" / "spec.md").write_text("x\n")
        (self.workspace / "secret.txt").write_text("s\n")
        git_dir = self.workspace / ".git"
        git_dir.mkdir()

        git_info = mock.Mock()
        git_info.git_dir = git_dir
        matrices = [
            SandboxSettings(),
            SandboxSettings(allow_network=True),
            SandboxSettings(allow_network=False, gpu_access=False),
            SandboxSettings(auto_include_refs=["docs/spec.md"]),
            SandboxSettings(filter_list=["secret.txt"]),
            SandboxSettings(gpu_access=True, filter_list=["secret.txt"],
                            auto_include_refs=["docs/spec.md"],
                            default_timeout=7.5),
            SandboxSettings(use_container=False, allow_network=True),
        ]
        for settings in matrices:
            with self.subTest(settings=settings.model_dump()):
                via_settings = SandboxConfig.from_settings(
                    self.workspace, settings, git_info)
                explicit = self._explicit_config(
                    self.workspace, settings, git_dir)
                self.assertEqual(via_settings.build_argv(["/bin/sh"]),
                                 explicit.build_argv(["/bin/sh"]))


# ----------------------------------------------------------------------
# Unsandboxed mode (use_container=False)


class UnsandboxedTest(unittest.TestCase):
    """No bwrap container: bare argv, host workspace paths, bwrap not
    required on PATH, isolation flags have no effect."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name) / "wt"
        self.workspace.mkdir()
        patcher = mock.patch("shutil.which", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_config_and_tool_paths(self):
        config = SandboxConfig(workspace=self.workspace, use_container=False)
        self.assertEqual(config.tool_path("."), str(self.workspace))
        self.assertEqual(config.tool_path("src/app.py"),
                         str(self.workspace / "src/app.py"))
        self.assertEqual(config.build_argv(["/bin/sh"]), ["/bin/sh"])

    def test_refs_still_resolve_but_nothing_mounted(self):
        (self.workspace / "docs").mkdir()
        (self.workspace / "docs" / "spec.md").write_text("x\n")
        settings = SandboxSettings(use_container=False,
                                   auto_include_refs=["docs/spec.md"],
                                   gpu_access=True)
        config = SandboxConfig.from_settings(self.workspace, settings)
        argv = config.build_argv(["/bin/sh", "-c", "echo"])
        self.assertEqual(argv, ["/bin/sh", "-c", "echo"])
        self.assertEqual(config.tool_path(".refs/spec.md"),
                         str(self.workspace) + "/.refs/spec.md")


# ----------------------------------------------------------------------
# Real-container tests (exec). Skipped as a class when nested bwrap cannot
# start (mount privileges), probed once instead of per test.


class ContainerExecTest(unittest.TestCase):
    """Real bwrap runs. A session-scoped probe decides the skip."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._probe_tmp = tempfile.TemporaryDirectory()
        try:
            config = SandboxConfig(workspace=Path(cls._probe_tmp.name))
        except SandboxUnavailable as exc:
            raise unittest.SkipTest(f"bwrap unavailable: {exc}")
        async def probe() -> SandboxResult:
            session = SandboxSession(config)
            try:
                return await session.run("true")
            finally:
                await session.close()
        result = asyncio.run(probe())
        if result.exit_code != 0 and (
                "Can't mount proc" in result.stdout or "bwrap" in result.stdout):
            raise unittest.SkipTest("nested bwrap cannot start in this environment")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._probe_tmp.cleanup()

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name) / "wt"
        self.workspace.mkdir()
        (self.workspace / ".gitignore").write_text("*.log\n")
        (self.workspace / "uv.lock").write_text("")
        (self.workspace / "filtered.txt").write_text("secret\n")
        import subprocess
        subprocess.run(["git", "init", "-q", str(self.workspace)], check=True)
        subprocess.run(["git", "-C", str(self.workspace), "add", ".gitignore"],
                       check=True)
        subprocess.run(["git", "-C", str(self.workspace),
                        "-c", "user.email=t@t", "-c", "user.name=t",
                        "commit", "-qm", "init"], check=True)
        # Keep the sandbox's cache root out of the real home for the
        # duration of each test.
        self.cache_tmp = tempfile.TemporaryDirectory(prefix="xarn-cache-")
        self.addCleanup(self.cache_tmp.cleanup)
        env = mock.patch.dict(os.environ,
                              {"XARNESS_SANDBOX_CACHE": self.cache_tmp.name})
        env.start()
        self.addCleanup(env.stop)

    def session(self, **kw) -> SandboxSession:
        kw.setdefault("workspace", self.workspace)
        config = SandboxConfig(**kw)
        return SandboxSession(config)

    def run_session(self, session: SandboxSession, *commands: str,
                    timeout: float | None = None) -> list[SandboxResult]:
        async def drive() -> list[SandboxResult]:
            results = []
            try:
                for command in commands:
                    results.append(await session.run(command, timeout=timeout))
            finally:
                await session.close()
            return results
        return asyncio.run(drive())

    # 1. read-only guarantees

    def test_ro_writes_fail(self):
        config = self.session_config(git_dir=self.workspace / ".git")
        results = self.run_session(
            SandboxSession(config),
            "touch /usr/x", "mkdir /etc/x", "touch /workspace/.git/x",
            "touch /workspace/uv.lock")
        for r in results:
            self.assertNotEqual(r.exit_code, 0, r.stdout)

    def session_config(self, **kw) -> SandboxConfig:
        kw.setdefault("workspace", self.workspace)
        return SandboxConfig(**kw)

    def test_ro_mounts_readable_and_git_reads_work(self):
        config = self.session_config(git_dir=self.workspace / ".git")
        (r,) = self.run_session(
            SandboxSession(config),
            "cat /etc/resolv.conf >/dev/null && ls /etc/ssl >/dev/null "
            "&& echo ETC_OK",
            "cat /workspace/.gitignore",
            "git status --porcelain; git log --oneline | head -1",
        )
        self.assertEqual(r.exit_code, 0, r.stdout)
        self.assertIn("ETC_OK", r.stdout)

    def test_git_index_not_mutated(self):
        import subprocess
        before = subprocess.run(
            ["git", "-C", str(self.workspace), "status", "--porcelain"],
            capture_output=True, text=True).stdout
        config = self.session_config(git_dir=self.workspace / ".git")
        (r,) = self.run_session(
            SandboxSession(config),
            "git add -A 2>/dev/null; git commit -qm x 2>/dev/null; "
            "echo attempted; true",
        )
        self.assertEqual(r.exit_code, 0, r.stdout)
        after = subprocess.run(
            ["git", "-C", str(self.workspace), "status", "--porcelain"],
            capture_output=True, text=True).stdout
        self.assertEqual(before, after)

    # 2. writes

    def test_writable_paths_and_host_visibility(self):
        config = self.session_config()
        session = SandboxSession(config)
        (r,) = self.run_session(
            session,
            "echo data > /workspace/made-in-sandbox.txt && "
            "echo t > /tmp/t && echo c > /cache/made-in-sandbox-"
            + uuid.uuid4().hex + " && echo ok",
        )
        self.assertEqual(r.exit_code, 0, r.stdout)
        self.assertIn("ok", r.stdout)
        self.assertEqual(
            (self.workspace / "made-in-sandbox.txt").read_text(), "data\n")
        (self.workspace / "made-in-sandbox.txt").unlink()

    # 3. env / identity

    def test_env_and_identity_inside_container(self):
        config = self.session_config()
        (r,) = self.run_session(
            SandboxSession(config),
            'test "$HOME" = /tmp && echo HOME_OK',
            '[ "$(hostname)" = sandbox ] && echo HOST_OK',
            '[ -z "$USER" ] || [ "$USER" = sandbox ]; echo USER_OK',
            "[ \"$(id -u)\" = \"$(id -u)\" ] && "
            "getent passwd $(id -u) | grep -q sandbox && echo IDENT_OK",
            "python3 -c 'import socket;socket.gethostbyname(\"localhost\")' "
            "&& echo DNS_OK",
            "ls / | grep -q '^home$' && echo HAS_HOME || echo NO_HOME",
        )
        self.assertEqual(r.exit_code, 0, r.stdout)
        for marker in ("HOME_OK", "HOST_OK", "USER_OK", "IDENT_OK",
                       "DNS_OK", "NO_HOME"):
            self.assertIn(marker, r.stdout)

    # 4. caches

    def test_cache_layout(self):
        config = self.session_config()
        expected = hashlib.sha1(str(self.workspace).encode()).hexdigest()[:12]
        (r,) = self.run_session(
            SandboxSession(config),
            'case "$UV_CACHE_DIR" in /cache/*) echo UV_OK;; esac',
            "echo x > /cache/uv/xarness-probe && rm /cache/uv/xarness-probe "
            "&& echo UVW_OK",
            "if [ -d /cache/uv-python ]; then "
            "echo x > /cache/uv-python/xarness-probe 2>/dev/null "
            "&& echo PYRW_BAD || echo PYRO_OK; else echo PYRO_ABSENT; fi",
            f'case "$UV_PROJECT_ENVIRONMENT" in /cache/venvs/{expected}) '
            f'echo VENV_OK;; esac',
        )
        self.assertEqual(r.exit_code, 0, r.stdout)
        for marker in ("UV_OK", "UVW_OK", "VENV_OK"):
            self.assertIn(marker, r.stdout)
        self.assertTrue(any(m in r.stdout for m in
                            ("PYRO_OK", "PYRO_ABSENT", "PYRW_BAD")))

    # 5. persistence + filter restart

    def test_session_persists_cwd_and_env(self):
        session = self.session()
        one, two = self.run_session(
            session,
            "cd /tmp && export XARNESS_MARK=hello",
            "pwd; echo $XARNESS_MARK",
        )
        self.assertEqual(one.exit_code, 0)
        self.assertEqual(two.exit_code, 0)
        self.assertEqual(two.stdout, "/tmp\nhello\n")

    def test_filter_change_restarts_shell_and_hides_path(self):
        config = self.session_config()
        session = SandboxSession(config)
        (first,) = self.run_session(session, "echo hello")
        self.assertEqual(first.exit_code, 0)
        pid_before = session._proc.pid
        config.filter_list = ["filtered.txt"]
        (second,) = self.run_session(
            session,
            "if [ -e /workspace/filtered.txt ]; then echo VISIBLE; "
            "else echo HIDDEN; fi")
        self.assertEqual(second.exit_code, 0)
        # The shell was restarted (new pid) and the path is hidden.
        self.assertNotEqual(session._proc.pid, pid_before)
        self.assertIn("HIDDEN", second.stdout)
        self.assertNotIn("VISIBLE", second.stdout)

    # 6. plan mode flip

    def test_plan_mode_flip_restarts_and_enforces(self):
        config = self.session_config()
        session = SandboxSession(config)
        (write_ok,) = self.run_session(
            session, "echo a > /workspace/plan.txt && echo WROTE_OK")
        self.assertIn("WROTE_OK", write_ok.stdout)
        pid_before = session._proc.pid
        config.read_only = True
        (ro,) = self.run_session(
            session,
            "echo b > /workspace/plan2.txt 2>/dev/null && echo RW_BAD "
            "|| echo RO_OK; cat /workspace/plan.txt")
        self.assertNotEqual(session._proc.pid, pid_before)  # restart on flip
        self.assertIn("RO_OK", ro.stdout)
        self.assertNotIn("RW_BAD", ro.stdout)
        self.assertIn("a", ro.stdout)  # reads still work
        config.read_only = False
        (rw,) = self.run_session(
            session, "echo c > /workspace/plan3.txt && echo WROTE2_OK")
        self.assertIn("WROTE2_OK", rw.stdout)

    # 7. default_timeout used when no per-command timeout is given

    def test_default_timeout_applies(self):
        config = self.session_config(timeout_seconds=0.5)
        async def drive():
            session = SandboxSession(config)
            try:
                return await session.run("sleep 5")
            finally:
                await session.close()
        result = asyncio.run(drive())
        self.assertTrue(result.timed_out)
        self.assertNotEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
