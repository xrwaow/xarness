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
        assert config.ref_path("docs") == "/tmp/refs/docs"

        # Tool paths anchor at the subtree: the agent's workspace IS the
        # --workspace dir, so any non-escaping relative path is valid and
        # maps inside the read-write subtree. Traversal and absolute paths
        # are always rejected.
        assert config.tool_root == "/workspace/f"
        assert config.tool_path("notes.txt") == "/workspace/f/notes.txt"
        assert config.tool_path(".refs/docs") == "/tmp/refs/docs"
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
        assert config.ref_path("docs") == "/tmp/refs/docs"
        assert config.tool_root == "/workspace"
        assert config.tool_path("app.py") == "/workspace/app.py"
        assert config.tool_path(".refs/docs") == "/tmp/refs/docs"
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


class ContainerSettingsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name) / "wt"
        self.workspace.mkdir()
        (self.workspace / ".gitignore").write_text("secret/\n")
        (self.workspace / "secret").mkdir()
        (self.workspace / "secret" / "x.txt").write_text("s\n")
        patcher = mock.patch("shutil.which", return_value="/usr/bin/bwrap")
        patcher.start()
        self.addCleanup(patcher.stop)

    def _git_init(self) -> None:
        import subprocess
        subprocess.run(["git", "init", "-q", str(self.workspace)], check=True)

    def test_gitignored_hidden_by_default(self):
        self._git_init()
        config = SandboxConfig(workspace=self.workspace)
        assert "secret" in config.hidden_paths
        argv = config.build_argv(["/bin/sh"])
        # The ignored path exists in the container only as a shadow mount
        # target: an empty placeholder bind-mounted over it.
        i = argv.index("/workspace/secret")
        assert argv[i - 2] == "--ro-bind"

    def test_gitignore_files_hidden_when_respected(self):
        self._git_init()
        (self.workspace / "sub").mkdir()
        (self.workspace / "sub" / ".gitignore").write_text("*.tmp\n")
        config = SandboxConfig(workspace=self.workspace)
        assert ".gitignore" in config.hidden_paths
        assert "sub/.gitignore" in config.hidden_paths
        argv = config.build_argv(["/bin/sh"])
        i = argv.index("/workspace/.gitignore")
        assert argv[i - 2] == "--ro-bind"

    def test_respect_gitignore_false_keeps_ignored_visible(self):
        self._git_init()
        config = SandboxConfig(workspace=self.workspace, respect_gitignore=False)
        assert config.hidden_paths == ()
        argv = config.build_argv(["/bin/sh"])
        # No shadow bind for the ignored dir; the workspace stays one bind.
        assert "/workspace/secret" not in argv

    def test_resolve_auto_refs_workspace_relative_and_missing(self):
        (self.workspace / "docs").mkdir()
        (self.workspace / "docs" / "spec.md").write_text("x\n")
        refs = SandboxConfig.resolve_auto_refs(self.workspace, ["docs/spec.md", "nope.md"])
        assert set(refs) == {"spec.md"}
        ref = refs["spec.md"]
        assert ref.host == self.workspace / "docs" / "spec.md"
        assert ref.mount == ".refs/spec.md"
        assert ref.read_only is True

    def test_resolve_auto_refs_basename_collision_disambiguated(self):
        (self.workspace / "docs").mkdir()
        (self.workspace / "docs" / "spec.md").write_text("a\n")
        other = Path(self._tmp.name) / "notes" / "spec.md"
        other.parent.mkdir()
        other.write_text("b\n")
        refs = SandboxConfig.resolve_auto_refs(self.workspace, ["docs/spec.md", str(other)])
        assert refs["spec.md"].host == self.workspace / "docs" / "spec.md"
        assert refs["notes-spec.md"].host == other

    def test_resolve_auto_refs_spec_mount_and_rw(self):
        (self.workspace / "docs").mkdir()
        (self.workspace / "docs" / "spec.md").write_text("x\n")
        refs = SandboxConfig.resolve_auto_refs(
            self.workspace,
            [{"path": "docs/spec.md", "mount": ".venv", "read_only": False}],
        )
        ref = refs["spec.md"]
        assert ref.mount == ".venv"
        assert ref.read_only is False

    def test_resolve_auto_refs_expands_home(self):
        home = Path(self._tmp.name) / "home"
        (home / "notes").mkdir(parents=True)
        (home / "notes" / "api.md").write_text("x\n")
        refs = SandboxConfig.resolve_auto_refs(self.workspace, [str(home / "notes" / "api.md")])
        assert refs["api.md"].host == home / "notes" / "api.md"

    def test_refs_tmpfs_and_binds_in_argv(self):
        (self.workspace / "docs").mkdir()
        (self.workspace / "docs" / "spec.md").write_text("x\n")
        other = Path(self._tmp.name) / "lib"
        other.mkdir()
        config = SandboxConfig(
            workspace=self.workspace,
            external_refs=SandboxConfig.resolve_auto_refs(
                self.workspace,
                [
                    "docs/spec.md",
                    {"path": str(other), "mount": ".venv", "read_only": False},
                ],
            ),
        )
        argv = config.build_argv(["/bin/sh"])
        # One tmpfs anchor for the refs area, nested under the container-only
        # /tmp tmpfs: bwrap creates its mountpoint inside that tmpfs, so the
        # host worktree is never touched (not even transiently).
        tmpfs_targets = [argv[i + 1] for i, a in enumerate(argv) if a == "--tmpfs"]
        # /tmp (container-only) first, the refs anchor nested inside it.
        assert tmpfs_targets == ["/tmp", "/tmp/refs"]
        i = argv.index("/tmp/refs/spec.md")
        assert argv[i - 2:i] == ["--ro-bind", str(self.workspace / "docs" / "spec.md")]
        i = argv.index("/workspace/.venv")
        assert argv[i - 2:i] == ["--bind", str(other)]
        # Nothing in the argv asks bwrap to create anything under the
        # workspace mount for refs.
        assert not any(a.startswith("/workspace/.refs") for a in argv)

    def test_writable_ref_forced_read_only_in_plan_mode(self):
        (self.workspace / "docs").mkdir()
        (self.workspace / "docs" / "spec.md").write_text("x\n")
        config = SandboxConfig(
            workspace=self.workspace,
            external_refs=SandboxConfig.resolve_auto_refs(self.workspace, ["docs/spec.md"]),
        )
        config.external_refs["spec.md"].read_only = False
        argv = config.build_argv(["/bin/sh"])
        i = argv.index("/tmp/refs/spec.md")
        assert argv[i - 2] == "--bind"
        config.read_only = True
        argv = config.build_argv(["/bin/sh"])
        i = argv.index("/tmp/refs/spec.md")
        assert argv[i - 2] == "--ro-bind"

    def test_mount_key_tracks_settings_and_refs(self):
        config = SandboxConfig(workspace=self.workspace)
        key = config.mount_key
        assert config.mount_key == key
        config.set_network_access(True)
        assert config.mount_key != key
        key = config.mount_key
        config.set_respect_gitignore(False)
        assert config.mount_key != key

    def test_set_respect_gitignore_recomputes(self):
        self._git_init()
        config = SandboxConfig(workspace=self.workspace)
        assert "secret" in config.hidden_paths
        config.set_respect_gitignore(False)
        assert config.hidden_paths == ()
        config.set_respect_gitignore(True)
        assert "secret" in config.hidden_paths

    def test_session_restarts_on_mount_key_change(self):
        self._git_init()
        config = SandboxConfig(workspace=self.workspace)
        session = SandboxSession(config)

        async def scenario():
            first = await session.run("echo one")
            if first.exit_code != 0 and "bwrap" in first.stdout:
                # Nested bwrap can't start in this environment (no mount
                # privileges) — the restart-on-mount-key logic needs a real
                # container, so skip instead of failing.
                self.skipTest("bwrap cannot start in this environment")
            assert first.exit_code == 0
            pid_one = session._proc.pid
            config.set_network_access(True)
            second = await session.run("echo two")
            assert second.exit_code == 0
            assert session._proc.pid != pid_one
            await session.close()

        asyncio.run(scenario())

    def test_gpu_access_binds_device_nodes(self):
        gpu_nodes = [p for p in Path("/dev").iterdir()
                     if p.name in ("dri", "kfd", "nvidiactl", "nvidia-modeset",
                                   "nvidia-uvm", "nvidia-uvm-tools", "nvidia-caps")
                     or p.name.startswith("nvidia") and p.name[6:].isdigit()]
        # An empty nvidia-caps dir is skipped (binding it would shadow a
        # useless empty directory over nothing).
        gpu_nodes = [p for p in gpu_nodes
                     if not p.is_dir() or any(p.iterdir())]
        if not gpu_nodes:
            self.skipTest("no GPU device nodes on this host")
        config = SandboxConfig(workspace=self.workspace)
        argv = config.build_argv(["/bin/sh"])
        self.assertNotIn("--dev-bind", argv)
        config.set_gpu_access(True)
        argv = config.build_argv(["/bin/sh"])
        assert "--dev-bind" in argv
        # Every existing GPU node is dev-bound at its real path; /dev itself
        # stays the empty container one. Missing nodes are skipped.
        bound = set()
        for i, tok in enumerate(argv):
            if tok == "--dev-bind":
                host = argv[i + 1]
                assert host.startswith("/dev/")
                assert argv[i + 2] == host
                bound.add(host)
        assert bound == {str(p) for p in gpu_nodes}


class GpuProbeTest(unittest.IsolatedAsyncioTestCase):
    """probe_gpu_access: nvidia-smi results mapped to info/warn; silent when
    there are no GPU nodes to probe (nested CI, GPU-less machines)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch("shutil.which", return_value="/usr/bin/bwrap")
        patcher.start()
        self.addCleanup(patcher.stop)

    async def _run(self, result, cuda_result=None, nodes_present=True,
                   caps_missing=False, module_present=True,
                   uvm_present=True) -> tuple[str, str] | None:
        from xarness.sandbox import probe_gpu_access

        config = SandboxConfig(workspace=Path(self._tmp.name))
        async def fake_run(config, command, input_bytes=None):
            if cuda_result is not None and command[0] == "python3":
                return cuda_result
            return result

        with mock.patch("xarness.sandbox.run_in_sandbox", fake_run), \
             mock.patch("xarness.sandbox._gpu_nodes_present",
                        return_value=nodes_present), \
             mock.patch("xarness.sandbox._caps_unusable",
                        return_value=caps_missing), \
             mock.patch("xarness.sandbox._module_sysfs_present",
                        side_effect=lambda m="nvidia": uvm_present
                        if m == "nvidia_uvm" else module_present):
            return await probe_gpu_access(config)

    async def test_no_gpu_nodes_means_no_probe(self):
        """Host without GPU nodes: no probe, no message."""
        assert await self._run(None, nodes_present=False) is None

    async def test_success_is_info(self):
        from xarness.sandbox import SandboxResult

        result = await self._run(
            SandboxResult(0, stdout="GPU 0: Tesla\n", stderr=""))
        assert result == ("info", "GPU access verified: GPU 0: Tesla")

    async def test_missing_caps_nodes_diagnosed(self):
        from xarness.sandbox import SandboxResult

        result = await self._run(
            SandboxResult(17, stdout="",
                          stderr="Failed to initialize NVML: "
                                 "GPU access blocked by the operating system\n"),
            caps_missing=True,
        )
        level, message = result
        assert level == "warn"
        assert "nvidia-caps" in message and "nvidia-cap1" in message
        assert "chmod" in message

    async def test_probe_sandbox_startup_failure(self):
        from xarness.sandbox import SandboxResult

        result = await self._run(SandboxResult(1, stdout="", stderr="bwrap: Can't mount proc on /proc: Operation not permitted\n"))
        level, message = result
        assert level == "warn"
        assert "failed to start" in message and "Can't mount proc" in message

    async def test_cgroup_denial_gets_nested_hint(self):
        from xarness.sandbox import SandboxResult

        result = await self._run(SandboxResult(
            6, stdout="",
            stderr="Failed to initialize NVML: "
                   "GPU access blocked by the operating system\n"))
        level, message = result
        assert level == "warn"
        assert "cgroup" in message and "nested" in message

    async def test_other_failure_reports_detail(self):
        from xarness.sandbox import SandboxResult

        result = await self._run(
            SandboxResult(1, stdout="", stderr="driver mismatch\n"),
            caps_missing=False, module_present=True)
        level, message = result
        assert level == "warn"
        assert "driver mismatch" in message


if __name__ == "__main__":
    unittest.main()


class GpuMountFixTest(unittest.TestCase):
    """With gpu_access on: /sys/module/nvidia is bound (cuInit needs it) and
    an unusable caps dir is shadowed (NVML falls back to direct access)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch("shutil.which", return_value="/usr/bin/bwrap")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_module_sysfs_bound_when_present(self):
        config = SandboxConfig(workspace=Path(self._tmp.name), gpu_access=True)
        argv = config.build_argv(["/bin/sh"])
        assert ("/sys/module/nvidia" in argv) == Path("/sys/module/nvidia").is_dir()

    def test_caps_dir_shadowed_when_unusable(self):
        from xarness.sandbox import _shadow_placeholders

        empty_file, empty_dir = _shadow_placeholders()
        config = SandboxConfig(workspace=Path(self._tmp.name), gpu_access=True)
        with mock.patch("xarness.sandbox._caps_unusable", return_value=True):
            argv = config.build_argv(["/bin/sh"])
        i = argv.index("/proc/driver/nvidia/capabilities")
        assert argv[i - 2:i] == ["--bind", empty_dir]

    def test_caps_dir_not_shadowed_when_usable(self):
        config = SandboxConfig(workspace=Path(self._tmp.name), gpu_access=True)
        with mock.patch("xarness.sandbox._caps_unusable", return_value=False):
            argv = config.build_argv(["/bin/sh"])
        assert "/proc/driver/nvidia/capabilities" not in argv


    async def test_missing_module_sysfs_diagnosed(self):
        from xarness.sandbox import SandboxResult

        result = await self._run(
            SandboxResult(1, stdout="", stderr="CUDA_ERROR_OS-ish failure\n"),
            caps_missing=False, module_present=False,
        )
        level, message = result
        assert level == "warn"
        assert "/sys/module/nvidia" in message and "304" in message


    async def test_nvml_ok_but_cuda_sysfs_missing(self):
        """nvidia-smi green + cuInit failing + no nvidia_uvm sysfs -> the
        precise parent-container message, not a green light."""
        from xarness.sandbox import SandboxResult

        with mock.patch("xarness.sandbox.GPU_PROBE_DEVICES", ("/dev/nvidiactl",)):
            result = await self._run(
                SandboxResult(0, stdout="GPU 0: Tesla\n", stderr=""),
                cuda_result=SandboxResult(4, stdout="cuInit 999\n", stderr=""),
                module_present=True, uvm_present=False,
            )
        level, message = result
        assert level == "warn"
        assert "nvidia_uvm" in message and "/sys/module" in message

    async def test_nvml_ok_cuda_ok_is_info(self):
        from xarness.sandbox import SandboxResult

        with mock.patch("xarness.sandbox.GPU_PROBE_DEVICES", ("/dev/nvidiactl",)):
            result = await self._run(
                SandboxResult(0, stdout="GPU 0: Tesla\n", stderr=""),
                cuda_result=SandboxResult(0, stdout="cuInit 0\n", stderr=""),
                module_present=True, uvm_present=True,
            )
        assert result == ("info", "GPU access verified: GPU 0: Tesla")
