"""Shared test helpers.

bwrap-real tests can only run where a nested bwrap can start (mount
privileges). When the environment can't (e.g. xarness running nested in a
container without CAP_SYS_ADMIN), real-sandbox tests skip instead of
spamming identical "Can't mount proc" failures.
"""

import pytest

from xarness import sandbox as sandbox_mod


def _bwrap_failed(result) -> bool:
    out = (getattr(result, "stdout", "") or "") + (getattr(result, "stderr", "") or "")
    return "Can't mount proc" in out or "Operation not permitted" in out and "bwrap" in out


@pytest.fixture(autouse=True)
def skip_when_bwrap_unusable(monkeypatch):
    orig_run_in_sandbox = sandbox_mod.run_in_sandbox

    async def run_in_sandbox(config, command, input_bytes=None):
        result = await orig_run_in_sandbox(config, command, input_bytes)
        if _bwrap_failed(result):
            pytest.skip("nested bwrap cannot start in this environment")
        return result

    monkeypatch.setattr(sandbox_mod, "run_in_sandbox", run_in_sandbox)

    # tools.py imports run_in_sandbox by name — patch its reference too.
    try:
        from xarness import tools
        if getattr(tools, "run_in_sandbox", None) is orig_run_in_sandbox:
            monkeypatch.setattr(tools, "run_in_sandbox", run_in_sandbox)
    except ImportError:
        pass

    orig_run = sandbox_mod.SandboxSession.run

    async def session_run(self, command, on_output=None, timeout=None):
        result = await orig_run(self, command, on_output=on_output, timeout=timeout)
        if _bwrap_failed(result):
            pytest.skip("nested bwrap cannot start in this environment")
        return result

    monkeypatch.setattr(sandbox_mod.SandboxSession, "run", session_run)
