"""Shared test helpers.

bwrap-real tests can only run where a nested bwrap can start (mount
privileges). When the environment can't (e.g. xarness running nested in a
container without CAP_SYS_ADMIN), real-sandbox tests skip instead of
spamming identical "Can't mount proc" failures.

The rest of this file is test-harness speed: the suite spends most of its
wall time in a handful of library behaviours that only matter for
debuggability or production shutdown, never for what a test asserts.
Each shim is session-scoped, touches only the test process, and restores
what it patched.
"""

import asyncio

import pytest

from xarness import sandbox as sandbox_mod


def _bwrap_failed(result) -> bool:
    out = (getattr(result, "stdout", "") or "") + (getattr(result, "stderr", "") or "")
    return "Can't mount proc" in out or "Operation not permitted" in out and "bwrap" in out


@pytest.fixture(autouse=True)
def isolated_state_dir(tmp_path_factory, monkeypatch):
    """Keep xarness's own state (external snapshot stores, per-session
    snapshot indexes) out of the real ``~/.local/state``: it is keyed by
    workspace path, so tests would otherwise leave files behind there."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path_factory.mktemp("state")))


@pytest.fixture(autouse=True)
def skip_when_bwrap_unusable(monkeypatch):
    orig_run_in_sandbox = sandbox_mod.run_in_sandbox

    async def run_in_sandbox(config, command, input_bytes=None, timeout=120.0):
        result = await orig_run_in_sandbox(config, command, input_bytes, timeout)
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


# ---------------------------------------------------------------------------
# Test-harness speed shims.
#
# `pytest -n auto` is the big lever; these make the serial run cheap too and
# matter for `-n0` single-file runs. They must be in place before the first
# test runs, so all of them are session-autouse.


@pytest.fixture(autouse=True, scope="session")
def _fast_textual_pauses():
    """Textual's Pilot sleeps in 20ms steps while waiting for idle, which
    dominates the TUI tests. Shrinking the step keeps the same idle
    detection (it still polls process CPU) at a fraction of the wall time.

    Don't shrink ``SLEEP_IDLE`` below the CPU-measurement noise floor: the
    idle test then never passes and every ``pause()`` spins to its 1s cap.
    """
    from textual import _wait

    previous = _wait.SLEEP_GRANULARITY
    _wait.SLEEP_GRANULARITY = 0.001
    try:
        yield
    finally:
        _wait.SLEEP_GRANULARITY = previous


@pytest.fixture(autouse=True, scope="session")
def _fast_asyncio_loops():
    """``unittest.IsolatedAsyncioTestCase`` hardcodes ``debug=True`` for the
    per-test event loop. Debug mode makes asyncio capture a source
    traceback for every Task and Handle, which is most of the time the TUI
    tests spend inside the loop (they create thousands of message
    callbacks). The assertions don't depend on debug-only warnings, so run
    the test loops without it.

    Only the runner setup is replaced; ``loop_factory`` is preserved when
    the interpreter has it (3.12+)."""
    from unittest import async_case

    def setup_runner(self):
        self._asyncioRunner = asyncio.Runner(
            debug=False, loop_factory=getattr(self, "loop_factory", None),
        )

    if not hasattr(async_case.IsolatedAsyncioTestCase, "_setupAsyncioRunner"):
        yield  # pragma: no cover - interpretive change upstream
        return
    previous = async_case.IsolatedAsyncioTestCase._setupAsyncioRunner
    async_case.IsolatedAsyncioTestCase._setupAsyncioRunner = setup_runner
    try:
        yield
    finally:
        async_case.IsolatedAsyncioTestCase._setupAsyncioRunner = previous


@pytest.fixture(autouse=True, scope="session")
def _fast_uvicorn_shutdown():
    """Uvicorn's graceful shutdown sleeps a fixed 100ms twice (once in
    ``shutdown``, once polling ``should_exit`` in ``main_loop``), so every
    test that starts a host pays ~0.2s of wall time *after* the assertion
    it cares about. Keep the same shutdown steps — stop listeners, shut
    down connections, close the lifespan — but don't wait for in-flight
    work no test has.

    Skipped when the uvicorn internals this shadows change shape."""
    try:
        import uvicorn.server as uvicorn_server
    except ImportError:  # pragma: no cover - uvicorn is a hard dep in practice
        yield
        return
    server_cls = uvicorn_server.Server
    if not all(hasattr(server_cls, name) for name in ("main_loop", "shutdown")):
        yield
        return

    async def main_loop(self):
        # Only a flag check; 1ms keeps an idle listener from spinning.
        while not self.should_exit:
            await asyncio.sleep(0.001)

    async def shutdown(self, sockets=None):
        for server in self.servers:
            server.close()
        for sock in sockets or []:
            sock.close()
        for connection in list(self.server_state.connections):
            connection.shutdown()
        if self.server_state.tasks and not self.force_exit:
            for task in list(self.server_state.tasks):
                task.cancel()
        for server in self.servers:
            await server.wait_closed()
        if not self.force_exit:
            await self.lifespan.shutdown()

    previous = (server_cls.main_loop, server_cls.shutdown)
    server_cls.main_loop = main_loop
    server_cls.shutdown = shutdown
    try:
        yield
    finally:
        server_cls.main_loop, server_cls.shutdown = previous


@pytest.fixture(autouse=True, scope="session")
def _shared_textual_parse_cache():
    """Every Textual ``App`` builds its own ``Stylesheet`` and re-parses the
    same 13KB of TCSS (plus each widget's DEFAULT_CSS) from scratch. Parsing
    is pure given the source and the CSS variables, so share the parsed
    ``RuleSet`` lists across app instances for the duration of the run.

    Keyed by the variables too: a test that changes theme gets a miss
    instead of another app's styles. Falls back to Textual's own per-
    instance cache if the internals move."""
    try:
        from textual.cache import LRUCache
        from textual.css.parse import parse
        from textual.css.stylesheet import Stylesheet
    except ImportError:  # pragma: no cover - older/newer textual
        yield
        return

    if not hasattr(Stylesheet, "_parse_rules"):
        yield  # pragma: no cover - interpretive change upstream
        return

    cache: LRUCache = LRUCache(2048)
    original = Stylesheet._parse_rules

    def parse_rules(self, css, read_from, is_default_rules=False,
                    tie_breaker=0, scope=""):
        try:
            variables = tuple(sorted(self._variables.items()))
            variable_tokens = self._variable_tokens
        except AttributeError:  # pragma: no cover - interpretive change upstream
            return original(
                self, css, read_from, is_default_rules, tie_breaker, scope,
            )
        key = (
            variables, css, read_from,
            is_default_rules, tie_breaker, scope,
        )
        try:
            return cache[key]
        except KeyError:
            pass
        rules = list(parse(
            scope, css, read_from,
            variable_tokens=variable_tokens,
            is_default_rules=is_default_rules,
            tie_breaker=tie_breaker,
        ))
        cache[key] = rules
        return rules

    Stylesheet._parse_rules = parse_rules
    try:
        yield
    finally:
        Stylesheet._parse_rules = original
