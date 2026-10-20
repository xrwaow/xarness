# xarness TODO — code review findings

Review baseline: `279 passed, 28 skipped` (`uv run --offline python -m pytest -q`).
References are `file:line` in the tree as of this review. Work top-down.

---

## P0 — correctness bugs (user-visible)

- [ ] **Plain `/undo` does not put the user message back in the input.**
  Click-to-undo does (it uses `_send_command()` and `_put_back_user_text()`),
  but the slash-command table wires `/undo` through `_post_command()`, which
  discards the command result. `Session._cmd_undo()` already returns
  `{"user_text": ...}`; the TUI never consumes it.
  - `src/xarness/tui/app.py:302-305` (`COMMANDS["undo"]` → `_post_command(Undo())`)
  - `src/xarness/tui/app.py:740-747` (`_post_command` drops the payload)
  - `src/xarness/tui/app.py:1084-1098` (`_run_undo_to` / `_put_back_user_text`,
    the working path)
  - `src/xarness/session.py:642-651` (`_cmd_undo` returns `user_text`)
  - Fix: make `/undo` use `await _send_command(Undo())` then
    `_put_back_user_text(payload)`, or unify `_post_command`/`_send_command`
    behind one helper that optionally delivers payloads.
  - Test: `/undo` with a non-empty user message leaves its text in
    `ChatInput`, matching click-to-undo.

- [ ] **`/retry` (and undo-to-message) lose attached images.**
  The rollback plan only carries `user_text`; retry resends with `images=None`.
  - `src/xarness/controller.py:103-119` (`RollbackPlan.user_text`, no images)
  - `src/xarness/session.py:707` (`_start_turn(plan.user_text, None)`)
  - `src/xarness/session.py:1084-1098` (`/undo_to` restores text only)
  - Fix: carry `images: list[ImageAttachment] | None` on `RollbackPlan`
    (from the user message) and pass it to `_start_turn`; include it in the
    `user_text` payload for undo-to-input.
  - Test: retry an image-bearing user message and assert the new user message
    reuses the same attachments.

- [ ] **`tool_output_limit` is reset on every tool-registry rebuild.**
  `_apply_sandbox_settings()` sets `self.tool_registry.max_output_chars` and
  then immediately calls `_rebuild_tools()`, which creates a fresh
  `ToolRegistry()` with the default 32768. `/mode` resets it too.
  - `src/xarness/session.py:150-151` (initial assignment)
  - `src/xarness/session.py:988-990` (set, then rebuild)
  - `src/xarness/session.py:1311-1317` (`_rebuild_registry` loses the value)
  - Fix: have `_rebuild_registry` apply `sandbox_settings.tool_output_limit`
    (or preserve the current `max_output_chars`) after construction; add a
    unit test around `/mode`, `/sandbox`, `/filter`.
  - Test: create with `tool_output_limit=4096`, issue `SetMode`, assert the
    registry still uses 4096.

- [ ] **Image content parts can map to the wrong attachment.**
  `Message.to_wire()` advances the attachment index once per image-token
  match, not once per matched attachment. A stray/duplicate token shifts all
  later images; the real token may be emitted as literal text.
  - `src/xarness/conversation.py:283-305` (token loop / `index`)
  - `src/xarness/tui/chat_input.py:289-300` (token ordering)
  - `src/xarness/tui/chat_input.py:247-253` (`_sync_attachments` can leave
    duplicate tokens for one deduped attachment)
  - Fix: only consume an attachment when the token actually corresponds to
    one (e.g. match by token text/position, not ordinal), or strip unmatched
    tokens. Ensure duplicate-image behaviour is defined.
  - Test: content with an unknown token before a real one sends the real
    image at the real token position, not before it.

- [ ] **First-run config autogeneration is never invoked.**
  `ensure_config()` exists and is tested in isolation, but no entrypoint calls
  it; a missing config still errors instead of creating the starter config.
  - `src/xarness/config.py:331-345` (`ensure_config`)
  - `src/xarness/cli.py:139-144` (`load_sectioned` before any creation)
  - Fix: call `ensure_config(DEFAULT_CONFIG_PATH)` at startup and surface its
    notices (the unused `startup_notices` path).
  - Test: run startup with `HOME` pointing at an empty temp dir; assert the
    starter config is created and chat can boot.

- [ ] **`/filter add .` hides the whole workspace.**
  `Path(".").parts` is empty, so the `..` check passes; `_filter_binds()`
  then ro-binds an empty placeholder over `/workspace/.`.
  - `src/xarness/session.py:856-865` (validation accepts `"."`)
  - `src/xarness/sandbox.py:613-633` (`_filter_binds`, target `/workspace/.`)
  - `src/xarness/config.py:150-170` (`filter_list` has no validator)
  - Fix: reject `.`/empty/absolute paths and traversal in `_cmd_filter` and
    validate `SandboxSettings.filter_list` in config (reuse the
    `RefSpec._validate_mount` rules).
  - Test: `/filter add .` errors; config `filter_list: ["."]` fails validation.

- [ ] **Multi-turn revert can partially modify the workspace while reporting
      “nothing was changed”.**
  `revert_turns()` applies newest-first and returns on the first conflict.
  Newer turns are already reverted, then `_apply_rollback()` raises and says
  nothing changed; history is not truncated.
  - `src/xarness/gitwork.py:607-620` (`revert_turns`)
  - `src/xarness/session.py:674-705` (`_apply_rollback`)
  - Fix: preflight all merge-trees and apply only if every turn is clean, or
    compute one combined before/after tree and revert once.
  - Test: turn A clean, turn B conflicting; assert every file is unchanged
    and the conversation is unchanged.

---

## P1 — robustness / lifecycle

- [ ] **`/sessions` can raise `UnboundLocalError` when the fetch fails.**
  - `src/xarness/tui/app.py:1159-1168` (`entries` assigned inside
    `_reporting`; used afterwards)
  - Fix: `entries = []` before the block or return on error.

- [ ] **Failed/cancelled compaction strands queued steering messages.**
  `_run_compact()` drains the queue only on success; on `RuntimeError` it
  returns early, and on cancellation it re-raises. `/flush` cancels only the
  turn task, so “send now” does nothing during compaction.
  - `src/xarness/session.py:712-743` (`_run_compact`, success-only drain)
  - `src/xarness/session.py:633-640` (`_cmd_flush_queued` ignores
    `_compact_task`)
  - `src/xarness/session.py:573-580` (`_cmd_cancel` cancels compaction)
  - Fix: drain/queue after every compaction exit path (success, failure,
    cancel), and make `/flush` cancel the compact task too.

- [ ] **Unexpected turn errors are silent.**
  `_run_turn()` catches only `CancelledError`; other exceptions escape the
  task and the client sees an idle status with no error line.
  `_run_compact()` catches only `RuntimeError`.
  - `src/xarness/session.py:1009-1114` (`_run_turn`, only `except
    asyncio.CancelledError` at 1086)
  - `src/xarness/session.py:722-733` (`_run_compact`)
  - Fix: catch `Exception`, emit an error `Notice`/`StreamError`, persist, and
    keep the partial round.

- [ ] **Plan/write mode is not persisted across resume.**
  - `src/xarness/session_store.py:80-103` (`SessionState` has no `mode`)
  - `src/xarness/session.py:206-244` (`Session.load` defaults to write)
  - `src/xarness/session.py:1293-1301` (`ensure_system_message` rewrites the
    prompt to the default mode)
  - Fix: persist `mode` and pass it into the loaded session, restoring the
    system prompt accordingly.

- [ ] **`supports_vision` is never enforced.**
  It is exposed in `/config` and documented as “the TUI refuses to attach
  images when false,” but no code checks it.
  - `src/xarness/config.py:77` (field)
  - `src/xarness/host.py:464` (exposed)
  - `src/xarness/tui/chat_input.py` / `src/xarness/tui/app.py` (no consumer)
  - Fix: gate paste/@-mention image attachment on the active profile's
    `supports_vision`; surface a clear notice when refused.

- [ ] **Stopping a session with queued messages can start a new turn after
      `stop()` returns.**
  `_run_turn()`'s `finally` starts the queued turn; `stop()` only awaits the
  old task. `_closed` is set but never read.
  - `src/xarness/session.py:1084-1113` (finally + `_start_turn`)
  - `src/xarness/session.py:200, 265-275` (`_closed` unused)
  - Fix: check `_closed` before starting a queued turn, and have `stop()`
    cancel/drain any newly created `_turn_task`.

- [ ] **`_start_unix_server()` can hang forever if uvicorn never starts.**
  - `src/xarness/host.py:603-620` (`while not server.started: sleep`)
  - Fix: poll `server_task.done()`/`server.should_exit`, add a timeout, and
    surface the task exception.

- [ ] **Persistence has a data race with live conversation mutation.**
  `_persist_state()` is evaluated on the event loop but returns a
  `SessionState` holding a live `Conversation`; `save_state()` then iterates
  it in a worker thread while the turn task can append messages.
  - `src/xarness/session.py:1340-1363`
  - Fix: take a detached snapshot (model-copy the messages) before
    `asyncio.to_thread`, or serialize in one step on an immutable value.

- [ ] **`/model` and `/sandbox` can crash after a swallowed config error.**
  `_reporting` catches the `ClientError`, then the handler dereferences
  `self._config_cache.models`/`.sandboxes`; the cache may still be `None`.
  - `src/xarness/tui/app.py:1194-1201` (`_pick_model`)
  - `src/xarness/tui/app.py:1234-1242` (`_pick_sandbox`)
  - Fix: return on fetch failure or use an empty `ConfigInfo()` fallback.

- [ ] **`Notice(level="warn")` renders as a plain notice.**
  - `src/xarness/tui/app.py:660-665`
  - Fix: map `warn` to `_line(..., kind="warn")` (and optionally `error` to
    `_error` as today).

- [ ] **Tracking instructions are shown even when tracking is disabled.**
  - `src/xarness/prompts.py:34-50` (`TRACKING_CLAUSE` added whenever a
    sandbox exists)
  - Fix: pass tracking state into `system_prompt_for()` and omit the clause
    when there is no `GitInfo`.

- [ ] **`SandboxSettings.merged_over()` shallow-copies mutable lists.**
  Inherited profiles share `auto_include_refs`/`filter_list` objects with the
  cached base; current call sites copy before mutating, but any future
  in-place edit will corrupt the cached config.
  - `src/xarness/config.py:176-181`
  - Fix: `model_copy(deep=True)` or copy the list fields explicitly.

- [ ] **Session names / paths are unvalidated (hardening).**
  `session_path(name)` is built by f-string; the API/CLI accept arbitrary
  names, and session/workspace directories are created with default mode.
  - `src/xarness/session_store.py:61-70` (`session_path`, `meta_path`)
  - `src/xarness/session_store.py:148-150` (`SESSIONS_DIR.mkdir` default mode)
  - Fix: restrict names to a safe charset (no separators/`..`), create the
    data dirs `0700`, files `0600`.

---

## P2 — dataflow / performance

- [ ] **`_turn_events` is unbounded (despite the “bounded ring buffer” doc).**
  Every event during a turn is appended, including one per streamed shell
  line; a mid-turn `snapshot()` serializes the backlog.
  - `src/xarness/session.py:17-25` (doc claim)
  - `src/xarness/session.py:324-347` (`_emit` append, no cap)
  - Fix: cap the buffer and coalesce `ToolOutputDelta` (keep tail/total
    count); never let a shell command grow it without limit.

- [ ] **Live tool output is unbounded end-to-end.**
  - `src/xarness/sandbox.py:826-898` (`SandboxSession.run` output list +
    `on_output`)
  - `src/xarness/session.py:1115-1139` (`_execute_call` sink)
  - `src/xarness/tui/flow.py:186-209` (`ToolItem.output += ...`)
  - `src/xarness/tools.py:81-85` (`_truncate_output` only at the end)
  - Fix: cap/stream with a rolling buffer; emit a truncation marker once.

- [ ] **`read_file` pulls the entire file even for previews/slices.**
  - `src/xarness/tools.py:246-258` (`sed -n start,10**9`; full file to
    stdout)
  - Fix: after counting lines, read only the needed range with
    `sed -n '${start},${end}p'`, or `head -n 200` for the non-Python preview.

- [ ] **`edit_file` echoes the whole updated file to stdout.**
  The shell `tee` writes the file contents to stdout; only the diff is
  returned in `ToolResult`, but the payload still crosses bwrap/harness.
  Result should contain only the diff (plus the short status line).
  - `src/xarness/tools.py:503-532`, especially `:514`
  - `src/xarness/tools.py:359-361` (the `write_file` pattern to copy:
    `tee -- "$1" >/dev/null`)
  - Fix: write silently (`["sh","-c",'tee -- "$1" >/dev/null', ...]`), keep
    `ToolResult.output` as `applied N edits ...\n<diff>` only; add a test
    asserting `write_result.stdout` is empty.

- [ ] **Persistence rewrites the full history after every tool result.**
  - `src/xarness/session.py:1115-1139` (`_execute_call` → `_persist`)
  - `src/xarness/session.py:1319-1363` (full `Message.model_dump`, including
    base64 images)
  - Fix: persist at round boundaries/turn end, or use an append-only/incremental
    format; at minimum avoid repeated full-image serialization.

- [ ] **Image data URLs are rebuilt on every request.**
  - `src/xarness/images.py:68-69` (`data_url`)
  - `src/xarness/conversation.py:302-305`
  - Fix: cache the encoded `data:...` string on the attachment.

- [ ] **Diff-bar refresh is not actually debounced and snapshots the whole
      workspace per tool.**
  - `src/xarness/tui/app.py:880-901` (`set_timer` each event; no coalescing)
  - `src/xarness/gitwork.py:440-446` / `728-736` (each call snapshots)
  - Fix: keep one timer handle and cancel/re-arm it; serialize refreshes
    (drop overlapping runs); derive stats from turn checkpoints when possible.

- [ ] **Presenter previous-entry lookup is O(n) per entry.**
  - `src/xarness/tui/flow.py:439-448` (`flat.index(entry)`)
  - `src/xarness/tui/flow.py:508-517` (`chunk_cache` keyed by width/epoch,
    never pruned)
  - Fix: maintain an entry→index map or pass the previous item; bound/evict
    `chunk_cache`.

- [ ] **`@`-mention search reruns `git ls-files` on every query.**
  - `src/xarness/file_search.py:27-58`
  - Fix: cache candidates per workspace with a short TTL/invalidation hook.

- [ ] **Sandbox argv is rebuilt for every one-shot file-tool call.**
  - `src/xarness/sandbox.py:660-730` (`build_argv`)
  - `src/xarness/tools.py:246-258`, `:503-532` (one/two bwrap spawns per
    read/edit)
  - Fix: cache the immutable argv prefix per mount key, or run file edits in
    the persistent shell.

- [ ] **`_truncate_output` can exceed the configured limit.**
  - `src/xarness/tools.py:81-85` (slice to `limit - len(note)` then append
    note)
  - Fix: clamp the final string to `limit` (truncate the note when needed).

---

## P3 — code quality / docs

- [ ] **Update the README to the current implementation.**
  It still documents the old config schema (`profiles`, `default_profile`,
  `container`, flat `provider`, `default_theme`), non-existent CLI forms
  (`xarness chat --session`, `xarness sessions list/delete`), tools
  `ls`/`glob`/`grep` and `web_search` (not registered), `--no-fs-tools`,
  reasoning-effort picking, and `supports_vision` enforcement.
  `/sandbox` and `/filter` are implemented but missing from the slash-command
  table. `docs/future.md` references a `subagents` config section that does
  not exist.
  - `README.md` (Configure / Run / Slash commands / Tools / `/model`
    sections)
  - `docs/future.md`
- [ ] **Remove dead paths:** `Session._closed` (`session.py:200,267`),
  `startup_notices` (never populated), `ensure_config` (unused until P0),
  `save_preferences(auto_compact=, profile=)` (never called from the TUI),
  `SandboxConfig.ref_path()` (unused), `_tool_header_detail` `web_search`
  branch.
- [ ] **Fix prompt typos:** “infered” and “tarteged” in `prompts.py`.
- [ ] **Make preference config writes atomic** (reuse the session-store
  temp-file + `os.replace` pattern): `src/xarness/config.py:388`.
- [ ] **Split the god objects:** `Session` (~1440 lines) and `AgentApp`
  (~1895 lines) mix transport/state/git/UI. Start by unifying
  `_post_command`/`_send_command` (the `/undo` bug is a direct symptom).
- [ ] **Minor correctness:** `_format_edit_diff` uses `splitlines()` and can
  report “no textual change” for newline-only edits
  (`src/xarness/tools.py:435-458`); `EditTool` reads+writes in two one-shot
  bwrap spawns and is racy against concurrent shell edits
  (`tools.py:503-514`).

---

## Suggested commit order

1. `/undo` payload, retry/undo images, `tool_output_limit` preservation.
2. Config autogeneration, `/sessions` crash, filter `"."` validation.
3. Atomic multi-turn revert.
4. Image token mapping, `supports_vision` gate.
5. Bound live tool output / turn-event buffer; fix persistence race.
6. Mode persistence, compaction queue paths, generic turn-error reporting.
7. Dataflow/perf batch: read/edit I/O, diff refresh, presenter indexing.
8. Docs, dead code, typos, config atomicity.

## Verification

- Run `uv run --offline python -m pytest -q` after each batch.
- Add focused regression tests for every P0 item:
  - plain `/undo` restores input text (parity with click-to-undo);
  - `/retry` keeps images;
  - `tool_output_limit` survives `/mode`, `/sandbox`, `/filter`;
  - image token/attachment mismatch;
  - `filter_list` rejects `"."` and `..`;
  - multi-turn revert conflict leaves every file and history intact;
  - `edit_file` returns only the diff and writes silently.
