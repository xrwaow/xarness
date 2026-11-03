# xarness

A terminal coding agent for OpenAI-compatible LLM APIs, built with
[Textual](https://github.com/Textualize/textual). Codex-CLI-style interface:
persistent input bar, streaming answers, expandable reasoning ("thoughts"),
live token/context accounting, and a sandboxed tool set the model uses to
read, edit, and run things in your workspace.

## Tools

When a filesystem sandbox is available, the model gets:

| Tool | Mode | What it does |
| --- | --- | --- |
| `read_file` | plan + write | Read a file (large files return a structural outline; read sections with `start_line`/`end_line`) |
| `write_file` | write | Create or overwrite a file |
| `edit_file` | write | Replace exact, unique strings in a file — pass a list of edits to change several places in one call; the result includes a diff of what changed |
| `run_bash` | plan + write | Run a shell command in a persistent sandboxed shell (cwd and env vars survive across calls). In plan mode the workspace is mounted read-only |
| `ask_user` | plan + write | Ask you a clarifying question in the TUI |

There is no `ls`, `glob`, `grep`, or `web_search` tool: use `run_bash`
with `rg`/`find`/`ls` (pipe search output through `head`). Compaction is not
a tool the model can call — it's automatic (`auto_compact`): when the context
window crosses the configured threshold, the conversation is summarized into
a handoff (the task, what's known so far, and what to do next) that continues
as the next user message. The summarization round continues the same
conversation with tools omitted, so the system prompt and history stay
byte-identical and the provider's prompt cache covers that prefix; the
"no tools, output only the handoff" notice is part of the appended user
message rather than a changed system prompt.

Switch between modes with `/mode`:

- **plan** — read-only: the model can inspect files with `read_file` and run
  read-only shell commands, but cannot edit files or write through the
  sandbox mount.
- **write** — the model can read, edit, and run shell commands.

### Sandbox

Filesystem and shell tools run inside [bubblewrap](https://github.com/containers/bubblewrap)
(`bwrap`): with Linux user/mount/PID namespaces the model gets a fresh,
mostly-empty container root, and only what xarness explicitly mounts exists
inside it. Install `bwrap` first (e.g. `sudo dnf install bubblewrap` on
Fedora); without it, filesystem/bash tools are disabled and the model just
chats.

Everything the model can reach:

| Mount | Mode | Purpose |
| --- | --- | --- |
| `/usr` (+ `/bin`, `/lib`, … symlinks) | read-only | toolchain binaries and libraries |
| `/etc/resolv.conf`, `/etc/ssl` | read-only | DNS/TLS when network access is on |
| `/proc`, `/dev`, `/tmp` | fresh/empty | isolated procfs, empty devtmpfs, tmpfs |
| workspace → `/workspace` | read-write, or read-only in plan mode | your real directory, edited in place |
| `.git` directory of your repo (or, without one, the private snapshot store at `/.xarness-snapshot-store`) | read-only | `git status`/`diff`/`log` work in the container; the harness does all git writes on the host |
| `uv.lock` | read-only | `uv` may read it to build the project env, never rewrite it |
| `.venv` | read-write | the project env the agent should use (see below) — your worktree's `.venv` is shadowed by it |
| shared caches (`uv`, `pip`, `cargo`, `npm`) | read-write | offline installs from warm host caches; `~/.cargo`, `~/.rustup`, `~/.local` are bound read-only so PATH tools still resolve |
| external refs → `/tmp/refs/<alias>` | read-only | the only bridge to host paths outside the workspace; writable mounts (e.g. a host-built `.venv`) are forced read-only in plan mode |
| `/tmp/refs` anchor | tmpfs | purely container-side (nested under the container `/tmp` tmpfs) — nothing, not even an empty folder, ever appears in your worktree; tools and shell both use `/tmp/refs/<alias>` |

Everything else — the rest of `/`, `/home`, other users' files — does not
exist as far as the model can tell. Stray writes outside the mounts land in
an ephemeral container layer that dies with the process; they never reach
the host.

**The project venv.** uv's env for the project lives outside the workspace
(`$UV_PROJECT_ENVIRONMENT`, under the shared cache and keyed by the
workspace path) and is bound over `/workspace/.venv`, so the agent finds a
working interpreter and console scripts at the conventional path —
`uv run pytest`, `.venv/bin/python`, `.venv/bin/pytest` all work. It is not
your worktree's `.venv`: that one was built on the host, and its interpreter
symlink and console-script shebangs name host paths that do not exist inside
the container (invoking them there fails with exit 127). Writes through
`/workspace/.venv` land in the cache, never in the worktree, so the agent can
`uv sync` without touching your files (a directory the sandbox had to create
for the mountpoint is removed again on exit). To use a
host-built venv inside the sandbox instead, mount it explicitly
(`RefSpec(path=..., mount=".venv")`) — ref mounts come after this one and
win.

**Plan mode** is the same container with the workspace and every ref bound
read-only, plus the editing tools removed from the tool set — so shell
commands cannot change files either.

**Live adjustments.** Everything mount-affecting (plan/write mode, refs) is
fingerprinted; when a setting changes at runtime (`/mode`),
the persistent shell restarts with the new mounts on its next command.
Network and GPU access are fixed at launch: `xarness --no-network --no-gpu`
runs the sandbox fully offline and without device access (each defaults to
the sandbox profile's setting — `allow_network: true`, `gpu_access: false`).

**Git guard.** When change tracking is active and `allow_git_write` is off
(and only then), a small generated `git` shim is installed first on `PATH`
—inside the container, and on the host `PATH` for unsandboxed sessions. It
is an allow-list: inspection subcommands pass (`status`, `diff`, `log`,
`show`, `blame`, `rev-parse`, `ls-files`, `cat-file`, `merge-base`,
`for-each-ref`, `config --get/--list`, the list forms of
`branch`/`tag`/`stash`/`remote`/`reflog`, and similar read-only ones), and
everything else is blocked — notably anything writing the index, refs, or
worktree: `checkout`, `switch`, `worktree`, `rebase`, `reset`,
`branch`/`tag` mutations, `stash`, `clean`, `merge`, `push`/`pull`, `add`,
`commit`, `update-ref`, `symbolic-ref`. A blocked call fails in place
(statements around it still run; `&&` stops there) instead of vetoing the
whole command string up front. Like the string scanner it replaced, this
is a best-effort accident guardrail, not a security boundary — the
boundary is the read-only `.git` mount / the mount namespace. All
snapshots, diffs, and reverts run host-side through throwaway indexes: the
user's index, HEAD, and refs are never written from inside the sandbox,
and what counts as "ignored" for change tracking is pinned at session
start, so editing ignore files in the container cannot shift what `/diff`
tracks.

Expose external files to the agent with `/refs add <path>` (or the
`auto_include_refs` sandbox setting); the model addresses them as
`/tmp/refs/<alias>` from both the file tools and the shell.
Plain entries are read-only; a `RefSpec` object can choose a different
mount and mark it writable.

## Change tracking and undo

The agent edits your files directly — there is no separate worktree and no
per-change accept gate before edits land. Instead, every session snapshots
your workspace state into a git tree object at startup (via a throwaway
index: your index, HEAD, and refs are never touched), and every turn
records two more: the state *before* the turn and the state *after* its
edits finished.

No setup is required: if the workspace isn't a git repo (or has no commits
yet), xarness keeps a private snapshot store under its own state directory
and drives git against it with an explicit `GIT_DIR` — no `.git` is created
in your workspace, and your files are never modified. No commits are ever
created either: snapshots are plain tree objects (plus an unreferenced
commit wrapper for the 3-way merge below). Opt out with `--no-init-repo`,
which disables change tracking entirely (the agent still edits the
directory directly; there is just no /diff or /undo for file changes).

While the agent works, a one-line summary above the input shows what it
changed (`Edited 2 files +26 -0`); click it to expand a per-file list
(`name  dir/  agent  +N -M`), and click a row to read that file's unified
diff. Each row is tagged with its source: `agent` for edits a turn made,
and `drift` (in the warning color) for changes made outside the session —
manual edits, or edits from a previous harness run while this session was
closed.

| Command | Action |
| --- | --- |
| `/diff` | Show the pending-changes panel; `/diff` again hides it |
| `/diff <path>` | Show one file's unified diff |
| `/accept` | Lock in the changes made so far: `/diff` resets, `/undo` can no longer revert past this point |
| `/reject` | Discard all changes made since the last `/accept` |
| `/undo` | Drop the last turn: reverse its file edits, put your message back in the input |
| `/retry` | Reverse the last turn's file edits and resend your message |

To undo further back, click any of your messages in the scrollback: a small
`↩ undo to here` affordance appears, and clicking it drops that message and
everything after it — each dropped turn's own file edits reversed, and the
message put back in the input. Messages the last compaction summarized away
can be clicked too (the pre-compaction history is restored first).

`/accept` is how you follow change-per-feature: let the agent build one
feature, `/accept` it, move on to the next. Everything before the last
accept is out of `/diff`'s and `/undo`'s reach — use git itself (a repo you
control) for history beyond that. `/reject` restores the workspace to the
last accepted snapshot, discarding everything since — including changes you
made yourself — while keeping the conversation going so the agent sees the
reverted files on its next turn.

If your `--workspace` is a subdirectory of a bigger repo, the session is
scoped to that subdirectory: the agent's sandbox mounts *your* directory as
its workspace root (`/workspace`) — the rest of the repo isn't mounted in
the container at all — and diffs/reverts cover only your directory.

What never enters a snapshot: paths your ignore rules already cover,
binary/media file types, files over 1 MB, and — in a project with a
`Cargo.toml` — cargo's build trees: `target/` and the sandbox's own
`target-sandbox/` (where `CARGO_TARGET_DIR` points, so every build inside
xarness lands there). Tracking those would bury `/diff` in object files the
agent's own `cargo build` just produced.

### Undo and retry

Each turn records the workspace state before it and the state after its
edits finished. `/undo` and `/retry` reverse *exactly that turn's diff* onto
the live workspace with a 3-way merge, removing every file change the turn
made — edits, deletions, and newly created files — while leaving changes
made since (by you, or by a previous harness run) untouched. If a manual
edit overlaps the lines the turn changed, the merge conflicts: the undo
reports it and changes nothing rather than picking a side.

**Only file edits are reversed.** Non-file side effects of `run_bash` —
package installs, background jobs, network calls, anything outside the
workspace — are *not* undone. `/undo` puts the removed message back into the
input box; `/retry` resends it immediately. Session token totals stay
cumulative: an undone turn's spend really happened, so it stays counted.
Both commands work across a compaction (the pre-compaction history is
restored). Without git tracking (`--no-init-repo` or a setup failure) they
still remove the messages and fix the token counts, but tell you the file
edits could not be reverted. With no turn to reverse, `/undo` is a no-op on
the filesystem.

## Sessions

Conversations autosave in `~/.local/share/xarness/sessions/sessions.db`
(SQLite: one state row plus one row per message; image blobs live under
`sessions/<name>/blobs/`) — disable with `--no-session`. Resume with
`/sessions` in the TUI, or attach/load by name at startup:

```sh
xarness --session my-session   # attach to an active session or load a saved one
xarness --sessions             # print active sessions and exit
```

Resuming reconnects to the session's change tracking (the persisted baseline
snapshot). A session saved by an older schema is refused with a "start a new
session" error rather than half-migrated. If the workspace changed outside
the session while it was closed, resuming shows what drifted and asks
whether to accept it as the new baseline or keep tracking against the last
known state — it never silently reverts or accepts. `/delete` removes the
saved session (its row and image blobs) and starts a fresh chat, and never
touches your workspace files.

## Install

Requires Python 3.11+ and `bwrap` on PATH for the filesystem tools.

```sh
pip install -e .
```

## Configure

Set your API key:

```sh
export OPENAI_API_KEY=sk-...
```

On first run xarness autogenerates a starter config at
`~/.config/xarness/config.json` (one OpenAI-shaped profile using the built-in
defaults) — edit it to point at your provider. If an existing config omits
optional values, xarness notes which defaults apply at startup.

Config is JSON at `~/.config/xarness/config.json`. The layout is sectioned:

```json
{
  "models": {
    "openai": {
      "base_url": "https://api.openai.com/v1",
      "api_key_env": "OPENAI_API_KEY",
      "model_id": "gpt-4.1",
      "supports_vision": true
    }
  },
  "sandboxes": {
    "default": {"allow_network": true, "gpu_access": true}
  },
  "presets": {"default": {"model": "openai", "sandbox": "default"}},
  "ui": {"theme": "ayu-darker"},
  "defaults": {"preset": "default"}
}
```

- `models` — named provider/model profiles. Each entry needs:
  - `base_url` — OpenAI-compatible endpoint (must be http/https).
  - `api_key_env` — environment variable holding the API key; never stored in
    the config itself. Set to `null` for providers that need no key (e.g.
    local models).
  - `model_id` — the model to request.
  - `display_name` — optional display name in the UI.
  - `max_context` — context-window size used for compaction (default 128000).
  - `cot_strength` — reasoning effort sent to the provider: `off`, `low`,
    `medium`, or `high` (no interactive picker; set it here).
  - `keep_reasoning` — send assistant reasoning back to the model on later
    rounds (default `true`).
  - `supports_vision` — the model accepts image attachments (default
    `false`). The TUI refuses to attach images when this is false. When
    `true`, paste an image file path into the input, `@`-mention it, or
    copy an image (screenshot, browser image) and press Ctrl+V in the
    input — the clipboard image attaches directly, no file needed (use
    Ctrl+V for text too; the terminal's Ctrl+Shift+V cannot carry images):
    each
    image becomes a `[🖼 name W×H #id]` token in the text (deletable as one
    unit). Images are downsampled to at most 1 megapixel and sent as base64
    on every round.
  - `auto_compact` — compact automatically when the context estimate passes
    `auto_compact_threshold` of `max_context` (default `false`).
  - `auto_compact_threshold` — fraction of the context window that triggers
    auto-compaction, e.g. `0.9` = 90% full; must be in (0, 1]
    (default `0.9`).
  - `provider` — optional OpenRouter provider routing, sent verbatim as the
    request's `provider` field.
- `tool_output_limit` — cap on the text of every tool result sent back to
  the model (default 32768; at least 256). It is a tool-registry setting: a
  `tool_output_limit` inside a sandbox profile is ignored (with a warning)
  and never round-trips into saved session settings.
- `sandboxes` — named sandbox profiles. `default` is the base; other
  profiles inherit from it and list only overrides. Fields include
  `allow_network`, `gpu_access`, `allow_git_write`, `use_container`,
  `enable_tracking`, `auto_include_refs`, `filter_list`, and
  `default_timeout`.
- `presets` — named `{model, sandbox}` pairs; `defaults.preset` chooses one.
- `ui.theme` — startup color theme (`ayu-darker`, `one-light`); `/theme`
  switches it live.

Switch model/sandbox at runtime with `/model` and `/sandbox`; they write the
choice back as the default preset/model.

## Run

```sh
xarness                              # default preset
xarness --model deepseek             # named model profile
xarness --sandbox offline            # named sandbox profile
xarness --preset work                # named preset
xarness --workspace ./some-project   # sandbox root (default: cwd)
xarness --no-network --no-gpu        # fully offline, no device access
xarness --session my-session         # attach/load a saved session
xarness --sessions                   # list active sessions
xarness --server                     # run the host in the foreground
```

## Slash commands

| Command | Action |
| --- | --- |
| `/tools` | List the tools the model currently has |
| `/model` | Choose the model profile |
| `/sandbox` | Choose this session's sandbox profile (idle only) |
| `/filter add <path>` / `/filter remove <path>` / `/filter list` | Add or remove a workspace path excluded from the sandbox mounts |
| `/sessions` | Resume a previous session |
| `/mode` | Switch between plan (read-only) and write mode |
| `/theme` | Choose a color theme |
| `/refs` | Add or remove this workspace's external reference paths. Ref paths are bound read-only inside the container at `/tmp/refs/<alias>` — the same path tool calls and the shell use (nothing is created in your worktree); they persist per workspace and are shared by every session opened there |
| `/new` | Start a new chat |
| `/delete` | Remove the saved session and start a fresh chat, leaving your workspace files untouched |
| `/undo` | Drop the last turn (only its own file edits reversed, message back in the input). Click an earlier message and confirm `↩ undo to here` to drop several turns at once. If the last thing that happened was a compaction, the first `/undo` restores the pre-compaction history instead (turn and files untouched); the next `/undo` removes the turn |
| `/retry` | Drop the last turn (file edits reverted) and resend its message |
| `/diff`, `/accept`, `/reject` | See "Change tracking and undo" above |

## Keys

| Key | Action |
| --- | --- |
| `Enter` | Send message |
| `Shift+Enter` (or `Alt+Enter`) | Newline in the input |
| `Ctrl+T` | Expand/collapse the most recent "Thought for Xs" block (clicking it works too) |
| `Ctrl+C` | Copy the current selection, or quit if nothing is selected |
| `Ctrl+Shift+C` | Copy the current selection (never quits) |
| `Escape` | Interrupt the agent mid-turn (press again to confirm) |

Note: `Shift+Enter` is only distinguishable from `Enter` on terminals with
extended keyboard support; `Alt+Enter` is the portable fallback.

## Behavior notes

- **States while waiting**: `Processing` from a request being sent until the
  first token of any kind, then `Thinking` (pulsing) while reasoning-channel
  tokens arrive (`reasoning_content` / `reasoning` delta fields), then the
  streamed answer. It appears only while an LLM request is in flight — tool
  calls that execute between rounds show their own `Running …` line, not the
  `Processing` strip. Models without a reasoning channel skip the thinking state.
  Compaction shows its own shimmering `Compacting` line while it summarizes.
- **Interrupting**: one `Escape` only arms the interrupt and shows a
  `press esc again to interrupt` prompt; a second `Escape` within a couple of
  seconds actually stops the turn, so a stray keypress can't kill work.
  Whatever the model had already written is kept — saved up to the last
  non-thinking block, so a trailing run of reasoning that never produced an
  answer is dropped — and a turn interrupted before any answer leaves the
  history untouched. A round that fails mid-stream (provider or transport
  error) keeps its partial answer the same way.
- **Thoughts**: after a turn with reasoning, a collapsed `▸ Thought for Xs`
  indicator stays in the scrollback; `Ctrl+T` or a click expands it inline.
  Reasoning text is kept in the conversation history regardless of visibility.
- **Scrolling**: the log follows new output only while you are at the bottom.
  Scroll up mid-answer and it stays where you put it (nothing yanks it back);
  scroll back to the bottom and it resumes following. Expanding a tool call
  whose output is a diff renders it with the theme's diff colors; expanding
  `write_file` shows the written content as a syntax-highlighted code block.
- **Token counts**: taken from the API's `usage` field when provided
  (`stream_options.include_usage` is requested); otherwise a clearly-labeled
  approximate local estimate is shown. The status bar keeps running session
  totals and context headroom against `max_context`. Compaction's own
  summarization round is included in the totals, and the compaction notice
  reports what it freed (`compacted: 12,450 → 1,830 tokens (freed 10,620)`).
  The handoff itself is shown in the transcript as a distinct padded block
  (also on resume), and `/undo` restores the pre-compaction history. Because
  compaction re-sends the live history instead of a flattened transcript, the
  summarization round is mostly a cache hit; only the handoff instruction is
  new input.

## Layout of the code

```
src/xarness/
  cli.py          argument parsing, boots the TUI, change-tracking setup
  config.py       pydantic config models, JSON loading, profile selection
  client.py       async httpx SSE client, reasoning-effort mapping
  compaction.py   context handoff: token estimates + summarizer round
  controller.py   one-round runner: stream events, wire history, usage totals
  host.py         session host: FastAPI over a unix socket, lifecycle, config resolution
  api.py          HTTP/SSE endpoints used by the TUI and other clients
  clientlib.py    async client used by the TUI (host spawn + API calls)
  conversation.py plain message history, wire-format faithful
  prompts.py      the one system-prompt renderer (mode + environment facts)
  tools.py        tool registry: read/write/edit, run_bash, ask_user
  sandbox.py      bubblewrap sandbox: one-shot file tools + persistent shell
  session_store.py  SQLite persistence: state row, message rows, image blobs
  file_search.py  bounded filename search backing @-mention autocomplete
  gitwork.py      tree-snapshot checkpoints, diff computation, read-only git shim
  theme.py        the color palettes
  tui/            Textual app, widgets, screens (picker/resume/confirm), stylesheet
```

The UI consumes an async generator of typed stream events and knows nothing
about HTTP; conversation state is plain data, and the tool registry is built
per session/mode, so adding tools doesn't touch widget code.

## Development

```sh
uv sync                          # create/refresh the project env
uv run pytest                    # full suite
uv run pytest -q tests/test_sandbox.py -k ContainerExec   # real bwrap runs
```

The container tests need `bwrap` and skip themselves when nested sandboxing is
unavailable. Everything else fakes `bwrap` and inspects the argv instead.
When xarness runs against its own repo, the sandbox exposes this env at
`.venv` (see “The project venv” above), so `.venv/bin/pytest` works inside
`run_bash` too.
