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
| `ls` | plan | List a directory's immediate entries (ignore rules applied) |
| `glob` | plan | Find files by glob pattern |
| `grep` | plan | Search file contents by regex |
| `write_file` | write | Create or overwrite a file |
| `edit_file` | write | Replace exact, unique strings in a file — pass a list of edits to change several places in one call; the result includes a diff of what changed |
| `run_bash` | write | Run a shell command in a persistent sandboxed shell (cwd, env vars, and background jobs survive across calls) |
| `ask_user` | plan + write | Ask you a clarifying question in the TUI |

`web_search` is disabled for now (the implementation isn't good enough to
ship). Compaction is not a tool the model can call — it's automatic
(`auto_compact`): when the context window crosses the configured threshold,
the conversation is summarized into a handoff (the task, what's known so
far, and what to do next) that continues as the next user message.

Switch between modes with `/mode`:

- **plan** — read-only: the model can inspect files (`read_file`, `ls`,
  `glob`, `grep`), but cannot edit or run state-changing commands.
- **write** — the model can read, edit, and run shell commands.

### Sandbox

Filesystem and shell tools run inside [bubblewrap](https://github.com/containers/bubblewrap)
(`bwrap`): the workspace is bound read-only or read-write depending on mode,
network access is off, and nothing the model can reach gets a raw socket into
the sandbox itself. Install `bwrap` first (e.g. `sudo dnf install bubblewrap`
on Fedora); without it, filesystem/bash tools are disabled and the model just
chats. Disable them explicitly with `--no-fs-tools`.

Expose external files read-only to the agent with `--ref ALIAS=PATH`
(repeatable); they appear under `.refs/ALIAS` in the workspace and are
read-only by convention.

## Change tracking and undo

The agent edits your files directly — there is no separate worktree and no
accept/reject step. Instead, every session snapshots your workspace state
into a git tree object at startup (via a throwaway index: your index, HEAD,
and refs are never touched), and every turn records two more: the state
*before* the turn and the state *after* its edits finished.

No setup is required: if the workspace isn't a git repo (or has no commits
yet), xarness runs `git init` itself so tracking works. Your files are never
modified by this, and no commits are ever created — snapshots are plain tree
objects (and an unreferenced commit wrapper for the 3-way merge below). Opt
out with `--no-init-repo` (the agent still edits the directory directly;
there is just no /diff or /undo for file changes).

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
scoped to that subdirectory: the agent's tools (ls/glob/grep, read/edit, and
the shell's start directory) are rooted at *your* directory — not the repo
root — tool writes land only inside it, and diffs/reverts cover only it.
The rest of the repo stays mounted read-only for context (and reachable via
`run_bash`), but it is not the agent's workspace.

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
input box; `/retry` resends it immediately. Both also roll back the turn's
token usage from the session totals, and work across a compaction (the
pre-compaction history is restored). Without git tracking (no repo,
`--no-init-repo`, or setup failure) they still remove the messages and fix
the token counts, but tell you the file edits could not be reverted. With no
turn to reverse, `/undo` is a no-op on the filesystem.

## Sessions

Conversations autosave under `~/.local/share/xarness/sessions/` (disable with
`--no-session`, or name one with `--session NAME`). Resume in the TUI with
`/sessions`, or from the command line:

```sh
xarness chat --session my-session   # resume if it exists, else start it
xarness sessions list
xarness sessions delete my-session
```

Resuming reconnects to the session's change tracking (the persisted baseline
snapshot); sessions created by older builds with worktree isolation resume
with fresh tracking instead. If the workspace changed outside the session
while it was closed, resuming shows what drifted and asks whether to accept
it as the new baseline or keep tracking against the last known state — it
never silently reverts or accepts. `/delete` removes the saved session file
and starts a fresh chat, and never touches your workspace files.

## Install

Requires Python 3.11+ and `bwrap` on PATH for the filesystem tools.

```sh
pip install -e .
```

## Configure

Copy the sample config and set your API key:

```sh
mkdir -p ~/.config/xarness
cp config.example.json ~/.config/xarness/config.json
export OPENAI_API_KEY=sk-...
```

Config is JSON at `~/.config/xarness/config.json` by default; override with
`--config`. Fields:

- `default_profile` — which profile to use when `--profile` is not passed
  (defaults to the only profile if exactly one is defined).
- `default_theme` — startup color theme; optional, defaults to `ayu-darker`.
  Options: `"ayu-darker"`, `"one-light"`; `/theme` switches it
  live.
- `container` — sandbox settings for the tools the model runs (all optional):

  ```json
  "container": {
    "network_access": false,
    "respect_gitignore": true,
    "gpu_access": false,
    "auto_include_refs": [
      "docs/spec.md",
      {"path": "~/prebuilt/venv", "mount": ".venv", "read_only": false}
    ]
  }
  ```

  - `network_access` — let tool commands inside the sandbox open network
    connections (default `false`). With it off, installs still work offline
    from the shared caches (`uv`, `pip`, `cargo`, `npm`).
  - `respect_gitignore` — paths matched by `.gitignore` / `.git/info/exclude`
    are hidden from every tool (they don't exist inside the sandbox), so the
    model never reads your build outputs or `.venv`. Set to `false` to expose
    ignored files (default `true`).
  - `gpu_access` — expose the host's GPU device nodes (`/dev/dri`, `/dev/kfd`,
    `/dev/nvidia*`, `/dev/nvidia-caps`) and driver sysfs/proc paths so
    CUDA/ROCm code can run in the container (default `false`). Driver
    userland comes from the read-only `/usr` bind; toggle it live with
    `/container`. When enabled, a one-shot probe (`nvidia-smi -L` inside the
    container) runs at startup and on toggle, and reports precisely why GPU
    access fails if it does: a startup notice warning is your friend here.
    Known nested-container pitfall: for **non-root** users NVML requires the
    NVIDIA capability device nodes (`/dev/nvidia-caps/nvidia-cap1`,
    `nvidia-cap2`); if the parent container passes `/dev/nvidia-caps` as an
    empty directory (or not at all), `nvidia-smi` fails with "GPU access
    blocked by the operating system" even though the plain `/dev/nvidia*`
    nodes open fine. Pass the cap nodes through (or run the parent as root,
    which bypasses the caps path) — no setting inside xarness can fix it.
  - `auto_include_refs` — host paths bound into the container at runtime, so
    the model can consult (or reuse) material outside the workspace. Two
    entry forms:
    - a plain path — bound **read-only** under `.refs/<alias>` (`.refs/` is a
      runtime-only tmpfs in the container, never a folder in your worktree);
    - an object `{"path", "mount", "read_only"}` — mount the host path at any
      workspace-relative `mount`, optionally writable (e.g. drop a host-built
      `.venv` into the project; writable mounts are forced read-only in plan
      mode).

    Paths may be absolute, `~`-expanded, or workspace-relative. The alias is
    the path's basename; two same-named files from different paths are
    disambiguated by prepending parent segments (`docs/spec.md` and
    `~/notes/spec.md` become `.refs/spec.md` and `.refs/notes-spec.md`).
    Missing paths are skipped.

  Everything here is also adjustable per session with `/container` in the
  TUI (settings popup); "save as defaults" there writes the current state
  back into the config. The config values are the startup defaults.
- `profiles` — list of profiles; switch with `--profile` (or `/model` in the
  TUI, which also sets reasoning effort). Each profile needs a unique `name`
  plus:
  - `base_url` — OpenAI-compatible endpoint (must be http/https).
  - `api_key_env` — environment variable holding the API key; never stored in
    the config itself. Set to `null` for providers that need no key (e.g.
    local models).
  - `model_id` — the model to request.
  - `shown_name` — optional display name in the UI.
  - `max_context` — context-window size used for compaction (default 128000).
  - `cot_strength` — reasoning effort: `off`, `low`, `medium`, or `high`.
  - `keep_reasoning` — send assistant reasoning back to the model on later
    rounds (default `true`).
  - `auto_compact` — compact automatically after each turn once the context
    estimate passes `auto_compact_threshold` of `max_context` (default
    `false`; `/auto_compact` toggles it at runtime).
  - `auto_compact_threshold` — fraction of the context window that triggers
    auto-compaction, e.g. `0.9` = 90% full; must be in (0, 1]
    (default `0.9`).
  - `provider` — optional OpenRouter provider routing, sent verbatim as the
    request's `provider` field, e.g.
    `{"order": ["openai", "together"], "allow_fallbacks": false}`.

A flat single-profile form (a `provider` object instead of `profiles`) also
works:

```json
{
  "provider": {
    "base_url": "https://api.openai.com/v1",
    "api_key_env": "OPENAI_API_KEY",
    "model_id": "gpt-4.1",
    "shown_name": "GPT-4.1",
    "max_context": 128000,
    "cot_strength": "medium",
    "keep_reasoning": true
  }
}
```

## Run

```sh
xarness                    # default profile
xarness --profile deepseek
xarness --workspace ./some-project    # sandbox root (default: cwd)
```

## Slash commands

| Command | Action |
| --- | --- |
| `/tools` | List the tools the model currently has |
| `/model` | Choose the model and reasoning effort |
| `/sessions` | Resume a previous session |
| `/mode` | Switch between plan (read-only) and write mode |
| `/container` | Container settings popup: network access, `.gitignore` shadowing, external references; "save as defaults" writes them to the config |
| `/theme` | Choose a color theme |
| `/new` | Start a new chat |
| `/delete` | Remove the saved session file and start a fresh chat, leaving your workspace files untouched |
| `/auto_compact` | Toggle automatic compaction when the context window passes the profile's `auto_compact_threshold` (checked after each turn) |
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

- **States while waiting**: `Processing` from request sent until the first
  token of any kind, then `Thinking` (pulsing) while reasoning-channel tokens
  arrive (`reasoning_content` / `reasoning` delta fields), then the streamed
  answer. Models without a reasoning channel skip the thinking state.
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
  Scroll up mid-answer and it stays where you put it (nothing yanks it back
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
  (also on resume), and `/undo` restores the pre-compaction history.

## Layout of the code

```
src/xarness/
  cli.py          argument parsing, boots the TUI, change-tracking setup
  config.py       pydantic config models, YAML loading, profile selection
  client.py       async httpx SSE client, reasoning-effort mapping
  events.py       typed stream events (the client/UI contract)
  controller.py   conversation state + event enrichment (no UI imports)
  conversation.py plain message history, wire-format faithful
  prompts.py      system prompts (general + plan/write mode clauses)
  tools.py        tool registry: read/write/edit, run_bash, ls/glob/grep, ask_user
  sandbox.py      bubblewrap sandbox: one-shot file tools + persistent shell
  session_store.py  save/load conversations as JSON, keyed by session name
  file_search.py  bounded filename search backing @-mention autocomplete
  gitwork.py      tree-snapshot checkpoints, diff computation, git filter
  theme.py        the color palettes
  tui/            Textual app, widgets, screens (picker/resume/confirm), stylesheet
```

The UI consumes an async generator of typed stream events and knows nothing
about HTTP; conversation state is plain data, and the tool registry is built
per session/mode, so adding tools doesn't touch widget code.
