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
| `run_bash` | write | Run a shell command in a persistent sandboxed shell (cwd, env vars, and background jobs survive across calls) |
| `web_search` | always | Search the web via the Brave Search API (needs `BRAVE_API_KEY` set) — runs in the harness process, never inside the sandbox |
| `ask` | always | Ask you a clarifying question in the TUI |
| `compact` | always | Summarize and truncate the conversation to reclaim context |

Switch between modes with `/mode`:

- **plan** — read-only: the model can inspect files and search the web, but
  cannot edit or run state-changing commands.
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

## Git-backed change tracking

Every session runs in an isolated `git worktree` — a second working directory
linked to your repository, checked out on its own branch (`agent/<session>`).
The agent's file edits and shell commands land there; your checkout, its
uncommitted changes, and its branch are never touched.

No setup is required: if the workspace isn't a git repo (or has no commits
yet), xarness runs `git init` itself and snapshots your current files as a
baseline commit — your files are never modified by this. Opt out with
`--no-init-repo` (the agent then edits the directory directly, and diff
tracking is unavailable). Untracked files are copied into the worktree so the
agent can see them — and re-synced at the start of every turn, so files you
add to your workspace mid-session become visible to the agent on its next
round; skip that with `--no-copy-untracked`.

While the agent works, a one-line summary above the input shows what it
changed (`Edited 2 files +26 -0`); click it to expand a per-file list
(`name  dir/  +N -M`), and click a row to read that file's unified diff.

`run_bash` also vetoes git commands that would interfere with the
harness-managed worktree/branch lifecycle.

| Command | Action |
| --- | --- |
| `/diff` | Show the pending-changes panel; `/diff` again hides it |
| `/diff <path>` | Show one file's unified diff |
| `/accept` | Merge the agent's branch into your branch (`--no-ff`), remove the worktree |
| `/reject` | Discard everything (asks you to confirm) |
| `/undo` | Drop the last turn: revert its file edits, put your message back in the input |
| `/retry` | Revert the last turn's file edits and resend your message |

`/accept` conflicts (your branch moved on since the session started) are
reported, never auto-resolved — the worktree stays so you can retry after
fixing things manually. Quitting without accepting or rejecting keeps the
worktree; resuming the session reconnects to it.

### Undo and retry

Each turn starts with a git checkpoint: the worktree's current state is
committed to the agent branch (or, when it is already clean, its HEAD is
recorded). `/undo` and `/retry` reset the worktree to that checkpoint,
removing every file change the turn made — edits, deletions, and newly
created files.

**Only file edits are reverted.** Non-file side effects of `run_bash` —
package installs, background jobs, network calls, anything outside the
worktree — are *not* undone. `/undo` puts the removed message back into the
input box; `/retry` resends it immediately. Both also roll back the turn's
token usage from the session totals, and work across a mid-turn `/compact`
(the pre-compaction history is restored). Without git isolation (no repo,
`--no-init-repo`, or setup failure) they still remove the messages and fix
the token counts, but tell you the file edits could not be reverted.

Because turns are checkpointed, `/accept` merges the branch's per-turn
checkpoint commits plus a final commit instead of one big diff — the history
reads as a per-turn log of the session.

## Sessions

Conversations autosave under `~/.local/share/xarness/sessions/` (disable with
`--no-session`, or name one with `--session NAME`). Resume in the TUI with
`/sessions`, or from the command line:

```sh
xarness chat --session my-session   # resume if it exists, else start it
xarness sessions list
xarness sessions delete my-session
```

Resuming reconnects to the session's worktree; if the worktree was removed
externally, xarness recreates a fresh one from the session's branch (uncommitted
agent changes from before are gone) — or, if it can't, disables filesystem
tools rather than letting the agent touch your real directory.

## Install

Requires Python 3.11+ and `bwrap` on PATH for the filesystem tools.

```sh
pip install -e .
```

## Configure

Copy the sample config and set your API key:

```sh
mkdir -p ~/.config/xarness
cp config.example.yaml ~/.config/xarness/config.yaml
export OPENAI_API_KEY=sk-...
```

Config lives at `~/.config/xarness/config.yaml` by default; override with
`--config`. The key is read from the environment variable named by
`api_key_env` — it is never stored in the YAML file. For providers that need
no key (e.g. local models), set `api_key_env: null`.

Multiple profiles are supported; switch with `--profile` (or `/model` in the
TUI, which also sets reasoning effort):

```sh
xarness chat --profile deepseek
```

If the file has no top-level `profiles:` key, it is treated as a single flat
profile (see the commented example in `config.example.yaml`).

## Run

```sh
xarness                    # default profile
xarness chat --profile deepseek
xarness chat --config ./my-config.yaml
xarness chat --workspace ./some-project    # sandbox root (default: cwd)
```

(bare `xarness` behaves like `xarness chat`)

## Slash commands

| Command | Action |
| --- | --- |
| `/tools` | List the tools the model currently has |
| `/model` | Choose the model and reasoning effort |
| `/sessions` | Resume a previous session |
| `/mode` | Switch between plan (read-only) and write mode |
| `/theme` | Choose a color theme |
| `/new` | Start a new chat |
| `/undo` | Drop the last turn (file edits reverted, message back in the input) |
| `/retry` | Drop the last turn (file edits reverted) and resend its message |
| `/diff`, `/accept`, `/reject` | See "Git-backed change tracking" above |

## Keys

| Key | Action |
| --- | --- |
| `Enter` | Send message |
| `Shift+Enter` (or `Alt+Enter`) | Newline in the input |
| `Ctrl+T` | Expand/collapse the most recent "Thought for Xs" block (clicking it works too) |
| `Ctrl+C` | Copy the current selection, or quit if nothing is selected |
| `Escape` | Interrupt the agent mid-turn |

Note: `Shift+Enter` is only distinguishable from `Enter` on terminals with
extended keyboard support; `Alt+Enter` is the portable fallback.

## Behavior notes

- **States while waiting**: `Processing` from request sent until the first
  token of any kind, then `Thinking` (pulsing) while reasoning-channel tokens
  arrive (`reasoning_content` / `reasoning` delta fields), then the streamed
  answer. Models without a reasoning channel skip the thinking state.
- **Thoughts**: after a turn with reasoning, a collapsed `▸ Thought for Xs`
  indicator stays in the scrollback; `Ctrl+T` or a click expands it inline.
  Reasoning text is kept in the conversation history regardless of visibility.
- **Scrolling**: the log follows new output only while you are at the bottom.
  Scroll up mid-answer and it stays where you put it (nothing yanks it back
  down); scroll back to the bottom and it resumes following. Expanding a tool
  call whose output is a diff renders it with the theme's diff colors.
- **Token counts**: taken from the API's `usage` field when provided
  (`stream_options.include_usage` is requested); otherwise a clearly-labeled
  approximate local estimate is shown. The status bar keeps running session
  totals and context headroom against `max_context`. Compaction's own
  summarization round is included in the totals, and the `compact` tool
  reports what it freed (`compacted: 12,450 → 1,830 tokens (freed 10,620)`).

## Layout of the code

```
src/xarness/
  cli.py          argument parsing, boots the TUI, worktree setup
  config.py       pydantic config models, YAML loading, profile selection
  client.py       async httpx SSE client, reasoning-effort mapping
  events.py       typed stream events (the client/UI contract)
  controller.py   conversation state + event enrichment (no UI imports)
  conversation.py plain message history, wire-format faithful
  prompts.py      system prompts (general + plan/write mode clauses)
  tools.py        tool registry: read/write/edit, run_bash, web_search, ask, compact
  sandbox.py      bubblewrap sandbox: one-shot file tools + persistent shell
  session_store.py  save/load conversations as JSON, keyed by session name
  file_search.py  bounded filename search backing @-mention autocomplete
  gitwork.py      worktree lifecycle, diff computation, accept/reject, git filter
  theme.py        the color palettes
  tui/            Textual app, widgets, screens (picker/resume/confirm), stylesheet
```

The UI consumes an async generator of typed stream events and knows nothing
about HTTP; conversation state is plain data, and the tool registry is built
per session/mode, so adding tools doesn't touch widget code.
