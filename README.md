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

## Change tracking and undo

The agent edits your files directly — there is no separate worktree and no
accept/reject step. Instead, every session snapshots your workspace state
into a git tree object at startup (via a throwaway index: your index, HEAD,
and refs are never touched), and every turn takes another snapshot.

No setup is required: if the workspace isn't a git repo (or has no commits
yet), xarness runs `git init` itself so tracking works. Your files are never
modified by this, and no commits are ever created — snapshots are plain tree
objects. Opt out with `--no-init-repo` (the agent still edits the directory
directly; there is just no /diff or /undo for file changes).

While the agent works, a one-line summary above the input shows what it
changed (`Edited 2 files +26 -0`); click it to expand a per-file list
(`name  dir/  +N -M`), and click a row to read that file's unified diff.

| Command | Action |
| --- | --- |
| `/diff` | Show the pending-changes panel; `/diff` again hides it |
| `/diff <path>` | Show one file's unified diff |
| `/accept` | Lock in the changes made so far: `/diff` resets, `/undo` can no longer revert past this point |
| `/reject` | Discard all changes made since the last `/accept` |
| `/undo` | Drop the last turn: revert its file edits, put your message back in the input |
| `/retry` | Revert the last turn's file edits and resend your message |

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

Each turn starts with a checkpoint: a git tree snapshot of the workspace's
current state (tracked changes, uncommitted changes, and untracked files —
gitignored files are excluded, matching classic git semantics). `/undo` and
`/retry` restore the workspace to that snapshot, removing every file change
the turn made — edits, deletions, and newly created files.

**Only file edits are reverted.** Non-file side effects of `run_bash` —
package installs, background jobs, network calls, anything outside the
workspace — are *not* undone. `/undo` puts the removed message back into the
input box; `/retry` resends it immediately. Both also roll back the turn's
token usage from the session totals, and work across a mid-turn `/compact`
(the pre-compaction history is restored). Without git tracking (no repo,
`--no-init-repo`, or setup failure) they still remove the messages and fix
the token counts, but tell you the file edits could not be reverted.

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
with fresh tracking instead.

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
| `/diff`, `/accept`, `/reject` | See "Change tracking and undo" above |

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
  cli.py          argument parsing, boots the TUI, change-tracking setup
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
  gitwork.py      tree-snapshot checkpoints, diff computation, git filter
  theme.py        the color palettes
  tui/            Textual app, widgets, screens (picker/resume/confirm), stylesheet
```

The UI consumes an async generator of typed stream events and knows nothing
about HTTP; conversation state is plain data, and the tool registry is built
per session/mode, so adding tools doesn't touch widget code.
