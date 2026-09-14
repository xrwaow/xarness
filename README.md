# xarness

A terminal chat client for OpenAI-compatible LLM APIs, built with
[Textual](https://github.com/Textualize/textual). Codex-CLI-style interface:
persistent input bar, streaming answers, expandable reasoning ("thoughts"),
and live token/context accounting.

This is the UI + config layer of a CLI agent tool — tool calling is not wired
up yet, but the conversation state and stream-event architecture are built to
accept it.

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
agent can see them; skip that with `--no-copy-untracked`.

While the agent works, a one-line summary above the input shows what it
changed (`Edited 2 files +26 -0`); click it to expand a per-file list
(`name  dir/  +N -M`), and click a row to read that file's unified diff.

| Command | Action |
| --- | --- |
| `/diff` | Show the pending-changes panel; `/diff` again hides it |
| `/diff <path>` | Show one file's unified diff |
| `/accept` | Merge the agent's branch into your branch (`--no-ff`), remove the worktree |
| `/reject` | Discard everything (asks you to type `reject` to confirm) |

`/accept` conflicts (your branch moved on since the session started) are
reported, never auto-resolved — the worktree stays so you can retry after
fixing things manually. Quitting without accepting or rejecting keeps the
worktree; resuming the session reconnects to it.

## Install

Requires Python 3.11+.

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

Multiple profiles are supported; switch with `--profile`:

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
```

(bare `xarness` behaves like `xarness chat`)

## Keys

| Key | Action |
| --- | --- |
| `Enter` | Send message |
| `Shift+Enter` (or `Alt+Enter`) | Newline in the input |
| `Ctrl+T` | Expand/collapse the most recent "Thought for Xs" block (clicking it works too) |
| `Ctrl+C` | Quit |

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
- **Token counts**: taken from the API's `usage` field when provided
  (`stream_options.include_usage` is requested); otherwise a clearly-labeled
  approximate local estimate is shown. The status bar keeps running session
  totals and context headroom against `max_context`.

## Layout of the code

```
src/xarness/
  cli.py          argument parsing, boots the TUI, worktree setup
  config.py       pydantic config models, YAML loading, profile selection
  client.py       async httpx SSE client, reasoning-effort mapping
  events.py       typed stream events (the client/UI contract)
  controller.py   conversation state + event enrichment (no UI imports)
  conversation.py plain message history, wire-format faithful (tool role ready)
  gitwork.py      worktree lifecycle, diff computation, accept/reject, git filter
  theme.py        the single color palette
  tui/            Textual app, widgets, stylesheet
```

The UI consumes an async generator of typed stream events and knows nothing
about HTTP; conversation state is plain data, so a tool-calling layer can
append tool-call/tool-result messages without touching widget code.
