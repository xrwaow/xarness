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
| `subagent` | plan + write | Run a fresh agent on one self-contained task in the same workspace and return its final answer (see below) |

Each tool's description names only the environment it runs in — the
container for a sandboxed session, your machine for `--sandbox false` — and
is fixed when the session's tool set is built, so the schema stays stable
for the provider's prompt cache.

`subagent` starts a child session with no memory of the current
conversation, no `ask_user` and no `subagent` tool of its own, and returns
the child's last non-empty assistant message. It shares the workspace and
container with its parent, so its edits are tracked (and undone, accepted or
rejected) exactly like any other agent's; it is stored as
`<parent-uuid>/agent-N`, appears in `/sessions` as `agent-N` under its parent, and you
can attach to watch it. `mode: "plan"` children are
read-only; a plan-mode agent can only start plan children. Which model they
run is the preset's `subagent_model`.

A subagent is an ordinary session: once it finishes it stays saved (unless
the run is `--no-session`), so you can open it from `/sessions` and drive it
like any other — it keeps its parent link, and the edits it made stay
labelled as its own. Reopening does not give it back the `subagent` or
`ask_user` tools (a child never nests and has nobody to ask). A message sent
to a child that is still running is queued as steering, exactly like one
sent to a running session. With `--no-session` nothing is saved for the
parent or the child, so a finished child cannot be reopened.

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
Network and GPU access are fixed at launch: `xarness --network false
--gpu false` runs the sandbox fully offline and without device access (each
flag takes an explicit `true`/`false`; leave it out to keep the sandbox
profile's setting — `allow_network: true`, `gpu_access: false`).
`--enable_tracking false` turns change tracking off for the session (no
checkpoints, diff bar or undo/revert), and `--sandbox false` runs the tools
unsandboxed (a host shell instead of the container).

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
user's index, HEAD, and refs are never written from inside the sandbox.

Expose external files to the agent with `/refs add <path>` (or the
`auto_include_refs` sandbox setting); the model addresses them as
`/tmp/refs/<alias>` from both the file tools and the shell.
Plain entries are read-only; a `RefSpec` object can choose a different
mount and mark it writable.

## Change tracking and undo

The agent edits your files directly — there is no separate worktree and no
per-change accept gate before edits land. Instead, the *workspace* keeps the
tracked state, shared by every agent working in it: a **baseline tree**
(what `/diff` measures against), an append-only **event log** of what each
tool call changed, and the set of **ignored paths** as of the baseline.

No setup is required: if the workspace isn't a git repo, xarness keeps a
private snapshot store under its own state directory and drives git
against it with an explicit `GIT_DIR` — no `.git` is created in your
workspace, and your files are never modified. (A repo with no commits yet
still counts as a repo: snapshots are trees and need no HEAD.) No commits
are ever created either: snapshots are plain tree objects. Opt out with
`--no-init-repo`, which disables change tracking entirely (the agent still
edits the directory directly; there is just no /diff or /undo for file
changes).

**One workspace, many agents.** The workspace is the repo root and its
tracked state belongs to it, so every session in that repo — a second
window, a subagent — shares one baseline, one pending diff and one
`/accept`. Paths are repo-root-relative everywhere and `/diff` covers the
whole repo, even when `--workspace repo/sub` points the sandbox's working
directory at a subdirectory.

**Where snapshots live.** Snapshot objects go into the object store of the
repo being tracked — your own `.git` — or, only when there is no repo, into
a private store under xarness's state directory
(`$XDG_STATE_HOME/xarness/repos/<hash>`, typically `~/.local/state/...`)
driven with an explicit `GIT_DIR`, so no `.git` ever appears in your
directory. Using yours means no second copy of your files: the objects are
content-addressed, so a blob git already stores is shared, not duplicated,
and files over 1 MB are never tracked at all (see below).

That they are unreferenced is also the catch: nothing in your repo points
at them, so your repo's own housekeeping can drop them — `git gc
--prune=now`, `git repack -ad`, or any aggressive prune discards
unreachable objects, and an open session's checkpoints are exactly that.
When that happens the session can no longer measure pending changes: the
pending-changes bar stops updating and is cleared on the next attach, with
no message — a failed summary push is treated as transient on purpose, so a
concurrent `index.lock` cannot make tracking look gone. `/diff` and a
per-file click are where it surfaces, as `diff failed: …`, and `/undo`
cannot reach past the point its history was collected. `/accept` — or a
session started fresh in that workspace — re-baselines from the files on
disk; until then `git diff` / `git status` in the repo still show the
pending edits. The conversation and the files themselves are unaffected.
`git fsck` lists the objects as dangling, and they never travel with
clone/fetch/push.

**What is recorded.** Every successful `write_file`/`edit_file` records the
path it wrote **and the provenance of that write**: the blob shas of the
file just before the call and just after it (either one absent when the path
did not exist). Every successful `run_bash` snapshots the workspace before
the command and again after it, and records the paths that differ plus the
ignored directories the command created (the two snapshots are the only
part serialized: two agents' commands can run at once). Nothing else is
recorded, and a command that changes nothing records nothing.

The pending diff is the real worktree measured against the baseline. Its
rows are labelled by who the change came from — for the agent looking at
it:

| Label | Meaning |
| --- | --- |
| `mine` | the path was written by one of *this* agent's `write_file`/`edit_file` calls |
| `agent` | another agent's explicit write, or any `run_bash` change |
| `auto` | an ignored directory an agent's command created (build output, `node_modules/`, …) |
| `drift` | anything no recorded call covers: your own edits, or changes that happened while no agent ran |

`mine` and `agent` rows both display the badge `agent`; the session's own
work is accent-colored and another agent's is dimmed.

**One file, several agents.** A path written explicitly reports *one
author's delta*, not the whole file's pending change: the row's `+A -D`, the
summary totals and the per-file diff all count the change between the blob
that author found and the blob it left, so two sessions on one file are
never credited with each other's hunks. The row that is not the viewer's
shows the last author's delta, and a file more than one session wrote
carries a count badge (the number of authors) so its totals read as shared.
A path with only `run_bash` or drift changes has no author and keeps the
whole pending delta.

The blob shas live in the same object store as the snapshots, so authorship
survives a restart; if a store is pruned under it the rows fall back to
whole-file counts instead of failing. `/accept` clears the event log along
with the pending set, so authorship starts over from the new baseline.

Explicit writes win over bash: if this agent edited a file and another
agent's `run_bash` then touched it, it stays `mine` here. Another session's
(or a subagent's) events are never drift — they are that agent's work,
`agent` to everyone else, including their parent. Because the before
snapshot is taken fresh for each command, an edit you made *before* it
stays `drift`: earlier changes are never credited to a later command.

**A bash window.** The before/after snapshots bracket the whole `run_bash`
call, so an edit *you* make while a command is running lands in that event
and is labelled `agent`. That is the one attribution hole, and it is
deliberate: no markers, no timestamps, no heuristics.

**Ignored paths never enter a snapshot.** Benchmarks showed that tracking
build output adds hundreds of MB of objects to your `.git` and makes
`/diff` about a thousand times slower, so ignored paths are read separately
(`git ls-files -o -i … --directory`, about a millisecond) and reported as
one row per directory (`target/: 412 files`, counted from one full listing
only when there are groups), never per file. A file *modified* inside an
already-ignored directory is not reported — that is intentional.

**Which paths are ignored** is resolved again on every snapshot, in three
layers: xarness's own defaults (build output and caches: `target/`, the
sandbox's `target-sandbox/` and `Cargo.lock` in a project with a `Cargo.toml`,
`node_modules/`, `dist/`, `build/`, `__pycache__/`, `.venv/`, `.next/`, plus
media/binary file types and `.xarness/`), then your repo's own ignore rules
(`.gitignore` including nested ones, `.git/info/exclude`, global excludes),
then the workspace's `.xarness/ignore` (gitignore syntax, ignore-only, no
re-includes). Because they are resolved per snapshot, a mid-session edit to
any of them takes effect at the next snapshot — and a path you *stop*
ignoring can therefore show up as `drift`, even though nothing changed on
disk.

While the agent works, a one-line summary above the input shows what it
changed (`Edited 2 files +26 -0`); click it to expand a per-file list
(`name  [new]  [2]  agent  +N -M`), and click a row to read that
file's unified diff. The numbers are the shown author's own delta and the
count between the name and the badge is how many sessions wrote the file
(see "One file, several agents" above).

The agent's own files lead, then a full-width rule separates them from the
three sections below — a dimmed `agent` (another session's work, and the
ignored groups agents created), `auto` (ignored directories agents created,
with their file count) and `drift` (changes no recorded call covers). All
three headers are always there, each reading label, `▸`/`▾` fold arrow, then
its own `+A -D` totals on the right. Clicking a header unfolds its rows
(indented below it), where a file row opens that file's unified diff; the
sections start folded, so the diff view shows only the agent's diffs by
default. An ignored group is a click target too: it has no baseline copy to
diff against, so it opens what is under it as added files (one `+++` per
file, read-only).

The right of the summary line carries one chip reading `agent`; clicking
it cycles the scope. With every session included the `age` is
accent-colored and the `nt` dimmed; scoped to this session the whole word
is accent-colored, and another session's (or a subagent's) files fold into
the `agent` section between this session's files and `auto`. The selection
is saved with the session (not in the config file), so reopening it —
later in this run or after a restart — shows the same scope, with a chip
clicked in a draft carrying over to the session it starts. The chip stays
a view filter: `/diff <path>`, `/accept`, `/reject` and the panel label's
file count still see everything pending.

| Command | Action |
| --- | --- |
| `/diff` | Show the pending-changes panel; `/diff` again hides it |
| `/diff <path>` | Show one file's unified diff (an ignored path or group reads as added) |
| `/accept` | Re-baseline the workspace at what the agents changed (leaving your own drift pending): `/diff` resets and `/undo` can no longer revert past this point |
| `/reject` | Restore the workspace to the baseline — including your own drift — and remove ignored directories the agents created |
| `/undo` | Drop the last turn: reverse its own file edits, put your message back in the input |
| `/retry` | Reverse the last turn's file edits and resend your message |

To undo further back, click any of your messages in the scrollback (the whole
message is the click target): a small `↩ undo to here` affordance appears
below it, and clicking that line (only its own text) drops the message and
everything after it — each dropped turn's own file edits reversed, and the
message put back in the input. Clicking anywhere else closes the affordance
without undoing. Messages the last compaction summarized away can be clicked
too (the pre-compaction history is restored first).

`/accept` is how you follow change-per-feature: let the agent build one
feature, `/accept` it, move on to the next. The new baseline is the old one
with every pending path that an agent changed (anything not `drift`)
replaced by its current content — one new tree sha, no per-file bookkeeping
— so your own unrecorded edits stay pending and are never absorbed.
Everything before the last accept is out of `/diff`'s and `/undo`'s reach —
use git itself (a repo you control) for history beyond that. `/accept`
never asks: your own drift is never absorbed, so there is nothing to warn
about, and the command's notice reports how many of the locked-in files
weren't this agent's work and how many more than one agent wrote.

`/reject` restores tracked paths to the baseline, discarding everything
since the last `/accept` — **including changes you made yourself**, which is
why the TUI confirms first when there is drift — while keeping the
conversation going so the agent sees the reverted files on its next turn.
It also removes the ignored directories a recorded command created, but
never one that already existed (a `node_modules/` you installed yourself
survives). Ignored files an agent edited explicitly cannot be restored,
since no baseline copy exists; the notice lists them.

### Undo and retry

Each turn's events name the paths it changed. `/undo` and `/retry` reverse
exactly those paths: for each removed turn, newest first, one of two things
happens per path. If the path still holds what the turn left, it is restored
from the turn's before-state (or deleted, when it did not exist before). If
something changed after the turn — you, or another agent — the turn's own
change is merged out of the live file instead: a three-way merge with the
turn's after-state as the base, the file on disk as ours and the turn's
before-state as theirs, so a later edit to a *different* part of the file is
kept and the turn's hunk is undone. Only overlapping hunks conflict (and a
path the turn created or deleted, which has no content to merge into). When
any path conflicts nothing is touched at all and the conflict list is
reported instead. It is therefore all-or-nothing per undo, and it never
touches another agent's paths.

A turn's checkpoint is stamped with the workspace epoch, so a turn that
spans an `/accept` or `/reject` is not revertible any more (the pre-accept
part is already in the new baseline; use `/reject` for the rest).

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
`sessions/<name>/blobs/`) — disable with `--no-session`. Each new session
gets a UUID as its name. Resume with `/sessions` in the TUI, or
attach/load by name at startup:

```sh
xarness --session my-session   # attach to an active session or load a saved one
xarness --sessions             # print active sessions and exit
xarness --kill                 # close every active session (each stays saved)
```

`/sessions` lists only the sessions of the workspace you are in: first the
live ones (running / waiting / ask / compacting / idle), then the saved ones.
Titles are padded to the widest title of the list, so the age, workspace and
+/- columns line up on every row. A subagent is stored as
`<parent-uuid>/agent-N`, shown as `agent-N` nested under its parent, and a
whole group moves to the live section
when any member is live (hosted children above idle ones) — so attaching to a
finished subagent brings its parent and siblings up with it. The
+/- columns are that session's own pending work plus whatever its subagents
wrote; a session blocked on the ask tool reads `ask` (it wants your answer),
one blocked on a tool call (including a running subagent) reads `waiting`,
not `running`.

Resuming reconnects to the session's change tracking: the tracked state is
per workspace, so a resumed session measures against the same baseline as
the one it left (and as any other agent in that repo). Whatever changed
while it was closed — your edits, another agent's — simply shows up in
`/diff` as `drift` or `agent`; nothing is reverted or accepted
implicitly. A session saved by an older schema is refused with a "start a
new session" error rather than half-migrated. `/delete` removes the saved
session (its rows and image blobs) and starts a fresh chat, and never
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
  `allow_network`, `gpu_access`, `allow_git_write`, `use_sandbox`,
  `enable_tracking`, `auto_include_refs`, `filter_list`, and
  `default_timeout`.
- `presets` — named `{model, sandbox}` pairs; `defaults.preset` chooses one.
  A preset may also carry `subagent_model`: the model its subagents run.
  Unset means the preset's own `model`, `"same"` means the spawning agent's
  model (which `--model` may have overridden), and any other value must be a
  key of `models` — a typo fails at startup, not mid-turn. Subagents share
  the parent's container, so there is no subagent sandbox setting.
- `ui.theme` — startup color theme (`ayu-darker`, `one-light`); `/theme`
  switches it live.

Switch model/sandbox at runtime with `/model` and `/sandbox`; they write the
choice back as the default preset/model.

## Run

```sh
xarness                              # default preset
xarness --model deepseek             # named model profile
xarness --sandbox offline            # named sandbox profile
xarness --sandbox true               # default profile, sandboxing on
xarness --sandbox false              # unsandboxed (host shell, no isolation)
xarness --preset work                # named preset
xarness --workspace ./some-project   # sandbox root (default: cwd)
xarness --network false --gpu false  # fully offline, no device access
xarness --enable_tracking false      # no checkpoints/diff bar/undo
xarness --session my-session         # attach/load a saved session
xarness --sessions                   # list active sessions
xarness --kill                       # close all active sessions (stays saved)
xarness --server                     # run the host in the foreground
```

The boolean flags all take an explicit value (`--network true`, `--gpu
false`, `--enable_tracking false`; `1/0`, `yes/no` and `on/off` work too)
and default to the sandbox profile's setting when omitted. `--sandbox`
takes either a profile name or `true`/`false`: with `true`/`false` no
profile is named, so the usual resolution applies (preset, then the
workspace's most recently used profile, then `default`) and only the
container on/off state is overridden.

## Slash commands

| Command | Action |
| --- | --- |
| `/tools` | List the tools the model currently has |
| `/model` | Choose the model profile |
| `/sandbox` | Choose this session's sandbox profile (idle only) |
| `/filter add <path>` / `/filter remove <path>` / `/filter list` | Add or remove a workspace path excluded from the sandbox mounts |
| `/sessions` | Resume a previous session of this workspace; a group (parent and its `agent-N` children) stays together, and moves to the live section when any member is live |
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
  approximate local estimate is shown. Each turn's footer shows that turn's
  own spend as `↑input ↓output` next to its timing line (`Worked for 5.0s
  (4.0s | 1.0s) · ↑12.3k ↓456`; a `~` marks an estimate). The status bar keeps
  running session totals and context headroom against `max_context`.
  Compaction's own
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
  tools.py        tool registry: read/write/edit, run_bash, ask_user, subagent
  sandbox.py      bubblewrap sandbox: one-shot file tools + persistent shell
  workspace.py    per-repo tracked state: baseline, event log, labels, accept/reject/undo
  session_store.py  SQLite persistence: state row, message rows, image blobs
  file_search.py  bounded filename search backing @-mention autocomplete
  gitwork.py      tree snapshots, tree diffs, the read-only git shim
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
