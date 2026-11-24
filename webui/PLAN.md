# xarness web UI — plan

A phone-first, no-build web client for the existing session host. It is a
**thin client of the same HTTP/SSE API the TUI uses** (`src/xarness/clientlib.py`
is the reference implementation): it renders one session's event stream and
sends commands. No server-side state is added.

## What the TUI is (what we are porting)

Layout: one scrolling transcript, a status bar at the bottom, and above the
input a stack of transient bars — generating bar, pending-changes summary,
ask question, suggestion popup, steer queue. The input is a multiline box;
pressing Enter sends, Shift+Enter adds a newline, Tab toggles plan/write mode.

Transcript rows, in the TUI's visual language:

| Row | Rendering | Source events |
| --- | --- | --- |
| user message | right-aligned surface bubble, real attachment thumbnails (no `🖼` token), click for `↩ undo to here` | `MessageAdded` (user), snapshot messages |
| thinking | activity-block row, shape + `Thought for 4s · 1.2k tokens`, click for the full reasoning (see *Activity blocks*) | `ReasoningDelta`, `RoundComplete` |
| answer | `•` marker + markdown prose and fenced code; live tail is plain text | `ContentDelta` |
| tool call | activity-block row, shape + name + argument detail + status, click for args/output (see *Activity blocks*) | `ToolCallStarted`, `ToolCallArgumentsDelta/Done`, `ToolOutputDelta`, `ToolFinished`, `ToolUsageChanged` |
| pending strip | `Processing 4s (esc to interrupt)`, shimmer | `ProcessingStarted`, status `compacting` |
| notice / warn / error | one line, `✗`/`!`/`✓` glyphs, muted/warning/error color | `Notice`, `StreamError`, client-side errors |
| turn footer | `Worked for 5.0s (4.0s | 1.0s) · ↑12.3k ↓456` | `TurnEnded.stats`, persisted `Timing.stats` |
| compaction divider | the handoff text, accent color | snapshot `is_summary` message |

The footer's two numbers are the turn's `busy | idle` split: time the model
spent generating vs time spent waiting on tools, compaction and teardown.

Palette (`src/xarness/theme.py`): `ayu-darker` by default and `one-light`
alternate; semantic keys (`bg`, `surface`, `border`, `text`, `muted`, `user`,
`reasoning`, `accent`, `accent2`, `status`, `error`, `success`, `warning`,
`diff_*`) become the CSS variables at `:root`.

Slash commands (`COMMANDS` in `src/xarness/tui/app.py`): `tools`, `model`,
`sessions`, `mode`, `sandbox`, `filter`, `refs`, `theme`, `new`, `delete`,
`diff`, `accept`, `reject`, `undo`, `retry`. Keys: Enter, Shift+Enter, Esc
(twice, to interrupt), Ctrl+C (copy/quit — browser-native here).

## Feature → API mapping

| Feature | Backend |
| --- | --- |
| attach / reconnect / rebuild | `GET /v1/sessions/{name}/events?range=[-40:]&slim=true` (SSE; first frame `snapshot` — a turn-aligned window of the newest N messages, tool bodies left for hydration — then `event`) |
| load earlier messages | `GET …/messages?range=[<start-40>:<start>]&slim=true` (a `MessagePage`; autoloaded near the top) |
| expand a block / row | `GET …/messages/{index}?content=false` (headers) / `content=true` (the round's output) |
| send message (no local echo) | `POST …/commands` `{type:"send_message"}` |
| draft chat → first session | `POST /v1/sessions` `{workspace?, model?, view}` (then attach) |
| home: live / recent / folders | `GET /v1/events` (SSE: `snapshot`, then `session_changed` / `sessions_removed`), `GET /v1/workspaces` |
| interrupt (double Esc / Stop button) | `{type:"cancel"}` |
| steer queue: discard / send now | `{type:"discard_queued"}` / `{type:"flush_queued"}` |
| ask tool answer / skip | `{type:"answer_ask", ask_id, answers}`; `AskResolved` closes it |
| undo, click-to-undo, retry | `{type:"rollback"}` / `{type:"rollback", message_id}` / `{type:"rollback", resend:true}`; result's `user_text` goes back in the input |
| `/mode` (and Tab) | `{type:"set_mode", mode}` |
| `/model` | `GET /v1/config`, `{type:"switch_model"}`, `PUT /v1/config/defaults` |
| `/sandbox` | `GET /v1/config` → `{type:"set_sandbox"}` |
| `/filter add|remove|list` | `{type:"filter"}` / `status.filter_list` |
| `/refs` | `GET|POST /v1/workspaces/refs` |
| `/tools` | `GET …/tools`; `{type:"set_mcp_server_scope"}`, `{type:"set_mcp_tool_enabled"}`; `GET …/tools` again to refresh |
| `/new` | detach locally (no request — the old session keeps running) |
| `/delete` | `DELETE /v1/sessions/{name}` |
| `/diff`, file rows, `/diff <path>` | `GET …/diff`, `GET …/diff?path=&author=`; in a draft `GET /v1/workspaces/diff?workspace=&scope=` (tracking is per workspace, so no session is needed) |
| diff scope chip | `{type:"set_view_state", view:{diff_scope}}` (a draft refetches the workspace endpoint instead) |
| `/accept`, `/reject` (+ drift confirm) | `{type:"accept"}` / `{type:"reject"}` |
| `/sessions` | `GET /v1/events` (same stream; `GET /v1/sessions` is the fallback poll) |
| one row's saved totals (hover) | `GET /v1/sessions/{name}/summary` (read-only: a saved session is not hosted to answer) |
| status bar / bars | `StatusChanged.status`, snapshot `status` |
| `@` file mention | `GET /v1/workspaces/files?workspace=&q=` |
| image attach | `PUT …/images/{sha256}` then `send_message.images` |
| `/theme` | client-only (CSS variables + `localStorage`) |
| connection indicator, reconnect backoff | `GET /v1/health` + SSE retry |

### Deliberately skipped (and why)

- **`/tools` add-a-server form.** Server listing, on/off scope cycling
  (`off`/`global`/`workspace`) and per-tool toggles are ported; adding a
  server needs the TUI's shell-style argument parser, and the config file is
  the documented primary path. The command exists, the UI just doesn't offer
  it.
- **Quit verdict / "continue headless".** A browser tab has no server-side
  session lifetime: closing it loses nothing, the host keeps the turn. No
  prompt is needed (and `GET …/quit` is not called).
- **Transcript virtualization.** The TUI's `FlowView` paints only visible
  rows. The browser's native scrolling handles a long log; we cap live tool
  output at the TUI's 64 KiB and collapse tool bodies, so a runaway command
  cannot bloat the DOM. The *history* is autoloaded in slim pages instead
  (see "Opening a long chat" below): attaching ships the newest window, and
  older pages carry shapes and headers until a block is expanded, so the DOM
  and the wire never hold a transcript nobody is reading.
- **Terminal-title/OSC, copy-on-quit, mouse-selection workarounds.** Browser
  equivalents (document title, native copy) replace them.
- **Theme persisted to the server config file.** `/model` persists its
  default through the existing endpoint; the theme is a per-browser
  preference in `localStorage` (`PUT /v1/config/defaults` has no theme field
  and adding one would change an existing response/request shape).
- **Image *paste from a terminal clipboard*** — a browser's paste event gives
  real image blobs, so `Ctrl+V`/file-picker attach is ported natively.

## Backend gaps (minimal, additive)

The host listens on a unix socket only, and `api.py` serves no HTML. Three
small additions, none changing an existing route or shape:

1. `api.create_app(host, *, token=None)` mounts `webui/` with `StaticFiles`
   **after** every router (so `/v1/*`, `/docs` win), and returns the app
   unchanged when the directory is absent.
2. When `token` is passed (only for the TCP listener), a middleware requires
   `Authorization: Bearer <token>` on `/v1/*` and answers `401` otherwise.
   The unix-socket app is built without a token — the TUI is untouched.
3. `host` gains an optional second listener: `xarness --web` runs the
   persistent host in the foreground over the unix socket *and* TCP
   (`--host`, `--port`, `--token`). A token is generated and printed when the
   bind address is not loopback and none was given.

## Structure

```
webui/
  index.html          shell: floating head, sidebar/drawer, log, composer, status bar
  style.css           palette variables, mobile-first layout, desktop @media
  app.js              wiring: state machine, events, keys, sessions, diff, ask
  api.js              typed-ish fetch client + SSE reader with reconnect
  state.js            transcript reducer (mirror of tui/transcript.py)
  render.js           DOM renderers per item kind + incremental updates
  markdown.js         sanitizing markdown → DOM (no innerHTML anywhere:
                      headings, lists, quotes, rules, pipe tables, code)
  diff.js             unified-diff → DOM, shared by prose, tool output and the bar
  commands.js         slash-command registry (mirror of COMMANDS)
  manifest.webmanifest, service-worker.js, icons/
```

No framework, no bundler, no npm. A markdown library would need vendoring and
a sanitizer audit; `markdown.js` builds DOM nodes from `textContent` in ~200
lines, which is both smaller and safe by construction. No CDN at runtime.

## Responsive plan

Mobile-first: single column, `100dvh` flex column, the log is the only
scrolling box, the composer is pinned with `env(safe-area-inset-bottom)`,
16px input text, ≥44px tap targets. The menu, session name and folder float
at the top-left of the conversation over a short gradient instead of sitting
in a full-width topbar. The sidebar is a slide-over drawer
behind a hamburger + scrim. Approve/answer controls live directly above the
input (thumb-reachable). `@media (min-width: 768px)` keeps the sidebar
visible, caps the message column and adds per-row metadata.

Auto-scroll only while the user is at the bottom; scrolling up raises a
"jump to latest" button. Streaming appends to the current message node;
reconnect always rebuilds from the fresh snapshot, so messages cannot
duplicate. All model/tool text goes through `textContent`.

## Verification

Fake OpenAI-compatible provider on localhost (streaming reasoning + content +
a `run_bash` and `ask_user` tool call), real host via `xarness --web`,
headless Chrome screenshots at 360, 768 and 1280 px, plus DOM checks for
horizontal overflow, the approval/ask flow, reconnect and a dead backend.

**Not verifiable in this development sandbox:** change tracking needs
`git init`, which the sandbox's read-only git shim blocks, so every session
runs with tracking off. The diff bar, `/diff`, `/accept`, `/reject` and
file-edit undo are written against the same endpoints and models as the
TUI's, but they were never exercised end to end here — the code paths that
do run (an empty `/diff` response hides the bar) are what was tested. The
diff **bar and patch view** of the phone round below were driven with an
injected `app.diff`/`app.diffPatch` state (open, current-row highlight,
whitespace, close by row and by header); a real tracked change was not.

## What the browser run changed

Things the plan got wrong or did not foresee, found by actually driving the
UI in Chrome:

- **Resuming.** A reload used to start a fresh session, because only a
  `#s=<name>` URL attached. The TUI reopens the workspace's last
  conversation, so the UI now does too: the last attached session is kept in
  `localStorage`, and the most recently used one of this workspace is the
  fallback. A new chat clears it.
- **A silent stream is not a dead host** — but a dead host leaves the SSE
  socket half-open and the browser never reports it. A watchdog looks at the
  stream's last activity; after 12s of silence it probes `GET /v1/health`
  and only then tears the stream down for the normal reconnect-with-backoff.
  Either way the reconnect starts from a fresh snapshot, so nothing
  duplicates.
- **The service worker was cache-first**, which pinned an old `app.js`
  forever after an update. It is now stale-while-revalidate for the shell
  (still: never `/v1/*`, never an event stream), network-first for
  navigations, and the cache name is versioned.
- **Tool rows did not expand** until the next stream event arrived; the
  click now rebuilds the body immediately.
- **The status line had no room at 768px.** Below 1024px it drops the
  `in/out` totals (keeping the context chip) instead of clipping the end of
  the line.
- **A stored token** was not used until the prompt was answered, so every
  reload asked again; it is now applied before the first request. The token
  button also stays visible while the host answers `401`, so a forgotten
  token can be re-entered without a reload.
- **A banner can lie.** The URLs used to print before the listeners were
  bound, so a bad `--host` (an address DHCP had moved, a port already taken)
  printed a URL and then died in a uvicorn `SystemExit` traceback. The bind
  is now probed first, the banner prints only once the listeners are up, and
  a bind failure is one line and exit status 2.

## Phone round: the style is ours now

TUI fidelity stopped being a goal: the transcript keeps the TUI's palette
and vocabulary, but rows are laid out for a thumb, not a terminal.

- **User messages rendered empty.** The host's `UserMessage` derives
  `content` from ordered `parts`, so the wire JSON of a user message has no
  `content` field — only `parts`. `state.js` read `message.content`; it now
  goes through `messageText()` (text parts joined) in both the snapshot and
  the `message_added` path.
- **Tool rows are cards.** The head is a grid: status dot, name, token
  count, caret on the first line and the dimmed argument detail clamped to
  two lines underneath (one line beside the name from 768px). The whole
  head is the 44px tap target. Expanded, arguments and output sit in their
  own bordered, scrollable boxes under `arguments` / `output` labels; the
  command line is long and used to be clipped.
- **Enter depends on the pointing device.** `(pointer: coarse)` (read live
  on every key, so a tablet with a keyboard gets it right) means Enter
  inserts a newline, the keyboard key is labelled *return* and **Send**
  sends; Ctrl/Cmd+Enter still sends. A mouse/keyboard keeps Enter = send
  and Shift+Enter = newline. The keyboard's own key label (`enterkeyhint`)
  follows the device.
- **Diffs lost their spacing.** Two causes: the patch view appended `"\n"`
  text nodes between block `<span>`s *inside a `<pre>`*, so every line was
  followed by an empty one, and a diff pasted into an assistant answer went
  through the markdown list parser (`-    return x` became a bullet and the
  indentation vanished). `diff.js` builds one block span per line with no
  separators (indentation and tabs intact), and `markdown.js` hands a run of
  patch lines to it instead of parsing them. Fenced ` ```diff ` blocks get
  the same treatment.
- **An open diff lives in a side panel.** Clicking a file row fetches its patch into a panel floating over the right edge, so the transcript keeps its width and never reflows (`diff.js` parses the hunks into old/new line-numbered rows, wraps long lines and colors the code with the same lexer as fenced blocks); below 900px the panel covers the screen. Clicking the same row again, `✕`, Esc, or folding the bar removes it; clicking a different row swaps the file in place (group rows included — they never had an `author` to match). The bar's body is capped at `46dvh` and scrolls inside, instead of pushing the log off a phone screen.

## Composer round

- **The composer is a bottom sheet flush with the screen.** Its height comes
  from `visualViewport.height` (`--app-h`), not `100dvh`: the layout
  viewport includes the area under mobile browser chrome and does not shrink
  for the software keyboard, which left a dead strip under the input and
  kept the composer from rising with the keyboard. On a phone the page no
  longer focuses the textarea on load (that popped the keyboard and scrolled
  the sheet by itself).
- **Send moved into the input row.** It used to sit alone at the right of a
  second row, which read as empty space under the input; that row is now the
  composer foot: model name and mode on the left, the context and token
  readouts pushed to the right edge.
- **The empty band above the input was the attachments strip.**
  `.attachments { display: flex }` beat the UA's `[hidden] { display: none }`,
  so the empty strip kept its 10px of padding even with nothing attached.
  `[hidden]` now wins, and the composer is 85px instead of 95px idle.
- **The slash palette is part of the sheet.** It is no longer an absolutely
  positioned card over the transcript: it is the top band of the composer
  panel, above the diff bar, and pushes the panel taller (`32dvh` max for
  the rows). Tapping a row runs it outright, so `/theme`, `/diff` and
  `/filter` open their picker/list on one tap. (Enter used to only complete
  an argument-taking command's word — see the next section.)
- **Diff file rows stack on a phone.** The path gets its own line (wrapped,
  not ellipsised) and the badges and `+/-` totals sit under it; on wider
  screens they stay on one line.
- **Shimmer.** The TUI's sweeping highlight is reproduced with a clipped
  CSS gradient on the pending label, the live `Thinking …` pill and a tool
  name while its arguments stream (`--shimmer-peak` is white on dark themes
  and black on light). The gradient is three times the label's width and
  only moves between `100%` and `0%`, so no glyph ever falls outside the
  painted area and goes transparent, and the band starts and ends off the
  label — half of each 3.4s cycle is a natural pause between sweeps rather
  than the band jumping straight back to the left edge.
- **`edit_file` and `read_file` summaries.** `edit_file` shows `+A -D path`
  (computed from salvaged partial JSON, so it grows while the arguments
  stream) and `read_file` just its path — a read has no additions to count —
  plus the window as a bare tuple: `[10, 20]`, `[10, …]` to the end of the
  file, `[…, 20]` from the top. (The TUI spells out `[start_line=10,
  end_line=20]`; on a phone that ran off the row for no gain — this is the one
  place the web row is terser than the TUI's.) A row whose name and summary fit
  on one phone line (all three file tools, most of the time) keeps its summary
  beside the name. `render.js` tries the one-line row and lets the browser
  decide — `scrollWidth` past `clientWidth` means the tail would be ellipsised,
  so the summary takes its own full-width line and wraps instead. Measuring the
  laid-out width, not counting characters: `read_file`'s `path [10, 20]` is a
  handful of characters longer than `edit_file`'s `+3 -3 path` but still fits
  beside the name on a phone. The expanded body of a streaming `write_file` shows its content
  and a streaming `edit_file` a provisional `-old`/`+new` diff, instead of
  the raw JSON call; the real diff takes over when the result arrives.
- **Chain-of-thought closes on tap.** The `Thought for …` pill was already a
  toggle, but its tap target was small and the open reasoning block itself
  did nothing. The pill is full-width (44px on a phone) and tapping the
  reasoning collapses it.
- **Auto-scroll is sticky, not distance-based.** The old rule turned follow
  off whenever the log was more than 40px from the bottom — so typing a
  newline (which shrinks the log) or a fast reasoning burst stopped it. Now
  only a real gesture (wheel, touch, pointer, scroll keys) within the log
  can take over; layout-driven scroll events and programmatic scrolling
  keep following. The jump-to-latest button reflects the same state.
- **The service worker no longer serves a stale shell.** It used to be
  stale-while-revalidate, which meant the first load after a change ran the
  previous `style.css`/`app.js` — one device (the phone, whose cache was
  older) could show an old UI while another looked right. Shell requests
  are network-first now, with the cache kept only as an offline fallback,
  and the SW is registered with `updateViaCache: 'none'` so the worker
  script itself is revalidated. Cache name is `xarness-ui-v5`. The host
  sends `Cache-Control: no-cache` on the shell too, so a device without a
  service worker (a phone on plain `http://` over the LAN is not a secure
  context) revalidates against the ETag instead of running a heuristically
  cached `style.css` for an hour.
- **Shimmer survives WebKit quirks.** Clipped text gains
  `box-decoration-break: clone` (and its `-webkit-` form), the known iOS Safari fix for
  gradient text that paints nothing (the glyphs are transparent, so a
  browser that drops the clip shows a blank label). `prefers-reduced-motion`
  still turns the sweep into a solid colour, and the pulsing status dot
  stays as an independent "this call is running" cue.
- **Tool summaries carry the diff colours.** `edit_file` (`+A -D path`) and
  `write_file` (`+N path`) render their counts as spans tinted with
  `--diff-add`/`--diff-del`, the way the TUI does it; the path stays muted.
  The counts sit at the row's right edge, with the state word (when there is
  one) just before them; the header's `+A -D`, time and token counts share the
  same right edge, so the numbers line up down the block.
  `read_file` shows no counts at all (nothing was added) — just the path and
  the requested range.
- **The reasoning pill is left-aligned with the tool rows.** It used to be
  indented to the tool *body*, which read as a stray offset; the chevron now
  sits in the tool rows' status-dot column and the `Thought for …` text
  starts exactly where a tool name does. The reasoning body keeps its
  indented, left-bordered block.
- **No scrollbar stripe in the composer.** Once a message passes the
  textarea's `max-height` the browser drew a grey scrollbar track down the
  right of the input, which looked like a rendering artefact (it appeared
  "randomly", depending on how many lines a message had — the cap is 40% of
  the viewport, so it arrives sooner with the keyboard open). The textarea
  still scrolls; it just no longer paints a track
  (`scrollbar-width: none` + a zero-width `::-webkit-scrollbar`).
- **Scroll follow, rewritten once more.** The old rule treated a scroll event
  as the user's only within one second of the *last gesture*, and a flick's
  momentum keeps firing scroll events after the finger is off the glass — so
  halfway through a fling the follow looked like a layout shift again and
  yanked the log back to the newest token. Now a gesture opens a 400ms
  window that *every* scroll event landing inside it extends, so momentum
  stays "the user's" until it settles, and our own pins are ignored for
  150ms after we write the position. Detach/re-attach use hysteresis (out at
  40px, back in at 8px) instead of a single 24px threshold, so scrolling up
  detaches once and coming back to the bottom re-attaches on its own — no
  tap on the jump button needed, and no flicker at the boundary while
  streaming. A drag shorter than two lines snaps back instead of detaching.
- **The shimmer band is white in every theme.** On light themes it used to
  be black (a dark band sweeping over dark text, which read as a smudge);
  `--shimmer-peak` is `#ffffff` in all four theme blocks now, so the sweep
  always looks like light moving across the label.
- **Tool names are 12px**, the size of the `Processing`/`Compacting` strip
  and of the reasoning pill, instead of 13px: the rows keep their tap height,
  but the label no longer sits a size above everything else in the log.

## Tables, palette and popups round

- **Markdown tables render as tables.** `markdown.js` recognises a GFM pipe
  table (header row, then a `---`/`:---`/`---:`/`:---:` delimiter row with the
  same number of cells) and builds a real `<table>`: outer pipes optional,
  `\|` and pipes inside code spans do not split a cell, short rows are padded
  and long ones clipped to the header's shape after headers, body rows in a
  `<tbody>` and alignment classes per column. It sits in a
  `div.table-wrap` that scrolls on its own, so a wide table never drags the
  transcript sideways on a phone; the header is quiet mono upper-case on the
  page background, body rows are hairline-separated with a barely-there zebra.
  Tables in the *live* tail still stream as plain text and become a table
  when the answer settles, exactly like paragraphs (the TUI does the same).
- **A code fence is just code.** No card, no tint, no border, no language
  header: a fenced block is a `<pre class="md-code">` of monospaced text on
  the same surface as the prose around it, scrolling sideways only when a
  line is genuinely too long. Inline code loses its little chip for the same
  reason — the mono face is the signal. The text is coloured by
  `highlight.js`, a small sticky-regex lexer that keeps the TUI's
  `CodeHighlightTheme` mapping (keywords `accent`, numbers/types `accent2`,
  strings `success`, comments muted italic, functions `status`) and paints
  through the same `--*` variables, so a `/theme` switch recolors code with
  no re-render. Bare and unknown fences stay plain, as in the TUI. A
  `diff`/`patch` fence is the one exception: it keeps its red/green lines,
  because there the colour is the information.
- **The composer foot reads left to right.** Model and mode sit on the left,
  where they describe what you are about to send; the context and session
  totals are pushed to the right edge by the ctx cell. No `|` separators, no
  chips: each mode is just its name in colour (write amber, plan blue), and a
  click toggles it. The attach button is gone: images are pasted into the input
  (the file picker was a second route to the same `addFiles`). Stop is a square
  glyph button, deliberately smaller than send, not the word "Stop".
- **A sent message is text, not a bordered card.** The user bubble keeps its
  surface and chat tail but lost the hairline border and the hover frame;
  clicking it still arms `↩ undo to here` (a background tint), which is the only
  state change.
- **One Enter opens the popup.** The old rule let Enter only *complete* an
  argument-taking command's word, so `/theme` and `/diff` (and `/filter`)
  needed a second Enter to show anything. Enter now always runs the
  highlighted row; Tab is the key that just fills the word in
  (`/filter `), leaving the arguments to be typed. The palette's footer line
  says which is which, along with the selected row's description, and it
  disappears on a phone where there is no Tab key.
- **`/filter` is a picker, not a syntax lesson.** Bare `/filter` opens a list
  of the session's excluded paths with a `+ add a path…` row (which asks for
  the path in a prompt sheet); `/filter add|remove <path>` still works when
  typed out.
- **The palette moves without rebuilding.** Arrow keys used to re-render the
  whole list (which reset the scroll and restarted the selection style);
  they now only move the `aria-selected` row and scroll it into view, and the
  selection is the plain hover background — the name brightens — with no
  marker bolted on the left. Hovering lights a row up but never steals the
  keyboard focus. A palette with no matches no longer swallows Enter: the
  text goes to the normal submit path.
- **Ctrl/Cmd+K no longer clears a draft.** Running a command from the palette
  only consumes the input when it actually holds a `/…` command, so opening
  the palette over a half-written message and picking `/sessions` leaves the
  message alone.
- **Popups are flat sheets.** The `<dialog>` has a header (title + ✕), a
  scrolling body and a pinned button row, a 3px corner (the same `--radius` as
  every other piece of chrome) and a short fade/rise animation (off under
  `prefers-reduced-motion`). Rows are plain lines of text: no border, no box,
  no bar, no check — hovering or arrowing to one is a flat square of the page
  background, and the row for the current session is only a shade brighter
  than its neighbours. On a phone the dialog becomes a bottom sheet: full
  width, squared off at the bottom, with the safe-area padding under the
  buttons and one wide button per action.
- **One alignment grid inside every sheet.** `--sheet-x` (16px) is the single
  left margin for the title, the text, section labels, notes, hints and every
  row; the body's `--row-x` inset is shared by row padding and section labels
  so they cancel out. Rows are `name (flex, ellipsised) — meta`; the meta
  cells are separated by spacing alone — no `·`, no `|` — so the values read
  as columns, not as a sentence.
- **The /tools sheet is the TUI's ToolsScreen, in web clothes.** Title
  `tools · N available`. Two panes, because the definitions are the long part:
  an `available tools` section at the top where each row is the name (plus the
  server name, only for a tool that comes from an MCP server — a repeated
  "built-in" is noise) and the description clamped to two lines, inside its
  own scroll box capped at `min(30dvh, 280px)`, marked by a hairline above and
  below rather than a card; then `mcp servers` below it, always on screen.
  Nothing is drawn as a box: a server is a block of lines separated from the
  next by a hairline — status dot, name, the scope as a word (`off` grey,
  `on (global)` green, `on (workspace)` accent) that a click cycles, the tool
  count, the target on the second line, then its tools as `[x] name` (muted
  name when off, disabled while the server is not connected). The description
  of an MCP tool is its row's `title`; it is not repeated under every
  checkbox. A failed or disabled server gets a note line saying what a click
  will do. The footer spells out the click cycle without separators. Toggles
  re-read `GET …/tools` and repaint in place, keeping the scroll position and
  the keyboard focus; a server that is `starting` is polled for a few seconds
  so the stale state does not linger.
- **/sessions is two values, not a report.** Each picker row is the name, then
  the age (tabular, muted) and the state word flush right (amber running,
  accent ask, blue waiting, green idle, muted saved). The workspace directory
  and `+n`/`-n` counts are gone: they are in the sidebar row, and in a picker
  they only pushed the thing you are choosing off the left edge. Live sessions
  come first under a `live` heading, saved ones under `saved`.
- **Keyboard in a picker.** Arrow keys move the highlight and take the focus
  with them, Enter picks, Esc/backdrop/✕ cancels. Confirm dialogs focus their
  primary button so Enter means "yes" (it used to focus the ✕ and mean
  "no"); Enter in an info sheet closes it, while a focused row button keeps
  its own Enter.
- The service-worker cache name moved to `xarness-ui-v20` so a device that
  already has the old shell picks this one up.

## Activity blocks: thinking and tools together

A turn's reasoning and tool calls used to be one row each. They are now one
**activity block** per contiguous run between assistant prose. A markdown
answer therefore closes the block, so the transcript reads: activity, answer,
activity, answer — the prose is the separator.

- **Shapes are SVG, and the colour says which tool.** Thinking is a blue
  triangle; every tool call is a square. Each built-in has its own hue
  (`TOOL_COLORS` in `render.js`: amber for bash, blue for `read_file`, green
  for `write_file`, purple for `edit_file`, …), so a long run reads as
  "bash, read, edit" without a legend; a tool the list has never seen hashes
  into the same palette and still gets a stable colour. An MCP tool keeps the
  square but takes a colour generated from its server name (`MCP_COLORS`), so
  one server is recognisable across a turn. Shapes carry status through
  `data-status` (running pulses, failed fades) rather than changing identity.
  Each shape is also drawn a little hand-made: a few degrees of rotation and
  skew with an independent stretch on each axis, and a thick outline that is
  ink-dark on the light theme and the panel border colour on the dark one
  (`--shape-stroke`). The variation is hashed from the session name plus the
  item's stable identity
  (message index and call id, never the in-memory id), so a reload redraws the
  same shapes and none of that art is stored anywhere.
- **The header is the run's summary.** Two tallies only: how many times the
  model thought and how many tools it ran, so it reads `Thought 9 times, Ran 6
  tools`. A run of one has nothing to tally, so it names the activity itself:
  `Thinking` until the reasoning stops and `Thought` after; for a tool,
  `Running read_file · src/app.py` until the call returns and `Ran …` after
  (the row would have carried that name and argument, and a solo call has no
  row). A longer run counts only what has settled,
  and the activity still in flight joins the line in the present tense (`, …
  Running`) until it is done — `Thought 2 times, Ran 1 tool, Running`, then
  `Thought 2 times, Ran 2 tools`. The per-tool breakdown lives in the expanded
  rows. Below the
  summary is **one SVG per activity, in the order it ran**, wrapping onto as
  many lines as it needs — not a `×N` over a grouped shape. A run of one (a
  single thought or a single call before the answer) keeps its shape on the
  summary line instead, so the block stays one line tall. The last shape is
  the current activity and pulses; every shape carries its own name/arguments
  as a tooltip, and the
  header tooltip carries the current one. On the right sits the block's **time**
  — no Done/Running word and no caret: the whole header is the toggle, so the
  only affordance it needs is `aria-expanded`. The stopwatch runs while the
  block is live and freezes when the answer closes it. It turns red only once
  the block has been closed *and* its last activity failed: while the block is
  still open nothing is conclusive, so a call that a later round works around
  never flashes the counter, and an earlier failure is shown on its own row.
  The diff counts, token spend and time form one right-aligned cluster on the
  summary line: nothing reserves a slot, so the cluster sits tight and slides
  with its content, while the shapes keep the full width of the line below.
  The expanded header sticks to the top of the log, so it is filled with the log
  background (opaque, not transparent) and dims to the surface colour on hover,
  masking the rows scrolling under it.
- **How the time is measured.** A live block is timed from its first activity
  to the moment a non-activity item closes it, so it includes the model and
  tool waiting: that is what "took" means. Tool calls are stamped in the view
  as they start and finish, since the protocol carries no tool timings. A block
  rebuilt from the log has no wall clock, so it adds up what the transcript
  knows — the reasoning durations — and prefixes `~` when tools are part of the
  sum (their time is unknown); with nothing measurable it shows no time at all.
- **Eight lines live, the rest scroll.** While the last activity is live and
  the header is collapsed, an eight-line (`calc(8 * 1.5em)`) window under the
  header shows its tail — reasoning text, tool output, or arguments/diff while
  output has not arrived. It follows the bottom unless the user scrolls up
  inside it, preserving the offset as more text streams in. The window is
  hidden once the block ends, but stays visible while the header is expanded
  during a live run.
- **Expand for the run, click a row for the item.** Tapping the header lists
  every activity as shape · name · arguments · state. A block with only one
  activity has no rows to list, so its header opens that activity directly
  (see below). The connector is carried
  entirely by the rows, so it starts a few pixels *below* the shapes row
  instead of poking up into the header, and the successive row lines overlap
  the 1px row gap so the stroke stays unbroken. Each row's shape paints above
  the connector, so the line runs behind the icons instead of across them.
  Tapping a row opens its arguments/output, or the full thought for a thinking
  row. Rows are 44px tap targets on phones.
- **Changes in the block show at a glance.** When the run wrote to a file, the
  header carries `+12 -3` at the start of its right cluster, on the summary line
  before the token counts and the time (all the same size, the counts still
  coloured): line totals summed over every successful `edit_file`/`write_file`
  in that block, coloured like the diff bar and with a sentence for the tooltip.
  A block that only read or ran commands shows nothing, and a failed edit is not
  counted. A slim page strips tool arguments, so the view hydrates a round as
  soon as an `edit_file`/`write_file` lands in it: the header total is there
  before anything is expanded.
- **Tokens too.** Next to the counts, muted, the block's spend:
  `↑3.4k ↓1.6k`. It sums two things: the **model round** that produced each
  item — `round_complete`
  reports input/output for the whole response, stored as `roundUsage` on the
  round's thought, or on its last tool call when the round thought nothing —
  and any **local spend a tool reported** for rounds it ran itself
  (`tool_usage_changed`). A thought with only a reasoning-token count falls
  back to that. `~` marks an approximate figure, a side nobody reported is
  dropped (`↓500`), and nothing is shown when the block reported nothing. Kept
  monochrome so the adjacent `+12 -3` stays the only coloured number in the
  header.
- **A round is one model response.** `round_complete` now closes the round's
  items: the thought and the streamed answer are dropped from the reducer's
  current slot, so the next round opens fresh ones. Before, a second reasoning
  phase appended to the first (already finished) thought live, while a snapshot
  of the same turn rebuilt one thought per assistant message — the two only
  agreed after a reload.
- **Live-only lines are not boundaries.** `app.js` pushes a `pending` item
  ("Processing") between rounds and drops it again; it is never replayed from
  the log. Treating it as a separator split one live turn into a block per
  round while a reload merged them, so `TRANSIENT_KINDS` keeps the block open
  across it and a new block is inserted above the pending line.
- **Grouping is a render concern.** `state.js` still emits the same
  `thinking`/`tool` items; `LogView` maps each item id to an `ActivityBlock`,
  so late updates (`round_complete` token counts) find their original block
  instead of creating a new one after its answer. Any other item closes the
  open block, except the live-only lines above. `thinkingItem` now starts
  collapsed, since the block header is the control.
- The service-worker cache name moved to `xarness-ui-v30`.

## Opening a long chat: autoload, shapes first

Attaching sends the newest window only (40 messages), and older history
loads itself as the reader reaches the top of the log — no button. A page is
*also slim*: tool arguments and output are dropped and reasoning keeps its
first 160 chars, so scrolling through an old chat is cheap.

- `GET …/events?range=…` / `GET /v1/sessions/{name}?range=…` window the
  snapshot (`message_offset`, `message_total`, `has_more`; no `range` =
  the whole history, the TUI's attach). `GET …/messages?range=&slim=`
  returns one older `MessagePage`; pass `[start-N:start]` to walk while
  `has_more`. Read-only, no command lock. The window is a Python-style
  slice (`[-40:]` is the newest 40; a negative bound counts from the end).
- `session.history_page` cuts only at user messages, so a tool call is never
  separated from its result (which would render as `interrupted`) and a
  turn's footer stays with it; a window that starts mid-turn grows back to
  that turn's user message.
- Why slim is enough: a collapsed activity block draws shapes, counts,
  LOC/usage totals, so it needs names, statuses and the head of the
  reasoning — not the text. Expanding the block fetches its rounds' headers
  (`GET …/messages/{index}?content=false`: reasoning plus tool arguments, no
  output); expanding a row fetches the output. `Transcript.applyBody` fills
  the items in place; `app.hydrate` dedupes a round's in-flight requests.
- `LogView.prepend` builds a page in a detached `LogView` and moves the
  nodes in above the transcript (nothing on screen is rebuilt; the live
  block keeps streaming), correcting `scrollTop` by the page's height so the
  reader stays put — or pinning to the bottom for a reader who never scrolled
  away from it. `onScroll` calls `loadEarlier` within 600px of the top.
- `history_replaced` still carries the full snapshot: `/undo`, compaction
  and `/retry` hand every client the rewritten transcript whole. The
  service-worker cache name moved to `xarness-ui-v34`.

## Body round: the call's content, not a card about it

The expanded row used to be a stack of labelled boxes — `arguments`,
`output`, `thought`, `content` — each an uppercase caption over a
rounded, bordered, surface-filled `<pre>`, indented 25px inside the row.
Three signals at once read as a card that could not decide what it was, the
captions repeated what the row already said (`output` above an output,
`thought` directly under "Thought"), and on a phone the indent plus the
card's padding plus its border spent ~40px of a 390px line before a single
glyph.

- **No captions.** `output` and `thought` said nothing the row did not; the
  argument JSON of an MCP call is dimmed instead of titled, and `content`
  is a diff. The body is just the content now.
- **No card.** The surface fill, the 1px border and the 3px radius are gone.
  The body hangs in the name column (25px, `--row-icon-w` + `--row-gap`, so
  it follows the mobile gap) and monospace plus colour do the framing. The
  row connector above is the only vertical line in a block: the thinking
  body and the live preview each used to draw their own `border-left`, a
  hair to the right of the connector — that was the second line.
- **Diffs are diffs.** `edit_file`'s result (an ack line plus a git patch) is
  split at `diff --git`, the ack dropped, and the patch handed to
  `diffTree`: old/new line numbers, a tinted `@@` header, and add/del rows
  with their backgrounds, lexed for the file type. While the call streams,
  the salvaged `edits` become a provisional `-old`/`+new` patch. `write_file`
  shows its content as `+` lines in the add tint, with the `created …`
  ack demoted to a note underneath.
- **run_bash reads as a terminal.** The command is the prompt line
  (`$ ls webui && npm test`, the prompt in the success colour), the output
  follows as-is, and the `[exit N]` the tool always appends becomes a
  one-line verdict — green `✓ exit 0`, red `✗ exit 2`. A failed call still
  renders the same way instead of falling back to raw JSON-in/pre-out, and a
  command that printed a patch (`git diff`) gets the diff view in place of
  its text. `read_file` lexes its result by extension, lifting the
  `[showing lines …]` footers out of the code; arguments are only shown for
  tools the UI has no presentation for.
- The dead transcript rules (`tool-head`, `tool-dot`, `tool-detail`,
  `tool-body`, `pill`, `thinking-body`) went with them; `.tool-name`'s
  weight moved into the `/tools` sheet's own rule. The service-worker cache
  name moved to `xarness-ui-v47`.

## Popup round: dialogs that belong to the same surface

The transcript rows got their idiom last round; the popups behind `/` had
not. They were an old layer: a palette that was a card inside a card, a
Cancel bar on every picker, and a help sheet drawn with hand-padded
monospace columns — the visual equivalent of aligning a table with spaces.

- **The palette is a band, not a card.** It carried `background: surface`
  and a 13px top radius inside the composer panel's 14px one, and each row
  was an 8px-radius filled box: three nested rounded rectangles for a list
  of commands. The fill and radius are gone — the band shows the panel it
  sits in, closed by one hairline — and rows are full-bleed with a flat
  background on hover. No left bar on the selected row: it was the same
  second-line mistake as round 1's duplicated connector. Selection is the
  row's fill plus its brighter name, nothing else.
- **Two columns, one grid.** The command and its description used to be a
  flex row, so every description started at a different x. The row is now a
  grid with a fixed `10ch` command column (measured in the mono font it
  renders, so the column really is ten characters), and a file mention —
  which has no description — takes the full width instead.
- **The foot teaches the call, not the row.** It repeated the selected
  row's description 30px under the same description. It now shows the
  usage line (`/filter [add|remove] <path>`) for the command under the
  cursor, plus the keys that work while the palette is open; the `args`
  column left the row entirely, where on a phone it was hidden anyway.
  A command that takes no arguments collapses the strip.
- **Pickers dismiss one way.** Every picker had a header ✕, a backdrop, an
  Esc handler *and* a full-width Cancel button — the loudest control in the
  sheet repeated the quietest three. The button is gone; the ✕ is 44px on a
  phone. Info sheets keep a Close button (thumb reach) but no longer paint it
  in the accent colour, which made "close" look like the main action.
- **The current row is readable in both themes.** `aria-current` only
  changed the name's colour, and in the light theme `--highlight` (#000) and
  `--text` (#1c1c1e) are nearly the same colour, so on the current session
  the marker vanished. It is now heavier as well as brighter, and the row
  you are on is painted selected from the moment the picker opens instead of
  waiting for the first pointer move.
- **The model row stopped eating its own name.** `Claude Sonnet 4.5` was
  squeezed to 45px — five characters — because the raw model id took 110px
  of a 345px row and rendered the same string in lower case. The id is gone
  below 560px; effort and vision stay, and the name has room.
- **Sections are structure.** An uppercase word floated above a gap. It now
  rides a hairline that runs to the sheet's edge (`::after`), so `live` and
  `saved` read as division rather than decoration. The sheet's head got the
  same hairline: the body scrolls under the title and there was nothing
  between them.
- **Help is a definition list.** The shortcut list was one `<pre>` whose
  columns were runs of spaces; at 390px the lines wrapped in the middle of
  the description and the alignment collapsed into noise. It is a `<dl>`
  grid now — keys in the mono face, descriptions in prose, columns sized by
  the widest key — and the command list is a second `<dl>` of `name`,
  `usage` (accent) and description. The notes are a paragraph instead of
  hand-wrapped `\n`, and the sentence about "every Thought for pill" is
  gone: that pill stopped existing last round.
- Redundant words went with the boxes: `/filter`'s remove rows said `✕` and
  `remove`; the field label stopped shouting in uppercase.

The service-worker cache name moved to `xarness-ui-v48`.

## Rail round: one stroke through the live tail

- The connector stopped at the last shape and the eight-line live tail below
  it started bare, so the thread the shapes draw broke at exactly the part
  that is still streaming. The tail now carries the same 1px stroke at the
  connector's x — `8px - (--row-icon-w + --row-gap)`, so it lands on the
  same pixel whether the phone gap (6px) or the desktop gap (8px) is in
  play. It is the block's rail continued, not a border on the tail's own
  edge: that edge was the second vertical line round 1 removed. When the
  tail is on screen the last row's stroke falls to the panel edge and the
  tail's top margin goes to 0, so the line is unbroken from the header to
  the last line of the tail.
- The block header no longer lights up under the pointer, and the block no
  longer draws a border when it opens. The header is a title, not a card:
  the shapes and the sticky header already say what is open, and a second
  surface behind the same text was noise. Row headers keep their hover —
  they are the thing you tap.

The service-worker cache name moved to `xarness-ui-v49`.

## Summary round: present tense while it runs

- The header said `Thinking` for a single thought even after it had finished,
  and `ran 1 tool` for a single call whether it was still running or not. A run
  of one now names its own state: `Thinking`/`Thought` and `Running`/`Ran`.
- A longer run tallied the activity in flight as if it were done, so `ran 2
  tools` appeared while the second call was still on the wire and `Thought 9
  times` counted a thought that had only just started. Only settled activities
  are tallied now; the last one joins the line as `Thinking`/`Running` and bumps
  its tally when it settles.

The service-worker cache name moved to `xarness-ui-v50`.

## Home round: a landing page instead of the last session

- A bare address used to reopen the most recent session, which is the wrong
  answer for a bookmark or a phone home-screen icon: it decided for you. The
  bare URL is now the home screen. `#s=<name>` still opens a conversation
  directly, and every attach writes that hash, so a reload of a chat still
  lands in the chat (a reload of the home stays on the home).
- The home screen carries what a landing page owes: **live** sessions (the
  ones the host is still hosting, with their state and the folder they are
  working in), **recent** ones (saved, newest first, with `all` when there are
  more than fit), and **folders** — the known workspaces from
  `GET /v1/workspaces`, plus whichever folder the draft will use when it is
  not in that list. A folder row selects the folder; it does not start a
  session, because the first message is what starts it. `another folder…`
  takes a path (the host creates it if it is missing).
- The draft's folder is now named instead of "draft": the topbar subtitle,
  the status tooltip, the sidebar's workspace row and a line on the home
  screen all show the path the first message will start in. It is the picked
  folder, else the newest session's workspace, else the most-used known one.
  The client sends exactly that path in `POST /v1/sessions`, so the label and
  the session cannot disagree.
- `/new` is the home screen too. It used to append "started a fresh chat" to
  an otherwise empty log, which removed the ascii banner; now the banner is
  the page, the notice is a toast, and the transcript stays empty until the
  first round lands.
- A detached page is not a disconnected one. The old state reuse showed the
  red `closed` dot while the host was perfectly reachable; it is a green
  `ready` now, and a notice addressed to a draft (a command that needs a
  session, a failed start) goes to a toast instead of a log nobody is
  looking at.
- The home screen is a document, not a dashboard: no cards, no tiles. The
  banner is a dim watermark, section labels stay in the uppercase mono of the
  sidebar, rows are full-bleed lines with a hover fill — the same vocabulary
  as the transcript, so the two surfaces do not look like different apps.

The service-worker cache name moved to `xarness-ui-v51`.

## Sidebar round: a column on the desktop, a drawer on the phone

- On a wide screen the sessions panel is now a real column: open by default,
  it pushes the conversation instead of covering it, with no scrim and no
  drop shadow. Collapsing it slides the column out and remembers the choice
  (`xarness.sidebar`); the phone always starts closed and closes the drawer
  after picking a session, a new chat, or a footer action.
- The drawer animates transform and scrim opacity (with `visibility` keeping
  the closed panel out of the tab order), fades a shadow in while open, and
  closes on Escape. `inert`/`aria-hidden` follow the state, focus moves to the
  close button on open and back to the menu button on close, and the menu
  button shows its expanded state.
- Styling follows the rest of the shell: the hamburger and close mark are
  1.6px stroke SVGs like send/stop, the header lines up with the topbar, the
  selected row gets an accent wash and a brighter name, the empty list has
  its own prose line, and the footer respects the bottom safe area.

The service-worker cache name moved to `xarness-ui-v52`.

## Floating-head round: no topbar, branches under their parent

- The full-width topbar is gone. The menu button, session name and folder sit
  at the top-left of the conversation over a short `bg`-to-transparent
  gradient; the connection dot stays at the right edge of that same line.
  The log and the home document pad past the 48px line, and sticky activity
  headers pin just below it instead of under it. The sidebar now starts at
  the top of the window, so its `sessions` head lines up with the menu.
- Sessions are drawn as a tree: a parent is followed immediately by its
  subagents, indented behind a 1px branch rail, instead of the children
  drifting through the flat recent list. Orphans whose parent is gone stay
  visible as roots. The same order drives the sidebar, the home lists and the
  `/sessions` picker; home parents carry a muted `2 agents` count and the
  recent list still limits itself by parent chats, so subagents do not crowd
  real sessions out.
- A subagent is named by its branch when open: `agent-3` in the title and
  `subagent · <folder>` in the subtitle, rather than the parent id leaking
  into both.

The service-worker cache name moved to `xarness-ui-v53`.

## Diff header count follows the scope chip

- The pending-diff header used to label itself from every row in the diff, so
  "Edited N files" behaved like the all-agents scope no matter which chip state
  was showing. The count is now taken from the same scope-aware set that feeds
  the +/− totals: this session's own rows while the chip reads `agent`, every
  session's rows while it is dimmed. Auto and drift rows stay in their own
  sections and never count, and the `· N ignored` suffix is unchanged.

The service-worker cache name moved to `xarness-ui-v54`.

## Tab badge: the page state on the icon

- The page keeps the artwork as its tab icon but adds a dot in the icon's
  bottom-right corner, drawn on a 64px canvas over `icons/icon-192.png` and
  swapped in through the `<link id="favicon">` as a PNG data URL. Nothing is
  added to the page itself: the tab is the one place a background session can
  still report from.
- The dot is gray at rest, the theme's `accent` while `ask_user` is waiting on
  an answer, and `success` green while a turn is `running`/`compacting`. It is
  painted from the same `status`/`ask` reads that drive the composer bars, so
  it holds its color instead of animating; a repaint only happens when the
  color, `data-theme` or `prefers-color-scheme` actually change.

The service-worker cache name moved to `xarness-ui-v55`.

## Pushed-list round: the host owns the row, the page only patches it

- `GET /v1/sessions` is a pure read now: no snapshot, no diff, no git. The
  pending +/− totals are cached on each hosted session and refreshed by the
  host off a workspace event, debounced into one `observe` + one `diff` per
  repo and fanned out to every session in it. The `stats` flag stays on the
  wire for older clients and does nothing. A poll that used to run one
  snapshot and N diffs under the workspace lock is now a dict walk.
- The list is pushed: `GET /v1/events` sends one `SessionsSnapshot`, then
  `session_changed` frames (one summary) and `sessions_removed` (names) as
  they happen — a create, a turn starting, a tool waiting, a pending ask,
  `clients_attached` moving, an unload or an eviction (both push the saved
  meta row: the file is still there, so the row only changes clothes), and a
  delete (which removes it). A reconnect always re-snapshots; there is no
  delta replay to get wrong. The web sidebar follows it and keeps the 5s
  poll as the fallback while the stream is down; the TUI's `/sessions`
  picker follows the same stream for exactly as long as the modal is open
  (closing it cancels the worker and the response), and neither pages a
  window any more — the stream carries the whole list, so the sidebar's
  scroll trigger and the picker's `on_end` fetch are gone.
- The `state` word is the host's: `saved` means "on disk, not hosted", and a
  hosted session reads `idle` — the web page's warm rows used to say `saved`.
  `hosted` is on the summary for clients that need the distinction without
  inferring it from `active`.
- Rows are patched, not rebuilt. The sidebar compares a structure signature
  (names, parents, order, live/saved grouping) and, when only a word, an age
  or a +/− total changed, rewrites that row's nodes in place. The home
  document's signature is structure only, so a stats tick no longer rebuilds
  every row. Picking a row paints the pushed title/status before the stream's
  first frame lands.
- Hovering no longer opens anything. The sidebar used to warm a row with
  `GET /v1/sessions/{name}?range=[-1:]`, and that request *hosts* the session:
  sweeping the list left a server view full of idle sessions nobody opened.
  A hover on a saved row now makes one read-only `GET …/summary`, which
  computes that row's own pending totals from the workspace's saved
  baseline — one scoped diff for this session and its saved subagents, never
  the repo-wide fan-out, cached five seconds per repo so a sweep costs one
  snapshot. Hosted rows get their totals from the pushed stream instead, and
  opening a session (a click) is the only thing that hosts it.
- The attach frame takes `slim=true`: headers and prose, empty tool bodies and
  arguments, hydrated per row as before. Loading a conversation was never the
  slow part; the frames are.
- MCP no longer gates anything on the read path. A session starts its servers
  in the background after create/load (`ensure_mcp` returns the shared task,
  never awaits it), so a broken server costs its `startup_timeout` in the
  background instead of a minute of "connecting". The first turn and `/tools`
  await that same task; a session with no server configured has nothing to
  await at all.

The service-worker cache name moved to `xarness-ui-v59`.

## Quiet round: the toggle belongs to the thing it opens

- **A block that is one activity has no list.** Tapping its header opened a
  one-row list that repeated the activity the header already named. The tap is
  the activity's now: open shows the whole thought, closed leaves just the
  header, and a tap while it runs closes the stream instead of stacking a row
  below it. It starts open while it streams and keeps whatever the reader left
  it as when it settles; a thought opened after it has settled reads from the
  top rather than from an old tail. The eight-line window is the body, and it
  gets `50dvh` once the stream is over.
- **A solo tool opens its own body, not a row around it.** A block with one
  call had the same two taps: header → one-row list → row → content. The header
  is the row now: once the call has settled, tapping it renders the terminal /
  file / patch exactly where the row's body would have been, on the live tail's
  rail and indent. While the call is still running the eight-line tail keeps
  the spot (it follows the output as it streams, which a rebuilt body could
  not do cheaply); a tool opened during the run swaps to its real body when it
  settles. The hidden list is still updated so hydration and line totals keep
  working.
- **The live tail no longer drags itself to the bottom.** The pin was only
  ever updated by a scroll event, and a programmatic/user scroll the browser
  had not reported yet read as "at the bottom" — one more reasoning line and
  the reader's place was gone. The pin is sampled right before the text is
  replaced. A scroll inside a nested box (that tail, a tool's output) also
  never reaches the log's own handler, so it could not take follow away
  either; a capturing listener on the log notices an upward scroll of a
  descendant and detaches, and the jump button says so.
- **An open row does not narrate itself.** A thinking row whose body is
  streaming no longer repeats its last line in the row header on every delta;
  the name and state stay, and the snippet comes back when the thought settles.
- **The session meta has no model name.** The model is in the composer foot
  and in `/model`; on a 280px row it was a third thing competing with the age
  and the folder. The list signature drops it too, so a model switch no longer
  rebuilds every row.
- **The diff bar is patched, not rebuilt.** A `diff_changed` frame used to
  throw away every row; now one node per row (keyed by kind + path + author) is
  reused, moved into order only when it is out of place, and rewritten in
  place. Scroll position, hover and focus survive an update, and folding or
  unfolding `auto`/`drift` with the reader scrolled down no longer jumps to the
  top.
- **The column is centered on the window, not on what is left of it.** Opening
  the desktop sessions panel used to re-center the log and the composer inside
  the narrower column, so the input slid sideways by half the panel. It now
  compensates by exactly that half (sampled frame by frame while the panel
  slides, since its real edge is what counts), and the CSS clamps the shift so
  a narrow window keeps an 8px gap to the panel instead of sliding under it.
- **The sidebar has no header of its own.** `sidebar-head` is gone: the first
  group line (`live`, or `saved` when nothing is live) is the header, at the
  height of the floating `main-head`, and it carries `+ new`. The menu button
  opens and closes the panel; the drawer still closes on the scrim, Esc, or
  picking a row, and focus lands on `+ new` when it opens on a phone.

The service-worker cache name moved to `xarness-ui-v60`.

## Quiet round: no bar, and nothing jumps under the pointer

- **There is no header bar.** `main-head` was an absolutely positioned strip
  inside `<main>`, so it only ever covered the conversation: opening the
  sessions panel left the title floating over the right-hand column and gave
  the panel a little band of its own. The menu and the title are now just a
  pinned cluster in the top-left corner of the page (the menu as a chip, the
  title as the panel's header), and the conversation runs to the top of the
  window on a desktop instead of being cut off below a band. On a phone there
  is no gutter for the controls to sit in, so the first line starts under them
  and later lines fade out beneath. The cluster keeps the pointer only on its
  controls, so the log stays reachable around it. With the panel collapsed on
  a desktop only the menu remains (the reference rail is a bare `☰`): the
  title is the panel's header, so it comes back with it instead of lying over
  the conversation. The title is capped at the panel's width, and the
  connection chip keeps the window's corner in both states instead of being
  pushed out of the panel by a long name (or teleporting across the window
  when the panel closes).
- **A narrow desktop still gives the first line room.** The column is centered
  on the window, so below about 976px its left edge (and the corner chip) would
  run under the pinned controls. Only in that range does the log
  start a header's height down, like the phone; on a wide window the column
  begins beside the controls and the conversation keeps the very top.
- **The panel is a surface, not an overlay.** It runs the full height of the
  window with a background of its own (`--sidebar-bg`, a little lighter than
  the page in dark and a little darker in light) and its edge continues from
  the bottom of the page to the very top — behind the pinned menu and title,
  which therefore read as the panel's own header. Its rows start under them.
- **The panel slides both ways.** The phone rule `.sidebar.open` repeats a
  `transform` transition, and its specificity beat the desktop panel's
  `margin-left` one: closing animated, opening snapped. The desktop override
  repeats the margin transition.
- **Sticky block headers pin flush at the top.** Sticky offsets are measured
  from the log's content box, so `top: 0` left the header floating a gutter
  lower with text sliding through the gap above it. `--pin-at` is where a
  header stops (under the pinned controls, never through them), `--log-top` is
  where the first line starts, and `--log-stick` is the difference — the value
  a sticky `top` needs — so a pinned header lands at `--pin-at` and an open
  row's header offsets from the block header the same way.
- **Tapping a block never moves it.** Anchoring does not reliably pick the
  header as the thing to hold when the list under it grows or shrinks, so
  expanding one shot the view up and collapsing one pulled its rows out from
  under the pointer. The tap records the header's position and the log's
  scroll is corrected by however far it drifted — both directions, not just
  the collapse case — and the correction is reported to the app so it is not
  mistaken for the reader leaving the bottom.
- **The jump button only appears when the reader really left.** A live tail
  that is cleared, hidden, or swapped for a shorter activity drops its own
  `scrollTop`, and the nested-scroll listener read that as an upward scroll:
  every new tool call or thought put "jump to latest" on screen while the
  reader was still at the bottom. A nested upward move only counts inside a
  gesture window (a wheel, touch, pointer or key press on the log); a tail
  being rebuilt is not one.
- **The split scope chip renders as `agent` again.** `replaceChildren` takes
  nodes, not an array: passing the two spans as one array stringified them
  into `[object HTMLSpanElement],[object HTMLSpanElement]` next to the diff
  totals.
- **A reachable host always ships the current app.** The worker fetches the
  shell with `cache: 'no-store'` and keeps its cache for offline only, so an
  open tab cannot keep running yesterday's `app.js` after a reload.

The service-worker cache name moved to `xarness-ui-v62`.

## Round: no status chip, and a dated history

- **The connection chip is gone from the corner.** `setConn` still keeps
  `connState` for reconnects and the watchdog, and it still reveals the token
  action when the host asks for one, but there is no dot or label in the
  header any more.
- **The menu is bare.** The chip background and the `aria-expanded` fill are
  gone; `☰` floats over the page with only the normal hover/focus feedback
  from `.icon-btn`. No colour change when the panel opens.
- **The title keeps its name and folder when the panel is closed.** The
  desktop rule that hid `.head-title` with the panel is removed. On a wide
  window the centered column starts past the cluster, so the conversation
  keeps the top; below 1340px, where the column would run under the pinned
  title, the collapsed log starts one header down (the same room the phone
  uses). The name is still capped at the panel's width and ellipsizes.
- **Saved sessions are filed by date.** The sidebar is `live` first, then
  `today`, `yesterday`, `last week`, `older`. Buckets are local calendar days,
  not elapsed hours, so they flip at midnight; a session uses
  `last_user_message_at` and falls back to `updated_at`. The bucket is part
  of the list's structure signature, so the DOM is rebuilt when a session
  crosses midnight. The sessions picker uses the same five headings.
- **The diff panel runs to the top too.** `diffview` was still offset by
  `--top-h`, a leftover from the header bar. It is now `top: 0` on both the
  desktop side sheet and the phone full-width sheet, with the safe-area inset
  moved into its own header so the close button clears the notch.
- **The queued bar sits over the composer.** `.composer > .steerbar` also
  matched the old `margin: 8px 0` shorthand, which out-specified the shared
  composer-column rule and reset its centered left margin to zero, so the bar
  stuck to the left edge while the input was centered. It now sets only the
  vertical margins, so the steer (and ask) bars share the column's width and
  offset, directly above the input.

The service-worker cache name moved to `xarness-ui-v63`.

## Round: scrolling away sticks, and nothing slides over a pinned header

- **A gesture up detaches at once.** `selfScrollAt` is stamped by every follow
  jump, and the 150ms "this scroll was ours" window therefore never expired
  while output streamed: the reader's wheel was swallowed and a jump already
  queued by the last delta dragged the log back down. A wheel, touch or key
  gesture upward now clears the follow in the handler itself, and the window
  only ignores scrolls that no gesture is asking for, so a flick back to the
  bottom still re-attaches mid-stream.
- **A tool's output no longer swallows the wheel.** The in-conversation scroll
  boxes (`.tool-code`, `.tool-cmd`, `.tool-diff`, `.activity-lines`) drop
  `overscroll-behavior: contain`: at their edge the scroll chains to the log
  instead of stopping dead.
- **A pinned header pins where the content starts.** Sticky offsets are measured
  from the log's content box, so pinning at `--pin-at` left a band of the log's
  own top padding scrolling above the header: a block's rows slid over the
  pinned summary that sits below them. `--pin-at`/`--log-stick` are gone. The
  log's top edge is `--log-top` itself (a `top`, not padding), so scrolled
  content is clipped there and a sticky header pins flush at `0`, right under
  the floating controls, with nothing passing above it.
- **An opened block drops the wall of shapes.** Its rows already carry every
  shape in order; pinned, the wall would only stand in front of them and make
  the header two rows taller.

The service-worker cache name moved to `xarness-ui-v64`.
