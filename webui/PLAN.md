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
| attach / reconnect / rebuild | `GET /v1/sessions/{name}/events` (SSE; first frame `snapshot`, then `event`) |
| send message (no local echo) | `POST …/commands` `{type:"send_message"}` |
| draft chat → first session | `POST /v1/sessions` `{workspace?, model?, view}` (then attach) |
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
| `/diff`, file rows, `/diff <path>` | `GET …/diff`, `GET …/diff?path=&author=` |
| diff scope chip | `{type:"set_view_state", view:{diff_scope}}` |
| `/accept`, `/reject` (+ drift confirm) | `{type:"accept"}` / `{type:"reject"}` |
| `/sessions` | `GET /v1/sessions?stats=true` (polled while visible) |
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
  cannot bloat the DOM.
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
  index.html          shell: topbar, sidebar/drawer, log, composer, status bar
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
16px input text, ≥44px tap targets. The sidebar is a slide-over drawer
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
- **The header is the run's summary.** Two tallies only: how many times the
  model thought and how many tools it ran, so it reads `Thought 9 times, ran 6
  tools`. The per-tool breakdown lives in the expanded rows. Below the
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
  every activity as shape · name · arguments · state. The connector runs down
  from the bottom of the header (the header drops its tap min-height while
  open, so it hugs the title/shapes) and is carried by a line on each row, so
  it starts just under the shapes rather than floating beside the first row.
  Each row's shape paints above that connector line, so the line runs behind
  the icons instead of across them.
  Tapping a row opens its arguments/output, or the full thought for a thinking
  row. Rows are 44px tap targets on phones.
- **Changes in the block show at a glance.** When the run wrote to a file, the
  header carries `+12 -3` between the summary and the time: line totals
  summed over every successful `edit_file`/`write_file` in that block, coloured
  like the diff bar and with a sentence for the tooltip. A block that only read
  or ran commands shows nothing, and a failed edit is not counted.
- **Tokens too.** Next to it, muted, the block's spend: `↑3.4k ↓1.6k`. It sums
  two things: the **model round** that produced each item — `round_complete`
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
