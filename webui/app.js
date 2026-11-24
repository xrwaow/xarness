/** The xarness web UI: a thin client of the host API, like the TUI. It renders
 * one session's SSE stream and sends commands. State lives in `state.js`, the
 * DOM in `render.js`, HTTP in `api.js`. */
import { api, ApiError } from './api.js';
import { byId, el, replace, toast } from './dom.js';
import { LogView, BANNER } from './render.js';
import { diffTree } from './diff.js';
import { Transcript, formatTokens } from './state.js';
import { COMMANDS, findCommand } from './commands.js';
import { showConfirm, showInfo, showPicker, showPrompt, sheetSection } from './ui.js';
import { sha256Hex } from './sha256.js';

const RECONNECT_MIN = 250,
  RECONNECT_MAX = 4000;
const INTERRUPT_ARM_SECONDS = 2.5,
  SESSIONS_REFRESH_MS = 5000;
// A hovered saved row fetches its own pending totals (the listing never runs
// git). Debounced so sweeping the list does not fire a request per row, and
// cached per row: the same numbers stay valid for half a minute.
const HOVER_MS = 120,
  SAVED_STATS_TTL_MS = 30000;
const STALL_MS = 12000,
  WATCHDOG_MS = 4000,
  PROBE_TIMEOUT_MS = 5000,
  TICK_MS = 250;
const DETACH_PX = 40,
  ATTACH_PX = 8;
// How much history one attach carries. A long chat must open as fast as a
// short one: the host sends the newest window (whole turns) and older pages
// are fetched only as the reader reaches for them, at the host's page size.
const HISTORY_PAGE = 40,
  // How close to the top of the log (px) triggers the next older page.
  LOAD_EARLIER_PX = 600;
const STORAGE = {
  token: 'xarness.token',
  theme: 'xarness.theme',
  model: 'xarness.model',
  workspace: 'xarness.workspace',
  sidebar: 'xarness.sidebar',
};
const THEMES = {
  'ayu-darker': { mode: 'dark', swatches: ['#121212', '#bfbdb6', '#d2a6fe', '#aad84c', '#ef7177'] },
  'one-light': { mode: 'light', swatches: ['#fafafa', '#000000', '#7c4dff', '#50a14f', '#e45649'] },
};
const DIFF_BADGE = { mine: 'agent', agent: 'agent', auto: 'auto', drift: 'drift' };
/** A `<button>` row with one `<span>` per class, built on first sight. */
const diffRowNode = (node, cls, parts) =>
  node ?? el('button', { class: cls, type: 'button' }, ...parts.map((part) => el('span', { class: part })));
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
// Resolve a CSS custom property to a color the canvas can use.
const cssColor = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#8a8986';
// The tab icon is the artwork plus a badge: the page's own state at a glance
// from a background tab. Geometry is a fraction of the icon's square.
const FAVICON = { size: 64, dot: 0.135, ring: 0.045, cx: 0.79, cy: 0.8, backdrop: '#121212' };
const commandRow = (command) => ({
  id: command.name,
  label: `/${command.name}`,
  args: command.usage || '',
  detail: command.description,
});

// ---------------------------------------------------------------- /tools
const MCP_SCOPE_NEXT = { off: 'global', global: 'workspace', workspace: 'off' };
const MCP_SCOPE_WORD = { off: 'off', global: 'on (global)', workspace: 'on (workspace)' };

function shorten(text, limit = 60) {
  const clean = String(text ?? '')
    .replace(/\s+/g, ' ')
    .trim();
  return clean.length <= limit ? clean : `${clean.slice(0, limit - 1)}…`;
}
const toolsHead = (label, count) =>
  el(
    'div',
    { class: 'tools-head' },
    el('span', { class: 'tools-label', text: label }),
    count === null || count === undefined ? null : el('span', { class: 'tools-count', text: String(count) }),
  );

function toolItem(tool, showSource) {
  const source = tool.source || 'builtin',
    fromMcp = source !== 'builtin';
  const description = tool.description?.trim() || '';
  return el(
    'div',
    { class: 'tool-item', role: 'listitem', title: description || null },
    el(
      'div',
      { class: 'tool-row' },
      el('span', { class: 'tool-name', text: tool.name }),
      showSource && fromMcp ? el('span', { class: 'tool-src', dataset: { mcp: 'true' }, text: source }) : null,
    ),
    description ? el('p', { class: 'tool-desc', text: description }) : null,
  );
}

function mcpToolRow(server, tool, act, connected) {
  const description = tool.description?.trim() || '';
  return el(
    'button',
    {
      class: 'mcp-tool',
      type: 'button',
      dataset: { on: String(!!tool.enabled), focus: `tool:${server.name}:${tool.name}` },
      disabled: !connected,
      'aria-pressed': String(!!tool.enabled),
      title: description || null,
      on: {
        click: () =>
          act({ type: 'set_mcp_tool_enabled', server: server.name, tool: tool.name, enabled: !tool.enabled }),
      },
    },
    el('span', { class: 'mcp-check', 'aria-hidden': 'true', text: tool.enabled ? '[x]' : '[ ]' }),
    el('span', { class: 'mcp-tool-name', text: tool.name }),
  );
}

function mcpServerBlock(server, act) {
  const tools = server.tools || [],
    connected = server.status === 'connected';
  const next = server.status === 'unavailable' ? server.scope : MCP_SCOPE_NEXT[server.scope] || 'off';
  const note =
    server.error ||
    (server.status === 'disabled'
      ? 'click the name to turn it on here'
      : server.status === 'unavailable'
        ? 'click the name to start it again'
        : '');
  return el(
    'div',
    { class: 'mcp-server' },
    el(
      'button',
      {
        class: 'mcp-head',
        type: 'button',
        dataset: { focus: `server:${server.name}` },
        title: `click to switch to ${MCP_SCOPE_WORD[next] || next}`,
        on: { click: () => act({ type: 'set_mcp_server_scope', name: server.name, scope: next }) },
      },
      el('span', { class: 'mcp-dot', dataset: { status: server.status }, 'aria-hidden': 'true' }),
      el('span', { class: 'mcp-name', text: server.name }),
      el('span', {
        class: 'mcp-scope',
        dataset: { scope: server.scope },
        text: MCP_SCOPE_WORD[server.scope] || server.scope,
      }),
      connected
        ? el('span', { class: 'mcp-count', text: `${tools.length} ${tools.length === 1 ? 'tool' : 'tools'}` })
        : null,
      el('span', { class: 'mcp-target', text: server.target ? shorten(server.target, 80) : '' }),
    ),
    note ? el('div', { class: 'mcp-note', dataset: { error: String(!!server.error) }, text: note }) : null,
    tools.length
      ? el('div', { class: 'mcp-tools' }, ...tools.map((tool) => mcpToolRow(server, tool, act, connected)))
      : null,
  );
}

function toolsSheetBody(info, act) {
  if (info.error) return [el('div', { class: 'sheet-note', text: info.error })];
  const nodes = [toolsHead('available tools', info.tools.length)];
  if (!info.tools.length)
    nodes.push(el('div', { class: 'sheet-note', text: 'no tools yet — send a message to start the session' }));
  else {
    const mixed = info.tools.some((tool) => (tool.source || 'builtin') !== 'builtin');
    nodes.push(el('div', { class: 'tools-list', role: 'list' }, ...info.tools.map((tool) => toolItem(tool, mixed))));
  }
  nodes.push(toolsHead('mcp servers', info.servers.length));
  if (!info.servers.length) nodes.push(el('div', { class: 'sheet-note', text: 'none configured' }));
  else nodes.push(el('div', { class: 'mcp-servers' }, ...info.servers.map((server) => mcpServerBlock(server, act))));
  nodes.push(el('div', { class: 'sheet-hint', text: 'click a server to cycle its scope; click a tool to toggle it' }));
  return nodes;
}

function relativeTime(iso) {
  if (!iso) return '?';
  const stamp = Date.parse(iso);
  if (Number.isNaN(stamp)) return '?';
  const seconds = Math.max(0, (Date.now() - stamp) / 1000),
    days = Math.floor(seconds / 86400);
  if (days) return `${days}d`;
  if (seconds < 60) return `${Math.max(1, Math.floor(seconds))}s`;
  const minutes = Math.floor(seconds / 60);
  return minutes < 60 ? `${minutes}m` : `${Math.floor(minutes / 60)}h`;
}
const innermostDir = (path) => (path ? path.replace(/\/+$/, '').split('/').pop() || path : '');
const shortName = (name) => (!name ? '' : name.includes('/') ? name.split('/').slice(-2).join('/') : name);
const byRecent = (a, b) =>
  (Date.parse(b.last_user_message_at || b.updated_at || 0) || 0) -
  (Date.parse(a.last_user_message_at || a.updated_at || 0) || 0);
const sessionLabel = (entry) => (entry.parent_id ? `↳ ${entry.name.split('/').pop()}` : entry.name);
const sessionWhen = (entry) => relativeTime(entry.last_user_message_at || entry.updated_at);
const bySectionThenRecent = (a, b) => (a.active !== b.active ? (a.active ? -1 : 1) : byRecent(a, b));
const startOfDay = (d, days = 0) => new Date(d.getFullYear(), d.getMonth(), d.getDate() - days).getTime();
/** Which heading a saved session falls under: local calendar days, not elapsed
 * hours, so "today" flips at midnight. */
function sessionBucket(entry, now = Date.now()) {
  const stamp = Date.parse(entry.last_user_message_at || entry.updated_at || 0) || 0,
    day = new Date(now);
  if (!stamp) return 'older';
  if (stamp >= startOfDay(day)) return 'today';
  if (stamp >= startOfDay(day, 1)) return 'yesterday';
  return stamp >= startOfDay(day, 7) ? 'week' : 'older';
}
const SESSION_BUCKETS = [['today', 'today'], ['yesterday', 'yesterday'], ['week', 'last week'], ['older', 'older']];
const SESSION_BUCKET_LABELS = Object.fromEntries(SESSION_BUCKETS);
/** Parent sessions with their subagent children directly underneath. A child
 * whose parent is no longer in the list stays a root, so it is never lost. */
function sessionGroups(sessions) {
  const known = new Set(sessions.map((entry) => entry.name)),
    children = new Map();
  for (const entry of sessions) {
    if (!entry.parent_id || !known.has(entry.parent_id)) continue;
    const list = children.get(entry.parent_id) || [];
    list.push(entry);
    children.set(entry.parent_id, list);
  }
  for (const list of children.values()) list.sort(byRecent);
  const seen = new Set(),
    groups = [];
  const walk = (root, rows = [], depth = 0) => {
    if (seen.has(root.name)) return;
    seen.add(root.name);
    rows.push({ entry: root, depth });
    for (const child of children.get(root.name) || []) walk(child, rows, depth + 1);
    return { root, rows };
  };
  // Orphans (their parent is not listed) stay visible as roots.
  const roots = sessions.filter((entry) => !entry.parent_id || !known.has(entry.parent_id));
  roots.sort(bySectionThenRecent);
  for (const root of roots) groups.push(walk(root));
  // Defensive: a cycle in the parent links still shows every session once.
  for (const entry of sessions) {
    if (!seen.has(entry.name)) groups.push(walk(entry));
  }
  return groups;
}
/** The fields that make a list need rebuilding (rather than patched in
 * place). Each surface adds the fields it draws. */
const groupShape = (groups, fields) =>
  groups.map((group) => [
    group.root.active,
    group.rows.map(({ entry, depth }) => [entry.name, depth, ...fields(entry, group)]),
  ]);

class App {

  constructor() {
    Object.assign(this, {
      transcript: new Transcript(),
      status: null,
      // The listing row of the session being attached, shown until its
      // snapshot lands (see viewStatus).
      pending: null,
      session: null,
      config: null,
      sessions: [],
      workspaces: [],
      // Bumped by every poll so a slow older response cannot replace a newer
      // list (see pollSessions).
      sessionsSeq: 0,
      connState: 'connecting',
      wide: window.matchMedia('(min-width: 768px)'),
      workspace: localStorage.getItem(STORAGE.workspace) || null,
      draftMode: 'write',
      draftModel: localStorage.getItem(STORAGE.model) || null,
      attachments: [],
      diff: null,
      diffOpen: false,
      diffPatch: null,
      diffExpanded: new Set(),
      // One DOM node per diff row, keyed by what it shows (see renderDiffBar).
      diffNodes: new Map(),
      diffScope: 'all',
      diffScopePending: null,
      ask: null,
      faviconCanvas: null,
      faviconBase: null,
      faviconKey: null,
      streamAbort: null,
      attachSeq: 0,
      lastActivity: 0,
      stalled: false,
      commandQueue: [],
      draining: false,
      pendingItem: null,
      pendingSeq: 0,
      phase: 'idle',
      historyOffset: 0,
      hasEarlier: false,
      loadingEarlier: false,
      bodyPending: new Set(),
      hintItems: [],
      interruptArmed: false,
      interruptTimer: null,
      paletteRows: [],
      paletteNodes: [],
      paletteIndex: 0,
      paletteKind: null,
      atTimer: null,
      atSeq: 0,
      followOutput: true,
      userScrollUntil: 0,
      selfScrollAt: 0,
      scrollQueued: false,
      lastNotice: '',
      lastNoticeAt: 0,
      homeSig: null,
      sessionNodes: new Map(),
      homeNodes: new Map(),
      listAbort: null,
      listLive: false,
      // Saved rows' totals, fetched one row at a time on hover (the listing
      // never runs git): name -> when it was fetched (ms).
      savedStats: new Map(),
      hoverTimer: null,
      log: byId('log'),
      home: byId('home'),
      input: byId('input'),
      diffView: byId('diffview'),
    });
    this.view = new LogView(this.log, {
      onUndoTo: (id) => this.undoTo(id),
      resolveImage: (image) => this.resolveImage(image),
      onNeedBody: (item, content) => this.hydrate(item, content),
      // A block that keeps its header still while it expands moves the log
      // itself; that scroll is not the reader leaving the bottom.
      onSelfScroll: () => (this.selfScrollAt = performance.now()),
    });
  }
  // ---------------------------------------------------------------- boot
  async start() {
    const storedToken = localStorage.getItem(STORAGE.token);
    if (storedToken) api.setToken(storedToken);
    this.applyTheme(localStorage.getItem(STORAGE.theme));
    this.initSidebar();
    this.bindEvents();
    this.watchPointer();
    this.renderTitle();
    this.renderStatus();
    this.renderBars();
    this.initFavicon();
    this.syncAttachments();
    this.setConn('connecting');
    this.view.renderAll([]);
    // Timers and listeners go up before connecting: with a `#s=` URL
    // `connect()` starts an event stream that lives as long as the session
    // is attached, so anything awaited after it would never run and the
    // sidebar would stay empty until some other action kicked a poll.
    setInterval(() => this.tick(), TICK_MS);
    // The list is pushed (GET /v1/events); the poll is only the fallback
    // while that stream is down. The home screen's folder counts are not on
    // the wire, so they get a slow nudge from the same timer.
    setInterval(() => {
      if (this.listLive) {
        if (!this.session) void this.refreshWorkspaces().then(() => this.renderHome());
      } else {
        this.pollSessions();
      }
    }, SESSIONS_REFRESH_MS);
    setInterval(() => this.checkStalled(), WATCHDOG_MS);
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden && !this.listLive) this.pollSessions();
    });
    window.addEventListener('hashchange', () => {
      const name = this.sessionFromHash();
      if (name) {
        if (name !== this.session) this.attach(name);
      } else if (this.session) this.detach();
    });
    if ('serviceWorker' in navigator)
      navigator.serviceWorker.register('service-worker.js', { updateViaCache: 'none' }).catch(() => {});
    if (!this.coarsePointer?.matches) this.input.focus();
    await this.connect();
    void this.watchSessions();
    this.pollSessions();
  }

  async connect() {
    let backoff = RECONNECT_MIN;
    for (;;) {
      try {
        const health = await api.health();
        if (health.protocol_version < 20)
          this.addLine(`the host speaks protocol v${health.protocol_version}; this UI may not understand it`, 'warn');
        await this.loadConfig();
        this.setConn('closed');
        await this.openCurrent();
        return;
      } catch (error) {
        if (error instanceof ApiError && error.unauthorized) {
          this.setConn('unauthed');
          if (await this.promptToken()) continue;
          this.addLine('the host needs a token: press "token" in the sidebar', 'warn');
          return;
        }
        const wasOffline = this.connState === 'offline';
        this.setConn('offline');
        if (!wasOffline) this.addLine(`cannot reach the host: ${error.message} — retrying`, 'error');
        await sleep(backoff);
        backoff = Math.min(backoff * 2, RECONNECT_MAX);
      }
    }
  }

  async openCurrent() {
    // A bare URL is the home screen: running and recent sessions, and the
    // folders a new one can start in. Only `#s=` opens a conversation.
    const name = this.sessionFromHash();
    // `attach` is a long-lived reconnect loop, not a one-shot load: await it
    // and this function (and `connect`, and `start`) never returns. Every
    // other caller fires it and forgets it; do the same here.
    if (name) void this.attach(name);
    else if (!this.session) {
      // Fill the landing lists before the first paint so the home screen never
      // flashes empty on the way in.
      await this.pollSessions();
      this.detach();
    }
  }

  async loadConfig() {
    try {
      this.config = await api.config();
    } catch (error) {
      this.config = null;
      throw error;
    }
    if (this.draftModel && !this.config.models.some((m) => m.name === this.draftModel)) this.draftModel = null;
    this.renderStatus();
    this.renderInfo();
  }

  sessionFromHash() {
    const match = /[#&]s=([^&]+)/.exec(location.hash || '');
    if (!match) return null;
    try {
      return decodeURIComponent(match[1]);
    } catch {
      return null;
    }
  }

  setHash(name) {
    const hash = name ? `#s=${encodeURIComponent(name)}` : '';
    if (location.hash === hash) return;
    history.replaceState(null, '', `${location.pathname}${location.search}${hash}`);
  }
  // ------------------------------------------------------------ DOM refs
  setConn(state) {
    // No chip in the corner any more: the state is kept for reconnects, and the
    // only place it still surfaces is the token action.
    this.connState = state;
    byId('token-btn').hidden = !api.token && state !== 'unauthed';
  }
  // ------------------------------------------------------------- events
  bindEvents() {
    const on = (id, event, fn, opts) => byId(id).addEventListener(event, fn, opts);
    on('menu-btn', 'click', () => this.toggleSidebar());
    on('scrim', 'click', () => this.toggleSidebar(false));
    on('send', 'click', () => this.submit());
    on('stop', 'click', () => this.interrupt(true));
    on('jump', 'click', () => this.scrollToLatest());
    on('tools-btn', 'click', () => {
      this.openTools();
      this.closeDrawer();
    });
    on('model-btn', 'click', () => {
      this.pickModel();
      this.closeDrawer();
    });
    on('theme-btn', 'click', () => {
      this.pickTheme();
      this.closeDrawer();
    });
    on('help-btn', 'click', () => {
      this.openHelp();
      this.closeDrawer();
    });
    on('token-btn', 'click', () => {
      this.tokenMenu();
      this.closeDrawer();
    });
    this.input.addEventListener('input', () => {
      this.resizeInput();
      this.updatePalette();
    });
    this.input.addEventListener('keydown', (event) => this.onInputKey(event));
    this.input.addEventListener('paste', (event) => {
      const files = event.clipboardData?.files;
      if (files && files.length) {
        event.preventDefault();
        this.addFiles(files);
      }
    });
    on('log', 'scroll', () => this.onScroll(), { passive: true });
    // A scroll inside a nested box (a live tail, a tool's output) never reaches
    // the log's handler, but it is still the reader taking over. Only a real
    // gesture counts: a tail that is cleared or swapped for a shorter activity
    // also drops its scrollTop, and that is not the reader leaving.
    this.nestedScroll = new WeakMap();
    this.log.addEventListener(
      'scroll',
      ({ target: node }) => {
        if (node === this.log) return;
        const previous = this.nestedScroll.get(node);
        this.nestedScroll.set(node, node.scrollTop);
        if (previous > node.scrollTop + 1 && performance.now() < this.userScrollUntil) this.detachFollow();
      },
      { capture: true, passive: true },
    );
    // A gesture the reader drives. An upward one stops following right here:
    // waiting for the scroll event loses the race against the jump a
    // last-in-flight delta has already queued, and that jump would drag the
    // log back to the bottom under the reader's hands.
    const gesture = (event) => {
      if (event.type === 'touchstart') {
        this.touchY = event.touches[0]?.clientY ?? null;
        return this.userScrolled(false);
      }
      if (event.type === 'touchmove') {
        const y = event.touches[0]?.clientY ?? null,
          up = y !== null && y > (this.touchY ?? y) + 2;
        this.touchY = y;
        return this.userScrolled(up);
      }
      const up =
        (event.type === 'wheel' && event.deltaY < 0) ||
        (event.type === 'keydown' && ['ArrowUp', 'PageUp', 'Home'].includes(event.key));
      this.userScrolled(up);
    };
    for (const event of ['wheel', 'touchstart', 'touchmove', 'pointerdown'])
      this.log.addEventListener(event, gesture, { passive: true });
    this.log.addEventListener('keydown', (event) => {
      if (['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End', ' '].includes(event.key)) gesture(event);
    });
    document.addEventListener('keydown', (event) => this.onGlobalKey(event));
  }

  // ------------------------------------------------------------- sidebar
  initSidebar() {
    const stored = localStorage.getItem(STORAGE.sidebar);
    const open = this.wide.matches ? stored !== 'closed' : false;
    this.setSidebar(open, { focus: false, persist: false });
    const onChange = (event) => {
      const want = event.matches ? localStorage.getItem(STORAGE.sidebar) !== 'closed' : false;
      this.setSidebar(want, { focus: false, persist: false });
    };
    if (this.wide.addEventListener) this.wide.addEventListener('change', onChange);
    else this.wide.addListener(onChange);
    window.addEventListener('resize', () => this.syncColumnShift());
  }

  sidebarOpen() {
    return byId('sidebar').classList.contains('open');
  }

  toggleSidebar(open = null, options) {
    this.setSidebar(open === null ? !this.sidebarOpen() : open, options);
  }

  setSidebar(open, { focus = true, persist = true } = {}) {
    const sidebar = byId('sidebar'),
      main = byId('main'),
      menu = byId('menu-btn');
    // The menu lives in the main column, which is inert while the phone drawer
    // is open: make it focusable again before returning focus to it.
    if (!open) main.inert = false;
    // Move the pointer's focus out before the panel leaves the tab order, so
    // closing never strands focus on an invisible control.
    if (!open && sidebar.contains(document.activeElement)) menu.focus({ preventScroll: true });
    sidebar.classList.toggle('open', open);
    sidebar.inert = !open;
    sidebar.setAttribute('aria-hidden', String(!open));
    document.documentElement.dataset.sidebar = open ? 'open' : 'closed';
    byId('scrim').classList.toggle('open', open);
    main.inert = open && !this.wide.matches;
    menu.setAttribute('aria-expanded', String(open));
    if (persist && this.wide.matches) localStorage.setItem(STORAGE.sidebar, open ? 'open' : 'closed');
    const landing = sidebar.querySelector('.session-new') || sidebar.querySelector('.session-row') || sidebar;
    if (open && focus && !this.wide.matches) landing.focus({ preventScroll: true });
    this.watchColumnShift();
  }

  /** The panel takes a column of the window while the conversation stays
   * centered on the window itself: the column shifts by half the panel's real
   * edge (it is animated by a margin), so its center lands where it was. The
   * CSS clamps the shift when the remaining width cannot hold both. */
  syncColumnShift() {
    const root = document.documentElement,
      shift = this.wide.matches ? Math.max(0, byId('sidebar').getBoundingClientRect().right) / 2 : 0;
    if (shift) root.style.setProperty('--col-shift', `${Math.round(shift)}px`);
    else root.style.removeProperty('--col-shift');
  }

  /** Follow the panel's edge for the length of its transition (its margin
   * animates for 200ms; 40 frames is comfortably past that). */
  watchColumnShift(frames = 40) {
    cancelAnimationFrame(this.columnRaf);
    this.syncColumnShift();
    if (frames > 1) this.columnRaf = requestAnimationFrame(() => this.watchColumnShift(frames - 1));
  }

  closeDrawer() {
    if (!this.wide.matches && this.sidebarOpen()) this.setSidebar(false);
  }

  onGlobalKey(event) {
    if (event.key === 'Escape' && !this.wide.matches && this.sidebarOpen()) {
      event.preventDefault();
      this.setSidebar(false);
      return;
    }
    if (event.key === 'Escape' && this.diffPatch) {
      event.preventDefault();
      this.closePatch();
      return;
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault();
      this.openCommandPalette();
    }
  }

  watchPointer() {
    this.coarsePointer = window.matchMedia('(pointer: coarse)');
    const update = () => this.applyInputDevice();
    if (this.coarsePointer.addEventListener) this.coarsePointer.addEventListener('change', update);
    this.applyInputDevice();
    this.trackViewport();
  }

  applyInputDevice() {
    this.touchFirst = !!this.coarsePointer?.matches;
    this.input.setAttribute('enterkeyhint', this.touchFirst ? 'enter' : 'send');
  }

  insertNewline() {
    const start = this.input.selectionStart ?? this.input.value.length,
      end = this.input.selectionEnd ?? start;
    this.input.value = `${this.input.value.slice(0, start)}\n${this.input.value.slice(end)}`;
    this.input.selectionStart = start + 1;
    this.input.selectionEnd = start + 1;
    this.resizeInput();
    this.updatePalette();
  }

  onInputKey(event) {
    if (event.key === 'Escape' && !this.wide.matches && this.sidebarOpen()) {
      event.preventDefault();
      event.stopPropagation();
      this.setSidebar(false);
      return;
    }
    const paletteOpen = !byId('palette').hidden && this.paletteRows.length > 0;
    const touchFirst = this.coarsePointer ? this.coarsePointer.matches : this.touchFirst;
    if (event.key === 'Enter') {
      if (paletteOpen && !event.shiftKey) {
        event.preventDefault();
        this.acceptPalette();
        return;
      }
      if (event.ctrlKey || event.metaKey) {
        event.preventDefault();
        this.submit();
        return;
      }
      if (touchFirst && !event.shiftKey && !event.altKey) {
        event.preventDefault();
        this.insertNewline();
        return;
      }
      if (!event.shiftKey && !event.altKey) {
        event.preventDefault();
        this.submit();
        return;
      }
      return;
    }
    if (event.key === 'Tab') {
      if (paletteOpen) {
        event.preventDefault();
        this.completePalette();
        return;
      }
      event.preventDefault();
      this.toggleMode();
      return;
    }
    if (event.key === 'Escape') {
      if (this.diffPatch) {
        event.preventDefault();
        this.closePatch();
        return;
      }
      if (paletteOpen) {
        event.preventDefault();
        this.closePalette();
        return;
      }
      event.preventDefault();
      this.interrupt(false);
      return;
    }
    if (paletteOpen && (event.key === 'ArrowDown' || event.key === 'ArrowUp')) {
      event.preventDefault();
      this.movePalette(event.key === 'ArrowDown' ? 1 : -1);
      return;
    }
    if (this.ask && !this.input.value && (event.key === 'ArrowDown' || event.key === 'ArrowUp')) {
      const options = this.ask.questions[this.ask.index]?.options || [];
      if (options.length > 1) {
        event.preventDefault();
        this.ask.selected =
          (this.ask.selected + (event.key === 'ArrowDown' ? 1 : -1) + options.length) % options.length;
        this.renderAsk();
      }
    }
  }
  // -------------------------------------------------------------- theme
  applyTheme(name) {
    const theme = name && THEMES[name] ? name : null;
    if (theme) document.documentElement.dataset.theme = THEMES[theme].mode;
    else delete document.documentElement.dataset.theme;
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute('content', !theme || THEMES[theme].mode === 'dark' ? '#121212' : '#fafafa');
    // the tab badge takes its colors from the palette the theme just changed
    this.faviconKey = null;
    this.syncFavicon();
  }

  handleTheme(arg) {
    if (!arg) {
      this.pickTheme();
      return;
    }
    if (!THEMES[arg]) {
      this.addLine(`unknown theme '${arg}'; available: ${Object.keys(THEMES).join(', ')}`, 'warn');
      return;
    }
    localStorage.setItem(STORAGE.theme, arg);
    this.applyTheme(arg);
    this.renderStatus();
    toast(`theme: ${arg}`);
  }

  async pickTheme() {
    const current = localStorage.getItem(STORAGE.theme);
    const rows = [
      ...Object.entries(THEMES).map(([name, theme]) => ({
        id: name,
        label: name,
        swatches: theme.swatches,
        current: name === current,
      })),
      { id: 'auto', label: 'follow system', current: !current },
    ];
    const choice = await showPicker({ title: 'Color theme', rows });
    if (!choice) return;
    if (choice === 'auto') {
      localStorage.removeItem(STORAGE.theme);
      this.applyTheme(null);
      return;
    }
    this.handleTheme(choice);
  }
  // -------------------------------------------------------------- auth
  async promptToken() {
    const token = await showPrompt({
      title: 'Host token',
      label: 'Bearer token',
      value: api.token || '',
      placeholder: 'printed by `xarness --web`',
      note: 'The host requires it for API calls. Stored in this browser only.',
    });
    if (!token) return null;
    api.setToken(token);
    localStorage.setItem(STORAGE.token, token);
    byId('token-btn').hidden = false;
    return token;
  }

  async tokenMenu() {
    const rows = [
      { id: 'change', label: 'Change token', detail: api.token ? 'set' : 'none' },
      { id: 'clear', label: 'Forget token (this browser)' },
    ];
    const choice = await showPicker({
      title: 'Host token',
      text: api.token ? 'This browser sends the stored token with every API call.' : 'No token stored.',
      rows,
    });
    if (choice === 'change') {
      if (await this.promptToken()) await this.reconnect();
    } else if (choice === 'clear') {
      api.setToken(null);
      localStorage.removeItem(STORAGE.token);
      toast('token forgotten');
      await this.reconnect();
    }
  }

  async reconnect() {
    this.setConn('connecting');
    try {
      await api.health();
      await this.loadConfig();
      await this.openCurrent();
    } catch (error) {
      if (error instanceof ApiError && error.unauthorized) {
        this.setConn('unauthed');
        this.addLine('the host needs a token: press "token" in the sidebar', 'warn');
        return;
      }
      this.addLine(`still cannot reach the host: ${error.message}`, 'error');
      this.setConn('offline');
    }
  }
  // ---------------------------------------------------------- sessions
  async pollSessions() {
    if (document.hidden) return;
    // Fallback only: the live path is GET /v1/events. A slow earlier poll
    // must not replace a newer list.
    const seq = ++this.sessionsSeq;
    let sessions;
    try {
      sessions = await api.sessions();
    } catch {
      return; /* retry next poll */
    }
    if (seq !== this.sessionsSeq) return;
    this.setSessions(sessions);
    if (this.session) return;
    await this.refreshWorkspaces();
    this.renderHome();
  }

  /** Replace the listing and repaint both surfaces. The sequence bump makes
   * an in-flight poll that started earlier stale: a pushed frame is always
   * newer than a listing response. */
  setSessions(sessions) {
    this.sessionsSeq += 1;
    this.sessions = sessions;
    this.homeSig = null;
    const groups = sessionGroups(sessions);
    this.renderSessions(groups);
    this.renderHome(groups);
  }

  /** The host-wide session-list stream: one snapshot, then deltas. State
   * changes arrive as they happen instead of at the next poll. */
  async watchSessions() {
    let backoff = RECONNECT_MIN;
    while (!this.listAbort?.signal.aborted) {
      const controller = new AbortController();
      this.listAbort = controller;
      try {
        await api.watchSessions({
          signal: controller.signal,
          onOpen: () => {
            this.listLive = true;
            backoff = RECONNECT_MIN;
          },
          onSnapshot: (snapshot) => this.applySessionSnapshot(snapshot.sessions || []),
          onEvent: (event) => this.applySessionsEvent(event),
        });
        if (controller.signal.aborted) return;
        throw new ApiError('the host closed the session stream', null);
      } catch (error) {
        this.listLive = false;
        if (error?.name === 'AbortError' || controller.signal.aborted) return;
        // Unauthorized means connect() is already asking for a token (or
        // gave up); keep retrying quietly instead of spamming the log.
        await sleep(backoff);
        backoff = Math.min(backoff * 2, RECONNECT_MAX);
      }
    }
  }

  applySessionSnapshot(sessions) {
    this.setSessions(sessions);
    if (!this.session && !this.workspaces.length)
      void this.refreshWorkspaces().then(() => this.renderHome());
  }

  applySessionsEvent(event) {
    if (event?.type === 'session_changed' && event.summary) this.upsertSummary(event.summary);
    else if (event?.type === 'sessions_removed' && event.names) {
      const gone = new Set(event.names);
      this.setSessions(this.sessions.filter((entry) => !gone.has(entry.name)));
    }
  }

  /** Upsert one listing row from a pushed summary. Existing DOM is patched
   * in place; only a membership/order change rebuilds the lists. */
  upsertSummary(summary) {
    const index = this.sessions.findIndex((entry) => entry.name === summary.name);
    if (index >= 0) this.sessions[index] = summary;
    else this.sessions.push(summary);
    this.setSessions(this.sessions);
  }

  async refreshWorkspaces() {
    try {
      this.workspaces = await api.workspaces();
    } catch {
      /* folders are a nicety; the session list still stands */
    }
  }

  updateSessionRow(entry) {
    const node = this.sessionNodes.get(entry.name);
    if (!node) return false;
    node.dataset.active = String(!!entry.active);
    if (entry.name === this.session) node.setAttribute('aria-current', 'true');
    else node.removeAttribute('aria-current');
    this.patchRow(node, entry);
    const adds = node.querySelector('.session-meta .adds');
    if (adds) adds.textContent = `+${entry.additions}`;
    const dels = node.querySelector('.session-meta .dels');
    if (dels) dels.textContent = `-${entry.deletions}`;
    return true;
  }

  /** The live cells every session row shares (sidebar and home). */
  patchRow(node, entry) {
    const state = node.querySelector('.session-state');
    if (state) {
      state.textContent = entry.state;
      state.dataset.state = entry.state;
    }
    const age = node.querySelector('.age');
    if (age) age.textContent = sessionWhen(entry);
  }

  renderSessions(groups = sessionGroups(this.sessions)) {
    const host = byId('session-list');
    const hasDiff = (entry) => !!(entry.tracking && (entry.additions || entry.deletions));
    const signature = JSON.stringify([
      groupShape(groups, (entry) => [!!entry.parent_id, entry.workspace, hasDiff(entry)]),
      // the bucket changes at midnight even when the membership does not
      groups.map((group) => sessionBucket(group.root)),
    ]);
    if (host.dataset.signature === signature) {
      for (const entry of this.sessions) this.updateSessionRow(entry);
      return;
    }
    host.dataset.signature = signature;
    const top = host.scrollTop;
    replace(host);
    this.sessionNodes.clear();
    if (!groups.length) {
      host.appendChild(
        el(
          'div',
          { class: 'session-empty' },
          el('span', { text: 'no sessions yet — send a message to start one' }),
          this.newChatButton(),
        ),
      );
      return;
    }
    // The panel has no header of its own: the first group line doubles as one,
    // so the new-chat action sits on it instead of in a bar above the list.
    let headed = false;
    const render = (items, label) => {
      if (!items.length) return;
      const head = el('div', { class: `session-group${headed ? '' : ' first'}` }, el('span', { class: 'group-label', text: label }));
      if (!headed) head.appendChild(this.newChatButton());
      headed = true;
      host.appendChild(head);
      for (const group of items)
        for (const { entry, depth } of group.rows) {
          const row = this.sessionRow(entry, depth);
          this.sessionNodes.set(entry.name, row);
          host.appendChild(row);
        }
    };
    // Running sessions stay together at the top; everything else is filed by
    // when it was last used, so the list reads like a chat history.
    render(groups.filter((group) => group.root.active), 'live');
    const saved = groups.filter((group) => !group.root.active);
    for (const [key, label] of SESSION_BUCKETS)
      render(saved.filter((group) => sessionBucket(group.root) === key), label);
    host.scrollTop = top;
  }

  newChatButton() {
    return el('button', {
      class: 'session-new',
      type: 'button',
      text: '+ new',
      title: 'Start a fresh chat (/new)',
      on: { click: () => (this.newChat(), this.closeDrawer()) },
    });
  }

  sessionRow(entry, depth = 0) {
    const tracked = entry.tracking && (entry.additions || entry.deletions),
      state = entry.state;
    return el(
      'button',
      {
        class: `session-row${depth ? ' session-child' : ''}`,
        type: 'button',
        role: 'listitem',
        dataset: { name: entry.name, active: String(!!entry.active), depth: String(depth) },
        'aria-current': entry.name === this.session ? 'true' : null,
        title: entry.parent_id ? `${entry.name}\nsubagent of ${entry.parent_id}` : entry.name,
        on: {
          click: () => {
            this.attach(entry.name);
            this.closeDrawer();
          },
          pointerenter: () => this.hoverSaved(entry),
        },
      },
      el(
        'span',
        { class: 'row-top' },
        el('span', { class: 'session-name', text: sessionLabel(entry) }),
        el('span', { class: 'session-state', dataset: { state }, text: state }),
      ),
      el(
        'span',
        { class: 'session-meta' },
        el('span', { class: 'age', text: sessionWhen(entry) }),
        entry.workspace ? el('span', { class: 'ws', text: innermostDir(entry.workspace) }) : null,
        tracked && entry.additions ? el('span', { class: 'adds', text: `+${entry.additions}` }) : null,
        tracked && entry.deletions ? el('span', { class: 'dels', text: `-${entry.deletions}` }) : null,
        entry.parent_id ? el('span', { text: 'subagent' }) : null,
      ),
    );
  }

  async openSessions() {
    // The stream keeps the list current; fall back to one fetch when it is
    // down (the sheet is an explicit "show me the list", so it may as well
    // try rather than show what the last poll happened to return).
    let sessions = this.sessions;
    if (!this.listLive) {
      try {
        sessions = await api.sessions();
      } catch {
        /* keep the list we have */
      }
    }
    const groups = sessionGroups(sessions),
      rows = [];
    let shown = null;
    for (const group of groups) {
      const name = group.root.active ? 'live' : SESSION_BUCKET_LABELS[sessionBucket(group.root)];
      for (const { entry } of group.rows) {
        const section = name === shown ? null : name;
        shown = name;
        rows.push({
          id: entry.name,
          label: sessionLabel(entry),
          current: entry.name === this.session,
          section,
          meta: [
            { text: sessionWhen(entry), class: 'age' },
            { text: entry.state, class: 'state', state: entry.state },
          ],
        });
      }
    }
    const choice = await showPicker({
      title: 'Sessions',
      text: rows.length ? '' : 'no sessions yet',
      className: 'sessions-picker',
      rows,
    });
    if (choice && choice !== this.session) this.attach(choice);
  }
  /** A saved row's pending totals are not in the listing (that read never
   * runs git), so one read-only per-session summary fetches them for this
   * row alone and patches it in place. Nothing here hosts a session: a
   * hosted row is kept current by the pushed stream instead. */
  hoverSaved(entry) {
    if (!entry || entry.hosted) return;
    const fetched = this.savedStats.get(entry.name);
    if (fetched && Date.now() - fetched < SAVED_STATS_TTL_MS) return;
    clearTimeout(this.hoverTimer);
    this.hoverTimer = setTimeout(async () => {
      const row = this.sessions.find((item) => item.name === entry.name);
      if (!row || row.hosted) return;  // opened meanwhile: the stream owns it
      this.savedStats.set(entry.name, Date.now());
      let summary;
      try {
        summary = await api.sessionSummary(entry.name);
      } catch {
        this.savedStats.delete(entry.name);
        return;
      }
      const index = this.sessions.findIndex((item) => item.name === summary.name);
      if (index < 0 || this.sessions[index].hosted) return;
      this.sessions[index] = { ...this.sessions[index], ...summary };
      // The structure signature carries "has +/-" so the row is rebuilt with
      // the columns when the totals stop being zero, else patched in place.
      this.renderSessions();
    }, HOVER_MS);
  }

  // --------------------------------------------------------- attach/SSE
  async attach(name) {
    if (!name) return;
    this.abortStream();
    this.session = name;
    this.view.session = name;
    this.setView('chat');
    this.setHash(name);
    // The pushed listing row paints the header until the snapshot lands.
    this.status = null;
    this.pending = this.sessions.find((entry) => entry.name === name) || null;
    this.renderChrome();
    this.resetDiff();
    this.diffScopePending = null;
    this.renderDiffBar();
    this.setHistory();
    this.setConn('connecting');
    const seq = ++this.attachSeq;
    const stale = () => seq !== this.attachSeq || this.session !== name;
    let backoff = RECONNECT_MIN,
      announced = false;
    while (!stale()) {
      const controller = new AbortController();
      this.streamAbort = controller;
      this.stalled = false;
      this.touchStream();
      try {
        await api.stream(name, {
          signal: controller.signal,
          messageRange: `[-${HISTORY_PAGE}:]`,
          // Headers + prose first; a row's tool output/args hydrate when it
          // is opened (the history pages already work this way).
          slim: true,
          onOpen: () => {
            this.touchStream();
            this.setConn('open');
            backoff = RECONNECT_MIN;
            announced = false;
          },
          onSnapshot: (snapshot) => {
            this.touchStream();
            this.loadSnapshot(snapshot);
          },
          onEvent: (event) => {
            this.touchStream();
            this.handleEvent(event);
          },
        });
        if (stale()) return;
        throw new ApiError('the host closed the event stream', null);
      } catch (error) {
        if (stale()) return;
        if (error?.name === 'AbortError' && !this.stalled) return;
        if (error instanceof ApiError && error.unauthorized) {
          this.setConn('unauthed');
          if (await this.promptToken()) continue;
          return;
        }
        if (error instanceof ApiError && error.permanent) {
          toast(`cannot open ${name}: ${error.message}`);
          this.detach();
          return;
        }
        this.setConn('offline');
        if (!announced) {
          this.addLine(
            this.stalled
              ? 'lost the host — the event stream went quiet, reconnecting'
              : `connection lost: ${error.message} — reconnecting`,
            'warn',
          );
          announced = true;
        }
        this.stalled = false;
        await sleep(backoff);
        backoff = Math.min(backoff * 2, RECONNECT_MAX);
      }
    }
  }

  touchStream() {
    this.lastActivity = Date.now();
  }

  async checkStalled() {
    if (!this.session || this.connState !== 'open' || this.stalled) return;
    if (Date.now() - this.lastActivity < STALL_MS) return;
    try {
      await api.health({ signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
      this.touchStream();
    } catch {
      this.stalled = true;
      this.streamAbort?.abort();
    }
  }

  abortStream() {
    this.attachSeq += 1;
    this.stalled = false;
    if (this.streamAbort) {
      this.streamAbort.abort();
      this.streamAbort = null;
    }
  }

  detach() {
    this.abortStream();
    this.session = null;
    this.view.session = '';
    this.status = null;
    this.pending = null;
    this.setHash(null);
    this.resetDiff();
    this.ask = null;
    this.phase = 'idle';
    this.setHistory();
    this.pendingItem = null;
    this.hintItems = [];
    this.transcript = new Transcript();
    // No stream is open, but the host is still reachable: ready, not the red
    // "closed" the old detached state used.
    if (this.connState !== 'offline' && this.connState !== 'unauthed') this.setConn('ready');
    this.homeSig = null;
    this.renderSessions();
    this.renderChrome();
    this.renderHome();
    this.pollSessions();
  }

  newChat() {
    const previous = this.session;
    this.detach();
    this.view.renderAll([]);
    if (previous) toast('left the chat — it keeps running in the background');
    this.input.focus();
  }

  // ---------------------------------------------------------------- home
  setView(view) {
    byId('main').dataset.view = view;
  }

  /** The folder the first message of a draft will start the session in: the
   * one picked on the home screen, else the folder of the most recent
   * session, else the most-used known folder. */
  startWorkspace() {
    if (this.workspace) return this.workspace;
    const latest = this.sessions.reduce(
      (best, entry) => (!best || byRecent(entry, best) < 0 ? entry : best),
      null,
    );
    return latest?.workspace || this.knownFolders()[0]?.path || null;
  }

  knownFolders() {
    return [...this.workspaces].sort(
      (a, b) => b.session_count - a.session_count || a.path.localeCompare(b.path),
    );
  }

  setWorkspace(path) {
    this.workspace = path || null;
    if (path) localStorage.setItem(STORAGE.workspace, path);
    else localStorage.removeItem(STORAGE.workspace);
    this.homeSig = null;
    this.renderChrome();
    this.renderHome();
    // A draft's /diff panel belongs to the folder it was opened for.
    if (!this.session && this.diff) void this.refreshDiff();
    this.input.focus();
  }

  async pickWorkspacePath() {
    const path = await showPrompt({
      title: 'New session folder',
      label: 'Path',
      value: this.startWorkspace() || '',
      placeholder: '/home/you/code/project',
      note: 'The host creates the folder if it does not exist yet.',
    });
    if (path) this.setWorkspace(path);
  }

  // Every known folder, plus the one the draft will actually use when it is
  // not in that list (a picked path, or an older session's workspace).
  folderList(start = this.startWorkspace()) {
    const rows = new Map();
    for (const ws of this.knownFolders()) rows.set(ws.path, Math.max(rows.get(ws.path) || 0, ws.session_count));
    if (start) rows.set(start, rows.get(start) || 0);
    return [...rows].map(([path, count]) => ({ path, count }));
  }

  homeSection(label, rows, action = null) {
    return el(
      'section',
      { class: 'home-section' },
      el(
        'div',
        { class: 'home-head' },
        el('h2', { class: 'home-label', text: label }),
        action
          ? el('button', { class: 'home-more', type: 'button', text: action.label, on: { click: action.run } })
          : null,
      ),
      ...rows,
    );
  }

  /** One row of the home document: a mark, a name and line, and trailing
   * meta. Sessions and folders differ only in what those cells say. */
  homeRow({ cls = 'home-row', dataset, title, current, run, mark = null, name, sub, meta }) {
    return el(
      'button',
      {
        class: cls,
        type: 'button',
        dataset,
        title,
        'aria-current': current ? 'true' : null,
        on: { click: run },
      },
      mark === null ? null : el('span', { class: 'home-mark', 'aria-hidden': 'true', text: mark }),
      el(
        'span',
        { class: 'home-row-main' },
        el('span', { class: 'home-row-name', text: name }),
        sub ? el('span', { class: 'home-row-sub', text: sub }) : null,
      ),
      meta,
    );
  }

  homeSessionRow(entry, depth = 0, childCount = 0) {
    // The folder is what a person recognises a chat by; the host's opaque
    // session name stays on the second line (with the model) rather than
    // being the loudest thing on the page. Subagents sit under their parent
    // and name their branch instead of repeating the parent's folder.
    const child = !!entry.parent_id,
      name = child ? entry.name.split('/').pop() : sessionLabel(entry),
      folder = innermostDir(entry.workspace);
    return this.homeRow({
      cls: `home-row${child ? ' home-child' : ''}`,
      dataset: { name: entry.name, ...(child ? { depth: String(depth) } : {}) },
      title: `${entry.name}${entry.parent_id ? `\nsubagent of ${entry.parent_id}` : entry.workspace ? `\n${entry.workspace}` : ''}`,
      run: () => this.attach(entry.name),
      mark: child ? '↳' : null,
      name: child ? name : folder || name,
      sub: [child ? 'subagent' : name, entry.model].filter(Boolean).join(' · '),
      meta: el(
        'span',
        { class: 'home-row-meta' },
        !child && childCount
          ? el('span', { class: 'agents', text: `${childCount} agent${childCount === 1 ? '' : 's'}` })
          : null,
        el('span', { class: 'session-state', dataset: { state: entry.state }, text: entry.state }),
        el('span', { class: 'age', text: sessionWhen(entry) }),
      ),
    });
  }

  homeFolderRow(folder, start = this.startWorkspace()) {
    const selected = folder.path === start,
      count = folder.count ? `${folder.count} ${folder.count === 1 ? 'session' : 'sessions'}` : '';
    return this.homeRow({
      current: selected,
      title: folder.path,
      run: () => this.setWorkspace(folder.path),
      mark: selected ? '▸' : '',
      name: innermostDir(folder.path) || folder.path,
      sub: folder.path,
      meta: count ? el('span', { class: 'home-row-meta', text: count }) : null,
    });
  }

  /** Patch the live cells of the already-built home rows (state, age).
   * Called on every pushed summary whose structure did not change. */
  updateHomeRows() {
    for (const entry of this.sessions) {
      const node = this.homeNodes.get(entry.name);
      if (node) this.patchRow(node, entry);
    }
  }

  /** The landing view for a bare URL and for /new: what is running, what to
   * resume, and the folders a fresh session can start in. */
  renderHome(groups = null) {
    if (this.session) {
      this.setView('chat');
      return;
    }
    this.setView('home');
    groups ??= sessionGroups(this.sessions);
    const start = this.startWorkspace(),
      active = groups.filter((group) => group.root.active),
      recent = groups.filter((group) => !group.root.active);
    const sig = JSON.stringify({
      // Structure only: state, totals and ages are patched in place below,
      // so a pushed state change never rebuilds the home DOM.
      sessions: groupShape(groups, (entry, group) => [
        entry.parent_id,
        entry.workspace,
        entry.model,
        entry.tracking,
        group.rows.length,
      ]),
      workspaces: this.workspaces.map((ws) => [ws.path, ws.session_count, ws.active_sessions]),
      start,
    });
    if (sig === this.homeSig) {
      this.updateHomeRows();
      return;
    }
    this.homeSig = sig;
    this.homeNodes.clear();

    const rows = (items) =>
      items.flatMap((group) =>
        group.rows.map(({ entry, depth }) => {
          const node = this.homeSessionRow(entry, depth, group.rows.length - 1);
          this.homeNodes.set(entry.name, node);
          return node;
        }),
      );
    const left = [];
    if (active.length) left.push(this.homeSection('live', rows(active)));
    if (recent.length) {
      const shown = recent.slice(0, 6);
      left.push(
        this.homeSection(
          'recent',
          rows(shown),
          recent.length > 6 ? { label: 'all', run: () => this.openSessions() } : null,
        ),
      );
    }

    const folderRows = this.folderList(start).map((folder) => this.homeFolderRow(folder, start));
    folderRows.push(this.homeRow({
      cls: 'home-row home-other',
      run: () => this.pickWorkspacePath(),
      mark: '+',
      name: 'another folder…',
    }));

    const panel = el(
      'div',
      { class: 'home-inner' },
      el(
        'header',
        { class: 'home-brand' },
        el('pre', { class: 'home-art', 'aria-hidden': 'true', text: BANNER }),
        el('p', { class: 'home-tag', text: 'chat with your coding agent' }),
      ),
      el(
        'p',
        { class: 'home-start' },
        el('span', { text: 'first message starts a new session in' }),
        el('b', { text: start || 'the host\u2019s folder', title: start || '' }),
      ),
      el(
        'div',
        { class: 'home-grid' },
        left.length ? el('div', { class: 'home-column' }, ...left) : null,
        el('div', { class: 'home-column' }, this.homeSection('folders', folderRows)),
      ),
    );
    const scroll = this.home.scrollTop;
    replace(this.home, panel);
    this.home.scrollTop = scroll;
  }

  async deleteCurrent() {
    const name = this.session;
    if (!name) {
      toast('no session to delete');
      return;
    }
    const ok = await showConfirm({
      title: 'Delete this session?',
      text: 'The saved conversation is removed and a fresh chat starts. Your workspace files are left exactly as they are.',
      confirmLabel: 'Delete',
      danger: true,
    });
    if (!ok) return;
    try {
      await api.deleteSession(name);
    } catch (error) {
      this.addLine(`/delete failed: ${error.message}`, 'error');
      return;
    }
    this.newChat();
    toast('session deleted — your files are untouched');
    this.pollSessions();
  }

  loadSnapshot(snapshot, keepStatus = false) {
    this.status = snapshot.status || (keepStatus ? this.status : null);
    this.pending = null;
    // A snapshot is the newest window (an older host sends no fields: all of it).
    this.setHistory(snapshot.message_offset || 0, !!snapshot.has_more);
    this.view.renderAll(this.transcript.load(snapshot));
    this.phase = 'idle';
    this.pendingItem = null;
    this.hintItems = [];
    this.syncCompacting(this.status?.state || 'idle');
    this.renderChrome();
    this.syncAsk();
    this.refreshDiff();
    this.scrollToLatest();
    // Layout can still settle this frame (fonts, markdown, a taller composer):
    // re-pin, and fill the screen only if the window did not.
    requestAnimationFrame(() => {
      if (this.followOutput) this.scheduleScroll();
      if (this.log.scrollHeight <= this.log.clientHeight) void this.loadEarlier();
    });
  }

  /** Track which window of history is loaded; forget in-flight bodies. */
  setHistory(offset = 0, hasEarlier = false) {
    Object.assign(this, { historyOffset: offset, hasEarlier, loadingEarlier: false });
    this.bodyPending.clear();
  }

  /** Autoload: when the reader nears the top, fetch the page in front of the
   * window and splice it in above, keeping their place on screen. */
  async loadEarlier() {
    const name = this.session;
    if (!name || !this.hasEarlier || this.loadingEarlier) return;
    if (this.log.scrollTop > LOAD_EARLIER_PX) return;
    this.loadingEarlier = true;
    try {
      const page = await api.messages(name, {
          range: `[${Math.max(0, this.historyOffset - HISTORY_PAGE)}:${this.historyOffset}]`,
          slim: true,
        }),
        messages = page.messages || [];
      if (this.session !== name) return;
      const height = this.log.scrollHeight,
        top = this.log.scrollTop;
      this.view.prepend(
        this.transcript.prependMessages(messages, { base: page.start ?? 0, lazy: !!page.slim }),
      );
      this.historyOffset = page.start ?? 0;
      this.hasEarlier = !!page.has_more && messages.length > 0;
      // Keep the reader on the message they were looking at (or on the bottom,
      // for one who never scrolled away); the stamp makes this scroll ours.
      this.selfScrollAt = performance.now();
      this.log.scrollTop = this.followOutput
        ? this.log.scrollHeight
        : top + this.log.scrollHeight - height;
      byId('jump').hidden = this.followOutput;
    } catch (error) {
      if (this.session === name) this.addLine(`could not load earlier messages: ${error.message}`, 'warn');
      this.hasEarlier = false;
    } finally {
      this.loadingEarlier = false;
      requestAnimationFrame(() => void this.loadEarlier());
    }
  }

  /** Fetch a slim round's text; `content` false means row headers only.
   * A filled item is never asked for again, so only the round's parallel
   * requests need deduping. */
  async hydrate(item, content = false) {
    const name = this.session,
      index = item.msgIndex,
      key = `${index}:${content ? 1 : 0}`;
    if (!name || !(index >= 0) || this.bodyPending.has(key)) return;
    this.bodyPending.add(key);
    try {
      const body = await api.body(name, index, content);
      if (this.session !== name) return;
      for (const item of this.transcript.applyBody(index, body.messages || [], content))
        this.view.update(item);
    } catch {
      /* stale index (a rewrite) or deleted session: keep the placeholder */
    } finally {
      this.bodyPending.delete(key);
    }
  }

  handleEvent(event) {
    const type = event?.type;
    if (type === 'status_changed') {
      this.status = event.status;
      const pending = this.diffScopePending;
      if (pending && event.status.view?.diff_scope === pending) this.diffScopePending = null;
      else if (pending) this.status.view = { ...event.status.view, diff_scope: pending };
      this.renderStatus();
      this.renderBars();
      this.renderInfo();
      this.syncCompacting(event.status.state);
      this.syncAsk();
      return;
    }
    if (type === 'notice') {
      this.addLine(event.text, event.level === 'error' ? 'error' : event.level === 'warn' ? 'warn' : 'info');
      return;
    }
    if (type === 'message_added') {
      this.sync(this.transcript.apply(event));
      return;
    }
    if (type === 'history_replaced') {
      this.loadSnapshot(event.snapshot, true);
      this.addLine(this.historyText(event));
      return;
    }
    if (type === 'checkpoints_changed') {
      this.refreshDiff();
      return;
    }
    if (type === 'ask_resolved') {
      if (this.ask && this.ask.askId === event.ask_id) this.closeAsk();
      return;
    }
    if (type === 'diff_changed') {
      this.showDiffSummary(event.summary);
      return;
    }
    if (type === 'turn_ended') {
      this.sync(this.transcript.apply(event));
      this.setPhase('idle');
      this.clearHintLines();
      return;
    }
    if (type === 'session_deleted') {
      const name = this.session;
      this.detach();
      toast(`session ${name} was deleted`);
      this.pollSessions();
      return;
    }
    if (type === 'processing_started') {
      this.sync(this.transcript.apply(event));
      this.setPhase('pending');
      return;
    }
    if (type === 'reasoning_delta' || type === 'content_delta' || type === 'tool_call_started')
      this.setPhase('streaming');
    // `round_complete` closes the round in the reducer (it nulls the thought and
    // the answer), so the round's own output has to be read before applying it.
    const silentRound =
      type === 'round_complete' && this.transcript.assistant === null && this.transcript.thinking === null;
    this.sync(this.transcript.apply(event));
    if (type === 'round_complete') {
      this.clearHintLines();
      if (silentRound && !event.has_tool_calls) this.addLine('empty response from model', 'error');
      this.setPhase('idle');
      this.renderStatus();
    } else if (type === 'stream_error') {
      this.setPhase('idle');
      this.addLine(event.message, 'error');
      this.clearHintLines();
    }
  }

  historyText(event) {
    if (event.reason === 'compaction') {
      const prefix = event.automatic ? 'auto-compacted' : 'compacted';
      if (!event.before_tokens) return `${prefix}: nothing to compact yet`;
      const freed = Math.max(0, event.before_tokens - event.after_tokens),
        count = (value) => value.toLocaleString('en-US');
      return `${prefix}: ${count(event.before_tokens)} → ${count(event.after_tokens)} tokens (freed ${count(freed)})`;
    }
    if (!event.scope && !event.resend) return 'compaction undone: pre-compaction history restored';
    const note =
      {
        reverted: 'file edits reverted',
        no_edits: 'no file edits to revert',
        no_tracking: 'no change tracking for this workspace — file edits could not be reverted',
      }[event.note] || 'file edits reverted';
    return event.resend ? `retrying: ${event.scope} rolled back (${note})` : `undone: ${event.scope} removed (${note})`;
  }
  // -------------------------------------------------------- live phase
  setPhase(phase) {
    if (phase === 'pending' || phase === 'compacting') {
      const label = phase === 'compacting' ? 'Compacting' : 'Processing';
      if (this.pendingItem && this.pendingItem.label === label) return;
      this.dropPending();
      this.phase = phase;
      this.pendingSeq += 1;
      this.pendingItem = {
        id: -this.pendingSeq,
        kind: 'pending',
        label,
        hint: 'esc to interrupt',
        started: performance.now() / 1000,
      };
      this.transcript.items.push(this.pendingItem);
      this.view.update(this.pendingItem);
      this.scheduleScroll();
      return;
    }
    this.phase = phase === 'streaming' ? 'streaming' : 'idle';
    this.dropPending();
  }

  syncCompacting(state) {
    if (state === 'compacting') this.setPhase('compacting');
    else if (this.phase === 'compacting') this.setPhase('idle');
  }

  removeItem(item) {
    this.transcript.items = this.transcript.items.filter((entry) => entry !== item);
    this.view.remove(item);
  }

  dropPending() {
    if (!this.pendingItem) return;
    const item = this.pendingItem;
    this.pendingItem = null;
    this.removeItem(item);
  }

  addLine(text, level = 'info') {
    if (!text) return;
    if (this.lastNotice === text && Date.now() - this.lastNoticeAt < 1500) return;
    this.lastNotice = text;
    this.lastNoticeAt = Date.now();
    // There is no transcript to write to on the home screen: a notice about
    // a draft (or a command that needs a session) still has to be seen.
    if (!this.session) {
      toast(text);
      return;
    }
    this.sync(this.transcript.addLine(text, level));
  }

  clearHintLines() {
    for (const item of this.hintItems) this.removeItem(item);
    this.hintItems = [];
  }

  sync(items) {
    if (!items || !items.length) return;
    for (const item of items) this.view.update(item);
    this.scheduleScroll();
  }
  // ------------------------------------------------------------ scrolling
  /** The reader is steering: a downward gesture only opens the window in which
   * the log can re-attach at the bottom, an upward one stops following now. */
  userScrolled(up = false) {
    this.userScrollUntil = performance.now() + 400;
    if (up) this.detachFollow();
  }

  /** The reader went up: stop following the output (the jump button says so). */
  detachFollow() {
    if (!this.followOutput) return;
    this.followOutput = false;
    byId('jump').hidden = false;
  }

  onScroll() {
    const now = performance.now();
    const distance = this.log.scrollHeight - this.log.scrollTop - this.log.clientHeight;
    // A scroll we caused ourselves (the follow jump, an expanding block keeping
    // its header still) says nothing about the reader: only a gesture makes a
    // scroll inside that window count, so a flick back to the bottom still
    // re-attaches while output keeps streaming.
    if (now - this.selfScrollAt < 150 && !(now < this.userScrollUntil)) {
      byId('jump').hidden = this.followOutput;
      return;
    }
    if (now < this.userScrollUntil) {
      this.userScrollUntil = now + 400;
      if (this.followOutput && distance > DETACH_PX) this.followOutput = false;
      else if (!this.followOutput && distance <= ATTACH_PX) this.followOutput = true;
    } else if (!this.followOutput && distance <= ATTACH_PX) this.followOutput = true;
    if (this.followOutput) this.scheduleScroll();
    byId('jump').hidden = this.followOutput;
    if (this.log.scrollTop < LOAD_EARLIER_PX) void this.loadEarlier();
  }

  scheduleScroll() {
    if (!this.followOutput || this.scrollQueued) return;
    this.scrollQueued = true;
    requestAnimationFrame(() => {
      this.scrollQueued = false;
      if (!this.followOutput) return;
      this.selfScrollAt = performance.now();
      this.log.scrollTop = this.log.scrollHeight;
      byId('jump').hidden = true;
    });
  }

  scrollToLatest() {
    this.followOutput = true;
    this.userScrollUntil = 0;
    this.selfScrollAt = performance.now();
    this.log.scrollTop = this.log.scrollHeight;
    byId('jump').hidden = true;
  }

  trackViewport() {
    const root = document.documentElement;
    let settleTimer = 0,
      layoutBaseline = root.clientHeight || window.innerHeight;
    const apply = () => {
      const viewport = window.visualViewport,
        layout = root.clientHeight || window.innerHeight;
      const visible = viewport ? viewport.height + viewport.offsetTop : window.innerHeight;
      const shrunk = !!viewport && visible < layoutBaseline - 80;
      const keyboardOpen = document.activeElement === this.input || shrunk;
      root.dataset.keyboard = keyboardOpen ? 'open' : 'closed';
      if (!keyboardOpen) layoutBaseline = layout;
      const resized = keyboardOpen && layout < layoutBaseline - 80;
      const height = resized ? Math.max(layout, visible) : visible;
      root.style.setProperty('--app-h', `${Math.round(height)}px`);
      if (this.followOutput) this.scheduleScroll();
    };
    const settle = () => {
      clearTimeout(settleTimer);
      let left = 4;
      const step = () => {
        apply();
        left -= 1;
        if (left > 0) settleTimer = setTimeout(step, 100);
      };
      step();
    };
    const setKeyboard = (open) => {
      document.documentElement.dataset.keyboard = open ? 'open' : 'closed';
      settle();
    };
    this.input.addEventListener('focus', () => setKeyboard(true));
    this.input.addEventListener('blur', () => setKeyboard(false));
    if (window.visualViewport) {
      window.visualViewport.addEventListener('resize', apply);
      window.visualViewport.addEventListener('scroll', apply);
    }
    window.addEventListener('resize', apply);
    window.addEventListener('orientationchange', apply);
    apply();
    this.trackComposer(root);
  }

  // The composer floats over the bottom of the log, so the log needs to know
  // how tall it is: --composer-h drives both the log's bottom padding (room to
  // scroll the last item clear of it) and its negative margin underneath.
  trackComposer(root) {
    const composer = byId('composer');
    let current = -1;
    const apply = () => {
      const height = Math.max(0, Math.round(composer.offsetHeight));
      if (height === current) return;
      current = height;
      root.style.setProperty('--composer-h', `${height}px`);
      // The log's bottom padding just changed under the reader: staying
      // pinned is the whole point of following the output.
      if (this.followOutput) this.scheduleScroll();
    };
    if (typeof ResizeObserver === 'function') new ResizeObserver(apply).observe(composer);
    window.addEventListener('resize', apply);
    window.addEventListener('orientationchange', apply);
    apply();
  }

  tick() {
    this.view.tick(this.transcript.items);
    if (this.pendingItem) this.scheduleScroll();
    // Late layout (an image, a font, highlighted code) must not leave the
    // reader short of the bottom while they follow the output.
    else if (this.followOutput && this.log.scrollHeight - this.log.scrollTop > this.log.clientHeight)
      this.scheduleScroll();
  }
  // -------------------------------------------------------------- status
  get viewStatus() {
    if (this.status) return this.status;
    // Between a tap and its snapshot, the pushed listing row stands in: the
    // title, model, folder and state are known before the transcript is.
    const config = this.config,
      pending = this.pending,
      target = pending?.model || this.draftModel || config?.default_model || null;
    const entry = (config?.models || []).find((model) => model.name === target);
    return {
      state: pending?.state === 'compacting' ? 'compacting' : pending && !['idle', 'saved'].includes(pending.state) ? 'running' : 'idle',
      mode: this.draftMode,
      model_name: target || '',
      model_display_name: entry?.display_name || config?.default_model_display_name || '—',
      model_id: '',
      cot_strength: entry?.cot_strength || 'medium',
      max_context: entry?.max_context || 128000,
      supports_vision: entry?.supports_vision || false,
      sandbox_name: config?.default_sandbox || null,
      workspace: pending?.workspace || this.startWorkspace(),
      usage_total: { input_tokens: 0, output_tokens: 0 },
      last_usage: { input_tokens: 0, output_tokens: 0, reasoning_tokens: 0 },
      queued: [],
      pending_ask: null,
      clients_attached: pending?.clients_attached || 0,
      filter_list: [],
      refs: [],
      tools: [],
      view: { diff_scope: this.diffScope },
    };
  }

  /** The header, footer and sidebar's live values, in one pass
   * (``renderStatus`` refreshes the title). */
  renderChrome() {
    this.renderStatus();
    this.renderBars();
    this.renderInfo();
  }

  renderStatus() {
    const status = this.viewStatus;
    this.syncFavicon();
    const used = (status.last_usage?.input_tokens || 0) + (status.last_usage?.output_tokens || 0);
    const max = status.max_context || 0,
      percentLeft = max ? Math.max(0, Math.round(100 * (1 - used / max))) : 100;
    const totalIn = status.usage_total?.input_tokens || 0,
      totalOut = status.usage_total?.output_tokens || 0;
    const bar = byId('statusbar');
    replace(
      bar,
      el('button', {
        class: 'model',
        type: 'button',
        text: status.model_display_name || '—',
        title: `${status.model_name ? `${status.model_display_name} (${status.model_name})` : status.model_display_name || 'model'} — switch model`,
        on: { click: () => this.pickModel() },
      }),
      el('button', {
        class: 'mode',
        type: 'button',
        dataset: { mode: status.mode },
        text: status.mode,
        title:
          status.mode === 'write'
            ? 'write mode — click to switch to plan (read-only)'
            : 'plan (read-only) mode — click to switch to write',
        on: { click: () => this.toggleMode() },
      }),
      el(
        'span',
        { class: 'ctx', title: `context ${formatTokens(used)} of ${formatTokens(max)} · ${percentLeft}% left` },
        `ctx ${formatTokens(used)}/${formatTokens(max)}`,
        el('span', { class: 'wide', text: ` · ${percentLeft}% left` }),
      ),
      el(
        'span',
        { class: 'io', title: `total input ${formatTokens(totalIn)} · total output ${formatTokens(totalOut)}` },
        ` · ↑${formatTokens(totalIn)} ↓${formatTokens(totalOut)}`,
      ),
    );
    bar.title = [
      status.model_display_name,
      status.model_name,
      status.mode,
      `effort ${status.cot_strength || 'medium'}`,
      `ctx ${formatTokens(used)}/${formatTokens(max)} (${percentLeft}% left)`,
      `in ${formatTokens(totalIn)} · out ${formatTokens(totalOut)}`,
      status.workspace,
    ]
      .filter(Boolean)
      .join(' · ');
    this.renderTitle();
  }

  renderInfo() {
    const status = this.viewStatus;
    const rows = [
      ['session', shortName(this.session) || 'new (not started)'],
      ['workspace', status.workspace || 'host cwd'],
      ['model', `${status.model_display_name || '—'}${status.model_name ? ` (${status.model_name})` : ''}`],
      ['effort', status.cot_strength || 'medium'],
      ['mode', status.mode],
      ['sandbox', status.use_sandbox === false ? 'off' : status.sandbox_name || '—'],
      [
        'tokens',
        `in ${formatTokens(status.usage_total?.input_tokens || 0)} · out ${formatTokens(status.usage_total?.output_tokens || 0)}`,
      ],
      ['clients', String(status.clients_attached ?? 0)],
    ];
    if (status.filter_list?.length) rows.push(['filtered', status.filter_list.join(', ')]);
    if (status.refs?.length) rows.push(['refs', status.refs.map((ref) => ref.mount).join(', ')]);
    replace(
      byId('info'),
      el('dl', {}, ...rows.flatMap(([key, value]) => [el('dt', { text: key }), el('dd', { text: value })])),
    );
  }

  renderTitle() {
    const status = this.viewStatus,
      folder = status.workspace || '',
      dir = innermostDir(folder),
      subagent = this.session?.includes('/') ? this.session.split('/').pop() : null,
      name = this.session ? subagent || shortName(this.session) : 'new chat';
    document.title = `xarness ${this.session ? name : dir || 'new chat'}`;
    byId('title').textContent = name;
    byId('title').title = this.session || folder || 'new chat';
    // Attached, the subtitle is the folder's short name. A draft names the
    // folder its first message will start the session in, not the word
    // "draft": the status bar and the composer are already showing values.
    byId('subtitle').textContent = this.session
      ? subagent
        ? `subagent · ${dir || 'the host\u2019s folder'}`
        : dir
      : folder || 'the host\u2019s folder';
    byId('subtitle').title = subagent ? `${this.session} · ${folder}` : folder;
  }

  // ------------------------------------------------------------ tab badge
  /** The browser tab icon is the artwork plus a dot in its bottom-right
   * corner: gray at rest, green while a turn runs, and the theme's accent
   * while `ask_user` waits on an answer. It is the one status a background
   * tab can still show. */
  initFavicon() {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = FAVICON.size;
    this.faviconCanvas = canvas;
    this.faviconBase = new Image();
    // repaint with the artwork once it decodes; until then the badge sits on
    // the plain panel color
    this.faviconBase.onload = () => {
      this.faviconKey = null;
      this.syncFavicon();
    };
    this.faviconBase.src = 'icons/icon-192.png';
    // following the system theme repaints without a /theme command
    matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change', () => {
      this.faviconKey = null;
      this.syncFavicon();
    });
    this.faviconKey = null;
    this.syncFavicon();
  }

  syncFavicon() {
    const canvas = this.faviconCanvas;
    if (!canvas) return;
    const status = this.viewStatus,
      state = this.ask ? 'asking' : status.state === 'running' || status.state === 'compacting' ? 'running' : 'idle',
      theme = document.documentElement.dataset.theme || '',
      scheme = matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light',
      key = `${state}:${theme}:${scheme}`;
    if (key === this.faviconKey) return;
    this.faviconKey = key;
    const color = state === 'asking' ? cssColor('--accent') : state === 'running' ? cssColor('--success') : cssColor('--muted');
    this.paintFavicon(color);
  }

  /** Draw the artwork and the badge at FAVICON.size. The ring is the icon's
   * own backdrop, so the dot reads as attached rather than painted over. */
  paintFavicon(color) {
    const canvas = this.faviconCanvas,
      ctx = canvas?.getContext('2d'),
      size = FAVICON.size;
    if (!ctx) return;
    ctx.clearRect(0, 0, size, size);
    const base = this.faviconBase;
    ctx.imageSmoothingQuality = 'high';
    if (base?.complete && base.naturalWidth) ctx.drawImage(base, 0, 0, size, size);
    else {
      ctx.fillStyle = FAVICON.backdrop;
      ctx.beginPath();
      if (typeof ctx.roundRect === 'function') ctx.roundRect(0, 0, size, size, size * 0.187);
      else ctx.rect(0, 0, size, size);
      ctx.fill();
    }
    const cx = size * FAVICON.cx,
      cy = size * FAVICON.cy,
      r = size * FAVICON.dot;
    ctx.beginPath();
    ctx.arc(cx, cy, r + size * FAVICON.ring, 0, Math.PI * 2);
    ctx.fillStyle = FAVICON.backdrop;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    const link = document.getElementById('favicon');
    if (!link) return;
    link.type = 'image/png';
    link.href = canvas.toDataURL('image/png');
  }

  renderBars() {
    const status = this.viewStatus,
      running = status.state === 'running' || status.state === 'compacting';
    byId('stop').hidden = !running;
    this.syncFavicon();
    byId('inputbar').dataset.plan = String(status.mode === 'plan');
    this.input.placeholder = this.ask ? 'Type your answer…  (Enter: send · Esc: skip)' : 'Send a message…';
    this.renderSteer();
    this.renderAsk();
    this.renderDiffBar();
  }
  // ----------------------------------------------------------- steer queue
  renderSteer() {
    const bar = byId('steerbar'),
      queued = this.viewStatus.queued || [];
    if (!queued.length) {
      bar.hidden = true;
      replace(bar);
      return;
    }
    bar.hidden = false;
    replace(
      bar,
      ...queued.map((entry) => el('div', { class: 'steer-item', text: `› ${entry.text}`, title: entry.text })),
      el(
        'div',
        { class: 'steer-foot' },
        el('button', {
          class: 'btn btn-ghost',
          type: 'button',
          text: 'send now',
          title: 'Cancel the running turn and send every queued message as one',
          on: { click: () => this.flushQueued() },
        }),
        el('button', {
          class: 'btn btn-ghost',
          type: 'button',
          text: 'discard',
          on: { click: () => this.postCommand({ type: 'discard_queued' }) },
        }),
        el('span', { class: 'steer-note', text: `${queued.length} queued · sent at the next round` }),
      ),
    );
  }
  // ------------------------------------------------------------- ask tool
  syncAsk() {
    const pending = this.viewStatus?.pending_ask;
    if (!pending) {
      if (this.ask) this.closeAsk();
      return;
    }
    if (this.ask && this.ask.askId === pending.ask_id) return;
    this.ask = { askId: pending.ask_id, questions: pending.questions || [], index: 0, answers: [], selected: 0 };
    this.renderBars();
    this.input.focus();
  }

  closeAsk() {
    this.ask = null;
    this.renderBars();
  }

  renderAsk() {
    const bar = byId('askbar');
    if (!this.ask) {
      bar.hidden = true;
      replace(bar);
      return;
    }
    const { questions, index } = this.ask,
      question = questions[index] || { question: '', options: [] },
      options = question.options || [];
    bar.hidden = false;
    replace(
      bar,
      el('div', { class: 'ask-head', text: `Question ${index + 1}/${questions.length}` }),
      el('p', { class: 'ask-question', text: question.question || '' }),
      options.length
        ? el(
            'div',
            { class: 'ask-options', role: 'radiogroup' },
            ...options.map((option, i) =>
              el(
                'button',
                {
                  class: 'ask-option',
                  type: 'button',
                  role: 'radio',
                  'aria-checked': String(i === this.ask.selected),
                  on: { click: () => this.answerAsk(option) },
                },
                el('span', { class: 'num', text: `${i + 1})` }),
                el('span', { text: option }),
              ),
            ),
          )
        : null,
      el('div', {
        class: 'ask-hint',
        text: options.length
          ? '↑/↓ choose · Enter picks · or type your own answer · Esc skips'
          : 'Enter sends your answer · Esc skips',
      }),
    );
  }

  answerAsk(answer) {
    if (!this.ask) return;
    const value = String(answer ?? '').trim();
    if (!value) return;
    this.ask.answers.push(value);
    this.ask.index += 1;
    this.ask.selected = 0;
    if (this.ask.index >= this.ask.questions.length) this.finishAsk(this.ask.answers);
    else this.renderAsk();
  }

  finishAsk(answers) {
    const { askId } = this.ask;
    this.ask = null;
    this.renderBars();
    this.postCommand({ type: 'answer_ask', ask_id: askId, answers });
  }

  skipAsk() {
    if (this.ask) this.finishAsk(null);
  }
  // ------------------------------------------------------- pending changes
  resetDiff() {
    this.diff = null;
    this.diffOpen = false;
    this.diffPatch = null;
    this.renderDiffView();
  }

  async refreshDiff() {
    try {
      this.showDiffSummary(await this.loadDiff());
    } catch {
      this.showDiffSummary(null);
    }
  }

  /** One DiffInfo: the attached session's, or — in a draft chat — the
   * workspace's own (tracking is per repo root, so it needs no session). */
  loadDiff({ path, author } = {}) {
    if (this.session) return api.diff(this.session, { path, author });
    const workspace = this.startWorkspace();
    if (!workspace) return Promise.resolve(null);
    return api.workspaceDiff(workspace, { path, author, scope: this.diffScope });
  }

  showDiffSummary(info) {
    const empty = !info || (!info.files?.length && !info.groups?.length);
    this.diff = empty ? null : info;
    if (empty) {
      this.diffOpen = false;
      this.diffPatch = null;
      this.diffExpanded.clear();
    } else if (this.diffPatch) {
      const path = this.diffPatch.path;
      const listed =
        (info.files || []).some((file) => file.path === path) || (info.groups || []).some((group) => group.prefix === path);
      if (!listed) this.diffPatch = null;
    }
    if (this.status?.view?.diff_scope && !this.diffScopePending) this.diffScope = this.status.view.diff_scope;
    if (!this.diffPatch) this.renderDiffView();
    this.renderDiffBar();
  }

  /** The bar's chrome and rows are built once and patched from then on: a
   * `diff_changed` frame arrives on every file change, and rebuilding the list
   * under the reader threw away their place. One node per row (keyed by what
   * it shows) is reused and rewritten in place. */
  ensureDiffBar() {
    if (this.diffHead) return;
    this.diffLabel = el('span', { class: 'diff-label-text' });
    this.diffIgnored = el('span', { class: 'dim' });
    this.diffScopeEl = el('button', {
      class: 'diff-scope',
      type: 'button',
      title: 'Toggle whose changes this counts (every session / this session)',
      on: { click: (event) => (event.stopPropagation(), this.toggleDiffScope()) },
    });
    this.diffScopeSig = null;
    this.diffAdds = el('span', { class: 'adds' });
    this.diffDels = el('span', { class: 'dels' });
    this.diffHead = el(
      'div',
      {
        class: 'diff-head',
        role: 'button',
        tabindex: '0',
        on: {
          click: () => this.toggleDiff(),
          keydown: (event) => {
            if (event.target !== event.currentTarget || !['Enter', ' '].includes(event.key)) return;
            event.preventDefault();
            this.toggleDiff();
          },
        },
      },
      el('span', { class: 'diff-label' }, this.diffLabel, this.diffIgnored),
      el('span', { class: 'diff-spacer' }),
      this.diffScopeEl,
      this.diffAdds,
      this.diffDels,
    );
    this.diffList = el('div', { class: 'diff-list' });
    this.diffHint = el(
      'div',
      { class: 'diff-hint' },
      'Run ', el('span', { class: 'cmd', text: '/accept' }), ' to lock these in, ',
      el('span', { class: 'cmd', text: '/reject' }), ' to discard them.',
    );
    this.diffBody = el('div', { class: 'diff-body' }, this.diffList, this.diffHint);
    this.diffNodes.clear();
    replace(byId('diffbar'), this.diffHead, this.diffBody);
  }

  /** Put `nodes` under `host` in order, touching only the ones out of place.
   * Moving a node that is already where it belongs would reset the list's
   * scroll for no reason. */
  setChildren(host, nodes) {
    for (let i = 0; i < nodes.length; i += 1) {
      const node = nodes[i];
      if (host.childNodes[i] !== node) host.insertBefore(node, host.childNodes[i] || null);
    }
    while (host.childNodes.length > nodes.length) host.removeChild(host.lastChild);
  }

  renderDiffBar() {
    const bar = byId('diffbar'),
      info = this.diff;
    if (!info) {
      if (bar.hidden) return;
      bar.hidden = true;
      bar.classList.remove('open');
      replace(bar);
      this.diffHead = this.diffBody = this.diffList = this.diffHint = null;
      this.diffNodes.clear();
      return;
    }
    this.ensureDiffBar();
    bar.hidden = false;
    bar.classList.toggle('open', this.diffOpen);
    const files = info.files || [],
      groups = info.groups || [],
      scope = this.diffScope;
    const isInline = (item) => item.kind === 'mine' || (item.kind === 'agent' && scope === 'all');
    // The totals follow the scope chip: this session's own rows while it reads
    // "agent", every session's rows while "agent" is dimmed. The auto/drift
    // sections never count toward either.
    const inlineFiles = files.filter(isInline);
    // The label count follows the same scope chip as the +/− totals instead of
    // counting every row; auto/drift and ignored groups stay out of it.
    const count = inlineFiles.length,
      label = `Edited ${count} file${count === 1 ? '' : 's'}`;
    const adds = inlineFiles.reduce((total, file) => total + (file.additions || 0), 0);
    const dels = inlineFiles.reduce((total, file) => total + (file.deletions || 0), 0);
    this.diffHead.setAttribute('aria-expanded', String(this.diffOpen));
    this.diffHead.title = this.diffOpen ? 'Hide the pending file list' : 'Show the pending file list';
    this.diffLabel.textContent = label;
    this.diffIgnored.textContent = info.ignored_files ? ` · ${info.ignored_files} ignored` : '';
    if (this.diffScopeSig !== scope) {
      this.diffScopeSig = scope;
      const chip = scope === 'all' ? [['hi', 'age'], ['dim', 'nt']] : [['hi', 'agent']];
      this.diffScopeEl.replaceChildren(...chip.map(([cls, text]) => el('span', { class: cls, text })));
    }
    this.diffAdds.textContent = `+${adds}`;
    this.diffDels.textContent = `-${dels}`;
    if (!this.diffOpen) {
      this.setChildren(this.diffList, []);
      return;
    }
    const sectionOf = (kind) => (kind === 'mine' || kind === 'agent' ? 'agent' : kind);
    const sections = scope === 'all' ? ['auto', 'drift'] : ['agent', 'auto', 'drift'];
    const sectionFiles = (section) => files.filter((file) => sectionOf(file.kind) === section && !isInline(file));
    const sectionGroups = (section) =>
      groups.filter((group) => sectionOf(group.kind) === section && !isInline(group));
    const leadingFiles = files.filter(isInline),
      leadingGroups = groups.filter(isInline);
    const columnWidth = (field, sign) => {
      let width = 1;
      for (const file of files) width = Math.max(width, `${sign}${file[field] || 0}`.length);
      for (const entries of [leadingFiles, ...sections.map(sectionFiles)]) {
        const sum = entries.reduce((total, file) => total + (file[field] || 0), 0);
        width = Math.max(width, `${sign}${sum}`.length);
      }
      return width;
    };
    this.diffList.style.setProperty('--add-w', `${columnWidth('additions', '+')}ch`);
    this.diffList.style.setProperty('--del-w', `${columnWidth('deletions', '-')}ch`);
    const used = new Set(),
      nodes = [],
      // Reuse the node that already shows this key: only its first sight builds.
      keep = (key, make) => {
        used.add(key);
        const node = make(this.diffNodes.get(key) || null);
        this.diffNodes.set(key, node);
        nodes.push(node);
      },
      fileRow = (file) =>
        keep(`file:${file.kind}:${file.path}:${file.author ?? ''}`, (node) => this.diffRow(file, node)),
      groupRow = (group) => keep(`group:${group.kind}:${group.prefix}`, (node) => this.diffGroup(group, node));
    leadingFiles.forEach(fileRow);
    leadingGroups.forEach(groupRow);
    if (leadingFiles.length || leadingGroups.length) keep('divider', () => el('div', { class: 'diff-divider' }));
    for (const section of sections) {
      const rows = sectionFiles(section),
        folders = sectionGroups(section),
        expanded = this.diffExpanded.has(section);
      keep(`section:${section}`, (node) => this.diffSection(section, rows, folders, node));
      if (!expanded) continue;
      rows.forEach(fileRow);
      folders.forEach(groupRow);
    }
    if (!files.length && !groups.length)
      keep('empty', () => el('div', { class: 'diff-empty', text: 'no pending changes' }));
    // A row that left the diff leaves the cache too: the key is the identity.
    for (const key of this.diffNodes.keys()) if (!used.has(key)) this.diffNodes.delete(key);
    this.setChildren(this.diffList, nodes);
  }

  /** A row is built the first time it is seen, then patched in place: the node
   * that already shows `file` is the one that keeps its place and its focus. */
  diffRow(file, node = null) {
    node = diffRowNode(node, 'diff-row', ['path', 'new', 'shared', 'badge', 'adds', 'dels']);
    const [path, isNew, shared, badge, adds, dels] = node.children,
      kind = file.kind === 'mine' ? 'mine' : file.kind,
      share = file.author === null && file.authors?.length > 1;
    node.title = file.path;
    node.onclick = () => this.openPatch(file.path, file.author ?? null);
    if (this.diffPatch?.path === file.path) node.setAttribute('aria-current', 'true');
    else node.removeAttribute('aria-current');
    path.textContent = file.path;
    isNew.textContent = 'new';
    isNew.hidden = !file.is_new;
    shared.hidden = !share;
    if (share) shared.textContent = String(file.authors.length);
    badge.className = `badge ${kind}`;
    badge.textContent = DIFF_BADGE[file.kind] || file.kind;
    adds.textContent = `+${file.additions || 0}`;
    dels.textContent = `-${file.deletions || 0}`;
    return node;
  }

  diffGroup(group, node = null) {
    node = diffRowNode(node, 'diff-row diff-group', ['path', 'badge', 'count']);
    const [path, badge, count] = node.children;
    node.title = group.prefix;
    node.onclick = () => this.openPatch(group.prefix, null);
    if (this.diffPatch?.path === group.prefix) node.setAttribute('aria-current', 'true');
    else node.removeAttribute('aria-current');
    path.textContent = group.prefix;
    badge.className = `badge ${group.kind}`;
    badge.textContent = DIFF_BADGE[group.kind] || group.kind;
    count.textContent = `${group.files} files`;
    return node;
  }

  /** A section header: `rows` are its files, `folders` the groups it hides. */
  diffSection(section, rows, folders, node = null) {
    node = diffRowNode(node, 'diff-section', ['name', 'spacer', 'count', 'adds', 'dels']);
    const expanded = this.diffExpanded.has(section),
      empty = !rows.length && !folders.length,
      count = folders.reduce((total, group) => total + (group.files || 0), 0),
      sum = (field) => rows.reduce((total, file) => total + (file[field] || 0), 0),
      [name, , countEl, adds, dels] = node.children;
    node.className = `diff-section ${section}${expanded ? ' expanded' : ''}${empty ? ' empty' : ''}`;
    node.title = expanded ? `Fold ${section}` : `Unfold ${section}`;
    node.setAttribute('aria-expanded', String(expanded));
    node.onclick = () => this.toggleDiffSection(section);
    name.textContent = section;
    countEl.textContent = count ? `${count} files` : '';
    adds.textContent = `+${sum('additions')}`;
    dels.textContent = `-${sum('deletions')}`;
    return node;
  }

  toggleDiffSection(section) {
    if (this.diffExpanded.has(section)) this.diffExpanded.delete(section);
    else this.diffExpanded.add(section);
    this.renderDiffBar();
  }

  closePatch() {
    if (!this.diffPatch) return;
    const path = this.diffPatch.path,
      active = document.activeElement,
      restore = !!active && (this.diffView.contains(active) || active.classList?.contains('diff-row'));
    this.diffPatch = null;
    this.renderDiffView();
    this.renderDiffBar();
    if (restore) [...byId('diffbar').querySelectorAll('.diff-row')].find((row) => row.title === path)?.focus();
  }

  /** The right-hand diff panel: what a file row opened, colored and with
   * line numbers. Clicking that same row again (or ✕/Esc, or folding the
   * bar) removes it; clicking another row swaps it in place. */
  renderDiffView() {
    const panel = this.diffView,
      patch = this.diffPatch;
    if (!patch || !patch.text) {
      if (!panel.hidden) {
        panel.hidden = true;
        replace(panel);
      }
      return;
    }
    const focusInside = panel.contains(document.activeElement);
    const samePath = panel.dataset.path === patch.path;
    const position = samePath ? (panel.querySelector('.diffview-body')?.scrollTop ?? 0) : 0;
    const tree = diffTree(patch.text);
    const file = tree.files[0];
    const notes = file?.notes || [];
    const sub = el(
      'div',
      { class: 'diffview-sub' },
      notes.includes('new') ? el('span', { class: 'tag', text: 'new file' }) : null,
      notes.includes('deleted') ? el('span', { class: 'tag', text: 'deleted' }) : null,
      notes.includes('binary') ? el('span', { class: 'tag', text: 'binary' }) : null,
      notes.some((note) => note.startsWith('renamed from ')) ? el('span', { class: 'tag', text: 'renamed' }) : null,
      tree.files.length > 1 ? el('span', { text: `${tree.files.length} files` }) : null,
      el('span', { class: 'adds', text: `+${tree.additions}` }),
      el('span', { class: 'dels', text: `-${tree.deletions}` }),
    );
    const body = el('div', { class: 'diffview-body', tabindex: '0' }, tree.node);
    const head = el(
      'div',
      { class: 'diffview-head' },
      el('div', { class: 'diffview-title' }, el('span', { class: 'path', text: patch.path, title: patch.path }), sub),
      el('button', {
        class: 'diffview-close',
        type: 'button',
        title: 'Close the diff (Esc)',
        'aria-label': `Close the diff for ${patch.path}`,
        text: '✕',
        on: { click: () => this.closePatch() },
      }),
    );
    replace(panel, head, body);
    panel.hidden = false;
    panel.dataset.path = patch.path;
    body.scrollTop = position;
    if (focusInside) body.focus({ preventScroll: true });
  }

  toggleDiff() {
    const hadFocus = document.activeElement?.classList?.contains('diff-head');
    this.diffOpen = !this.diffOpen;
    if (!this.diffOpen && this.diffPatch) {
      this.diffPatch = null;
      this.renderDiffView();
    }
    this.renderDiffBar();
    if (hadFocus) byId('diffbar').querySelector('.diff-head')?.focus();
  }

  async toggleDiffScope() {
    const scope = this.diffScope === 'all' ? 'mine' : 'all';
    this.diffScope = scope;
    this.diffPatch = null;
    this.renderDiffView();
    this.renderDiffBar();
    if (!this.session) {
      // A draft has no session to save the choice on (it rides out on the
      // create request); the workspace endpoint builds its rows anyway.
      return this.refreshDiff();
    }
    this.diffScopePending = scope;
    try {
      await api.command(this.session, { type: 'set_view_state', view: { diff_scope: scope } });
    } catch (error) {
      this.diffScopePending = null;
      this.diffScope = this.status?.view?.diff_scope || 'all';
      this.addLine(error.message, 'error');
      this.renderDiffBar();
      return;
    }
    await this.refreshDiff();
  }

  async openPatch(path, author) {
    if (this.diffPatch?.path === path) {
      this.closePatch();
      return;
    }
    try {
      const info = await this.loadDiff({ path, author });
      if (!info || !String(info.text || '').trim()) {
        this.addLine(`no pending changes for ${path}`, 'warn');
        return;
      }
      this.diffPatch = { path, author, text: info.text };
      this.diffOpen = true;
      this.renderDiffBar();
      this.renderDiffView();
      this.diffView.querySelector('.diffview-body')?.focus({ preventScroll: true });
    } catch (error) {
      this.addLine(`diff failed: ${error.message}`, 'error');
    }
  }

  async showDiffCommand(arg) {
    if (!arg) {
      // Toggle: anything showing — the file list, or an open patch — hides;
      // otherwise (re)read the pending set and expand it. A draft chat reads
      // it from the workspace, since tracking is per repo root.
      if (this.diffOpen) return this.toggleDiff();
      await this.refreshDiff();
      if (!this.diff) return this.addLine('no pending changes to diff', 'warn');
      return this.toggleDiff();
    }
    await this.refreshDiff();
    const file = this.diff?.files.find((entry) => entry.path === arg || entry.path.endsWith(`/${arg}`));
    await this.openPatch(file ? file.path : arg, file ? (file.author ?? null) : null);
  }

  accept() {
    if (this.session) this.postCommand({ type: 'accept' });
  }

  async reject() {
    if (!this.session) return;
    const drift = (this.diff?.files || []).filter((file) => file.kind === 'drift');
    if (drift.length) {
      const ok = await showConfirm({
        title: 'Reject all pending changes?',
        text: `This also reverts ${drift.length} file(s) you changed yourself (they are not the agent's work).`,
        detail: drift
          .slice(0, 12)
          .map((file) => file.path)
          .join('\n'),
        confirmLabel: 'Reject all',
        danger: true,
      });
      if (!ok) return;
    }
    this.postCommand({ type: 'reject' });
  }
  // ------------------------------------------------------------- commands
  postCommand(command) {
    this.commandQueue.push(command);
    void this.drainCommands();
  }

  async drainCommands() {
    if (this.draining) return;
    this.draining = true;
    try {
      while (this.commandQueue.length) {
        const command = this.commandQueue.shift();
        try {
          await this.sendCommand(command);
        } catch (error) {
          this.addLine(error.message || String(error), 'error');
        }
      }
    } finally {
      this.draining = false;
    }
  }

  sendCommand(command) {
    return this.session ? api.command(this.session, command) : null;
  }

  async ensureSession() {
    if (this.session) return true;
    const request = {
      workspace: this.startWorkspace() || undefined,
      model: this.draftModel || undefined,
      view: { diff_scope: this.diffScope },
    };
    try {
      const summary = await api.createSession(request);
      this.draftModel = null;
      localStorage.removeItem(STORAGE.model);
      this.attach(summary.name);
      if (this.draftMode !== 'write') this.postCommand({ type: 'set_mode', mode: this.draftMode });
      this.draftMode = 'write';
      await this.pollSessions();
      return true;
    } catch (error) {
      this.addLine(`could not start a session: ${error.message}`, 'error');
      return false;
    }
  }

  resizeInput() {
    this.input.style.height = 'auto';
    this.input.style.height = `${Math.min(this.input.scrollHeight, window.innerHeight * 0.4)}px`;
    this.syncSendButton();
    this.scheduleScroll();
  }

  // dim the send icon while there is nothing to send (every programmatic
  // change to the textarea goes through resizeInput, and attachment changes go
  // through syncAttachments, so this stays in sync)
  syncSendButton() {
    const send = byId('send');
    if (send) send.dataset.filled = String(Boolean(this.input.value.trim() || this.attachments.length));
  }

  async submit() {
    const text = this.input.value.trim();
    if (this.ask) {
      if (text) {
        this.answerAsk(text);
        this.input.value = '';
        this.resizeInput();
      } else {
        const options = this.ask.questions[this.ask.index]?.options || [];
        if (options.length) this.answerAsk(options[this.ask.selected] ?? options[0]);
      }
      return;
    }
    if (!text) {
      this.flushQueued();
      return;
    }
    this.scrollToLatest();
    if (text.startsWith('/') && this.runSlash(text)) {
      this.input.value = '';
      this.resizeInput();
      this.syncAttachments();
      return;
    }
    const images = this.attachments;
    this.input.value = '';
    this.resizeInput();
    this.attachments = [];
    this.syncAttachments();
    await this.sendMessage(text, images);
  }

  async sendMessage(text, images) {
    if (!(await this.ensureSession())) {
      this.restoreUnsent(text, images);
      return;
    }
    const refs = [];
    for (const image of images || []) {
      try {
        await api.uploadImage(this.session, image.data, image.id);
        refs.push({ id: image.id, name: image.name, mime: image.mime, width: image.width, height: image.height });
      } catch (error) {
        this.addLine(`could not upload ${image.name}: ${error.message}`, 'error');
      }
    }
    try {
      await this.sendCommand({ type: 'send_message', text, images: refs.length ? refs : null });
    } catch (error) {
      this.addLine(error.message, 'error');
      this.restoreUnsent(text, images);
    }
    this.pollSessions();
  }

  restoreUnsent(text, images) {
    if (this.input.value.trim()) return;
    this.input.value = text;
    this.attachments = images || [];
    this.resizeInput();
    this.syncAttachments();
    this.input.focus();
  }

  flushQueued() {
    const status = this.viewStatus;
    if (status.queued?.length && status.state === 'running') this.postCommand({ type: 'flush_queued' });
  }

  runSlash(text) {
    const body = text.slice(1),
      space = body.search(/\s/);
    const name = space < 0 ? body : body.slice(0, space),
      arg = space < 0 ? '' : body.slice(space + 1).trim();
    const command = findCommand(name);
    if (!command) return false;
    try {
      command.run(this, arg);
    } catch (error) {
      this.addLine(error.message || String(error), 'error');
    }
    return true;
  }
  // ----------------------------------------------------------- interaction
  interrupt(explicit) {
    if (this.ask) {
      this.skipAsk();
      return;
    }
    const status = this.viewStatus,
      busy = status.state === 'running' || status.state === 'compacting';
    if (!busy) {
      if (explicit) this.addLine('nothing is running', 'warn');
      return;
    }
    if (!explicit && !this.interruptArmed) {
      this.interruptArmed = true;
      const items = this.transcript.addLine('press esc again to interrupt', 'warn');
      this.hintItems.push(...items);
      this.sync(items);
      clearTimeout(this.interruptTimer);
      this.interruptTimer = setTimeout(() => this.disarmInterrupt(), INTERRUPT_ARM_SECONDS * 1000);
      return;
    }
    this.disarmInterrupt();
    this.postCommand({ type: 'cancel' });
  }

  disarmInterrupt() {
    this.interruptArmed = false;
    clearTimeout(this.interruptTimer);
    this.interruptTimer = null;
  }

  toggleMode() {
    if (!this.session) {
      this.draftMode = this.draftMode === 'write' ? 'plan' : 'write';
      this.renderStatus();
      this.renderBars();
      return;
    }
    this.postCommand({ type: 'set_mode', mode: this.viewStatus.mode === 'write' ? 'plan' : 'write' });
  }

  async undo() {
    if (!this.session) return;
    const result = await this.sendCommand({ type: 'rollback' });
    if (result?.user_text) this.restoreUnsent(result.user_text, []);
  }

  async undoTo(messageId) {
    if (!this.session || !messageId) return;
    try {
      const result = await this.sendCommand({ type: 'rollback', message_id: messageId });
      if (result?.user_text) this.restoreUnsent(result.user_text, []);
    } catch (error) {
      this.addLine(error.message, 'error');
    }
  }

  retry() {
    if (this.session) this.postCommand({ type: 'rollback', resend: true });
  }

  async pickModel() {
    if (!this.config) {
      this.addLine('/model: no config from the host yet', 'warn');
      return;
    }
    const current = this.viewStatus.model_name;
    const rows = this.config.models.map((model) => ({
      id: model.name,
      label: model.display_name || model.name,
      current: model.name === current,
      meta: [
        { text: model.name, class: 'model-id' },
        { text: `effort ${model.cot_strength}`, class: 'effort' },
        { text: model.supports_vision ? 'vision' : '', class: 'vision' },
      ],
    }));
    const choice = await showPicker({ title: 'Switch model', rows });
    if (!choice) return;
    if (this.session) this.postCommand({ type: 'switch_model', name: choice });
    else {
      this.draftModel = choice;
      localStorage.setItem(STORAGE.model, choice);
      this.renderStatus();
    }
    try {
      this.config = await api.setDefaultProfile(choice);
      this.renderInfo();
    } catch (error) {
      this.addLine(`could not save default model: ${error.message}`, 'error');
    }
  }

  async pickSandbox() {
    if (!this.config) {
      this.addLine('/sandbox: no config from the host yet', 'warn');
      return;
    }
    if (!this.session) {
      this.addLine('/sandbox: send a message first — the profile is per session', 'warn');
      return;
    }
    const current = this.viewStatus.sandbox_name;
    const rows = this.config.sandboxes.map((sandbox) => ({
      id: sandbox.name,
      label: sandbox.name,
      current: sandbox.name === current,
    }));
    const choice = await showPicker({ title: 'Sandbox profile', text: 'Applies when the session is idle.', rows });
    if (choice) this.postCommand({ type: 'set_sandbox', name: choice });
  }

  filterCommand(arg) {
    const text = (arg || '').trim();
    if (!text) {
      void this.openFilter();
      return;
    }
    this.handleFilter(text);
  }

  async openFilter() {
    if (!this.session) {
      this.addLine('/filter: send a message first — filters are per session', 'warn');
      return;
    }
    const filters = this.viewStatus.filter_list || [];
    const rows = [
      { id: 'add', label: '+ add a path…', detail: filters.length ? 'exclude another path' : 'exclude a path' },
      ...filters.map((path) => ({ id: `remove:${path}`, label: `✕ ${path}` })),
    ];
    const choice = await showPicker({
      title: 'Sandbox filters',
      text: 'Paths listed here are not mounted into the sandbox.',
      rows,
    });
    if (!choice) return;
    if (choice === 'add') {
      const path = await showPrompt({
        title: 'Filter a path',
        label: 'Host path',
        placeholder: '/home/me/project/secrets',
        note: 'Excluded from the sandbox mounts in this session.',
      });
      if (path) this.postCommand({ type: 'filter', path, add: true });
      return;
    }
    this.postCommand({ type: 'filter', path: choice.slice('remove:'.length), add: false });
  }

  handleFilter(arg) {
    const [action = 'list', ...rest] = arg.split(/\s+/),
      path = rest.join(' ').trim();
    if (action === 'add' && path) this.postCommand({ type: 'filter', path, add: true });
    else if (action === 'remove' && path) this.postCommand({ type: 'filter', path, add: false });
    else if (action === 'list') {
      const filters = this.viewStatus.filter_list || [];
      this.addLine(
        filters.length ? `filtered out of the sandbox mounts: ${filters.join(', ')}` : 'no paths filtered out',
      );
    } else this.addLine('usage: /filter add <path> | /filter remove <path> | /filter list', 'error');
  }

  workspaceForRefs() {
    return this.viewStatus.workspace || this.workspace || null;
  }

  async openRefs() {
    const workspace = this.workspaceForRefs();
    if (!workspace) {
      this.addLine('no workspace to manage references for', 'error');
      return;
    }
    const body = el('div');
    const update = async (path, add) => {
      try {
        await api.updateWorkspaceRefs(workspace, path, add);
      } catch (error) {
        this.addLine(error.message, 'error');
      }
      await render();
    };
    const render = async () => {
      replace(body);
      let refs = [];
      try {
        refs = await api.workspaceRefs(workspace);
      } catch (error) {
        body.appendChild(el('div', { class: 'sheet-note', text: error.message }));
        return;
      }
      body.appendChild(el('div', { class: 'sheet-note', text: workspace }));
      if (!refs.length) body.appendChild(el('div', { class: 'sheet-note', text: 'no references yet' }));
      for (const ref of refs) {
        body.appendChild(
          el(
            'div',
            { class: 'sheet-row' },
            el('span', { class: 'name', text: ref.path }),
            el('span', { class: 'detail', text: ref.mount }),
            el('button', {
              class: 'btn btn-ghost',
              type: 'button',
              text: 'remove',
              on: { click: () => update(ref.path, false) },
            }),
          ),
        );
      }
      body.appendChild(
        el('button', {
          class: 'btn',
          type: 'button',
          style: 'margin-top:6px',
          text: '+ add a path',
          on: {
            click: async () => {
              const path = await showPrompt({
                title: 'Add an external reference',
                label: 'Host path',
                note: 'Bound read-only at /tmp/refs/<alias> for this workspace.',
              });
              if (!path) return;
              await update(path, true);
              void this.openRefs();
            },
          },
        }),
      );
    };
    await render();
    await showInfo({ title: 'External references (/refs)', body });
  }

  async openTools() {
    if (!this.session) {
      this.addLine('send a message first — the tool set is built with the session', 'error');
      return;
    }
    const body = el('div', { class: 'tools-sheet' });
    let title = 'tools',
      polls = 0,
      load;
    const act = async (command) => {
      try {
        await this.sendCommand(command);
      } catch (error) {
        this.addLine(error.message, 'error');
      }
      polls = 0;
      await load(false);
    };
    const paint = (info) => {
      title = info.error ? 'tools' : `tools · ${info.tools.length} available`;
      if (!body.isConnected && byId('sheet').open) return;
      byId('sheet-title').textContent = title;
      const scroller = byId('sheet-body'),
        top = scroller.scrollTop;
      const listTop = body.querySelector('.tools-list')?.scrollTop || 0;
      const focused = body.contains(document.activeElement) ? document.activeElement.dataset?.focus || '' : '';
      replace(body, ...toolsSheetBody(info, act));
      scroller.scrollTop = top;
      const list = body.querySelector('.tools-list');
      if (list) list.scrollTop = listTop;
      if (focused) {
        const again = [...body.querySelectorAll('[data-focus]')].find((node) => node.dataset.focus === focused);
        again?.focus({ preventScroll: true });
      }
    };
    load = async (followStarting = true) => {
      let info;
      try {
        info = await api.tools(this.session);
      } catch (error) {
        paint({ tools: [], servers: [], error: error.message });
        return;
      }
      paint(info);
      const starting = info.servers.some((server) => server.status === 'starting');
      if (followStarting && starting && polls < 12) {
        polls += 1;
        setTimeout(() => {
          if (body.isConnected) void load();
        }, 800);
      }
    };
    replace(body, el('div', { class: 'sheet-note', text: 'loading…' }));
    await load();
    await showInfo({ title, body });
  }

  openHelp() {
    const keys = [
      ['Enter', 'run the highlighted command, otherwise send the message'],
      ['Shift+Enter', 'a new line without sending'],
      ['Tab', 'complete the command, otherwise toggle plan / write'],
      ['Esc', 'close a popup; twice interrupts a running turn'],
      ['/', 'the command palette'],
      ['@', 'mention a workspace file'],
      ['Ctrl/Cmd+K', 'the command palette, from anywhere'],
      ['↑↓', 'move through the palette, a picker or the agent’s options'],
    ];
    const body = el('div', { class: 'help' });
    body.appendChild(
      el(
        'dl',
        { class: 'help-keys' },
        ...keys.map(([key, what]) => [el('dt', {}, el('kbd', { text: key })), el('dd', { text: what })]),
      ),
    );
    body.appendChild(sheetSection('commands'));
    body.appendChild(
      el(
        'dl',
        { class: 'help-cmds' },
        ...COMMANDS.map((command) => [
          el('dt', { text: `/${command.name}` }),
          el(
            'dd',
            {},
            command.usage ? el('span', { class: 'help-args', text: command.usage }) : null,
            el('span', { text: command.description }),
          ),
        ]),
      ),
    );
    body.appendChild(sheetSection('notes'));
    body.appendChild(
      el('p', {
        class: 'help-note',
        text:
          'Every command runs on one Enter. A command that opens a picker (/model, /theme, /filter, ' +
          '/sessions, /tools) shows it right away; Tab completes the word if you want to type an ' +
          'argument. Tool rows in the transcript show what they did inline — tap one to fold it away.',
      }),
    );
    void showInfo({ title: 'xarness web UI', text: 'The same commands and views as the terminal client.', body });
  }
  // -------------------------------------------------------------- palette
  openCommandPalette() {
    this.paletteKind = 'slash';
    this.paletteIndex = 0;
    this.renderPalette(COMMANDS.map(commandRow));
    this.input.focus();
  }

  updatePalette() {
    const value = this.input.value,
      caret = this.input.selectionStart ?? value.length;
    if (value.startsWith('/') && !/\s/.test(value)) {
      const query = value.slice(1).toLowerCase();
      const matches = COMMANDS.filter((command) => command.name.startsWith(query));
      matches.sort((a, b) => (a.name === query ? -1 : 0) - (b.name === query ? -1 : 0));
      this.paletteKind = 'slash';
      this.paletteIndex = 0;
      this.renderPalette(matches.map(commandRow), query);
      return;
    }
    const before = value.slice(0, caret),
      at = before.lastIndexOf('@');
    if (at >= 0 && !/[\s]/.test(before.slice(at + 1)) && (at === 0 || /\s/.test(before[at - 1]))) {
      const query = before.slice(at + 1);
      this.paletteKind = 'at';
      this.paletteIndex = 0;
      this.paletteAt = at;
      clearTimeout(this.atTimer);
      this.atTimer = setTimeout(() => this.searchFiles(query), 150);
      return;
    }
    this.closePalette();
  }

  async searchFiles(query) {
    const workspace = this.viewStatus.workspace || this.workspace;
    if (!workspace) return;
    const seq = (this.atSeq += 1);
    try {
      const files = await api.searchFiles(workspace, query);
      if (seq !== this.atSeq || this.paletteKind !== 'at') return;
      this.paletteIndex = 0;
      this.renderPalette(files.map((file) => ({ id: file, label: file })));
    } catch {
      this.closePalette();
    }
  }

  renderPalette(rows, query = '') {
    const palette = byId('palette'),
      list = byId('palette-list');
    this.paletteRows = rows;
    if (!rows.length) {
      this.paletteIndex = 0;
      this.paletteNodes = [];
      replace(
        list,
        el('div', { class: 'palette-empty', text: query ? `no matches for “${query}”` : 'nothing to show' }),
      );
      this.renderPaletteFoot(null);
      palette.hidden = false;
      this.scheduleScroll();
      return;
    }
    this.paletteIndex = Math.min(this.paletteIndex, rows.length - 1);
    this.paletteNodes = rows.map((row, index) =>
      el(
        'button',
        {
          class: 'palette-row',
          type: 'button',
          role: 'option',
          'aria-selected': String(index === this.paletteIndex),
          dataset: { index: String(index) },
          on: {
            pointerenter: () => {
              if (this.paletteIndex !== index) this.selectPalette(index);
            },
            click: () => {
              this.paletteIndex = index;
              this.acceptPalette();
            },
          },
        },
        el('span', { class: 'palette-name', text: row.label }),
        row.detail ? el('span', { class: 'palette-desc', text: row.detail }) : null,
      ),
    );
    replace(list, ...this.paletteNodes);
    this.renderPaletteFoot(rows[this.paletteIndex]);
    palette.hidden = false;
    this.scheduleScroll();
  }

  // The foot teaches the selected command's call signature (the row already
  // says what it does) plus the keys that work while the palette is open.
  renderPaletteFoot(row) {
    const foot = byId('palette-foot');
    if (!foot) return;
    const hint = this.paletteKind === 'at' ? '↵ insert' : '↵ run · ⇥ complete · esc close';
    replace(
      foot,
      el('span', { class: 'palette-usage', text: row?.args ? `${row.label} ${row.args}` : '' }),
      el('span', { class: 'palette-hint', text: hint }),
    );
  }

  selectPalette(index) {
    if (!this.paletteRows.length || !this.paletteNodes.length) return;
    const count = this.paletteNodes.length;
    this.paletteIndex = ((index % count) + count) % count;
    this.paletteNodes.forEach((node, at) => node.setAttribute('aria-selected', String(at === this.paletteIndex)));
    const node = this.paletteNodes[this.paletteIndex];
    if (node.scrollIntoView) node.scrollIntoView({ block: 'nearest' });
    this.renderPaletteFoot(this.paletteRows[this.paletteIndex]);
  }

  movePalette(delta) {
    this.selectPalette(this.paletteIndex + delta);
  }

  closePalette() {
    byId('palette').hidden = true;
    this.paletteKind = null;
    this.paletteRows = [];
    this.paletteNodes = [];
  }

  completePalette() {
    const row = this.paletteRows[this.paletteIndex];
    if (!row) {
      this.closePalette();
      return;
    }
    if (this.paletteKind === 'at') {
      this.acceptPalette();
      return;
    }
    const command = findCommand(row.id);
    this.closePalette();
    this.input.value = `/${row.id}${command?.usage ? ' ' : ''}`;
    this.resizeInput();
    this.input.focus();
    const position = this.input.value.length;
    this.input.setSelectionRange(position, position);
  }

  acceptPalette() {
    const row = this.paletteRows[this.paletteIndex],
      kind = this.paletteKind;
    if (!row) {
      this.closePalette();
      return;
    }
    if (kind === 'at') {
      const value = this.input.value,
        at = this.paletteAt ?? value.lastIndexOf('@'),
        caret = this.input.selectionStart ?? value.length;
      const insert = `${row.id} `;
      this.input.value = `${value.slice(0, at)}${insert}${value.slice(caret)}`;
      this.closePalette();
      this.input.focus();
      const position = at + insert.length;
      this.input.setSelectionRange(position, position);
      this.resizeInput();
      return;
    }
    const command = findCommand(row.id),
      typed = this.input.value.startsWith('/');
    this.closePalette();
    if (typed) {
      this.input.value = '';
      this.resizeInput();
    }
    if (!this.touchFirst) this.input.focus();
    if (command) command.run(this, '');
  }
  // ---------------------------------------------------------------- images
  async resolveImage(image) {
    if (!image?.id || !this.session) return null;
    try {
      return await api.imageUrl(this.session, image.id);
    } catch {
      return null;
    }
  }

  async addFiles(fileList) {
    const files = [...(fileList || [])];
    if (!files.length) return;
    if (!this.viewStatus.supports_vision) {
      this.addLine('the active model does not support image input (supports_vision is false in the config)', 'warn');
      return;
    }
    for (const file of files) {
      if (!file.type.startsWith('image/')) {
        this.addLine(`${file.name}: not an image`, 'warn');
        continue;
      }
      try {
        const data = new Uint8Array(await file.arrayBuffer()),
          id = sha256Hex(data),
          size = await imageSize(data, file.type);
        this.attachments.push({
          id,
          name: file.name || 'image',
          mime: file.type,
          width: size.width,
          height: size.height,
          data,
          url: URL.createObjectURL(file),
        });
      } catch (error) {
        this.addLine(`${file.name}: ${error.message}`, 'error');
      }
    }
    this.syncAttachments();
  }

  syncAttachments() {
    const host = byId('attachments'),
      images = this.attachments;
    host.hidden = !images.length;
    replace(
      host,
      ...images.map((image, index) =>
        el(
          'span',
          { class: 'attachment' },
          el('img', { src: image.url, alt: '' }),
          el('span', { class: 'name', text: image.name }),
          el('span', { text: `${image.width}×${image.height}` }),
          el('button', {
            type: 'button',
            text: '✕',
            'aria-label': `Remove ${image.name}`,
            on: {
              click: () => {
                this.attachments.splice(index, 1);
                this.syncAttachments();
              },
            },
          }),
        ),
      ),
    );
    this.resizeInput();
  }
}

async function imageSize(data, mime) {
  try {
    const bitmap = await createImageBitmap(new Blob([data], { type: mime }));
    const size = { width: bitmap.width, height: bitmap.height };
    bitmap.close?.();
    return size;
  } catch {
    return { width: 0, height: 0 };
  }
}
const app = new App();
window.xarness = app;
app.start().catch((error) => app.addLine(`startup failed: ${error.message}`, 'error'));
