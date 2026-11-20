/** The xarness web UI: a thin client of the host API, like the TUI. It renders
 * one session's SSE stream and sends commands. State lives in `state.js`, the
 * DOM in `render.js`, HTTP in `api.js`. */
import { api, ApiError } from './api.js';
import { byId, el, replace, toast } from './dom.js';
import { LogView } from './render.js';
import { diffTree } from './diff.js';
import { Transcript, formatTokens } from './state.js';
import { COMMANDS, commandUsage, findCommand } from './commands.js';
import { showConfirm, showInfo, showPicker, showPrompt, sheetSection } from './ui.js';
import { sha256Hex } from './sha256.js';

const RECONNECT_MIN = 250,
  RECONNECT_MAX = 4000;
const INTERRUPT_ARM_SECONDS = 2.5,
  SESSIONS_REFRESH_MS = 5000;
const STALL_MS = 12000,
  WATCHDOG_MS = 4000,
  PROBE_TIMEOUT_MS = 5000,
  TICK_MS = 250;
const DETACH_PX = 40,
  ATTACH_PX = 8;
const STORAGE = {
  token: 'xarness.token',
  theme: 'xarness.theme',
  model: 'xarness.model',
  workspace: 'xarness.workspace',
  session: 'xarness.session',
};
const THEMES = {
  'ayu-darker': { mode: 'dark', swatches: ['#121212', '#bfbdb6', '#d2a6fe', '#aad84c', '#ef7177'] },
  'one-light': { mode: 'light', swatches: ['#fafafa', '#000000', '#7c4dff', '#50a14f', '#e45649'] },
};
const DIFF_BADGE = { mine: 'agent', agent: 'agent', auto: 'auto', drift: 'drift' };
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
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

class App {

  constructor() {
    Object.assign(this, {
      transcript: new Transcript(),
      status: null,
      session: null,
      config: null,
      sessions: [],
      connState: 'connecting',
      workspace: localStorage.getItem(STORAGE.workspace) || null,
      draftMode: 'write',
      draftModel: localStorage.getItem(STORAGE.model) || null,
      attachments: [],
      diff: null,
      diffOpen: false,
      diffPatch: null,
      diffExpanded: new Set(),
      diffScope: 'all',
      diffScopePending: null,
      ask: null,
      streamAbort: null,
      attachSeq: 0,
      lastActivity: 0,
      stalled: false,
      commandQueue: [],
      draining: false,
      pendingItem: null,
      pendingSeq: 0,
      phase: 'idle',
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
      log: byId('log'),
      input: byId('input'),
      diffView: byId('diffview'),
    });
    this.view = new LogView(this.log, {
      onUndoTo: (id) => this.undoTo(id),
      resolveImage: (image) => this.resolveImage(image),
    });
  }
  // ---------------------------------------------------------------- boot
  async start() {
    const storedToken = localStorage.getItem(STORAGE.token);
    if (storedToken) api.setToken(storedToken);
    this.applyTheme(localStorage.getItem(STORAGE.theme));
    this.bindEvents();
    this.watchPointer();
    this.renderTitle();
    this.renderStatus();
    this.renderBars();
    this.syncAttachments();
    this.setConn('connecting');
    this.view.renderAll([]);
    await this.connect();
    this.pollSessions();
    setInterval(() => this.tick(), TICK_MS);
    setInterval(() => this.pollSessions(), SESSIONS_REFRESH_MS);
    setInterval(() => this.checkStalled(), WATCHDOG_MS);
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden) this.pollSessions();
    });
    window.addEventListener('hashchange', () => {
      const name = this.sessionFromHash();
      if (name && name !== this.session) this.attach(name);
    });
    if ('serviceWorker' in navigator)
      navigator.serviceWorker.register('service-worker.js', { updateViaCache: 'none' }).catch(() => {});
    if (!this.coarsePointer?.matches) this.input.focus();
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
    const name = this.sessionFromHash() || this.session;
    if (name) this.attach(name);
    else await this.resumeLast();
  }

  async resumeLast() {
    let entries = this.sessions;
    if (!entries.length) {
      try {
        entries = await api.sessions({ workspace: this.workspace || undefined });
        this.sessions = entries;
        this.renderSessions();
      } catch {
        return false;
      }
    }
    if (!entries.length) return false;
    const remembered = localStorage.getItem(STORAGE.session);
    const chosen =
      entries.find((entry) => entry.name === remembered) ||
      entries.find((entry) => entry.active) ||
      [...entries].sort(byRecent)[0];
    if (!chosen) return false;
    void this.attach(chosen.name);
    return true;
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
  setConn(state, label = null) {
    this.connState = state;
    const node = byId('conn');
    node.dataset.state = state;
    byId('token-btn').hidden = !api.token && state !== 'unauthed';
    const text =
      { connecting: 'connecting', open: 'live', offline: 'reconnecting…', closed: 'idle', unauthed: 'token needed' }[
        state
      ] || state;
    byId('conn-label').textContent = label || text;
  }
  // ------------------------------------------------------------- events
  bindEvents() {
    const on = (id, event, fn, opts) => byId(id).addEventListener(event, fn, opts);
    on('menu-btn', 'click', () => this.toggleSidebar());
    on('close-sidebar', 'click', () => this.toggleSidebar(false));
    on('scrim', 'click', () => this.toggleSidebar(false));
    on('new-chat', 'click', () => this.newChat());
    on('send', 'click', () => this.submit());
    on('stop', 'click', () => this.interrupt(true));
    on('jump', 'click', () => this.scrollToLatest());
    on('tools-btn', 'click', () => this.openTools());
    on('model-btn', 'click', () => this.pickModel());
    on('theme-btn', 'click', () => this.pickTheme());
    on('help-btn', 'click', () => this.openHelp());
    on('token-btn', 'click', () => this.tokenMenu());
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
    const gesture = () => this.userScrolled();
    for (const event of ['wheel', 'touchstart', 'touchmove', 'pointerdown'])
      this.log.addEventListener(event, gesture, { passive: true });
    this.log.addEventListener('keydown', (event) => {
      if (['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End', ' '].includes(event.key)) gesture();
    });
    document.addEventListener('keydown', (event) => this.onGlobalKey(event));
  }

  toggleSidebar(open = null) {
    const sidebar = byId('sidebar');
    const next = open === null ? !sidebar.classList.contains('open') : open;
    sidebar.classList.toggle('open', next);
    byId('scrim').hidden = !next;
    byId('menu-btn').setAttribute('aria-expanded', String(next));
  }

  onGlobalKey(event) {
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
    try {
      this.sessions = await api.sessions({ stats: true });
      this.renderSessions();
    } catch {
      /* retry next poll */
    }
  }

  renderSessions() {
    const host = byId('session-list');
    replace(host);
    const sorted = [...this.sessions].sort((a, b) => (a.active !== b.active ? (a.active ? -1 : 1) : byRecent(a, b)));
    if (!sorted.length) {
      host.appendChild(el('div', { class: 'session-group', text: 'no sessions yet — send a message to start one' }));
      return;
    }
    const render = (entries, label) => {
      if (!entries.length) return;
      host.appendChild(el('div', { class: 'session-group', text: label }));
      for (const entry of entries) host.appendChild(this.sessionRow(entry));
    };
    render(
      sorted.filter((entry) => entry.active),
      'live',
    );
    render(
      sorted.filter((entry) => !entry.active),
      'saved',
    );
  }

  sessionRow(entry) {
    const tracked = entry.tracking && (entry.additions || entry.deletions),
      state = entry.state;
    return el(
      'button',
      {
        class: `session-row${entry.parent_id ? ' session-child' : ''}`,
        type: 'button',
        role: 'listitem',
        dataset: { active: String(!!entry.active) },
        'aria-current': entry.name === this.session ? 'true' : null,
        title: entry.name,
        on: { click: () => this.attach(entry.name) },
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
        el('span', { text: sessionWhen(entry) }),
        entry.workspace ? el('span', { class: 'ws', text: innermostDir(entry.workspace) }) : null,
        entry.model ? el('span', { text: entry.model }) : null,
        tracked && entry.additions ? el('span', { class: 'adds', text: `+${entry.additions}` }) : null,
        tracked && entry.deletions ? el('span', { class: 'dels', text: `-${entry.deletions}` }) : null,
        entry.parent_id ? el('span', { text: 'subagent' }) : null,
      ),
    );
  }

  async openSessions() {
    const sorted = [...this.sessions].sort((a, b) => (a.active !== b.active ? (a.active ? -1 : 1) : byRecent(a, b)));
    const liveCount = sorted.filter((entry) => entry.active).length;
    const rows = sorted.map((entry, index) => {
      const live = index < liveCount,
        state = live ? entry.state : 'saved';
      return {
        id: entry.name,
        label: sessionLabel(entry),
        current: entry.name === this.session,
        section: index === 0 && liveCount ? 'live' : index === liveCount && liveCount < sorted.length ? 'saved' : null,
        meta: [
          { text: sessionWhen(entry), class: 'age' },
          { text: state, class: 'state', state },
        ],
      };
    });
    const choice = await showPicker({
      title: 'Sessions',
      text: rows.length ? '' : 'no sessions yet',
      className: 'sessions-picker',
      rows,
    });
    if (choice && choice !== this.session) this.attach(choice);
  }
  // --------------------------------------------------------- attach/SSE
  async attach(name) {
    if (!name) return;
    this.abortStream();
    this.session = name;
    localStorage.setItem(STORAGE.session, name);
    this.setHash(name);
    this.renderTitle();
    this.resetDiff();
    this.diffScopePending = null;
    this.renderDiffBar();
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
          this.addLine(`cannot open ${name}: ${error.message}`, 'error');
          this.setConn('closed');
          this.session = null;
          localStorage.removeItem(STORAGE.session);
          this.setHash(null);
          this.renderTitle();
          return;
        }
        this.setConn('offline', 'reconnecting…');
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
    localStorage.removeItem(STORAGE.session);
    this.status = null;
    this.setHash(null);
    this.resetDiff();
    this.ask = null;
    this.phase = 'idle';
    this.pendingItem = null;
    this.hintItems = [];
    this.transcript = new Transcript();
    this.setConn('closed');
    this.renderTitle();
  }

  newChat() {
    const previous = this.session;
    this.detach();
    this.view.renderAll([]);
    if (previous)
      this.addLine(
        'started a fresh chat — the previous session keeps running in the background (/sessions to go back)',
      );
    this.renderStatus();
    this.renderBars();
    this.input.focus();
  }

  async deleteCurrent() {
    const name = this.session;
    if (!name) {
      this.addLine('no session to delete', 'warn');
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
    this.addLine('deleted: saved session removed — your files are untouched');
    this.pollSessions();
  }

  loadSnapshot(snapshot, keepStatus = false) {
    this.status = snapshot.status || (keepStatus ? this.status : null);
    this.view.renderAll(this.transcript.load(snapshot));
    this.phase = 'idle';
    this.pendingItem = null;
    this.hintItems = [];
    this.syncCompacting(this.status?.state || 'idle');
    this.renderTitle();
    this.renderStatus();
    this.renderBars();
    this.renderInfo();
    this.syncAsk();
    this.refreshDiff();
    this.scrollToLatest();
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
      this.view.renderAll([]);
      this.addLine(`session ${name} was deleted`);
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
  userScrolled() {
    this.userScrollUntil = performance.now() + 400;
  }

  onScroll() {
    const now = performance.now();
    const distance = this.log.scrollHeight - this.log.scrollTop - this.log.clientHeight;
    if (now - this.selfScrollAt < 150) {
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
    };
    if (typeof ResizeObserver === 'function') new ResizeObserver(apply).observe(composer);
    window.addEventListener('resize', apply);
    window.addEventListener('orientationchange', apply);
    apply();
  }

  tick() {
    this.view.tick(this.transcript.items);
    if (this.pendingItem) this.scheduleScroll();
  }
  // -------------------------------------------------------------- status
  get viewStatus() {
    if (this.status) return this.status;
    const config = this.config,
      target = this.draftModel || config?.default_model || null;
    const entry = (config?.models || []).find((model) => model.name === target);
    return {
      state: 'idle',
      mode: this.draftMode,
      model_name: target || '',
      model_display_name: entry?.display_name || config?.default_model_display_name || '—',
      model_id: '',
      cot_strength: entry?.cot_strength || 'medium',
      max_context: entry?.max_context || 128000,
      supports_vision: entry?.supports_vision || false,
      sandbox_name: config?.default_sandbox || null,
      workspace: this.workspace,
      usage_total: { input_tokens: 0, output_tokens: 0 },
      last_usage: { input_tokens: 0, output_tokens: 0, reasoning_tokens: 0 },
      queued: [],
      pending_ask: null,
      clients_attached: 0,
      filter_list: [],
      refs: [],
      tools: [],
      view: { diff_scope: this.diffScope },
    };
  }

  renderStatus() {
    const status = this.viewStatus;
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
      ['session', shortName(this.session) || 'draft (no session yet)'],
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
      dir = innermostDir(status.workspace) || 'draft';
    document.title = `xarness ${this.session ? shortName(this.session) : dir}`;
    byId('title').textContent = this.session ? shortName(this.session) : 'new chat';
    byId('title').title = this.session || 'new chat';
    byId('subtitle').textContent = dir;
  }

  renderBars() {
    const status = this.viewStatus,
      running = status.state === 'running' || status.state === 'compacting';
    byId('stop').hidden = !running;
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
      this.showDiffSummary(this.session ? await api.diff(this.session) : null);
    } catch {
      this.showDiffSummary(null);
    }
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

  renderDiffBar() {
    const bar = byId('diffbar'),
      info = this.diff;
    if (!info) {
      bar.hidden = true;
      replace(bar);
      bar.classList.remove('open');
      return;
    }
    bar.hidden = false;
    bar.classList.toggle('open', this.diffOpen);
    const count = info.files.length,
      label = `Edited ${count} file${count === 1 ? '' : 's'}`;
    const files = info.files || [],
      groups = info.groups || [],
      scope = this.diffScope;
    const isInline = (item) => item.kind === 'mine' || (item.kind === 'agent' && scope === 'all');
    // The totals follow the scope chip: this session's own rows while it reads
    // "agent", every session's rows while "agent" is dimmed. The auto/drift
    // sections never count toward either.
    const inlineFiles = files.filter(isInline);
    const adds = inlineFiles.reduce((total, file) => total + (file.additions || 0), 0);
    const dels = inlineFiles.reduce((total, file) => total + (file.deletions || 0), 0);
    const head = el(
      'div',
      {
        class: 'diff-head',
        role: 'button',
        tabindex: '0',
        'aria-expanded': String(this.diffOpen),
        title: this.diffOpen ? 'Hide the pending file list' : 'Show the pending file list',
        on: {
          click: () => this.toggleDiff(),
          keydown: (event) => {
            if (event.target !== event.currentTarget) return;
            if (event.key !== 'Enter' && event.key !== ' ') return;
            event.preventDefault();
            this.toggleDiff();
          },
        },
      },
      el(
        'span',
        { class: 'diff-label' },
        label,
        info.ignored_files ? el('span', { class: 'dim', text: ` · ${info.ignored_files} ignored` }) : null,
      ),
      el('span', { class: 'diff-spacer' }),
      el(
        'button',
        {
          class: 'diff-scope',
          type: 'button',
          title: 'Toggle whose changes this counts (every session / this session)',
          on: {
            click: (event) => {
              event.stopPropagation();
              this.toggleDiffScope();
            },
          },
        },
        this.diffScope === 'all'
          ? [el('span', { class: 'hi', text: 'age' }), el('span', { class: 'dim', text: 'nt' })]
          : el('span', { class: 'hi', text: 'agent' }),
      ),
      el('span', { class: 'adds', text: `+${adds}` }),
      el('span', { class: 'dels', text: `-${dels}` }),
    );
    const body = el('div', { class: 'diff-body' });
    if (this.diffOpen) {
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
      const list = el('div', { class: 'diff-list' });
      list.style.setProperty('--add-w', `${columnWidth('additions', '+')}ch`);
      list.style.setProperty('--del-w', `${columnWidth('deletions', '-')}ch`);
      for (const file of leadingFiles) list.appendChild(this.diffRow(file));
      for (const group of leadingGroups) list.appendChild(this.diffGroupRow(group));
      if (leadingFiles.length || leadingGroups.length) list.appendChild(el('div', { class: 'diff-divider' }));
      for (const section of sections) {
        const sectioned = sectionFiles(section),
          grouped = sectionGroups(section),
          expanded = this.diffExpanded.has(section);
        const ignored = grouped.reduce((total, group) => total + (group.files || 0), 0);
        const adds = sectioned.reduce((total, file) => total + (file.additions || 0), 0);
        const dels = sectioned.reduce((total, file) => total + (file.deletions || 0), 0);
        const empty = !sectioned.length && !grouped.length;
        list.appendChild(
          el(
            'button',
            {
              class: `diff-section ${section}${expanded ? ' expanded' : ''}${empty ? ' empty' : ''}`,
              type: 'button',
              'aria-expanded': String(expanded),
              title: expanded ? `Fold ${section}` : `Unfold ${section}`,
              on: { click: () => this.toggleDiffSection(section) },
            },
            el('span', { class: 'name', text: section }),
            el('span', { class: 'spacer' }),
            el('span', { class: 'count', text: ignored ? `${ignored} files` : '' }),
            el('span', { class: 'adds', text: `+${adds}` }),
            el('span', { class: 'dels', text: `-${dels}` }),
          ),
        );
        if (!expanded) continue;
        for (const file of sectioned) list.appendChild(this.diffRow(file));
        for (const group of grouped) list.appendChild(this.diffGroupRow(group));
      }
      if (!files.length && !groups.length)
        list.appendChild(el('div', { class: 'diff-empty', text: 'no pending changes' }));
      body.appendChild(list);
      body.appendChild(
        el(
          'div',
          { class: 'diff-hint' },
          'Run ',
          el('span', { class: 'cmd', text: '/accept' }),
          ' to lock these in, ',
          el('span', { class: 'cmd', text: '/reject' }),
          ' to discard them.',
        ),
      );
    }
    replace(bar, head, body);
  }

  toggleDiffSection(section) {
    if (this.diffExpanded.has(section)) this.diffExpanded.delete(section);
    else this.diffExpanded.add(section);
    this.renderDiffBar();
  }

  diffRow(file) {
    const current = this.diffPatch?.path === file.path,
      kind = file.kind === 'mine' ? 'mine' : file.kind;
    return el(
      'button',
      {
        class: 'diff-row',
        type: 'button',
        'aria-current': current ? 'true' : null,
        title: file.path,
        on: { click: () => this.openPatch(file.path, file.author ?? null) },
      },
      el('span', { class: 'path', text: file.path }),
      file.is_new ? el('span', { class: 'new', text: 'new' }) : null,
      file.author === null && file.authors?.length > 1
        ? el('span', { class: 'shared', text: String(file.authors.length) })
        : null,
      el('span', { class: `badge ${kind}`, text: DIFF_BADGE[file.kind] || file.kind }),
      el('span', { class: 'adds', text: `+${file.additions || 0}` }),
      el('span', { class: 'dels', text: `-${file.deletions || 0}` }),
    );
  }

  diffGroupRow(group) {
    const current = this.diffPatch?.path === group.prefix;
    return el(
      'button',
      {
        class: 'diff-row diff-group',
        type: 'button',
        title: group.prefix,
        'aria-current': current ? 'true' : null,
        on: { click: () => this.openPatch(group.prefix, null) },
      },
      el('span', { class: 'path', text: group.prefix }),
      el('span', { class: `badge ${group.kind}`, text: DIFF_BADGE[group.kind] || group.kind }),
      el('span', { class: 'count', text: `${group.files} files` }),
    );
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
    if (!this.session) return;
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
    if (!this.session) return;
    if (this.diffPatch?.path === path) {
      this.closePatch();
      return;
    }
    try {
      const info = await api.diff(this.session, { path, author });
      if (!String(info.text || '').trim()) {
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
    if (!this.session) {
      this.addLine('/diff: no session attached — send a message first', 'error');
      return;
    }
    if (!arg) {
      await this.refreshDiff();
      if (!this.diff) {
        this.addLine('no pending changes to diff', 'warn');
        return;
      }
      this.diffOpen = true;
      this.renderDiffBar();
      return;
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
      workspace: this.workspace || undefined,
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
      ...filters.map((path) => ({ id: `remove:${path}`, label: `✕ ${path}`, detail: 'remove' })),
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
      'Enter            run the highlighted command, else send',
      'Shift+Enter      newline',
      'Tab              complete the command, else toggle plan / write',
      'Esc              close a popup; twice interrupts a running turn',
      '/                slash commands',
      '@                mention a workspace file',
      'Ctrl/Cmd+K       command palette',
      '↑↓ / ←→          move in the palette and in a picker',
    ].join('\n');
    const body = el('div');
    body.appendChild(el('pre', { class: 'keys', text: keys }));
    body.appendChild(sheetSection('commands'));
    body.appendChild(el('pre', { class: 'keys', text: COMMANDS.map((command) => commandUsage(command)).join('\n') }));
    body.appendChild(sheetSection('notes'));
    body.appendChild(
      el('pre', {
        class: 'keys',
        text:
          'Every command runs on one Enter; a command that opens a\n' +
          'picker (/model, /theme, /filter, /sessions, /tools) shows it\n' +
          'right away. Tab completes the word if you want to type an\n' +
          'argument. Tool rows and every "Thought for" pill expand on tap.',
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
        row.args ? el('span', { class: 'palette-args', text: row.args }) : null,
        row.detail ? el('span', { class: 'palette-desc', text: row.detail }) : null,
      ),
    );
    replace(list, ...this.paletteNodes);
    this.renderPaletteFoot(rows[this.paletteIndex]);
    palette.hidden = false;
    this.scheduleScroll();
  }

  renderPaletteFoot(row) {
    const foot = byId('palette-foot');
    if (!foot) return;
    const hint =
      this.paletteKind === 'at' ? '↵ insert · ↑↓ choose · esc close' : '↵ run · ⇥ complete · ↑↓ choose · esc close';
    replace(
      foot,
      row?.detail
        ? el('span', { class: 'palette-note', text: row.detail })
        : el('span', { class: 'palette-note', text: row?.label || '' }),
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
