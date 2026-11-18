/** DOM renderers for transcript items. One element per item, updated in
 * place: streaming appends only settled chunks and rewrites the live tail. */
import { el } from './dom.js';
import { STATUS, formatDuration, formatElapsed, formatTokens, formatUsage } from './state.js';
import { appendMarkdown } from './markdown.js';
import { looksLikePatch, patchView } from './diff.js';

export function tokenText(image) {
  return `[🖼 ${image.name} ${image.width}×${image.height} #${String(image.id || '').slice(0, 4)}]`;
}
const GLYPHS = { error: '✗ ', warn: '! ', success: '✓ ' };
const now = () => performance.now() / 1000;
const parseArgs = (text) => {
  try {
    const value = JSON.parse(text);
    return value && typeof value === 'object' && !Array.isArray(value) ? value : null;
  } catch {
    return null;
  }
};
const shorten = (text, limit = 60) => {
  const one = String(text).split(/\s+/).filter(Boolean).join(' ');
  return one.length <= limit ? one : `${one.slice(0, limit - 1)}…`;
};

function partialValues(text, key, unescape = false) {
  const re = new RegExp(`"${key}"\\s*:\\s*"`, 'g'),
    out = [];
  let match;
  while ((match = re.exec(text)) !== null) {
    let i = match.index + match[0].length,
      value = '';
    while (i < text.length) {
      const char = text[i];
      if (char === '\\') {
        const next = text[i + 1];
        if (next === undefined) {
          i += 1;
          break;
        }
        if (unescape) {
          if (next === 'n') value += '\n';
          else if (next === 't') value += '\t';
          else if (next !== 'r') value += next;
        }
        i += 2;
        continue;
      }
      if (char === '"') break;
      value += char;
      i += 1;
    }
    out.push(value);
  }
  return out;
}

function partialString(text, key, unescape = false) {
  const values = partialValues(text, key, unescape);
  return values.length ? values[0] : '';
}

function textLines(text) {
  if (!text) return [];
  const parts = String(text).replace(/\r\n/g, '\n').split('\n');
  if (parts[parts.length - 1] === '') parts.pop();
  return parts;
}
const lineCount = (text) => textLines(text).length;

function diffCounts(oldText, newText) {
  const a = textLines(oldText),
    b = textLines(newText);
  let start = 0;
  while (start < a.length && start < b.length && a[start] === b[start]) start += 1;
  let endA = a.length,
    endB = b.length;
  while (endA > start && endB > start && a[endA - 1] === b[endB - 1]) {
    endA -= 1;
    endB -= 1;
  }
  const midA = a.slice(start, endA),
    midB = b.slice(start, endB);
  if (!midA.length) return [midB.length, 0];
  if (!midB.length) return [0, midA.length];
  if (midA.length * midB.length > 250000) return [midB.length, midA.length];
  let prev = new Uint32Array(midB.length + 1),
    cur = new Uint32Array(midB.length + 1);
  for (let i = 1; i <= midA.length; i += 1) {
    for (let j = 1; j <= midB.length; j += 1) {
      cur[j] = midA[i - 1] === midB[j - 1] ? prev[j - 1] + 1 : Math.max(prev[j], cur[j - 1]);
    }
    [prev, cur] = [cur, prev];
    cur.fill(0);
  }
  const common = prev[midB.length];
  return [midB.length - common, midA.length - common];
}

function editPairs(argsText) {
  const args = parseArgs(argsText);
  if (args) {
    if (Array.isArray(args.edits)) {
      return args.edits
        .filter((edit) => edit && typeof edit.old_string === 'string' && edit.old_string)
        .map((edit) => ({
          old_string: edit.old_string,
          new_string: typeof edit.new_string === 'string' ? edit.new_string : '',
        }));
    }
    if (typeof args.old_string === 'string' && args.old_string) {
      return [{ old_string: args.old_string, new_string: typeof args.new_string === 'string' ? args.new_string : '' }];
    }
    return [];
  }
  const olds = partialValues(argsText, 'old_string', true);
  if (!olds.length) return [];
  const news = partialValues(argsText, 'new_string', true);
  return olds.map((old, i) => ({ old_string: old, new_string: news[i] || '' }));
}

function editTotals(argsText) {
  let adds = 0,
    dels = 0,
    found = false;
  for (const pair of editPairs(argsText)) {
    const [a, d] = diffCounts(pair.old_string, pair.new_string);
    adds += a;
    dels += d;
    found = true;
  }
  return found ? [adds, dels] : null;
}

function editDiffText(argsText) {
  const pairs = editPairs(argsText);
  if (!pairs.length) return '';
  const lines = [];
  pairs.forEach((pair, index) => {
    lines.push(`@@ edit ${index + 1} @@`);
    for (const line of textLines(pair.old_string)) lines.push(`-${line}`);
    for (const line of textLines(pair.new_string)) lines.push(`+${line}`);
  });
  return lines.join('\n');
}

export function displayName(name) {
  if (!name || !name.startsWith('mcp__')) return name || 'tool';
  const [server, tool] = name.slice(5).split('__');
  return tool ? `${server} · ${tool}` : server;
}

export function toolDetail(name, argsText, outputText = '') {
  const args = parseArgs(argsText);
  const path = args && typeof args.path === 'string' ? args.path : partialString(argsText, 'path');
  if (name === 'write_file') {
    if (args && typeof args.content === 'string') return `+${lineCount(args.content)} ${args.path || ''}`.trim();
    const partial = partialString(argsText, 'content', true);
    if (partial) return `+${lineCount(partial)} ${path}`.trim();
    return (args ? args.path : path) || '';
  }
  if (name === 'edit_file') {
    const totals = editTotals(argsText);
    return `${totals && (totals[0] || totals[1]) ? `+${totals[0]} -${totals[1]}` : ''} ${path}`.trim();
  }
  if (name === 'read_file') {
    const start = args?.start_line,
      end = args?.end_line;
    const range =
      start !== undefined && start !== null
        ? ` [${start}, ${end ?? '…'}]`
        : end !== undefined && end !== null
          ? ` […, ${end}]`
          : '';
    return `${path}${range}`.trim();
  }
  if (args) {
    for (const key of [
      'path',
      'file_path',
      'filename',
      'command',
      'query',
      'url',
      'pattern',
      'name',
      'prompt',
      'message',
    ]) {
      if (typeof args[key] === 'string' && args[key]) return shorten(args[key], 80);
    }
    if (name.startsWith('mcp__') && Object.keys(args).length) return shorten(JSON.stringify(args), 72);
    return '';
  }
  for (const key of ['path', 'file_path', 'command', 'query', 'url', 'pattern']) {
    const value = partialString(argsText, key);
    if (value) return shorten(value, 80);
  }
  return '';
}

// ------------------------------------------------------------ activity shapes
// Every thought/tool call gets a shape. Thinking is a triangle in the theme's
// status colour; a built-in tool call is a square whose hue says which tool it
// was, so a long run reads as "bash, read, edit" without a legend; an MCP tool
// call is a circle in its server's deterministic colour, so one server is
// recognisable across a turn and reads apart from the built-ins at a glance.
// The shapes are SVG, not glyphs, so they have real fills and a stroke in the
// page colour.
const SVG_NS = 'http://www.w3.org/2000/svg';
const MCP_COLORS = ['#d8648f', '#2fa3a3', '#c08a1f', '#7f6cf0', '#4f9b64', '#d46b3c', '#b168d6', '#4c89d6'];
// Mid-tone, saturated hues: legible on #fafafa and on #121212 alike.
const TOOL_COLORS = {
  run_bash: '#e0a526', bash: '#e0a526',   // golden amber
  read_file: '#8bc34a',                   // lime
  write_file: '#2ebd85',                  // emerald
  edit_file: '#b86bff',                   // violet
  list_dir: '#14b8a6',                    // teal
  glob: '#ff8a4c',                        // orange
  grep: '#f2609b',                        // pink
  web_search: '#9b7bff',                  // purple-indigo
  fetch_url: '#c9a227',                   // mustard
  ask_user: '#f05a5a',                    // coral red
};
const TOOL_FALLBACK_COLORS = [...new Set(Object.values(TOOL_COLORS))];
const SHAPE_PATHS = {
  thinking: 'M8 2.5 14.2 13.4H1.8Z',
  tool: 'M2.7 2.7h10.6v10.6H2.7Z',
  mcp: 'M8 2.6a5.4 5.4 0 1 0 0 10.8a5.4 5.4 0 1 0 0-10.8Z',
};
const toolFailed = (item) => item.status === STATUS.FAILED || item.status === STATUS.PARSE_ERROR;
const serverOf = (name) => (name && name.startsWith('mcp__') ? name.slice(5).split('__')[0] : null);

function hashText(text) {
  let hash = 0;
  for (const ch of String(text || '')) hash = (hash * 31 + ch.codePointAt(0)) >>> 0;
  return hash;
}

function mcpColor(server) {
  return MCP_COLORS[hashText(server) % MCP_COLORS.length];
}

// A tool the UI has never heard of still gets a stable colour of its own.
function toolColor(name) {
  return TOOL_COLORS[name] || TOOL_FALLBACK_COLORS[hashText(name) % TOOL_FALLBACK_COLORS.length];
}

function activityMeta(item) {
  if (item.kind === 'thinking') return { kind: 'thinking' };
  const server = serverOf(item.name);
  if (server) return { kind: 'mcp', server };
  return { kind: 'tool', name: item.name };
}

function activityStatus(item) {
  if (item.kind === 'thinking') return item.done ? 'done' : 'thinking';
  if (item.status === STATUS.MAKING) return 'running';
  if (toolFailed(item)) return 'failed';
  return 'ok';
}

function shapeIcon(meta, status = 'ok') {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', '0 0 16 16');
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('focusable', 'false');
  svg.setAttribute('class', `shape shape-${meta.kind}`);
  svg.dataset.status = status;
  if (meta.kind === 'mcp') svg.style.color = mcpColor(meta.server);
  else if (meta.kind === 'tool') svg.style.color = toolColor(meta.name);
  const path = document.createElementNS(SVG_NS, 'path');
  path.setAttribute('d', SHAPE_PATHS[meta.kind] || SHAPE_PATHS.tool);
  path.setAttribute('fill', 'currentColor');
  path.setAttribute('stroke', 'var(--bg)');
  path.setAttribute('stroke-width', '1.5');
  path.setAttribute('stroke-linejoin', 'round');
  svg.appendChild(path);
  return svg;
}

function activitySummary(items) {
  // Two tallies only: how many times it thought and how many tools it ran. The
  // per-tool breakdown lives in the expanded rows, not in this one line.
  let thoughts = 0;
  let tools = 0;
  for (const item of items) {
    if (item.kind === 'thinking') thoughts += 1;
    else tools += 1;
  }
  const parts = [];
  if (thoughts) parts.push(`thought ${thoughts} ${thoughts === 1 ? 'time' : 'times'}`);
  if (tools) parts.push(`ran ${tools} ${tools === 1 ? 'tool' : 'tools'}`);
  const text = parts.join(', ');
  return text ? text[0].toUpperCase() + text.slice(1) : '';
}

function activityCurrent(item) {
  if (!item) return { name: '', detail: '' };
  if (item.kind === 'thinking') return { name: thinkingText(item), detail: '' };
  return {
    name: `${STATUS_VERB[item.status] || ''}${displayName(item.name)}`,
    detail: toolFailed(item) ? '' : toolDetail(item.name, item.args, item.output),
  };
}

function shapeTitle(item) {
  if (item.kind === 'thinking') {
    if (!item.done) return 'thinking';
    return item.duration !== null ? `thought · ${formatDuration(item.duration)}` : 'thought';
  }
  const detail = toolFailed(item) ? '' : toolDetail(item.name, item.args, item.output);
  return `${displayName(item.name)}${detail ? ` · ${detail}` : ''}`;
}

function activityPreviewText(item) {
  if (!item) return '';
  if (item.kind === 'thinking') return item.text || '';
  if (item.output) return item.output;
  const args = parseArgs(item.args);
  if (item.name === 'write_file') {
    const content = args && typeof args.content === 'string' ? args.content : partialString(item.args, 'content', true);
    if (content) return content;
  }
  if (item.name === 'run_bash' || item.name === 'bash') {
    const command = args && typeof args.command === 'string' ? args.command : partialString(item.args, 'command', true);
    if (command) return command;
  }
  if (item.name === 'edit_file') {
    const diff = editDiffText(item.args);
    if (diff) return diff;
  }
  return args ? JSON.stringify(args, null, 2) : item.args || '';
}

// Line totals for every edit made inside one activity block: `+12 -3`, or null
// when the block did not touch a file.
function activityLocStats(items) {
  let adds = 0,
    dels = 0,
    touched = false;
  for (const item of items) {
    if (item.kind !== 'tool' || toolFailed(item)) continue;
    if (item.name === 'edit_file') {
      const totals = editTotals(item.args);
      if (!totals) continue;
      adds += totals[0];
      dels += totals[1];
      touched = true;
    } else if (item.name === 'write_file') {
      const args = parseArgs(item.args);
      const content = args && typeof args.content === 'string' ? args.content : partialString(item.args, 'content', true);
      if (!content) continue;
      adds += lineCount(content);
      touched = true;
    }
  }
  return touched ? { adds, dels } : null;
}

// Tokens the block spent. Two sources: `roundUsage`, the model round that
// produced the item (input/output of the whole response), and `usage`, the
// local spend a tool reported for rounds it ran itself. A thought with nothing
// but a reasoning-token count falls back to that. `~` when any figure is
// approximate.
function activityUsage(items) {
  let input = 0,
    output = 0,
    approximate = false,
    found = false;
  const add = (usage) => {
    input += usage.input_tokens || 0;
    output += usage.output_tokens || 0;
    approximate = approximate || !!usage.approximate;
    found = true;
  };
  for (const item of items) {
    if (item.roundUsage) add(item.roundUsage);
    if (item.usage) add(item.usage);
    if (!item.roundUsage && !item.usage && item.kind === 'thinking' && item.tokens) {
      output += item.tokens;
      found = true;
    }
  }
  return found ? { input_tokens: input, output_tokens: output, approximate } : null;
}

function thinkingSnippet(item) {
  const line = String(item.text || '')
    .split('\n')
    .reverse()
    .find((entry) => entry.trim());
  return shorten(line || (item.done ? 'no reasoning text' : 'thinking…'), 88);
}
const STATUS_VERB = {
  [STATUS.MAKING]: '',
  [STATUS.PARSE_ERROR]: 'Failed ',
  [STATUS.OK]: '',
  [STATUS.FAILED]: 'Failed ',
};

function createNode(item) {
  switch (item.kind) {
    case 'user': {
      const text = el('div', { class: 'text' }),
        preview = typeof this.resolveImage === 'function';
      renderUserText(text, item, preview);
      const images = preview ? userImages(item.images || [], this.resolveImage) : null;
      const bubble = el('div', { class: 'bubble', tabindex: '0', role: 'button' }, text, images);
      if (!text.textContent.trim()) {
        text.hidden = true;
        if (images) images.style.marginTop = '0';
      }
      bubble.addEventListener('click', () => {
        if (!item.messageId) return;
        item.armed = !item.armed;
        node.classList.toggle('armed', item.armed);
      });
      const undo = el('button', { class: 'undo', type: 'button', text: '↩ undo to here' });
      undo.addEventListener('click', () => {
        item.armed = false;
        bubble.parentElement.classList.remove('armed');
        this.onUndoTo?.(item.messageId);
      });
      const node = el('article', { class: 'item item-user' }, bubble, undo);
      node._text = text;
      return node;
    }
    case 'answer': {
      const live = el('div', { class: 'md-live' }),
        prose = el('div', { class: 'prose' }, live);
      const node = el('article', { class: 'item item-answer' }, el('div', { class: 'body' }, prose));
      node._prose = prose;
      node._live = live;
      node._chunks = 0;
      return node;
    }
    case 'pending': {
      const label = el('span', { class: 'pending-label', text: item.label }),
        elapsed = el('span', { class: 'pending-elapsed' });
      const hint = el('span', { class: 'pending-hint', text: `(${item.hint})` });
      const node = el('article', { class: 'item item-pending' }, label, elapsed, hint);
      node._elapsed = elapsed;
      node._labelText = label;
      return node;
    }
    case 'line':
      return el('article', {
        class: 'item item-line',
        dataset: { kind: item.level },
        text: `${GLYPHS[item.level] || ''}${item.text}`,
      });
    case 'footer':
      return el('article', { class: 'item item-footer', text: footerText(item) });
    case 'divider': {
      const body = el('div', { class: 'handoff-body', text: item.text });
      const head = el(
        'button',
        { class: 'handoff-head', type: 'button' },
        el('span', { class: 'pill-icon', text: '▸' }),
        el('span', { text: 'earlier conversation, summarized' }),
      );
      const node = el('article', { class: 'item item-divider' }, el('div', { class: 'handoff' }, head, body));
      head.addEventListener('click', () => node.querySelector('.handoff').classList.toggle('open'));
      return node;
    }
    default:
      return el('article', { class: 'item item-line', text: String(item.kind) });
  }
}

function renderUserText(host, item, preview = false) {
  host.replaceChildren();
  const byToken = new Map();
  for (const image of item.images || []) byToken.set(tokenText(image), image);
  const re = /\[🖼 [^\]\n]+\]/g;
  let last = 0,
    match;
  while ((match = re.exec(item.text)) !== null) {
    if (match.index > last) host.appendChild(document.createTextNode(item.text.slice(last, match.index)));
    if (!preview || !byToken.has(match[0])) host.appendChild(el('span', { class: 'chip', text: match[0] }));
    last = match.index + match[0].length;
  }
  if (last < item.text.length) host.appendChild(document.createTextNode(item.text.slice(last)));
  for (const image of item.images || []) {
    const token = tokenText(image);
    if (!preview && !item.text.includes(token))
      host.appendChild(el('div', {}, el('span', { class: 'chip', text: token })));
  }
}

function userImages(images, resolve) {
  if (!images.length) return null;
  const wrap = el('div', { class: 'user-images' });
  for (const image of images) {
    const img = el('img', {
      class: 'user-image',
      alt: image.name || 'attached image',
      title: `${image.name || 'image'} ${image.width || '?'}×${image.height || '?'}`,
    });
    img.hidden = true;
    if (image.width && image.height) img.style.aspectRatio = `${image.width} / ${image.height}`;
    wrap.appendChild(img);
    Promise.resolve()
      .then(() => resolve(image))
      .then((url) => {
        if (!url) throw new Error('image unavailable');
        img.src = url;
        img.hidden = false;
      })
      .catch(() => img.replaceWith(el('span', { class: 'chip', text: tokenText(image) })));
  }
  return wrap;
}

function thinkingText(item) {
  if (!item.done) {
    const elapsed = item.started !== null ? formatElapsed(now() - item.started) : '';
    return elapsed ? `Thinking ${elapsed}` : 'Thinking';
  }
  const text = item.duration !== null ? `Thought for ${formatDuration(item.duration)}` : 'Thought for …';
  return item.tokens ? `${text} · ${formatTokens(item.tokens)} tokens` : text;
}

function footerText(item) {
  let text = `Worked for ${formatDuration(item.total)}`;
  if (item.busy !== null && item.busy !== undefined && item.idle !== null && item.idle !== undefined) {
    text += ` (${formatDuration(item.busy)} | ${formatDuration(item.idle)})`;
  }
  if (item.usage) text += ` · ${formatUsage(item.usage)}`;
  return text;
}
const COUNTS_RE = /^(\+\d+)(?: (-\d+))?(?: ([\s\S]+))?$/;

function setDetail(node, text) {
  node.replaceChildren();
  const match = text ? COUNTS_RE.exec(text) : null;
  if (!match) {
    node.textContent = text || '';
    return;
  }
  node.appendChild(el('span', { class: 'loc-add', text: match[1] }));
  if (match[2]) node.appendChild(el('span', { class: 'loc-del', text: ` ${match[2]}` }));
  if (match[3]) node.appendChild(el('span', { class: 'loc-path', text: ` ${match[3]}` }));
}

function renderToolBody(item, body) {
  const failed = toolFailed(item);
  const args = parseArgs(item.args);
  let argsNode = null,
    argsLabel = 'arguments';
  if (!failed && item.name === 'write_file') {
    const content = args && typeof args.content === 'string' ? args.content : partialString(item.args, 'content', true);
    if (content) {
      argsNode = el('pre', { class: 'tool-args', text: content });
      argsLabel = 'content';
    }
  } else if (!failed && item.name === 'edit_file' && !item.output) {
    const diff = editDiffText(item.args);
    if (diff) {
      argsNode = patchView(diff);
      argsLabel = 'diff';
    }
  }
  const skipArgs = failed || item.name === 'edit_file' || (item.name === 'read_file' && !!item.output);
  if (!argsNode && !skipArgs && item.args.trim()) {
    argsNode = args
      ? el('pre', { class: 'tool-args', text: JSON.stringify(args, null, 2) })
      : el('pre', { class: 'tool-args', text: item.args });
  }
  if (argsNode) {
    argsNode.setAttribute('aria-label', 'tool arguments');
    body.appendChild(
      el('div', { class: 'tool-block' }, el('span', { class: 'tool-label', text: argsLabel }), argsNode),
    );
  }
  if (item.output) {
    const out = !failed && looksLikePatch(item.output) ? patchView(item.output) : el('pre', { text: item.output });
    out.setAttribute('aria-label', 'tool output');
    body.appendChild(el('div', { class: 'tool-block' }, el('span', { class: 'tool-label', text: 'output' }), out));
  }
  if (!item.args.trim() && !item.output)
    body.appendChild(el('div', { class: 'tool-note', text: 'no output yet' }));
}

/** One run of thinking + tool calls, between two pieces of assistant prose.
 *
 * Collapsed it is a header (aggregate shapes, a generated summary, the current
 * activity and its state) plus an eight-line window on what the current
 * activity is writing. Expanded it lists every activity as shape · name ·
 * arguments; each row opens its own arguments/output. */
class ActivityBlock {

  constructor(view) {
    this.view = view;
    this.items = [];
    this.rows = new Map();
    this.expanded = false;
    this.closed = false;
    this.previewPinned = true;
    this.previewItemId = null;
    this.shapes = el('span', { class: 'activity-shapes' });
    this.shapeSig = null;
    this.titleText = el('span', { class: 'activity-title-text' });
    this.title = el('span', { class: 'activity-title' }, this.titleText);
    this.inlineShape = null;
    this.loc = el('span', { class: 'activity-loc' });
    this.locSig = null;
    this.usage = el('span', { class: 'activity-usage' });
    this.usageSig = null;
    this.time = el('span', { class: 'activity-time' });
    this.startedAt = null;
    this.frozen = null;
    this.measured = false;
    this.times = new Map();
    this.head = el(
      'button',
      { class: 'activity-head', type: 'button', 'aria-expanded': 'false' },
      el('span', { class: 'activity-copy' }, this.title, this.shapes),
      this.loc,
      this.usage,
      this.time,
    );
    this.list = el('div', { class: 'activity-list', role: 'list' });
    this.panel = el('div', { class: 'activity-panel' }, this.list);
    this.lines = el('pre', { class: 'activity-lines', 'aria-label': 'current activity' });
    this.preview = el('div', { class: 'activity-preview' }, this.lines);
    this.node = el('section', { class: 'item item-activity activity' }, this.head, this.panel, this.preview);
    this.head.addEventListener('click', () => {
      this.expanded = !this.expanded;
      this.refresh();
    });
    this.lines.addEventListener(
      'scroll',
      () => {
        const distance = this.lines.scrollHeight - this.lines.scrollTop - this.lines.clientHeight;
        this.previewPinned = distance <= 4;
      },
      { passive: true },
    );
  }

  get last() {
    return this.items[this.items.length - 1] || null;
  }

  add(item) {
    this.items.push(item);
    if (this.startedAt === null) this.startedAt = now();
    this.stamp(item);
    const row = this.createRow(item);
    this.rows.set(item.id, row);
    this.list.appendChild(row.node);
    this.updateRow(row, item);
    this.refresh();
  }

  update(item) {
    const row = this.rows.get(item.id);
    if (!row) {
      this.add(item);
      return;
    }
    // the reducer mutates items in place, but a replacement object must also take
    // over the slot in `items` so the summary and the loc totals stay current.
    if (row.item !== item) {
      const index = this.items.indexOf(row.item);
      if (index >= 0) this.items[index] = item;
      row.item = item;
    }
    this.stamp(item);
    this.updateRow(row, item);
    this.refresh();
  }

  // A block is timed from its first activity to the moment it closes. Tool calls
  // are stamped as they start and finish, so a replayed block (which arrives
  // without timings) can still add up whatever the transcript knows.
  stamp(item) {
    if (item.kind === 'thinking') {
      if (!item.done) this.measured = true;
      return;
    }
    if (item.kind !== 'tool') return;
    const entry = this.times.get(item.id);
    const running = item.status === STATUS.MAKING;
    if (!entry) {
      if (running) {
        this.times.set(item.id, { start: now(), end: null });
        this.measured = true;
      }
      return;
    }
    if (!running && entry.end === null) entry.end = now();
  }

  // Seconds a block took: the measured wall clock while it ran, the sum of its
  // parts (marked approximate) when it came back from the log.
  timeText() {
    if (!this.closed && this.measured && this.startedAt !== null)
      return formatDuration(Math.max(0, now() - this.startedAt));
    if (this.frozen !== null) return formatDuration(Math.max(0, this.frozen));
    let sum = 0,
      parts = 0,
      complete = true;
    for (const item of this.items) {
      let seconds = null;
      if (item.kind === 'thinking') seconds = typeof item.duration === 'number' ? item.duration : null;
      else {
        const entry = this.times.get(item.id);
        if (entry && entry.end !== null) seconds = entry.end - entry.start;
      }
      if (seconds === null) complete = false;
      else {
        sum += seconds;
        parts += 1;
      }
    }
    if (!parts) return '';
    return `${complete ? '' : '~'}${formatDuration(sum)}`;
  }

  remove(item) {
    const row = this.rows.get(item.id);
    if (!row) return;
    row.node.remove();
    this.rows.delete(item.id);
    this.items = this.items.filter((entry) => entry !== item && entry !== row.item);
    if (!this.items.length) {
      this.node.remove();
      if (this.view.openBlock === this) this.view.openBlock = null;
    } else this.refresh();
  }

  close() {
    if (this.closed) return;
    this.closed = true;
    if (this.measured && this.startedAt !== null) this.frozen = now() - this.startedAt;
    this.refresh();
  }

  createRow(item) {
    const shape = el('span', { class: 'activity-row-icon' });
    const name = el('span', { class: 'activity-row-name' });
    const args = el('span', { class: 'activity-row-args' });
    const state = el('span', { class: 'activity-row-state' });
    const caret = el('span', { class: 'activity-row-caret', text: '▸' });
    const head = el(
      'button',
      { class: 'activity-row-head', type: 'button', 'aria-expanded': 'false' },
      shape,
      el('span', { class: 'activity-row-label' }, name, args),
      state,
      caret,
    );
    const body = el('div', { class: 'activity-row-body' });
    const node = el('div', { class: 'activity-row', role: 'listitem' }, head, body);
    const row = { node, head, shape, name, args, state, caret, body, item, shapeKey: null, signature: null };
    head.addEventListener('click', () => {
      row.item.expanded = !row.item.expanded;
      this.updateRow(row, row.item);
    });
    return row;
  }

  updateRow(row, item) {
    const status = activityStatus(item);
    const meta = activityMeta(item);
    const shapeKey = `${meta.kind}:${meta.server || ''}:${status}`;
    if (row.shapeKey !== shapeKey) {
      row.shapeKey = shapeKey;
      row.shape.replaceChildren(shapeIcon(meta, status));
    }
    row.state.dataset.state = status;
    row.node.dataset.status = status;
    if (item.kind === 'thinking') {
      row.name.textContent = item.done ? 'Thought' : 'Thinking';
      row.args.textContent = thinkingSnippet(item);
      if (item.done) {
        const stats = [];
        if (item.duration !== null) stats.push(formatDuration(item.duration));
        if (item.tokens) stats.push(formatTokens(item.tokens));
        row.state.textContent = stats.join(' · ');
      } else row.state.textContent = 'thinking';
    } else {
      const failed = toolFailed(item);
      row.name.textContent = displayName(item.name);
      row.name.classList.toggle('failed', failed);
      setDetail(row.args, failed ? '' : toolDetail(item.name, item.args, item.output));
      row.state.textContent =
        item.status === STATUS.MAKING ? 'running' : failed ? 'failed' : item.usage ? formatUsage(item.usage) : '';
    }
    row.head.setAttribute('aria-expanded', item.expanded ? 'true' : 'false');
    row.node.classList.toggle('open', !!item.expanded);
    const signature =
      item.kind === 'thinking'
        ? `${item.expanded}:${item.text.length}`
        : `${item.expanded}:${item.status}:${item.args.length}:${item.output.length}`;
    if (row.signature === signature) return;
    row.signature = signature;
    row.body.replaceChildren();
    if (!item.expanded) return;
    if (item.kind === 'thinking') {
      row.body.appendChild(
        el(
          'div',
          { class: 'tool-block' },
          el('span', { class: 'tool-label', text: 'thought' }),
          el('pre', { class: 'tool-args', text: item.text || '(no reasoning text)' }),
        ),
      );
      return;
    }
    renderToolBody(item, row.body);
  }

  // `+12 -3` sits between the summary and the state word, and only appears when
  // the block actually wrote to a file.
  updateLoc() {
    const stats = activityLocStats(this.items);
    const signature = stats ? `${stats.adds}:${stats.dels}` : '';
    if (this.locSig === signature) return;
    this.locSig = signature;
    this.loc.replaceChildren();
    this.loc.hidden = !stats;
    if (!stats) {
      this.loc.removeAttribute('title');
      return;
    }
    this.loc.appendChild(el('span', { class: 'loc-add', text: `+${stats.adds}` }));
    if (stats.dels) this.loc.appendChild(el('span', { class: 'loc-del', text: ` -${stats.dels}` }));
    this.loc.title = `${stats.adds} line${stats.adds === 1 ? '' : 's'} added${
      stats.dels ? `, ${stats.dels} removed` : ''
    }`;
  }

  updateUsage() {
    const usage = activityUsage(this.items);
    const shown = usage && (usage.input_tokens || usage.output_tokens) ? usage : null;
    const signature = shown ? `${shown.input_tokens}:${shown.output_tokens}:${shown.approximate}` : '';
    if (this.usageSig === signature) return;
    this.usageSig = signature;
    this.usage.replaceChildren();
    this.usage.hidden = !shown;
    if (!shown) {
      this.usage.removeAttribute('title');
      return;
    }
    if (shown.approximate) this.usage.appendChild(el('span', { class: 'usage-approx', text: '~' }));
    if (shown.input_tokens)
      this.usage.appendChild(el('span', { class: 'usage-in', text: `↑${formatTokens(shown.input_tokens)}` }));
    if (shown.output_tokens)
      this.usage.appendChild(
        el('span', { class: 'usage-out', text: `${shown.input_tokens ? ' ' : ''}↓${formatTokens(shown.output_tokens)}` }),
      );
    this.usage.title = `${shown.approximate ? '~' : ''}${shown.input_tokens} tokens in, ${shown.output_tokens} out`;
  }

  refresh() {
    const last = this.last;
    this.node.classList.toggle('open', this.expanded);
    this.head.setAttribute('aria-expanded', this.expanded ? 'true' : 'false');
    const status = last ? activityStatus(last) : 'ok';
    const running = status === 'thinking' || status === 'running';
    const live = !this.closed && running;
    // the block reports the state of the activity it is on, not the worst one it
    // ever ran: a failure earlier in the run is visible on its own row. Red is
    // only conclusive once the run has moved past the block (an answer closed
    // it), so a call that the next round works around never flashes the time.
    const state = this.closed && status === 'failed' ? 'failed' : live ? 'running' : 'ok';
    this.node.classList.toggle('live', live);
    this.time.textContent = this.timeText();
    this.time.hidden = !this.time.textContent;
    this.time.dataset.state = state;
    this.node.dataset.state = state;

    // One SVG per activity, in the order it ran, wrapping under the summary.
    const itemStatus = (item) => {
      const value = activityStatus(item);
      return this.closed && (value === 'thinking' || value === 'running') ? 'ok' : value;
    };
    const parts = this.items.map((item) => ({ item, meta: activityMeta(item), status: itemStatus(item) }));
    // a lone activity puts its shape on the summary line; a run of them stacks the
    // shapes underneath it, one per call.
    const inline = parts.length === 1;
    const shapeSig = `${inline}:${parts
      .map(({ item, meta, status }) => `${item.id}:${meta.kind}:${meta.server || ''}:${status}`)
      .join('|')}`;
    if (this.shapeSig !== shapeSig) {
      this.shapeSig = shapeSig;
      this.shapes.replaceChildren();
      this.inlineShape?.remove();
      this.inlineShape = null;
      this.title.classList.toggle('single', inline);
      for (const { item, meta, status } of parts) {
        const shape = el('span', { class: 'activity-shape', title: shapeTitle(item) }, shapeIcon(meta, status));
        if (inline) {
          this.inlineShape = shape;
          this.title.appendChild(shape);
        } else this.shapes.appendChild(shape);
      }
    }
    this.shapes.hidden = inline;
    this.titleText.textContent = activitySummary(this.items);
    this.updateLoc();
    this.updateUsage();

    const current = activityCurrent(last);
    this.head.title = `${current.name}${current.detail ? ` · ${current.detail}` : ''}`;

    const text = activityPreviewText(last);
    this.preview.hidden = !text;
    if (this.previewItemId !== (last?.id ?? null)) {
      this.previewItemId = last?.id ?? null;
      this.previewPinned = true;
    }
    if (this.lines.textContent !== text) {
      const offset = this.lines.scrollTop;
      this.lines.textContent = text;
      this.lines.scrollTop = this.previewPinned ? this.lines.scrollHeight : offset;
    } else if (this.previewPinned) this.lines.scrollTop = this.lines.scrollHeight;
  }
}

// Live-only transcript lines that must not break an activity block in two: they
// are pushed while a run is in flight and never replayed from the log, so
// treating them as a boundary would split a segment live and merge it on reload.
const TRANSIENT_KINDS = new Set(['pending']);

export class LogView {

  constructor(root, { onUndoTo = null, resolveImage = null } = {}) {
    this.root = root;
    this.nodes = new Map();
    this.blockOf = new Map();
    this.openBlock = null;
    this.transientNode = null;
    this.onUndoTo = onUndoTo;
    this.resolveImage = resolveImage;
    this.welcome = el(
      'div',
      { class: 'welcome' },
      el('pre', {
        text: `... ...  ....   ... ..  .. ...     ....   ....   ....
 '|..'  '' .||   ||' ''  ||  ||  .|...|| ||. '  ||. '
  .|.   .|' ||   ||      ||  ||  ||      . '|.. . '|..
.|  ||. '|..'|' .||.    .||. ||.  '|...' |'..|' |'..|'`,
      }),
      el('p', { text: 'send a message to start · / for commands · @ for files' }),
    );
  }

  create(item) {
    const node = createNode.call(this, item);
    node.dataset.id = String(item.id);
    this.nodes.set(item.id, node);
    return node;
  }

  update(item) {
    if (item.kind === 'thinking' || item.kind === 'tool') {
      let block = this.blockOf.get(item.id);
      if (!block) {
        block = this.openBlock;
        if (!block) {
          block = new ActivityBlock(this);
          // a live-only "Processing" line sits at the tail of the transcript; the
          // block belongs above it so the two stay in chronological order.
          if (this.transientNode?.isConnected) this.root.insertBefore(block.node, this.transientNode);
          else this.root.appendChild(block.node);
          this.openBlock = block;
          if (this.welcome.isConnected) this.welcome.remove();
        }
        this.blockOf.set(item.id, block);
        block.add(item);
      } else {
        block.update(item);
      }
      const row = block.rows.get(item.id);
      if (row) this.nodes.set(item.id, row.node);
      return row?.node || null;
    }
    let node = this.nodes.get(item.id);
    // Only an item the view has never shown can end a run. The reducer re-syncs
    // items it already emitted (a round settling re-stamps its answer and its
    // thought), and those sit *above* the open block: treating such a replay as
    // a boundary would cut the segment in two here and merge it on a reload.
    if (!node) {
      // live-only status lines are not a boundary either: they are never
      // replayed from the log, so closing here would split a segment on a live
      // run and merge it on a reload.
      if (!TRANSIENT_KINDS.has(item.kind) && this.openBlock) {
        this.openBlock.close();
        this.openBlock = null;
      }
      node = this.create(item);
      this.root.appendChild(node);
      if (this.welcome.isConnected) this.welcome.remove();
    }
    this.transientNode = TRANSIENT_KINDS.has(item.kind) ? node : null;
    switch (item.kind) {
      case 'user':
        node.classList.toggle('armed', !!item.armed);
        break;
      case 'answer': {
        for (let i = node._chunks; i < item.chunks.length; i += 1) {
          const fragment = document.createDocumentFragment();
          appendMarkdown(fragment, item.chunks[i]);
          node._live.before(fragment);
        }
        node._chunks = item.chunks.length;
        if (node._live.textContent !== item.live) node._live.textContent = item.live;
        break;
      }
      case 'pending':
        node._labelText.textContent = item.label;
        node._elapsed.textContent = formatElapsed(now() - item.started);
        break;
      case 'footer':
        node.textContent = footerText(item);
        break;
      default:
        break;
    }
    return node;
  }

  renderAll(items) {
    this.reset();
    for (const item of items) this.update(item);
    if (!items.length) this.root.appendChild(this.welcome);
  }

  reset() {
    this.root.replaceChildren();
    this.nodes.clear();
    this.blockOf.clear();
    this.openBlock = null;
    this.transientNode = null;
  }

  remove(item) {
    if (this.transientNode?.dataset.id === String(item.id)) this.transientNode = null;
    const block = this.blockOf.get(item.id);
    if (block) {
      block.remove(item);
      this.blockOf.delete(item.id);
      this.nodes.delete(item.id);
      return;
    }
    const node = this.nodes.get(item.id);
    if (!node) return;
    node.remove();
    this.nodes.delete(item.id);
  }

  tick(items) {
    for (const item of items)
      if (item.kind === 'pending' || (item.kind === 'thinking' && !item.done)) this.update(item);
    // keep the block's stopwatch (and its pulsing last shape) moving while a long
    // tool runs and no events arrive.
    if (this.openBlock) this.openBlock.refresh();
  }
}
