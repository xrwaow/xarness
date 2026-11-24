/** DOM renderers for transcript items. One element per item, updated in
 * place: streaming appends only settled chunks and rewrites the live tail. */
import { el } from './dom.js';
import { STATUS, formatDuration, formatElapsed, formatTokens, formatUsage } from './state.js';
import { appendMarkdown } from './markdown.js';
import { diffTree, looksLikePatch, patchView } from './diff.js';
import { appendHighlighted, specFor } from './highlight.js';

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
// The shapes are SVG, not glyphs, so they have real fills and an ink-coloured
// stroke, and each one is nudged, stretched and skewed like a hand-drawn sticker.
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

// The shape art below is a pure function of the session name and the shape's
// stable identity, so a reload redraws exactly the same stickers without any of
// it being stored. `unitHash` just spreads the hash over [0, 1).
function unitHash(seed, salt) {
  // FNV-1a over seed + salt, then a final avalanche so neighbouring salts are
  // no more alike than neighbouring seeds.
  let hash = 2166136261 >>> 0;
  for (const ch of `${seed}\u0000${salt}`) {
    hash = Math.imul(hash ^ ch.codePointAt(0), 16777619) >>> 0;
  }
  hash = Math.imul(hash ^ (hash >>> 16), 0x85ebca6b) >>> 0;
  hash ^= hash >>> 13;
  return (hash >>> 0) / 4294967296;
}

// Shapes sit on the grid but are never quite straight: a little rotation, an
// independent stretch on each axis and a hint of skew read as drawn by hand.
const SHAPE_MAX_ROTATION = 12; // degrees
const SHAPE_MAX_SKEW = 4; // degrees
const SHAPE_STRETCH = 0.16; // ± on a 1.0 scale

function shapeVariation(session, identity) {
  const seed = `${session || ''}|${identity || ''}`;
  const spread = (salt, amount) => (unitHash(seed, salt) * 2 - 1) * amount;
  return {
    rotation: spread('rotation', SHAPE_MAX_ROTATION),
    skew: spread('skew', SHAPE_MAX_SKEW),
    scaleX: 1 + spread('scale-x', SHAPE_STRETCH),
    scaleY: 1 + spread('scale-y', SHAPE_STRETCH),
  };
}

// Persisted items keep the message index and the provider's call id across a
// reload; the in-memory item id does not, so seed the art from the former.
function shapeIdentity(item) {
  const at = item.msgIndex === undefined || item.msgIndex === null ? `d${item.id}` : `m${item.msgIndex}`;
  return item.kind === 'tool' ? `${at}:${item.callId || item.name || ''}` : `${at}:think`;
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

function shapeIcon(meta, status = 'ok', session = '', identity = '') {
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
  // The thick outline is what makes the shapes look like cartoon stickers
  // rather than UI glyphs. It is ink-dark on the light theme and the panel
  // border colour on the dark one (see --shape-stroke).
  path.setAttribute('stroke', 'var(--shape-stroke)');
  path.setAttribute('stroke-width', '2.1');
  path.setAttribute('stroke-linejoin', 'round');
  const { rotation, skew, scaleX, scaleY } = shapeVariation(session, identity);
  path.setAttribute(
    'transform',
    `translate(8 8) rotate(${rotation.toFixed(2)}) skewX(${skew.toFixed(2)}) ` +
      `scale(${scaleX.toFixed(3)} ${scaleY.toFixed(3)}) translate(-8 -8)`,
  );
  svg.appendChild(path);
  return svg;
}

// A thought is over when its reasoning stops, a tool when its call returns.
function activitySettled(item) {
  return item.kind === 'thinking' ? !!item.done : item.status !== STATUS.MAKING;
}

function activitySummary(items) {
  // Two tallies only: how many times it thought and how many tools it ran. The
  // per-tool breakdown lives in the expanded rows, not in this one line.
  let thoughts = 0;
  let tools = 0;
  let live = null;
  for (const item of items) {
    if (!activitySettled(item)) {
      live = item;
      continue;
    }
    if (item.kind === 'thinking') thoughts += 1;
    else tools += 1;
  }
  // A run of one has nothing to tally, so it names the activity itself. There
  // is no row to carry the tool's name and argument, so the header does: a
  // single call would otherwise read just `Ran`.
  if (items.length === 1) {
    const sole = items[0];
    if (sole.kind === 'thinking') return live ? 'Thinking' : 'Thought';
    const name = displayName(sole.name);
    const failed = !live && toolFailed(sole);
    const verb = live ? 'Running' : failed ? 'Failed' : 'Ran';
    let detail = failed ? '' : toolDetail(sole.name, sole.args, sole.output);
    // The block's right cluster already carries the edit counts, the way a
    // row's did: the header only needs the path beside the tool's name.
    if (detail && (sole.name === 'edit_file' || sole.name === 'write_file')) {
      const counted = DIFF_COUNTS_RE.exec(detail);
      if (counted) detail = counted[2] || '';
    }
    return detail ? `${verb} ${name} · ${detail}` : `${verb} ${name}`;
  }
  const parts = [];
  if (thoughts === 1) parts.push('Thought');
  else if (thoughts > 1) parts.push(`Thought ${thoughts} times`);
  if (tools === 1) parts.push('Ran 1 tool');
  else if (tools > 1) parts.push(`Ran ${tools} tools`);
  // The activity still in flight has not been thought or run yet, so it joins
  // the line in the present tense instead of bumping a tally.
  if (live) parts.push(live.kind === 'thinking' ? 'Thinking' : 'Running');
  return parts.join(', ');
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
      // the summary the host hands over is markdown, so it reads like prose here
      // instead of a wall of preformatted text.
      const body = el('div', { class: 'handoff-body prose' });
      appendMarkdown(body, item.text);
      const head = el(
        'button',
        { class: 'handoff-head', type: 'button' },
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

function isBash(item) {
  return item.kind === 'tool' && (item.name === 'run_bash' || item.name === 'bash');
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

/* edit_file/write_file read `+A -D path`: the path stays next to the tool name
 * and the diff counts are pulled out to the row's right edge, by the state word.
 * A detail that is only a path (nothing counted yet) is left as it is. */
const DIFF_COUNTS_RE = /^(\+\d+(?: -\d+)?)(?:\s+([\s\S]*))?$/;

function setRowDetail(row, text, name, failed) {
  const counted = !failed && (name === 'edit_file' || name === 'write_file');
  const match = counted && text ? DIFF_COUNTS_RE.exec(text) : null;
  if (!match) {
    setDetail(row.args, text);
    row.loc.replaceChildren();
    return;
  }
  setDetail(row.args, match[2] || '');
  setDetail(row.loc, match[1]);
}

// ------------------------------------------------------------- tool bodies
// An expanded row shows the call's *content* — the terminal, the file it read
// or wrote, the patch it applied — not a report about it. So there are no
// "arguments"/"output" headings above the boxes (the row already names the
// tool and its argument) and no card around them either: the body hangs off
// the row's rail, monospace and colour do the framing.

const BASH_TOOLS = new Set(['run_bash', 'bash']);
const EXIT_RE = /^\[exit (\d+)\]$/;
const SHOWING_RE = /^\[(?:showing lines|this file is larger)[^\]]*\]$/;
const EDIT_ACK_RE = /^applied .* to .*$/;

function noteLine(text, extra = '') {
  return el('div', { class: `tool-note ${extra}`.trim(), text });
}

function languageOf(path) {
  const base = String(path || '')
    .split('/')
    .pop();
  const dot = base.lastIndexOf('.');
  const ext = (dot > 0 ? base.slice(dot + 1) : '').toLowerCase();
  return ext && specFor(ext) ? ext : '';
}

/** A block of code, lexed for its file type when the path says which. */
function codeBlock(text, path, extra = '') {
  const pre = el('pre', { class: `tool-code ${extra}`.trim() });
  appendHighlighted(pre, text, languageOf(path));
  return pre;
}

/** `+line` for every line of a file that was just written: a new-file diff,
 * with the file's own syntax colours inside the add tint. */
function addedBlock(text, path) {
  const pre = el('pre', { class: 'diff tool-added' });
  const lang = languageOf(path);
  const lines = textLines(text);
  if (!lines.length) pre.appendChild(el('span', { class: 'diff-line', text: '(empty file)' }));
  for (const line of lines) {
    const row = el('span', { class: 'diff-line add' }, el('span', { class: 'diff-sign', text: '+' }));
    if (lang) appendHighlighted(row, line, lang);
    else row.appendChild(document.createTextNode(line));
    pre.appendChild(row);
  }
  return pre;
}

/** A tool's result as prose + patch: edit_file acks first, then diffs. */
function splitPatch(text) {
  const source = String(text || '');
  const at = source.search(/^(?:diff --git |--- \S)/m);
  if (at < 0 || !looksLikePatch(source.slice(at))) return { note: source.trim(), patch: '' };
  return { note: source.slice(0, at).trim(), patch: source.slice(at).trim() };
}

function patchBlock(patch) {
  const tree = diffTree(patch);
  tree.node.classList.add('tool-diff');
  return tree.node;
}

function bashBody(item, body, args) {
  const command = (
    args && typeof args.command === 'string' ? args.command : partialString(item.args, 'command', true)
  ).replace(/\s+$/, '');
  if (command.trim()) {
    const cmd = el('pre', { class: 'tool-cmd' });
    cmd.appendChild(el('span', { class: 'tool-prompt', text: '$ ' }));
    cmd.appendChild(document.createTextNode(command));
    body.appendChild(cmd);
  }
  // run_bash ends every result with `[exit N]`; pulled out of the output it
  // becomes the one-line verdict the terminal never gives you for free.
  let output = item.output || '';
  let exit = '';
  if (output) {
    const lines = textLines(output);
    const last = (lines[lines.length - 1] || '').trim();
    const match = EXIT_RE.exec(last);
    if (match) {
      exit = match[1];
      lines.pop();
      output = lines.join('\n');
    }
  }
  if (output.trim()) {
    body.appendChild(looksLikePatch(output) ? patchBlock(output) : el('pre', { class: 'tool-code', text: output }));
  }
  if (exit)
    body.appendChild(
      el(
        'div',
        { class: `tool-exit ${exit === '0' ? 'ok' : 'fail'}` },
        el('span', { class: 'mark', text: exit === '0' ? '✓' : '✗' }),
        el('span', { text: `exit ${exit}` }),
      ),
    );
  if (!command.trim() && !item.output) body.appendChild(noteLine('waiting for the command…'));
  else if (!output.trim() && !exit && item.status === STATUS.MAKING) body.appendChild(noteLine('running…'));
}

function writeBody(item, body, args) {
  const failed = toolFailed(item);
  const content =
    args && typeof args.content === 'string' ? args.content : partialString(item.args, 'content', true);
  if (failed) {
    if (item.output) body.appendChild(el('pre', { class: 'tool-code tool-error', text: item.output }));
    else body.appendChild(noteLine('write failed'));
    return;
  }
  if (content) body.appendChild(addedBlock(content, args?.path || toolPath(item)));
  else if (!item.args.trim()) body.appendChild(noteLine('waiting for the content…'));
  // "created webui/notes.md (80 bytes)" — the part the diff cannot say.
  if (item.output.trim()) body.appendChild(noteLine(item.output.replace(/\s+$/, '')));
}

function editBody(item, body) {
  const failed = toolFailed(item);
  const { note, patch } = splitPatch(item.output);
  if (failed) {
    if (item.output) body.appendChild(el('pre', { class: 'tool-code tool-error', text: item.output }));
    else body.appendChild(noteLine('edit failed'));
    return;
  }
  if (patch) {
    body.appendChild(patchBlock(patch));
    if (note && !EDIT_ACK_RE.test(note)) body.appendChild(noteLine(note));
    return;
  }
  // A result without a patch says why ("(no textual change)") — showing the
  // provisional patch over it would claim an edit that never happened.
  if (note) {
    body.appendChild(noteLine(note));
    return;
  }
  // No result yet: the streamed arguments already spell out the change, so
  // show that as a provisional patch instead of the raw JSON call.
  const provisional = editDiffText(item.args);
  if (provisional) {
    body.appendChild(patchView(provisional, { className: 'tool-provisional' }));
    return;
  }
  if (item.args.trim()) body.appendChild(el('pre', { class: 'tool-code tool-args', text: item.args }));
  else body.appendChild(noteLine('waiting for the edit…'));
}

function readBody(item, body) {
  if (toolFailed(item)) {
    body.appendChild(el('pre', { class: 'tool-code tool-error', text: item.output || 'read failed' }));
    return;
  }
  const output = item.output;
  if (!output.trim()) {
    body.appendChild(noteLine(item.args.trim() ? 'no output yet' : 'waiting for the path…'));
    return;
  }
  // Large reads come with bracketed footers and an outline preamble that is
  // prose, not code: lift them out before lexing.
  const notes = [];
  const code = textLines(output)
    .filter((line) => {
      if (!SHOWING_RE.test(line)) return true;
      notes.push(line.slice(1, -1));
      return false;
    })
    .join('\n');
  const outline = /^File outline retrieved\./m.test(output) || /^# File outline for /m.test(output);
  body.appendChild(
    outline ? el('pre', { class: 'tool-code tool-args', text: output }) : codeBlock(code, toolPath(item)),
  );
  for (const entry of notes) body.appendChild(noteLine(entry));
}

function genericBody(item, body, args) {
  const failed = toolFailed(item);
  const argsText = args ? JSON.stringify(args, null, 2) : String(item.args || '').trim();
  if (argsText) body.appendChild(el('pre', { class: 'tool-code tool-args', text: argsText }));
  if (item.output) {
    const out =
      !failed && looksLikePatch(item.output)
        ? patchBlock(item.output)
        : el('pre', { class: `tool-code${failed ? ' tool-error' : ''}`, text: item.output });
    body.appendChild(out);
  }
  if (!argsText && !item.output) body.appendChild(noteLine('no output yet'));
}

function toolPath(item) {
  const args = parseArgs(item.args);
  return args && typeof args.path === 'string' ? args.path : partialString(item.args, 'path');
}

function renderToolBody(item, body) {
  const args = parseArgs(item.args);
  if (BASH_TOOLS.has(item.name)) return bashBody(item, body, args);
  if (item.name === 'write_file') return writeBody(item, body, args);
  if (item.name === 'edit_file') return editBody(item, body);
  if (item.name === 'read_file') return readBody(item, body);
  return genericBody(item, body, args);
}

/** One run of thinking + tool calls, between two pieces of assistant prose.
 *
 * Collapsed it is a header (aggregate shapes, a generated summary, the current
 * activity and its state) plus an eight-line window on what the current
 * activity is writing. Expanded it lists every activity as shape · name ·
 * arguments; each row opens its own body (the terminal, the code, the
 * patch).
 *
 * A block with a single activity is the exception: there is no run to list, so
 * the tap belongs to the activity itself. A thought opens its reasoning, a tool
 * opens the terminal / file / patch the row would have opened. */
class ActivityBlock {

  constructor(view) {
    this.view = view;
    this.items = [];
    this.rows = new Map();
    this.expanded = false;
    this.closed = false;
    // A solo activity's header opens the activity itself. `null` means nobody
    // has said either way: the text follows the stream, shown while it runs.
    this.soloMode = null;
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
    // the diff counts, token spend and time flow as one right-aligned cluster:
    // nothing reserves a slot, so the row sits tight and slides as it changes.
    this.stats = el('span', { class: 'activity-stats' }, this.loc, this.usage, this.time);
    this.startedAt = null;
    this.frozen = null;
    this.measured = false;
    this.times = new Map();
    this.head = el(
      'button',
      { class: 'activity-head', type: 'button', 'aria-expanded': 'false' },
      el('span', { class: 'activity-copy' }, this.title),
      this.stats,
      this.shapes,
    );
    this.list = el('div', { class: 'activity-list', role: 'list' });
    this.panel = el('div', { class: 'activity-panel' }, this.list);
    this.lines = el('pre', { class: 'activity-lines', 'aria-label': 'current activity' });
    // A solo tool's opened body goes here (the eight-line window is only the
    // live tail); the two share the preview's rail and margin.
    this.body = el('div', { class: 'activity-body' });
    this.bodyItemId = null;
    this.bodySignature = null;
    this.preview = el('div', { class: 'activity-preview' }, this.lines, this.body);
    this.node = el('section', { class: 'item item-activity activity' }, this.head, this.panel, this.preview);
    if (typeof ResizeObserver === 'function') {
      this.headObserver = new ResizeObserver(() => this.syncHeadHeight());
      this.headObserver.observe(this.head);
    }
    this.head.addEventListener('click', () => this.toggleExpanded());
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

  /** One activity and nothing else: there is no run to list, only its content. */
  get solo() {
    return this.items.length === 1;
  }

  /** The block is still working: a tool is running or a thought is streaming. */
  isLive() {
    const status = this.closed || !this.last ? 'ok' : activityStatus(this.last);
    return status === 'thinking' || status === 'running';
  }

  /** Whether a solo activity's content is on screen right now. */
  get soloVisible() {
    if (!this.solo || this.soloMode === 'closed') return false;
    return this.soloMode === 'open' || this.isLive();
  }

  // Toggling changes the height of the log under the header, and the browser's
  // own anchoring moves the scroll to match — when it moves at all. The header
  // is the thing that was tapped, so it must stay put: measure it before the
  // change, then correct the log's scroll by however far it drifted.
  toggleExpanded() {
    // A solo activity has nothing to list: open shows its content, closed hides
    // it, and a tap while it runs closes the stream.
    if (this.solo) {
      const show = !this.soloVisible;
      this.soloMode = show ? 'open' : 'closed';
      this.refresh();
      // A settled thought reads from its beginning; a live tail follows the end.
      if (show && !this.isLive()) {
        this.previewPinned = false;
        this.lines.scrollTop = 0;
      }
      return;
    }
    const log = this.node.closest('.log');
    const anchor = this.head.getBoundingClientRect().top;
    this.expanded = !this.expanded;
    this.refresh();
    if (this.expanded)
      for (const item of this.items)
        if (item.textLoaded === false || item.argsLoaded === false)
          this.view.onNeedBody?.(item, false);
    if (!log) return;
    const drift = this.head.getBoundingClientRect().top - anchor;
    if (drift) {
      log.scrollTop += drift;
      this.view.onSelfScroll?.();
    }
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

  syncHeadHeight() {
    const open = this.node.classList.contains('open');
    this.node.style.setProperty('--activity-head-h', open ? `${this.head.offsetHeight}px` : '0px');
  }

  /** Elapsed time for a bash row: live while it runs, frozen once it settles. */
  toolTime(item) {
    if (!isBash(item)) return '';
    const entry = this.times.get(item.id);
    if (!entry) return '';
    if (item.status === STATUS.MAKING) return formatElapsed(Math.max(0, now() - entry.start));
    if (entry.end === null) return '';
    return formatDuration(Math.max(0, entry.end - entry.start));
  }

  createRow(item) {
    const shape = el('span', { class: 'activity-row-icon' });
    const name = el('span', { class: 'activity-row-name' });
    const args = el('span', { class: 'activity-row-args' });
    const loc = el('span', { class: 'activity-row-loc' });
    const state = el('span', { class: 'activity-row-state' });
    const tail = el('span', { class: 'activity-row-tail' }, state, loc);
    const head = el(
      'button',
      { class: 'activity-row-head', type: 'button', 'aria-expanded': 'false' },
      shape,
      el('span', { class: 'activity-row-label' }, name, args),
      tail,
    );
    const body = el('div', { class: 'activity-row-body' });
    const node = el('div', { class: 'activity-row', role: 'listitem' }, head, body);
    const row = { node, head, shape, name, args, loc, state, body, item, shapeKey: null, signature: null };
    head.addEventListener('click', () => {
      row.item.expanded = !row.item.expanded;
      this.updateRow(row, row.item);
      // A slim page left the output out; the row is where it is actually
      // read, so that is when its text is fetched.
      if (row.item.expanded && row.item.kind === 'tool' && row.item.outputLoaded === false)
        this.view.onNeedBody?.(row.item, true);
    });
    return row;
  }

  updateRow(row, item) {
    const status = activityStatus(item);
    const meta = activityMeta(item);
    const shapeKey = `${meta.kind}:${meta.server || ''}:${status}`;
    if (row.shapeKey !== shapeKey) {
      row.shapeKey = shapeKey;
      row.shape.replaceChildren(shapeIcon(meta, status, this.view.session, shapeIdentity(item)));
    }
    row.state.dataset.state = status;
    row.node.dataset.status = status;
    if (item.kind === 'thinking') {
      row.name.textContent = item.done ? 'Thought' : 'Thinking';
      // While the row is open its body streams the same reasoning: the snippet
      // would only echo its last line and move every delta.
      row.args.textContent = item.expanded && !item.done ? '' : thinkingSnippet(item);
      row.loc.replaceChildren();
      if (item.done) {
        const stats = [];
        if (item.duration !== null) stats.push(formatDuration(item.duration));
        if (item.tokens) stats.push(`${formatTokens(item.tokens)} tok`);
        row.state.textContent = stats.join(' · ');
      } else row.state.textContent = '';
    } else {
      const failed = toolFailed(item);
      row.name.textContent = displayName(item.name);
      row.name.classList.toggle('failed', failed);
      setRowDetail(row, failed ? '' : toolDetail(item.name, item.args, item.output), item.name, failed);
      let text = item.status === STATUS.MAKING ? 'running' : failed ? 'failed' : item.usage ? formatUsage(item.usage) : '';
      const timer = this.toolTime(item);
      if (timer) text = text ? `${text} · ${timer}` : timer;
      row.state.textContent = text;
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
      // the row says "Thought" — a "thought" heading under it said nothing
      row.body.appendChild(
        item.text.trim()
          ? el('pre', { class: 'tool-code tool-thought', text: item.text })
          : noteLine('no reasoning text'),
      );
      return;
    }
    renderToolBody(item, row.body);
  }

  // A slim history page ships tool calls without their arguments, so a
  // collapsed block could not total its edits (or its writes) until someone
  // opened the round. Fetch the arguments as soon as an edit lands in the
  // block; the request is per round and deduped while it is in flight.
  hydrateEdits() {
    for (const item of this.items)
      if (
        item.kind === 'tool' &&
        item.argsLoaded === false &&
        (item.name === 'edit_file' || item.name === 'write_file')
      )
        this.view.onNeedBody?.(item, false);
  }

  // `+12 -3` leads the header's right cluster, and only appears when the block
  // actually wrote to a file.
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

  // A solo tool's body is rebuilt only when its text changes: streaming deltas
  // arrive many times a second, and replacing the whole terminal each time
  // would throw the reader's place away.
  renderSoloBody(item) {
    const signature = `${item.status}:${item.args.length}:${item.output.length}`;
    if (this.bodyItemId === item.id && this.bodySignature === signature) return;
    this.bodyItemId = item.id;
    this.bodySignature = signature;
    this.body.replaceChildren();
    renderToolBody(item, this.body);
  }

  clearSoloBody() {
    if (this.bodySignature === null && !this.body.childElementCount) return;
    this.body.replaceChildren();
    this.bodyItemId = null;
    this.bodySignature = null;
  }

  refresh() {
    const last = this.last,
      solo = this.solo,
      // A solo activity's "open" is its content being on screen: it starts open
      // while it streams, and the reader's tap decides after that.
      open = solo ? this.soloVisible : this.expanded;
    this.node.classList.toggle('solo', solo);
    this.node.classList.toggle('open', open);
    this.head.setAttribute('aria-expanded', open ? 'true' : 'false');
    this.syncHeadHeight();
    this.hydrateEdits();
    // a running bash row counts its seconds up on every tick
    for (const row of this.rows.values())
      if (isBash(row.item) && row.item.status === STATUS.MAKING) this.updateRow(row, row.item);
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
        const shape = el(
          'span',
          { class: 'activity-shape', title: shapeTitle(item) },
          shapeIcon(meta, status, this.view.session, shapeIdentity(item)),
        );
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

    // A settled solo tool shows its actual body (the terminal, the file, the
    // patch) instead of a one-row list. While it is live the eight-line tail is
    // the content: it follows the output as it streams, which a rebuilt body
    // could not do cheaply.
    const bodyShown = solo && open && last?.kind === 'tool' && activitySettled(last);
    if (bodyShown) {
      if (last.outputLoaded === false || last.argsLoaded === false) this.view.onNeedBody?.(last, true);
      this.renderSoloBody(last);
    } else this.clearSoloBody();
    this.lines.hidden = bodyShown;
    const hasBody = bodyShown && this.body.childElementCount > 0;
    const text = bodyShown ? '' : activityPreviewText(last),
      wasHidden = this.preview.hidden;
    this.preview.hidden = solo ? !open || (!hasBody && !text) : !text;
    if (this.previewItemId !== (last?.id ?? null)) {
      this.previewItemId = last?.id ?? null;
      this.previewPinned = true;
    }
    if (this.lines.textContent !== text) {
      // Sample the pin now, not only in the scroll listener: a reader who
      // scrolled the tail up must stay there when the next line lands, even if
      // that scroll event has not been delivered yet. A hidden box measures
      // nothing, so coming back to a live stream goes to its tail.
      const offset = this.lines.scrollTop;
      this.previewPinned = wasHidden ? this.isLive() : this.lines.scrollHeight - offset - this.lines.clientHeight <= 4;
      this.lines.textContent = text;
      this.lines.scrollTop = this.previewPinned ? this.lines.scrollHeight : offset;
    } else if (this.previewPinned || (wasHidden && !this.preview.hidden && this.isLive())) {
      this.previewPinned = true;
      this.lines.scrollTop = this.lines.scrollHeight;
    }
  }
}

// The xarness wordmark: four hand-drawn rows, dimmed under the fold so it
// reads as a watermark, not a logo. Shared by the transcript's empty state
// and the home screen so there is one banner, not two.
export const BANNER = `... ...  ....   ... ..  .. ...     ....   ....   ....
 '|..'  '' .||   ||' ''  ||  ||  .|...|| ||. '  ||. '
  .|.   .|' ||   ||      ||  ||  ||      . '|.. . '|..
.|  ||. '|..'|' .||.    .||. ||.  '|...' |'..|' |'..|'`;

// Live-only transcript lines that must not break an activity block in two: they
// are pushed while a run is in flight and never replayed from the log, so
// treating them as a boundary would split a segment live and merge it on reload.
const TRANSIENT_KINDS = new Set(['pending']);

export class LogView {

  constructor(root, { onUndoTo = null, resolveImage = null, onNeedBody = null, onSelfScroll = null, session = '' } = {}) {
    this.root = root;
    // Seeds every shape's hand-drawn variation. The app sets it on attach; a
    // prepend buffer inherits it from the view it is built from.
    this.session = session;
    this.nodes = new Map();
    this.blockOf = new Map();
    this.openBlock = null;
    this.transientNode = null;
    this.onUndoTo = onUndoTo;
    this.resolveImage = resolveImage;
    this.onNeedBody = onNeedBody;
    // Tells the app that a scroll it is about to see came from here, not the
    // reader, so the follow state (and its jump button) is left alone.
    this.onSelfScroll = onSelfScroll;
    this.welcome = el(
      'div',
      { class: 'welcome' },
      el('pre', {
        text: BANNER,
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

  /** Insert a page of older items above everything else, built in a detached
   * buffer so historical items never join the live block or its updates. */
  prepend(items) {
    if (!items?.length) return;
    const buffer = new LogView(document.createElement('div'), this);
    buffer.renderAll(items);
    // childNodes is live: copy it before moving the run in front of the log.
    this.root.prepend(...buffer.root.childNodes);
    for (const map of ['nodes', 'blockOf']) for (const [id, value] of buffer[map]) this[map].set(id, value);
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
