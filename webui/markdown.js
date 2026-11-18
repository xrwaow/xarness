/** Small sanitising markdown renderer: every node is built with textContent, so
 * model output can never become markup. Links only for http(s). */
import { el } from './dom.js';
import { patchEnd, patchView, startsPatch } from './diff.js';
import { appendHighlighted } from './highlight.js';

export function splitSettled(text) {
  const fence = /```[\s\S]*?```/g;
  let last = 0,
    match;
  while ((match = fence.exec(text)) !== null) last = match.index + match[0].length;
  return [text.slice(0, last), text.slice(last)];
}
const INLINE_RE = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*\n]+\*|_[^_\n]+_|\[[^\]]+\]\([^)\s]+\))/g;
const SAFE_URL_RE = /^https?:\/\//i;

export function appendInline(parent, text) {
  const source = String(text);
  let index = 0,
    match;
  INLINE_RE.lastIndex = 0;
  while ((match = INLINE_RE.exec(source)) !== null) {
    if (match.index > index) parent.appendChild(document.createTextNode(source.slice(index, match.index)));
    const token = match[0];
    if (token.startsWith('`')) parent.appendChild(el('code', { text: token.slice(1, -1) }));
    else if (token.startsWith('**')) parent.appendChild(el('strong', {}, token.slice(2, -2)));
    else if (token.startsWith('[')) {
      const close = token.indexOf(']'),
        label = token.slice(1, close),
        url = token.slice(close + 2, -1);
      if (SAFE_URL_RE.test(url))
        parent.appendChild(el('a', { href: url, target: '_blank', rel: 'noopener noreferrer', text: label }));
      else parent.appendChild(document.createTextNode(label));
    } else parent.appendChild(el('em', {}, token.slice(1, -1)));
    index = match.index + token.length;
  }
  if (index < source.length) parent.appendChild(document.createTextNode(source.slice(index)));
  return parent;
}

function codeBlock(language, code) {
  if (/^(diff|patch)$/i.test(language || '')) return patchView(code, { className: 'md-code' });
  // Syntax colors come from the palette (highlight.js); a bare fence stays
  // plain, the way the TUI renders it.
  return appendHighlighted(el('pre', { class: 'md-code' }), code, language);
}
const TABLE_DELIM_CELL_RE = /^:?-+:?$/;
const HEADING_RE = /^(#{1,6})[ \t]+(.*)$/;
const HR_RE = /^[ \t]*([-*_])[ \t]*(?:\1[ \t]*){2,}$/;
const LIST_RE = /^[ \t]*([-*+]|\d+[.)])[ \t]+(.*)$/;
const QUOTE_RE = /^[ \t]*>[ \t]?(.*)$/;
const FENCE_OPEN_RE = /^[ \t]*```(.*)$/;
const FENCE_CLOSE_RE = /^[ \t]*```/;

function splitTableRow(line) {
  const source = String(line),
    cells = [];
  let cell = '',
    inCode = false;
  for (let i = 0; i < source.length; i += 1) {
    const char = source[i];
    if (char === '\\' && source[i + 1] === '|') {
      cell += '|';
      i += 1;
    } else if (char === '`') {
      inCode = !inCode;
      cell += char;
    } else if (char === '|' && !inCode) {
      cells.push(cell);
      cell = '';
    } else cell += char;
  }
  cells.push(cell);
  if (cells.length > 1 && !cells[0].trim()) cells.shift();
  if (cells.length > 1 && !cells[cells.length - 1].trim()) cells.pop();
  return cells.map((text) => text.trim());
}

function tableAt(lines, i) {
  const head = lines[i],
    delim = lines[i + 1];
  if (!head || delim === undefined || !head.includes('|') || !delim.includes('|')) return null;
  const heads = splitTableRow(head),
    delims = splitTableRow(delim);
  if (!heads.length || delims.length !== heads.length || !delims.every((c) => TABLE_DELIM_CELL_RE.test(c))) return null;
  const align = delims.map((cell) => {
    const left = cell.startsWith(':'),
      right = cell.endsWith(':');
    return left && right ? 'center' : right ? 'right' : 'left';
  });
  const rows = [];
  let end = i + 2;
  while (end < lines.length) {
    const line = lines[end];
    if (!line.trim() || !line.includes('|') || FENCE_OPEN_RE.test(line) || HEADING_RE.test(line)) break;
    const cells = splitTableRow(line).slice(0, heads.length);
    while (cells.length < heads.length) cells.push('');
    rows.push(cells);
    end += 1;
  }
  return { heads, align, rows, end };
}

function tableBlock({ heads, align, rows }) {
  const alignClass = (i) => (align[i] && align[i] !== 'left' ? `align-${align[i]}` : null);
  const cell = (tag, text, i) => {
    const node = el(tag, { class: alignClass(i) });
    appendInline(node, text);
    return node;
  };
  return el(
    'div',
    { class: 'table-wrap' },
    el(
      'table',
      { class: 'md-table' },
      el('thead', {}, el('tr', {}, ...heads.map((text, i) => cell('th', text, i)))),
      el('tbody', {}, ...rows.map((row) => el('tr', {}, ...row.map((text, i) => cell('td', text, i))))),
    ),
  );
}

export function appendMarkdown(root, text) {
  const lines = String(text ?? '')
    .replace(/\r\n/g, '\n')
    .split('\n');
  let i = 0;
  while (i < lines.length) {
    const line = lines[i],
      fence = line.match(FENCE_OPEN_RE);
    if (fence) {
      const body = [];
      i += 1;
      while (i < lines.length && !FENCE_CLOSE_RE.test(lines[i])) {
        body.push(lines[i]);
        i += 1;
      }
      i += 1;
      root.appendChild(codeBlock(fence[1].trim(), body.join('\n')));
      continue;
    }
    if (!line.trim()) {
      i += 1;
      continue;
    }
    if (startsPatch(lines, i)) {
      const end = patchEnd(lines, i);
      if (end > i) {
        root.appendChild(patchView(lines.slice(i, end).join('\n'), { className: 'prose-diff' }));
        i = end;
        continue;
      }
    }
    const table = tableAt(lines, i);
    if (table) {
      root.appendChild(tableBlock(table));
      i = table.end;
      continue;
    }
    const heading = line.match(HEADING_RE);
    if (heading) {
      const node = el(`h${Math.min(heading[1].length, 6)}`);
      appendInline(node, heading[2]);
      root.appendChild(node);
      i += 1;
      continue;
    }
    if (HR_RE.test(line)) {
      root.appendChild(el('hr'));
      i += 1;
      continue;
    }
    if (QUOTE_RE.test(line)) {
      const body = [];
      while (i < lines.length && QUOTE_RE.test(lines[i])) {
        body.push(lines[i].match(QUOTE_RE)[1]);
        i += 1;
      }
      root.appendChild(el('blockquote', {}, body.join('\n')));
      continue;
    }
    if (LIST_RE.test(line)) {
      const list = el(/^\d/.test(line.match(LIST_RE)[1]) ? 'ol' : 'ul');
      while (i < lines.length && LIST_RE.test(lines[i])) {
        appendInline(list.appendChild(el('li')), lines[i].match(LIST_RE)[2]);
        i += 1;
      }
      root.appendChild(list);
      continue;
    }
    const para = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !FENCE_OPEN_RE.test(lines[i]) &&
      !HEADING_RE.test(lines[i]) &&
      !LIST_RE.test(lines[i]) &&
      !QUOTE_RE.test(lines[i]) &&
      !HR_RE.test(lines[i]) &&
      !tableAt(lines, i)
    ) {
      para.push(lines[i]);
      i += 1;
    }
    const node = el('p');
    para
      .join('\n')
      .split('\n')
      .forEach((chunk, index) => {
        if (index) node.appendChild(el('br'));
        appendInline(node, chunk);
      });
    root.appendChild(node);
  }
  return root;
}
