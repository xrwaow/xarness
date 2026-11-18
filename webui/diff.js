/** Patch rendering, shared by the diff bar, tool output and prose. One block
 * span per line, with no "\n" text nodes between them (a <pre> would render
 * each of those as an empty line). */
import { el } from './dom.js';

const META_RE =
  /^(diff --git |index |--- |\+\+\+ |new file |deleted file |rename |similarity index|dissimilarity index|old mode|new mode|Binary files )/;
const HUNK_RE = /^@@ /;
const HUNK_START_RE = /^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@/;

export function diffLineKind(line) {
  if (META_RE.test(line) || line.startsWith('\\')) return 'meta';
  if (HUNK_RE.test(line)) return 'hunk';
  if (line.startsWith('+')) return 'add';
  if (line.startsWith('-')) return 'del';
  return '';
}

export function isPatchLine(line) {
  return (
    line.startsWith(' ') ||
    line.startsWith('+') ||
    line.startsWith('-') ||
    line.startsWith('\\') ||
    HUNK_RE.test(line) ||
    META_RE.test(line)
  );
}

export function startsPatch(lines, i) {
  const line = lines[i] || '';
  return (
    /^diff --git /.test(line) || (/^--- /.test(line) && /^\+\+\+ /.test(lines[i + 1] || '')) || HUNK_START_RE.test(line)
  );
}

export function patchEnd(lines, i) {
  let j = i;
  while (j < lines.length && lines[j] !== '' && isPatchLine(lines[j])) j += 1;
  return j;
}

export function looksLikePatch(text) {
  return /^diff --git /m.test(text) || (/^--- \S.*\n\+\+\+ \S/m.test(text) && /^@@ /m.test(text));
}

export function patchView(text, { className = '' } = {}) {
  const pre = el('pre', { class: `diff ${className}`.trim() });
  for (const line of String(text ?? '')
    .replace(/\r\n/g, '\n')
    .replace(/\n$/, '')
    .split('\n')) {
    pre.appendChild(el('span', { class: `diff-line ${diffLineKind(line)}`.trim(), text: line }));
  }
  return pre;
}
