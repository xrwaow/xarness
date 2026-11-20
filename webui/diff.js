/** Patch rendering, shared by the diff bar, tool output and prose. One block
 * span per line, with no "\n" text nodes between them (a <pre> would render
 * each of those as an empty line). */
import { el } from './dom.js';
import { appendHighlighted, specFor } from './highlight.js';

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

// ---------------------------------------------------------------- diff view

/** The `a/… b/…` pair of a `diff --git` line. */
function gitPaths(line) {
  const match = /^diff --git "?a\/(.*?)"? "?b\/(.*?)"?$/.exec(line);
  if (match) return { oldPath: match[1], path: match[2] };
  const rest = line.startsWith('diff --git ') ? line.slice('diff --git '.length) : '';
  const mid = rest.indexOf(' b/');
  if (mid >= 0) return { oldPath: rest.slice(2, mid), path: rest.slice(mid + 3) };
  return { oldPath: '', path: rest.slice(2) };
}

/** The fence tag for a file path, or '' when the file type has no lexer. */
function languageFor(path) {
  const base = String(path ?? '')
    .split('/')
    .pop()
    .replace(/"$/, '');
  const dot = base.lastIndexOf('.');
  const ext = (dot >= 0 ? base.slice(dot + 1) : base).toLowerCase();
  return specFor(ext) ? ext : '';
}

/** Structure a unified patch for the side-by-side-ish view: per file, rows
 * carrying both line numbers, with git's bookkeeping (index, ---, +++, mode)
 * folded into a note on the file header instead of painted as lines.
 *
 * Hunk bodies are consumed by the counts in their `@@` header, so a removed
 * `-- foo` line (rendered `--- foo`) can never be mistaken for a file marker.
 * A patch without `diff --git` headers still works: everything lands in one
 * unnamed file. */
export function parsePatch(text) {
  const files = [];
  let file = null;
  const startFile = (path = '', oldPath = '') => {
    file = { path, oldPath, notes: [], rows: [], additions: 0, deletions: 0, binary: false };
    files.push(file);
    return file;
  };
  let oldNo = 0,
    newNo = 0,
    oldLeft = 0,
    newLeft = 0;
  const lines = String(text ?? '')
    .replace(/\r\n/g, '\n')
    .replace(/\n$/, '')
    .split('\n');
  for (const line of lines) {
    if (line.startsWith('diff --git ')) {
      const { path, oldPath } = gitPaths(line);
      startFile(path, oldPath);
      oldLeft = newLeft = 0;
      continue;
    }
    const hunk = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/.exec(line);
    if (hunk) {
      if (!file) startFile();
      oldNo = Number(hunk[1]);
      newNo = Number(hunk[3]);
      oldLeft = hunk[2] === undefined ? 1 : Number(hunk[2]);
      newLeft = hunk[4] === undefined ? 1 : Number(hunk[4]);
      file.rows.push({ kind: 'hunk', text: line });
      continue;
    }
    if (!file) startFile();
    if (line.startsWith('\\')) {
      file.rows.push({ kind: 'meta', text: line });
      continue;
    }
    if (oldLeft > 0 || newLeft > 0) {
      const sign = line[0];
      if (sign === ' ' || sign === '') {
        file.rows.push({ kind: 'ctx', text: line.slice(1), oldNo, newNo });
        oldNo += 1;
        newNo += 1;
        oldLeft -= 1;
        newLeft -= 1;
        continue;
      }
      if (sign === '+') {
        file.rows.push({ kind: 'add', text: line.slice(1), newNo });
        newNo += 1;
        newLeft -= 1;
        file.additions += 1;
        continue;
      }
      if (sign === '-') {
        file.rows.push({ kind: 'del', text: line.slice(1), oldNo });
        oldNo += 1;
        oldLeft -= 1;
        file.deletions += 1;
        continue;
      }
    }
    if (line.startsWith('+++ ')) {
      const path = line.slice(4).replace(/^"?b\//, '').replace(/"$/, '').replace(/\t.*$/, '');
      if (path && path !== '/dev/null' && !file.path) file.path = path;
      else if (path === '/dev/null') file.notes.push('deleted');
      continue;
    }
    if (line.startsWith('--- ')) {
      const path = line.slice(4).replace(/^"?a\//, '').replace(/"$/, '').replace(/\t.*$/, '');
      if (path && path !== '/dev/null' && !file.oldPath) file.oldPath = path;
      continue;
    }
    if (line.startsWith('index ') || line.startsWith('new mode ') || line.startsWith('old mode ') || line === '') continue;
    if (line.startsWith('new file')) {
      file.notes.push('new');
      continue;
    }
    if (line.startsWith('deleted file')) {
      file.notes.push('deleted');
      continue;
    }
    if (line.startsWith('rename from ')) {
      file.notes.push(`renamed from ${line.slice('rename from '.length)}`);
      continue;
    }
    if (line.startsWith('rename to ') || line.startsWith('similarity index') || line.startsWith('dissimilarity index'))
      continue;
    if (/^Binary files .* differ$/.test(line)) {
      file.binary = true;
      file.notes.push('binary');
      continue;
    }
    file.rows.push({ kind: 'meta', text: line });
  }
  return {
    files: files.filter((entry) => entry.rows.length || entry.notes.length || entry.path),
    additions: files.reduce((total, entry) => total + entry.additions, 0),
    deletions: files.reduce((total, entry) => total + entry.deletions, 0),
  };
}

const NOTE_TEXT = { new: 'new file', deleted: 'deleted', binary: 'binary' };

function fileHeading(file) {
  const note = file.notes.find((entry) => NOTE_TEXT[entry]);
  const rename = file.notes.find((entry) => entry.startsWith('renamed from '));
  return el(
    'div',
    { class: 'dv-file-head', title: file.path },
    el('span', { class: 'path', text: file.path || '(unnamed file)' }),
    rename ? el('span', { class: 'tag', text: 'renamed' }) : null,
    note ? el('span', { class: 'tag', text: NOTE_TEXT[note] }) : null,
    el('span', { class: 'spacer' }),
    el('span', { class: 'adds', text: `+${file.additions}` }),
    el('span', { class: 'dels', text: `-${file.deletions}` }),
  );
}

function fileLines(file) {
  const wrap = el('div', { class: 'dv-lines' });
  const lang = languageFor(file.path || file.oldPath);
  let oldWidth = 1,
    newWidth = 1;
  for (const row of file.rows) {
    if (row.oldNo) oldWidth = Math.max(oldWidth, String(row.oldNo).length);
    if (row.newNo) newWidth = Math.max(newWidth, String(row.newNo).length);
  }
  wrap.style.setProperty('--old-w', `${oldWidth}ch`);
  wrap.style.setProperty('--new-w', `${newWidth}ch`);
  for (const row of file.rows) {
    if (row.kind === 'hunk' || row.kind === 'meta') {
      wrap.appendChild(el('div', { class: `dv-line ${row.kind}` }, el('span', { class: 'dv-note', text: row.text })));
      continue;
    }
    const sign = row.kind === 'add' ? '+' : row.kind === 'del' ? '-' : ' ';
    const code = el('span', { class: 'dv-code' });
    if (lang) appendHighlighted(code, row.text, lang);
    else code.textContent = row.text;
    wrap.appendChild(
      el(
        'div',
        { class: `dv-line ${row.kind}` },
        el('span', { class: 'dv-num old', text: row.oldNo === undefined ? '' : String(row.oldNo) }),
        el('span', { class: 'dv-num new', text: row.newNo === undefined ? '' : String(row.newNo) }),
        el('span', { class: 'dv-sign', text: sign }),
        code,
      ),
    );
  }
  if (file.binary || !file.rows.length) {
    wrap.appendChild(
      el('div', { class: 'dv-line meta' }, el('span', { class: 'dv-note', text: file.binary ? 'binary file — no preview' : 'no hunks' })),
    );
  }
  return wrap;
}

/** A patch as a scrollable, colored diff with old/new line numbers. File
 * headers only appear when the patch spans several files; the caller already
 * names a one-file title. Returns the node plus the counts for the header. */
export function diffTree(text) {
  const parsed = parsePatch(text);
  const root = el('div', { class: 'dv' });
  const multi = parsed.files.length > 1;
  for (const file of parsed.files.length ? parsed.files : [{ path: '', notes: [], rows: [], additions: 0, deletions: 0 }]) {
    const section = el('section', { class: 'dv-file' });
    if (multi) section.appendChild(fileHeading(file));
    section.appendChild(fileLines(file));
    root.appendChild(section);
  }
  return { node: root, ...parsed };
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
