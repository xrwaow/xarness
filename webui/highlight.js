/** Syntax coloring for fenced code blocks.
 *
 * The TUI colors code with a palette-driven Pygments theme
 * (`theme.CodeHighlightTheme` in src/xarness/theme.py): keywords in `accent`,
 * type-ish keywords and numbers in `accent2`, strings in `success`, comments
 * in muted italic, function/decorator/class names in `status`, attributes in
 * `accent2`, tags in bold `accent`. The browser has no Pygments, so this is a
 * small sticky-regex lexer per language family that keeps that mapping and
 * paints through the same `--accent`/`--success`/... variables — code follows
 * `/theme` with no re-render, exactly like the TUI.
 *
 * Every node is a text node or a span built with `el`, never markup: model
 * output can never become HTML, the same rule markdown.js follows.
 *
 * A bare fence stays plain (as in the TUI): guessing a language paints prose
 * in arbitrary token colors. An unknown language is plain too.
 */
import { el } from './dom.js';

const set = (words) => new Set(words.trim().split(/\s+/));

// ------------------------------------------------------------------ words
const PY_KEYWORDS = set(`
  and as assert async await break class continue def del elif else except
  finally for from global if import in is lambda match case nonlocal not or
  pass raise return try while with yield
`);
const PY_BUILTINS = set(`
  abs all any ascii bin bool breakpoint bytearray bytes callable chr
  classmethod compile complex delattr dict dir divmod enumerate eval exec
  filter float format frozenset getattr globals hasattr hash help hex id input
  int isinstance issubclass iter len list locals map max memoryview min next
  object oct open ord pow print property range repr reversed round set setattr
  slice sorted staticmethod str sum super tuple type vars zip
`);
const PY_CONSTANTS = set(`True False None NotImplemented Ellipsis`);
const PY_PSEUDO = set(`self cls`);

const C_KEYWORDS = set(`
  abstract as async await base break case catch class const continue debugger
  declare default delete do else enum export extends extern fallthrough final
  finally for foreach friend function goto if implements import in instanceof
  interface internal is lambda let macro namespace new noexcept of operator
  override package private protected public readonly ref require return sealed
  short signed sizeof static struct super switch synchronized template this
  throw throws try type typedef typename typeof union unsafe unsigned use
  using var virtual volatile while with yield
`);
const RUST_EXTRA = set(`as crate dyn fn impl loop match mod move mut pub trait where`);
const GO_EXTRA = set(`chan defer func go range select`);
const RUST_KEYWORDS = new Set([...C_KEYWORDS, ...RUST_EXTRA]);
const GO_KEYWORDS = new Set([...C_KEYWORDS, ...GO_EXTRA]);
const C_TYPES = set(`
  any auto bool boolean byte char double f32 f64 float i8 i16 i32 i64 i128
  int isize long never number object short signed str string u8 u16 u32 u64
  u128 unknown unsigned usize void
`);
const C_CONSTANTS = set(`true false null nil undefined nullptr NaN Infinity`);

const SH_KEYWORDS = set(`
  if then else elif fi for while until do done case esac in function select
  time coproc return exit local export readonly declare unset shift source
  alias set eval exec trap wait break continue
`);
const SH_BUILTINS = set(`
  echo cd pwd printf read test type command builtin let getopts mapfile pushd
  popd dirs alias unalias history jobs kill bg fg nohup sleep true false
`);

const SQL_KEYWORDS = set(`
  add all alter and as asc begin between by cascade case check column commit
  constraint create default delete desc distinct drop else end exists foreign
  from full grant group having if in index inner insert into is join key left
  like limit not null offset on or order outer primary references recursive
  revoke right rollback select set table temporary then transaction union
  unique update values view when where with
`);

// Declarations whose next word is the declared name.
const DECL_FN = set(`def function fn func fun sub method proc`);
const DECL_CLASS = set(`class struct enum interface trait type namespace module union impl`);

// ------------------------------------------------------------------ rules
/** Tail every spec ends with. One character at a time so a punctuation run
 * stops before a delimiter a later-started token needs ("(`x`)"): the
 * tokenizer merges same-type runs back together.
 *
 * Whitespace stays plain and one text node per run, so long code does not
 * become one node per character. */
const TAIL = [{ type: null, re: /\s+/y }, { type: null, re: /[^\s\w]/y }];

const ident = (re) => ({ type: '@ident', re });

const PY_SPEC = {
  rules: [
    { type: 'comment', re: /#[^\n]*/y },
    {
      type: (text) => (/"""|'''/.test(text) ? 'doc' : 'string'),
      re: /(?:[rRbBuUfF]{1,2})?(?:"""[\s\S]*?(?:"""|$)|'''[\s\S]*?(?:'''|$)|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')/y,
    },
    { type: 'decorator', re: /@[\w.]+/y },
    {
      type: 'number',
      re: /\b(?:0[xX][0-9a-fA-F_]+|0[bB][01_]+|0[oO][0-7_]+|(?:\d[\d_]*(?:\.[\d_]*)?|\.\d[\d_]*)(?:[eE][+-]?\d+)?j?)/y,
    },
    ident(/[A-Za-z_]\w*/y),
    ...TAIL,
  ],
  ident(text, { afterDot }) {
    if (PY_KEYWORDS.has(text)) return 'keyword';
    if (PY_CONSTANTS.has(text)) return 'keyword-alt';
    if (PY_PSEUDO.has(text)) return 'pseudo';
    if (PY_BUILTINS.has(text)) return 'builtin';
    if (afterDot) return 'attribute';
    return 'plain';
  },
};

const C_NUMBER_RE =
  /\b(?:(?:0[xX][0-9a-fA-F_]+)|(?:0[bB][01_]+)|(?:0[oO][0-7_]+)|(?:\d[\d_]*(?:\.[\d_]*)?|\.\d[\d_]*)(?:[eE][+-]?\d+)?)(?:[A-Za-z_]\w*)?/y;

const C_RULES = [
  { type: 'comment', re: /\/\/[^\n]*/y },
  { type: 'comment', re: /\/\*[\s\S]*?(?:\*\/|$)/y },
  { type: 'comment', re: /#(?:include|define|undef|if|ifdef|ifndef|else|elif|endif|pragma|error|warning|line)\b[^\n]*/y },
  { type: 'string', re: /`(?:\\[\s\S]|[^`\\])*`?/y },
  { type: 'string', re: /"(?:\\[\s\S]|[^"\\\n])*"?/y },
  { type: 'string', re: /'(?:\\[\s\S]|[^'\\\n])*'?/y },
  { type: 'decorator', re: /@[A-Za-z_]\w*/y },
  { type: 'number', re: C_NUMBER_RE },
  { type: 'attribute', re: /[$A-Za-z_]\w*(?=\s*:)/y },
  ident(/[$A-Za-z_]\w*/y),
  ...TAIL,
];

function cIdentWith(keywords) {
  return (text, { afterDot }) => {
    if (keywords.has(text)) return 'keyword';
    if (C_TYPES.has(text) || C_CONSTANTS.has(text)) return 'keyword-alt';
    if (afterDot) return 'attribute';
    if (text.length > 1 && /^[A-Z]/.test(text)) return 'class';
    return 'plain';
  };
}

const C_SPEC = { rules: C_RULES, ident: cIdentWith(C_KEYWORDS) };

// Rust lifetimes ('a, 'static) must be claimed before the char-literal rule.
const RUST_SPEC = {
  rules: [...C_RULES.slice(0, 3), { type: 'keyword-alt', re: /'[a-z_]\w*(?!')/y }, ...C_RULES.slice(3)],
  ident: cIdentWith(RUST_KEYWORDS),
};

const GO_SPEC = { rules: C_RULES, ident: cIdentWith(GO_KEYWORDS) };

const SH_SPEC = {
  rules: [
    { type: 'comment', re: /#[^\n]*/y },
    { type: 'string', re: /"(?:\\.|[^"\\\n])*"?/y },
    { type: 'string', re: /'[^'\n]*'?/y },
    { type: 'variable', re: /\$\{[^}\n]*\}|\$[A-Za-z_]\w*|\$[@#?*!$0-9-]/y },
    { type: 'number', re: /\b\d+(?:\.\d+)?\b/y },
    ident(/[A-Za-z_][\w-]*/y),
    ...TAIL,
  ],
  ident(text) {
    if (SH_KEYWORDS.has(text)) return 'keyword';
    if (SH_BUILTINS.has(text)) return 'builtin';
    return 'plain';
  },
};

const JSON_SPEC = {
  rules: [
    { type: 'comment', re: /\/\/[^\n]*/y },
    { type: 'attribute', re: /"(?:\\.|[^"\\\n])*"(?=\s*:)/y },
    { type: 'string', re: /"(?:\\.|[^"\\\n])*"?/y },
    { type: 'keyword-alt', re: /\b(?:true|false|null)\b/y },
    { type: 'number', re: /-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b/y },
    ident(/[A-Za-z_]\w*/y),
    ...TAIL,
  ],
  ident: () => 'plain',
};

const CONFIG_SPEC = {
  rules: [
    { type: 'comment', re: /#[^\n]*/y },
    { type: 'attribute', re: /[A-Za-z_][\w.-]*(?=[ \t]*[:=])/y },
    { type: 'string', re: /"(?:\\[\s\S]|[^"\\\n])*"?|'[^'\n]*'?/y },
    { type: 'keyword-alt', re: /\b(?:true|false|null|yes|no|on|off)\b/y },
    { type: 'number', re: /\b\d+(?:\.\d+)?\b/y },
    ...TAIL,
  ],
  ident: () => 'plain',
};

const MARKUP_SPEC = {
  rules: [
    { type: 'comment', re: /<!--[\s\S]*?(?:-->|$)/y },
    { type: 'comment', re: /<!\[CDATA\[[\s\S]*?(?:\]\]>|$)/y },
    { type: 'keyword-alt', re: /<![A-Za-z][^>\n]*>?/y },
    { type: 'tag', re: /<\/?[A-Za-z][\w:.-]*/y },
    { type: 'tag', re: /\/?>/y },
    { type: 'attribute', re: /[A-Za-z_:][\w:.-]*(?==)/y },
    { type: 'string', re: /"(?:[^"\n]*)"?|'[^'\n]*'?/y },
    { type: 'number', re: /&[#\w]+;/y },
    ...TAIL,
  ],
  ident: () => 'plain',
};

const CSS_SPEC = {
  rules: [
    { type: 'comment', re: /\/\*[\s\S]*?(?:\*\/|$)/y },
    { type: 'keyword', re: /@[\w-]+/y },
    { type: 'keyword-alt', re: /!\s*important\b/y },
    { type: 'string', re: /"(?:\\[\s\S]|[^"\\\n])*"?|'[^'\n]*'?/y },
    // `.class` selectors before properties, so `.a:hover` is not read as the
    // property `a`; a property's colon is followed by a space, a pseudo's is not.
    { type: 'tag', re: /\.[A-Za-z_-][\w-]*/y },
    { type: 'attribute', re: /[-A-Za-z]+(?=[ \t]*:[ \t])/y },
    { type: 'number', re: /#[0-9a-fA-F]{3,8}\b|\b\d+(?:\.\d+)?(?:[a-z%]+)?\b/y },
    { type: 'tag', re: /[A-Za-z_-][\w-]*/y },
    ...TAIL,
  ],
  ident: () => 'plain',
};

const SQL_SPEC = {
  rules: [
    { type: 'comment', re: /--[^\n]*/y },
    { type: 'comment', re: /\/\*[\s\S]*?(?:\*\/|$)/y },
    { type: 'string', re: /'(?:''|[^'])*'?/y },
    { type: 'string', re: /"(?:""|[^"])*"?/y },
    { type: 'number', re: /\b\d+(?:\.\d+)?\b/y },
    ident(/[A-Za-z_]\w*/y),
    ...TAIL,
  ],
  ident(text) {
    return SQL_KEYWORDS.has(text.toLowerCase()) ? 'keyword' : 'plain';
  },
};

const SPECS = {
  python: PY_SPEC,
  clike: C_SPEC,
  rust: RUST_SPEC,
  go: GO_SPEC,
  shell: SH_SPEC,
  json: JSON_SPEC,
  config: CONFIG_SPEC,
  markup: MARKUP_SPEC,
  css: CSS_SPEC,
  sql: SQL_SPEC,
};

const ALIASES = {
  py: 'python',
  python: 'python',
  python3: 'python',
  js: 'clike',
  javascript: 'clike',
  jsx: 'clike',
  mjs: 'clike',
  cjs: 'clike',
  ts: 'clike',
  typescript: 'clike',
  tsx: 'clike',
  c: 'clike',
  h: 'clike',
  cpp: 'clike',
  'c++': 'clike',
  cc: 'clike',
  cxx: 'clike',
  hpp: 'clike',
  java: 'clike',
  kotlin: 'clike',
  kt: 'clike',
  cs: 'clike',
  csharp: 'clike',
  go: 'go',
  golang: 'go',
  swift: 'clike',
  dart: 'clike',
  scala: 'clike',
  rust: 'rust',
  rs: 'rust',
  sh: 'shell',
  bash: 'shell',
  zsh: 'shell',
  fish: 'shell',
  shell: 'shell',
  console: 'shell',
  json: 'json',
  jsonc: 'json',
  json5: 'json',
  yaml: 'config',
  yml: 'config',
  toml: 'config',
  ini: 'config',
  conf: 'config',
  env: 'config',
  dotenv: 'config',
  properties: 'config',
  html: 'markup',
  xhtml: 'markup',
  htm: 'markup',
  xml: 'markup',
  svg: 'markup',
  vue: 'markup',
  svelte: 'markup',
  css: 'css',
  scss: 'css',
  sass: 'css',
  less: 'css',
  sql: 'sql',
  postgres: 'sql',
  postgresql: 'sql',
  mysql: 'sql',
  sqlite: 'sql',
};

/** The lexer for a fence info string ("python title=x" → python), or null. */
export function specFor(language) {
  const tag = String(language ?? '')
    .trim()
    .split(/\s+/)[0]
    .toLowerCase()
    .replace(/^language-/, '');
  return (tag && SPECS[ALIASES[tag]]) || null;
}

/** Split source into `{ text, type }` runs; type is '' for plain text. */
export function tokenize(source, spec) {
  const tokens = [];
  let i = 0;
  let pending = null; // 'function' | 'class' while a declaration name is due
  let prevChar = '';
  const push = (text, type) => {
    if (!text) return;
    const last = tokens[tokens.length - 1];
    if (last && last.type === (type || '')) last.text += text;
    else tokens.push({ text, type: type || '' });
  };
  while (i < source.length) {
    let matched = false;
    for (const rule of spec.rules) {
      rule.re.lastIndex = i;
      const found = rule.re.exec(source);
      if (!found || !found[0]) continue;
      const text = found[0];
      let type = typeof rule.type === 'function' ? rule.type(text, { pending, prevChar }) : rule.type;
      if (type === '@ident') {
        type = text === 'this' ? 'keyword-alt' : spec.ident(text, { pending, afterDot: prevChar === '.', prevChar });
        if (type === 'plain' && pending) type = pending;
      }
      push(text, type === 'plain' ? '' : type);
      if (text.trim()) {
        pending = type === 'keyword' ? (DECL_FN.has(text) ? 'function' : DECL_CLASS.has(text) ? 'class' : null) : null;
        prevChar = text[text.length - 1];
      }
      i += text.length;
      matched = true;
      break;
    }
    if (matched) continue;
    // No rule matched (a stray character): emit it and step one code point.
    const codePoint = source.codePointAt(i);
    const size = codePoint > 0xffff ? 2 : 1;
    const text = source.slice(i, i + size);
    push(text, '');
    prevChar = text;
    i += size;
  }
  return tokens;
}

/** Append `code` to `pre`, colored for `language` when that language is known.
 * Plain for a bare or unknown fence. */
export function appendHighlighted(pre, code, language) {
  const source = String(code ?? '');
  const spec = specFor(language);
  if (!spec) {
    pre.appendChild(document.createTextNode(source));
    return pre;
  }
  for (const token of tokenize(source, spec)) {
    if (!token.type) pre.appendChild(document.createTextNode(token.text));
    else pre.appendChild(el('span', { class: `tok-${token.type}`, text: token.text }));
  }
  return pre;
}
