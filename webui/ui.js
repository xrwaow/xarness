/** Dialogs (native <dialog>) and view helpers. One sheet at a time: opening
 * another cancels the first. Resolves a value when it closes. */
import { byId, el } from './dom.js';

const sheet = byId('sheet');
let activeFinish = null;

export function openSheet({ title, text = '', body = null, buttons = [], enterConfirms = false, onOpen = null }) {
  return new Promise((resolve) => {
    if (activeFinish) activeFinish(null);
    byId('sheet-title').textContent = title || '';
    const textNode = byId('sheet-text');
    textNode.hidden = !text;
    textNode.textContent = text || '';
    const bodyNode = byId('sheet-body');
    bodyNode.replaceChildren();
    if (body) bodyNode.appendChild(body);
    bodyNode.scrollTop = 0;
    const buttonNode = byId('sheet-buttons');
    buttonNode.replaceChildren();
    let finished = false,
      primary = null,
      primaryNode = null;
    const finish = (value) => {
      if (finished) return;
      finished = true;
      if (activeFinish === finish) activeFinish = null;
      sheet.oncancel = sheet.onclick = sheet.onkeydown = null;
      byId('sheet-close').onclick = null;
      if (sheet.open) sheet.close();
      resolve(value);
    };
    for (const button of buttons) {
      const node = el('button', {
        type: 'button',
        class: `btn${button.primary ? ' btn-primary' : ''}${button.danger ? ' btn-danger' : ''}`,
        text: button.label,
        on: { click: () => finish(button.value) },
      });
      if (button.primary) {
        primary = button;
        primaryNode = node;
      }
      buttonNode.appendChild(node);
    }
    byId('sheet-close').onclick = () => finish(null);
    sheet.oncancel = (event) => {
      event.preventDefault();
      finish(null);
    };
    sheet.onclick = (event) => {
      if (event.target === sheet) finish(null);
    };
    sheet.onkeydown = (event) => {
      if (!enterConfirms || event.key !== 'Enter' || event.defaultPrevented) return;
      const tag = event.target?.tagName;
      if (tag === 'BUTTON' || tag === 'INPUT' || tag === 'TEXTAREA') return;
      event.preventDefault();
      finish(primary ? primary.value : null);
    };
    if (sheet.open) sheet.close();
    activeFinish = finish;
    sheet.showModal();
    if (enterConfirms && primaryNode) primaryNode.focus({ preventScroll: true });
    if (onOpen) onOpen(finish);
  });
}

export function showPicker({ title, rows, text = '', note = '', footer = '', className = '' }) {
  const body = el('div', { class: className });
  if (note) body.appendChild(el('div', { class: 'sheet-note', text: note }));
  let finishSheet = null;
  let selected = Math.max(
    0,
    rows.findIndex((row) => row.current),
  );
  const nodes = [];
  const select = (index, { focus = false } = {}) => {
    if (!nodes.length) return;
    selected = (index + nodes.length) % nodes.length;
    nodes.forEach((node, at) => node.classList.toggle('selected', at === selected));
    nodes[selected].scrollIntoView?.({ block: 'nearest' });
    if (focus) nodes[selected].focus({ preventScroll: true });
  };
  for (const row of rows) {
    if (row.section) body.appendChild(el('div', { class: 'sheet-section', text: row.section }));
    const index = nodes.length;
    const node = el(
      'button',
      {
        type: 'button',
        class: 'sheet-row',
        'aria-current': row.current ? 'true' : null,
        on: { click: () => finishSheet?.(row.id), pointerenter: () => select(index) },
      },
      el('span', { class: 'name', text: row.label }),
      row.swatches
        ? el(
            'span',
            { class: 'swatches' },
            ...row.swatches.map((color) => el('span', { class: 'swatch', style: `background:${color}` })),
          )
        : null,
      row.meta && row.meta.length
        ? el(
            'span',
            { class: 'meta' },
            ...row.meta.map((cell) =>
              el('span', {
                class: `cell ${cell.class || ''}`.trim(),
                dataset: cell.state ? { state: cell.state } : null,
                text: cell.text,
              }),
            ),
          )
        : row.detail
          ? el('span', { class: 'detail', text: row.detail })
          : null,
    );
    nodes.push(node);
    body.appendChild(node);
  }
  if (!rows.length) body.appendChild(el('div', { class: 'sheet-note', text: '(nothing to choose)' }));
  if (footer) body.appendChild(el('div', { class: 'sheet-hint', text: footer }));
  const onKey = (event) => {
    if (!rows.length) return;
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      select(selected + (event.key === 'ArrowDown' ? 1 : -1), { focus: true });
    } else if (event.key === 'Enter') {
      event.preventDefault();
      finishSheet?.(rows[selected]?.id ?? null);
    }
  };
  sheet.addEventListener('keydown', onKey);
  // No Cancel button: Esc, the backdrop and the header ✕ all dismiss a picker,
  // and a full-width bar would read as the sheet's main action.
  return openSheet({
    title,
    text,
    body,
    buttons: [],
    onOpen: (finish) => {
      finishSheet = finish;
      // paint the current row as selected before the pointer moves, so the
      // first arrow key is a visible step from somewhere
      select(selected, { focus: true });
    },
  })
    .finally(() => sheet.removeEventListener('keydown', onKey))
    .then((value) => (value === null || value === undefined ? null : value));
}

export function showConfirm({
  title,
  text = '',
  detail = '',
  confirmLabel = 'OK',
  cancelLabel = 'Cancel',
  danger = false,
}) {
  return openSheet({
    title,
    text,
    body: detail ? el('pre', { class: 'keys', text: detail }) : null,
    buttons: [
      { label: cancelLabel, value: null },
      { label: confirmLabel, value: true, primary: !danger, danger },
    ],
    enterConfirms: true,
  }).then((value) => value === true);
}

export function showPrompt({ title, label = '', value = '', placeholder = '', note = '' }) {
  const input = el('input', { type: 'text', value, placeholder, autocomplete: 'off', spellcheck: 'false' });
  const body = el('div');
  if (note) body.appendChild(el('div', { class: 'sheet-note', text: note }));
  body.appendChild(el('label', { class: 'sheet-field' }, label, input));
  const promise = openSheet({
    title,
    body,
    buttons: [
      { label: 'Cancel', value: null },
      { label: 'OK', value: 'ok', primary: true },
    ],
    enterConfirms: false,
  });
  input.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter') return;
    event.preventDefault();
    byId('sheet-buttons').lastElementChild?.click();
  });

  requestAnimationFrame(() => {
    input.focus();
    input.select();
  });
  return promise.then((result) => (result === 'ok' ? input.value.trim() : null));
}

export function showInfo({ title, text = '', body = null }) {
  return openSheet({
    title,
    text,
    body,
    buttons: [{ label: 'Close', value: null }],
    enterConfirms: true,
  });
}

export function sheetSection(text) {
  return el('div', { class: 'sheet-section', text });
}
