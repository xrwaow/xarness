/** DOM helpers: nodes only, never markup. */

export function el(tag, props = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') n.className = v;
    else if (k === 'text') n.textContent = v;
    else if (k === 'dataset') Object.assign(n.dataset, v);
    else if (k === 'on') for (const [e, f] of Object.entries(v)) n.addEventListener(e, f);
    else if (k in n) n[k] = v;
    else n.setAttribute(k, v);
  }
  return append(n, kids);
}

export function append(parent, kids) {
  for (const c of kids.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    parent.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  }
  return parent;
}

export function replace(parent, ...kids) {
  parent.replaceChildren();
  return append(parent, kids);
}

export function byId(id) {
  const n = document.getElementById(id);
  if (!n) throw new Error(`missing element #${id}`);
  return n;
}
let toastTimer = null;

export function toast(message, ms = 2600) {
  const n = byId('toast');
  n.textContent = message;
  n.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    n.hidden = true;
    toastTimer = null;
  }, ms);
}
