/** Host API client: one method per endpoint of `src/xarness/api.py`, mirroring
 * `clientlib.py`. SSE is read with fetch so the auth header can be sent. */
const JSON_HEADERS = { 'Content-Type': 'application/json' };

export class ApiError extends Error {

  constructor(message, status = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }

  get unauthorized() {
    return this.status === 401;
  }

  get permanent() {
    return this.status !== null && this.status >= 400 && this.status < 500 && this.status !== 429;
  }
}

async function errorDetail(response) {
  let detail;
  try {
    detail = (await response.json()).detail;
  } catch {
    return `${response.status} ${response.statusText}`;
  }
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (!item || typeof item !== 'object') return String(item);
        const loc = (item.loc || []).join('.');
        return loc ? `${loc}: ${item.msg}` : String(item.msg);
      })
      .join('; ');
  }
  return String(detail ?? `${response.status} ${response.statusText}`);
}

function query(params) {
  const search = new URLSearchParams();
  for (const [k, v] of Object.entries(params || {}))
    if (v !== undefined && v !== null && v !== '') search.set(k, String(v));
  const text = search.toString();
  return text ? `?${text}` : '';
}
const sessionPath = (name) => `/v1/sessions/${encodeURIComponent(name)}`;

export class Api {

  constructor({ token = null } = {}) {
    this.token = token;
    this._images = new Map();
  }

  setToken(token) {
    this.token = token;
    this._images.clear();
  }

  headers(extra = {}) {
    const h = { ...extra };
    if (this.token) h.Authorization = `Bearer ${this.token}`;
    return h;
  }

  async request(method, path, { json, body, headers = {}, signal } = {}) {
    const init = { method, headers: this.headers(headers), signal, cache: 'no-store' };
    if (json !== undefined) {
      Object.assign(init.headers, JSON_HEADERS);
      init.body = JSON.stringify(json);
    } else if (body !== undefined) init.body = body;
    let response;
    try {
      response = await fetch(path, init);
    } catch (error) {
      if (error?.name === 'AbortError') throw error;
      throw new ApiError(`host unreachable (${error.message})`, null);
    }
    if (!response.ok) throw new ApiError(await errorDetail(response), response.status);
    if (!(response.headers.get('content-type') || '').includes('application/json')) return response.text();
    return response.json();
  }

  get(path, params, { signal } = {}) {
    return this.request('GET', `${path}${query(params)}`, { signal });
  }

  health({ signal } = {}) {
    return this.get('/v1/health', undefined, { signal });
  }

  config() {
    return this.get('/v1/config');
  }

  setDefaultProfile(name) {
    return this.request('PUT', '/v1/config/defaults', { json: { default_profile: name } });
  }

  searchFiles(workspace, q) {
    return this.get('/v1/workspaces/files', { workspace, q });
  }

  workspaceRefs(workspace) {
    return this.get('/v1/workspaces/refs', { workspace });
  }

  updateWorkspaceRefs(workspace, path, add) {
    return this.request('POST', '/v1/workspaces/refs', { json: { workspace, path, add } });
  }

  sessions({ active, workspace, stats = false } = {}) {
    return this.get('/v1/sessions', {
      active: active === undefined ? undefined : String(active).toLowerCase(),
      workspace,
      stats: stats ? 'true' : undefined,
    });
  }

  createSession(request) {
    return this.request('POST', '/v1/sessions', { json: request });
  }

  deleteSession(name) {
    return this.request('DELETE', sessionPath(name));
  }

  diff(name, { path, author } = {}) {
    return this.get(`${sessionPath(name)}/diff`, { path, author });
  }

  tools(name) {
    return this.get(`${sessionPath(name)}/tools`);
  }

  command(name, command) {
    return this.request('POST', `${sessionPath(name)}/commands`, { json: command });
  }

  async uploadImage(name, data, imageId) {
    await this.request('PUT', `${sessionPath(name)}/images/${imageId}`, {
      body: data,
      headers: { 'Content-Type': 'application/octet-stream' },
    });
  }

  async imageUrl(name, imageId) {
    const key = `${name}/${imageId}`;
    if (this._images.has(key)) return this._images.get(key);
    const response = await fetch(`${sessionPath(name)}/images/${imageId}`, {
      headers: this.headers(),
      cache: 'force-cache',
    });
    if (!response.ok) throw new ApiError('image not found', response.status);
    const url = URL.createObjectURL(await response.blob());
    this._images.set(key, url);
    return url;
  }

  async stream(name, { signal, onSnapshot = () => {}, onEvent = () => {}, onOpen = () => {} } = {}) {
    const response = await fetch(`${sessionPath(name)}/events`, {
      headers: this.headers({ Accept: 'text/event-stream' }),
      signal,
      cache: 'no-store',
    });
    if (!response.ok) throw new ApiError(await errorDetail(response), response.status);
    onOpen();
    if (!response.body) throw new ApiError('the host sent no stream body', 500);
    const reader = response.body.getReader(),
      decoder = new TextDecoder();
    let buffer = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let boundary = buffer.indexOf('\n\n');
      while (boundary >= 0) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        let kind = 'event';
        const data = [];
        for (const line of frame.split('\n')) {
          if (!line || line.startsWith(':')) continue;
          if (line.startsWith('event:')) kind = line.slice(6).trim();
          else if (line.startsWith('data:')) data.push(line.slice(5).trim());
        }
        if (data.length) {
          let payload = null;
          try {
            payload = JSON.parse(data.join('\n'));
          } catch {
            payload = null;
          }
          if (payload) {
            if (kind === 'snapshot') onSnapshot(payload);
            else if (kind === 'event') onEvent(payload);
          }
        }
        boundary = buffer.indexOf('\n\n');
      }
    }
  }
}
export const api = new Api();
