/* App-shell cache. Never touches /v1/*: API responses and event streams must
 * always come from the host. */
const CACHE = 'xarness-ui-v64';
const SHELL = [
  './',
  'index.html',
  'style.css',
  'app.js',
  'api.js',
  'state.js',
  'render.js',
  'markdown.js',
  'highlight.js',
  'diff.js',
  'dom.js',
  'ui.js',
  'commands.js',
  'sha256.js',
  'manifest.webmanifest',
  'icons/icon.svg',
  'icons/icon-192.png',
  'icons/icon-512.png',
];
self.addEventListener('install', (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE);
      await Promise.all(SHELL.map((url) => cache.add(url).catch(() => {})));
      self.skipWaiting();
    })(),
  );
});
self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      for (const key of await caches.keys()) if (key !== CACHE) await caches.delete(key);
      await self.clients.claim();
    })(),
  );
});
self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  if (url.pathname.startsWith('/v1/')) return;
  if ((request.headers.get('accept') || '').includes('text/event-stream')) return;
  if (request.mode === 'navigate') {
    // Network first; the cache is only an offline fallback.
    event.respondWith(
      (async () => {
        try {
          const response = await fetch(request, { cache: 'no-store' });
          const cache = await caches.open(CACHE);
          cache.put('index.html', response.clone()).catch(() => {});
          return response;
        } catch (error) {
          const cached = await caches.match('index.html');
          if (cached) return cached;
          throw error;
        }
      })(),
    );
    return;
  }
  event.respondWith(
    (async () => {
      try {
        // No-store: a cached app.js is a stale UI, and this path only runs when
        // the host is reachable. The shell cache is the offline fallback.
        const response = await fetch(request, { cache: 'no-store' });
        if (response.ok && response.type === 'basic') {
          const cache = await caches.open(CACHE);
          cache.put(request, response.clone()).catch(() => {});
        }
        return response;
      } catch (error) {
        const cached = await caches.match(request);
        if (cached) return cached;
        throw error;
      }
    })(),
  );
});
