/* 禾医 PWA Service Worker
   作用：① 让"添加到主屏幕"后能独立打开 ② 弱网/断网时仍能打开界面
   注意：只缓存静态外壳，**不缓存 /api/** 的响应（诊断结果必须实时） */
const CACHE = 'heyi-shell-v1';
const SHELL = ['/mobile', '/static/manifest.webmanifest', '/static/icon.svg'];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE)
      .then(c => c.addAll(SHELL).catch(() => {}))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys()
      .then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);

  // 诊断/问答接口一律走网络，绝不缓存
  if (url.pathname.startsWith('/api/')) return;

  // 仅处理同源 GET
  if (e.request.method !== 'GET' || url.origin !== location.origin) return;

  e.respondWith(
    fetch(e.request)
      .then(resp => {
        const copy = resp.clone();
        caches.open(CACHE).then(c => c.put(e.request, copy)).catch(() => {});
        return resp;
      })
      .catch(() => caches.match(e.request).then(r => r || caches.match('/mobile')))
  );
});
