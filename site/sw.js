// JIMBO service worker: app shell + model payload cached; nightly data network-first with offline fallback.
const CACHE='jimbo-698d2f259c';
const SHELL=['./','index.html','manifest.webmanifest','icons/icon-192.png','icons/icon-512.png','data/dash.json?v=698d2f259c'];
self.addEventListener('install',e=>{ e.waitUntil(caches.open(CACHE).then(c=>c.addAll(SHELL)).then(()=>self.skipWaiting())); });
self.addEventListener('activate',e=>{ e.waitUntil(caches.keys().then(ks=>Promise.all(ks.filter(k=>k!==CACHE).map(k=>caches.delete(k)))).then(()=>self.clients.claim())); });
self.addEventListener('fetch',e=>{
  const u=new URL(e.request.url); if(e.request.method!=='GET') return;
  const fonts=/fonts\.(googleapis|gstatic)\.com$/.test(u.hostname);
  if(u.origin!==location.origin&&!fonts) return;                       // GitHub API/raw: always live
  if(fonts||u.search.includes('v=')){                                   // versioned payload and fonts: cache first
    e.respondWith(caches.match(e.request).then(r=>r||fetch(e.request).then(n=>{ const c=n.clone(); caches.open(CACHE).then(x=>x.put(e.request,c)); return n; })));
    return; }
  e.respondWith(fetch(e.request).then(n=>{ const c=n.clone(); caches.open(CACHE).then(x=>x.put(e.request,c)); return n; })
    .catch(()=>caches.match(e.request,{ignoreSearch:true})));          // index + nightly JSON: network first, offline fallback
});
