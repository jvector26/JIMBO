"""Build the JIMBO phone app (GitHub Pages PWA) from the dashboard template.
  python code/build_dashboard.py && python code/build_app.py [SITE_DIR]   (default /home/claude/gh/site)
Writes: index.html (template code/dash_v2.html, app loader code/appsrc/app.js), sw.js, manifest.webmanifest, icons/,
data/dash.json (preseason model payload, taken from dashboard.html). site/data/latest.json + history.json come from the
nightly run (pipeline/nightly.py), never from here."""
import hashlib, os, shutil, sys
SITE = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/gh/site"
SRC = "code/appsrc"
dash = open("dashboard.html").read()
tag = '<script id="data" type="application/json">'
i = dash.index(tag) + len(tag); j = dash.index("</script>", i); payload = dash[i:j]
V = hashlib.sha1(payload.encode()).hexdigest()[:10]
os.makedirs(os.path.join(SITE, "data"), exist_ok=True); os.makedirs(os.path.join(SITE, "icons"), exist_ok=True)
open(os.path.join(SITE, "data", "dash.json"), "w").write(payload)
t = open("code/dash_v2.html").read()
assert t.count(tag + "__DATA__</script>") == 1
t = t.replace(tag + "__DATA__</script>", tag + "</script>")
head = """<link rel="manifest" href="manifest.webmanifest">
<link rel="apple-touch-icon" href="icons/apple-touch-icon.png">
<link rel="icon" type="image/png" sizes="192x192" href="icons/icon-192.png">
<meta name="theme-color" content="#4B1F8C">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="JIMBO">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<style>#loading{position:fixed;inset:0;display:grid;place-items:center;background:#FBF8F1;z-index:50;font-family:system-ui,sans-serif;color:#4B1F8C;font-weight:600}
#loading img{width:96px;height:96px;border-radius:22px;display:block;margin:0 auto 12px}</style>
</head>"""
assert t.count("</head>") == 1; t = t.replace("</head>", head)
t = t.replace("<body>", '<body>\n<div id="loading"><div><img src="icons/icon-192.png" alt=""><p>Loading…</p></div></div>', 1)
app = open(os.path.join(SRC, "app.js")).read().replace("__V__", V)
t = t.replace("</body>", "<script>\n" + app + "</script>\n</body>", 1)
open(os.path.join(SITE, "index.html"), "w").write(t)
open(os.path.join(SITE, "sw.js"), "w").write(open(os.path.join(SRC, "sw.js")).read().replace("__V__", V))
shutil.copy(os.path.join(SRC, "manifest.webmanifest"), SITE)
for f in os.listdir(os.path.join(SRC, "icons")): shutil.copy(os.path.join(SRC, "icons", f), os.path.join(SITE, "icons", f))
print(f"app built in {SITE}: payload {len(payload)/1e6:.2f} MB, version {V}")
