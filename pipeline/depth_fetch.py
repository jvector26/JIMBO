"""Fetch the current RealGM NBA depth chart (league page) -> live/{S}/depth/realgm_{YYYYMMDD}.html.gz.
Writes live/{S}/depth/fetch_log.txt with every attempt (Actions logs are not readable from the sandbox)."""
import argparse, datetime as dt, gzip, os, time
p = argparse.ArgumentParser(); p.add_argument("--season", type=int, default=2026)
p.add_argument("--until", help="YYYY-MM-DD: do nothing after this date (US Eastern); nightly passes the day before the opener")
a = p.parse_args()
if a.until and dt.datetime.now(dt.timezone(dt.timedelta(hours=-4))).strftime("%Y-%m-%d") > a.until:
    print(f"after {a.until}: depth chart frozen (last pre-opener chart stays in force)"); raise SystemExit(0)
URL = "https://basketball.realgm.com/nba/depth-charts"
os.makedirs(f"live/{a.season}/depth", exist_ok=True)
log = []; html = None
def ok(b): return b is not None and b"Depth Chart</h2>" in b or (b is not None and b"depth_starters" in b)
# Cloudflare challenges runner IPs intermittently (2026-09-23: one run all 403, the next 200) -> 3 rounds, 60 s apart
for rnd in range(3):
    for imp in ("chrome", "chrome120", "safari17_0", "firefox"):
        try:
            from curl_cffi import requests as cr
            r = cr.get(URL, impersonate=imp, timeout=60)
            log.append(f"round {rnd} curl_cffi {imp} {r.status_code} {len(r.content)} {r.content[:60]!r}")
            if r.status_code == 200 and ok(r.content): html = r.content; break
        except Exception as e: log.append(f"curl_cffi {imp} error {e}")
        time.sleep(3)
    if html is not None: break
    if rnd < 2: time.sleep(60)
if html is None:
    import requests
    for ua in ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36", "JIMBO"):
        try:
            r = requests.get(URL, timeout=60, headers={"User-Agent": ua})
            log.append(f"requests {ua[:10]} {r.status_code} {len(r.content)} {r.content[:120]!r}")
            if r.status_code == 200 and ok(r.content): html = r.content; break
        except Exception as e: log.append(f"requests error {e}")
def wayback_latest():
    import requests
    frm = (dt.datetime.utcnow() - dt.timedelta(days=10)).strftime("%Y%m%d")
    r = requests.get("http://web.archive.org/cdx/search/cdx", params={"url": "basketball.realgm.com/nba/depth-charts", "output": "json",
                     "from": frm, "filter": "statuscode:200", "fl": "timestamp,original"}, timeout=90)
    rows = r.json()[1:] if r.status_code == 200 and r.text.strip() else []
    return rows[-1] if rows else None
if html is None:  # RealGM sits behind a Cloudflare challenge for cloud IPs -> go through the Wayback Machine
    import requests
    try:
        last = wayback_latest(); log.append(f"wayback latest {last}")
        stale = last is None or (dt.datetime.utcnow() - dt.datetime.strptime(last[0], "%Y%m%d%H%M%S")).total_seconds() > 30 * 3600
        if stale:
            r = requests.get("https://web.archive.org/save/" + URL, timeout=180, headers={"User-Agent": "JIMBO (github.com/jvector26/JIMBO)"})
            log.append(f"save-page-now {r.status_code}"); time.sleep(20)
            last = wayback_latest() or last; log.append(f"wayback latest after save {last}")
        if last:
            r = requests.get(f"https://web.archive.org/web/{last[0]}id_/{last[1]}", timeout=120)
            log.append(f"wayback fetch {r.status_code} {len(r.content)}")
            if r.status_code == 200 and ok(r.content): html = r.content; log.append(f"capture {last[0]}")
    except Exception as e: log.append(f"wayback error {e}")
d = dt.datetime.now(dt.timezone(dt.timedelta(hours=-4))).strftime("%Y%m%d")
if html is not None:
    fn = f"live/{a.season}/depth/realgm_{d}.html.gz"
    with gzip.open(fn, "wb") as f: f.write(html)
    log.append(f"saved {fn} {len(html)}")
open(f"live/{a.season}/depth/fetch_log.txt", "w").write(f"{dt.datetime.utcnow().isoformat()}Z\n" + "\n".join(log) + "\n")
print("\n".join(log))
