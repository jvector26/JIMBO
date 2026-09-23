"""Fetch the current RealGM NBA depth chart (league page) -> live/{S}/depth/realgm_{YYYYMMDD}.html.gz.
Writes live/{S}/depth/fetch_log.txt with every attempt (Actions logs are not readable from the sandbox)."""
import argparse, datetime as dt, gzip, os, time
p = argparse.ArgumentParser(); p.add_argument("--season", type=int, default=2026); a = p.parse_args()
URL = "https://basketball.realgm.com/nba/depth-charts"
os.makedirs(f"live/{a.season}/depth", exist_ok=True)
log = []; html = None
def ok(b): return b is not None and b"Depth Chart</h2>" in b or (b is not None and b"depth_starters" in b)
for imp in ("chrome", "chrome120", "safari17_0", "firefox"):
    try:
        from curl_cffi import requests as cr
        r = cr.get(URL, impersonate=imp, timeout=60)
        log.append(f"curl_cffi {imp} {r.status_code} {len(r.content)} {r.content[:120]!r}")
        if r.status_code == 200 and ok(r.content): html = r.content; break
    except Exception as e: log.append(f"curl_cffi {imp} error {e}")
    time.sleep(3)
if html is None:
    import requests
    for ua in ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36", "JIMBO"):
        try:
            r = requests.get(URL, timeout=60, headers={"User-Agent": ua})
            log.append(f"requests {ua[:10]} {r.status_code} {len(r.content)} {r.content[:120]!r}")
            if r.status_code == 200 and ok(r.content): html = r.content; break
        except Exception as e: log.append(f"requests error {e}")
d = dt.datetime.now(dt.timezone(dt.timedelta(hours=-4))).strftime("%Y%m%d")
if html is not None:
    fn = f"live/{a.season}/depth/realgm_{d}.html.gz"
    with gzip.open(fn, "wb") as f: f.write(html)
    log.append(f"saved {fn} {len(html)}")
open(f"live/{a.season}/depth/fetch_log.txt", "w").write(f"{dt.datetime.utcnow().isoformat()}Z\n" + "\n".join(log) + "\n")
print("\n".join(log))
