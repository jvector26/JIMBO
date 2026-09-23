"""Fetch the current RealGM NBA depth chart (league page) -> live/{S}/depth/realgm_{YYYYMMDD}.html.gz"""
import argparse, datetime as dt, gzip, os, time
p = argparse.ArgumentParser(); p.add_argument("--season", type=int, default=2026); a = p.parse_args()
URL = "https://basketball.realgm.com/nba/depth-charts"
html = None
try:
    from curl_cffi import requests as cr
    r = cr.get(URL, impersonate="chrome", timeout=60)
    if r.status_code == 200: html = r.content
except Exception as e: print("curl_cffi failed", e)
if html is None:
    import requests
    for i in range(3):
        r = requests.get(URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200: html = r.content; break
        time.sleep(5)
if html is None or b"Depth Chart" not in html: raise SystemExit("depth chart fetch failed")
d = dt.datetime.now(dt.timezone(dt.timedelta(hours=-4))).strftime("%Y%m%d")
os.makedirs(f"live/{a.season}/depth", exist_ok=True)
fn = f"live/{a.season}/depth/realgm_{d}.html.gz"
with gzip.open(fn, "wb") as f: f.write(html)
print("saved", fn, len(html))
