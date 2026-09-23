"""Fetch archived preseason NBA depth charts (RealGM; ESPN discovery as fallback) from the Wayback Machine.
Runs on GitHub Actions (archive.org is blocked from the Claude sandbox).
Output: research/depth_wayback/captures.csv (every capture found) and html/{season}_{src}_{timestamp}.html.gz
Season S = season starting in year S (2014 = 2014-15)."""
import csv, gzip, os, sys, time, datetime as dt
import requests

OUT = "research/depth_wayback"
OPENER = {2014: "2014-10-28", 2015: "2015-10-27", 2016: "2016-10-25", 2017: "2017-10-17", 2018: "2018-10-16",
          2019: "2019-10-22", 2020: "2020-12-22", 2021: "2021-10-19", 2022: "2022-10-18", 2023: "2023-10-24",
          2024: "2024-10-22", 2025: "2025-10-21"}
SOURCES = {  # src -> CDX prefix
    "realgm": "basketball.realgm.com/nba/depth",
    "espn": "www.espn.com/nba/depth",
}
S = requests.Session(); S.headers["User-Agent"] = "JIMBO-research (github.com/jvector26/JIMBO)"

def get(url, **kw):
    for i in range(6):
        try:
            r = S.get(url, timeout=90, **kw)
            if r.status_code == 200: return r
            if r.status_code in (404, 403): return r
        except Exception as e:
            print("  retry", i, e, flush=True)
        time.sleep(5 * (i + 1))
    return None

def cdx(prefix, frm, to):
    r = get("http://web.archive.org/cdx/search/cdx", params=dict(url=prefix, matchType="prefix", output="json",
            fl="timestamp,original,statuscode,length", filter="statuscode:200", **{"from": frm, "to": to}))
    if r is None or r.status_code != 200: return []
    rows = r.json()
    return [dict(zip(rows[0], x)) for x in rows[1:]] if rows else []

def main():
    os.makedirs(f"{OUT}/html", exist_ok=True)
    caps = []
    for season, op in OPENER.items():
        o = dt.date.fromisoformat(op)
        frm = (o - dt.timedelta(days=60)).strftime("%Y%m%d"); to = (o + dt.timedelta(days=3)).strftime("%Y%m%d")
        for src, pre in SOURCES.items():
            rows = cdx(pre, frm, to); time.sleep(2)
            for x in rows: x.update(season=season, src=src, opener=op)
            caps += rows
            print(season, src, len(rows), "captures", flush=True)
    with open(f"{OUT}/captures.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["season", "src", "opener", "timestamp", "original", "statuscode", "length"])
        w.writeheader(); w.writerows(caps)
    # choose RealGM league-wide page captures: latest before opener, and ~7/14/28 days before (distinct days)
    for season, op in OPENER.items():
        o = dt.date.fromisoformat(op)
        for src in SOURCES:
            c = [x for x in caps if x["season"] == season and x["src"] == src]
            if src == "realgm":
                c = [x for x in c if x["original"].split("?")[0].rstrip("/").lower().endswith("depth-charts")] or c
            else:
                c = [x for x in c if "/team/" not in x["original"]] or c
            c = [x for x in c if dt.datetime.strptime(x["timestamp"][:8], "%Y%m%d").date() < o]
            if not c: print(season, src, "NO pre-opener capture", flush=True); continue
            picks = {}
            for lag in (1, 7, 14, 28):
                tgt = o - dt.timedelta(days=lag)
                best = min(c, key=lambda x: abs((dt.datetime.strptime(x["timestamp"][:8], "%Y%m%d").date() - tgt).days))
                picks[best["timestamp"][:8]] = best
            for day, x in sorted(picks.items()):
                fn = f"{OUT}/html/{season}_{src}_{x['timestamp']}.html.gz"
                if os.path.exists(fn): continue
                r = get(f"https://web.archive.org/web/{x['timestamp']}id_/{x['original']}")
                time.sleep(3)
                if r is None or r.status_code != 200: print("  fail", season, src, x["timestamp"], flush=True); continue
                with gzip.open(fn, "wb") as f: f.write(r.content)
                print("  saved", fn, len(r.content), flush=True)

if __name__ == "__main__":
    main()
