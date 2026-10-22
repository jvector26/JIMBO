"""One-off (2026-10-22): rebuild live/2026/pregame.csv from the git history of site/data/latest.json, applying the same
rule as pipeline/nightly.py (last run before tip wins; a missing line keeps the last one seen). Opening-night snapshots
lacked some lines (ESPN abbreviation bug fixed 10-20 evening), so a game with no line in any pre-tip snapshot takes the
final row of live/2026/odds/{date}.csv (closing line, kept after tip by the merge fix). PHI@NYK 10-20 had no line
anywhere; filled by hand after running this (NYK -4.5, the DraftKings line noted in the session 28 opening-night check).
  python tools/backfill_pregame.py   (run from the repo root)
"""
import json, os, subprocess, sys, pandas as pd
sys.argv = sys.argv[:1]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
ns = {}
src = open("pipeline/nightly.py").read()
exec(src[src.index("def home_line"):src.index('pg_f = ')], {"re": __import__("re")}, ns)
home_line = ns["home_line"]
EA = {"SA": "SAS", "GS": "GSW", "NY": "NYK", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
commits = subprocess.check_output(["git", "log", "--reverse", "--format=%h", "--since=2026-10-19", "--", "site/data/latest.json"], text=True).split()
rec = {}
for c in commits:
    L = json.loads(subprocess.check_output(["git", "show", f"{c}:site/data/latest.json"]))
    run = pd.Timestamp(L["generated_utc"].replace("Z", ":00")) if L.get("generated_utc") else None
    if run is None: continue
    for g in L["tonight"]:
        tip = pd.Timestamp(g["time_utc"].replace("Z", "")) if g.get("time_utc") else None
        if tip is None or tip <= run: continue
        ln, lt = home_line(g.get("line"), g["home"], g["away"]), g.get("line")
        p = rec.get(g["game"])
        if ln is None and p and p["line"] is not None: ln, lt = p["line"], p["line_txt"]
        rec[g["game"]] = {"game": g["game"], "date": int(L["game_day"].replace("-", "")), "time_utc": g["time_utc"], "home": g["home"],
                          "away": g["away"], "model": g["margin"], "p_home": g["p_home"], "line": ln, "line_txt": lt,
                          "run_utc": L["generated_utc"]}
for r in rec.values():
    if r["line"] is not None: continue
    f = f"live/2026/odds/{r['date']}.csv"
    if not os.path.exists(f): continue
    o = pd.read_csv(f); o["home"] = o.home.replace(EA); o["away"] = o.away.replace(EA)
    o = o[(o.home == r["home"]) & (o.away == r["away"])]
    if len(o):
        det = " ".join(EA.get(w, w) for w in str(o.details.iloc[-1]).split(" "))
        r["line"], r["line_txt"] = home_line(det, r["home"], r["away"]), det + " (odds file)"
df = pd.DataFrame(rec.values()).sort_values(["date", "time_utc", "game"])
df.to_csv("live/2026/pregame.csv", index=False)
print(df.to_string(index=False))
