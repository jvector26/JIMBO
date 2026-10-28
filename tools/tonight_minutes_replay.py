"""Replay tonight's-minutes variants over this season's finished game days (2026-10-28, Cormac Ryan question).
For each asof date with final games: run pipeline/nightly.py --asof D with JIMBO_GAME_VARIANT=v, read tonight_min.csv, score
vs actual box minutes (every rostered player of the teams playing; not in the box or DNP = 0) and the game margin vs final.
  python3 tools/tonight_minutes_replay.py OUTDIR [variants, default: ,a,b,ab,abc,abd]
"""
import glob, os, subprocess, sys, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); L = os.path.join(ROOT, "live", "2026")
OUT = sys.argv[1]; VAR = (sys.argv[2] if len(sys.argv) > 2 else ",a,b,ab,abc,abd").split(",")
meta = pd.read_csv(os.path.join(L, "games_meta.csv")); meta = meta[meta.status == 3]
dates = sorted(meta.date_et.unique())
def mins(x):
    if not isinstance(x, str) or not x.startswith("PT"): return 0.0
    m, s_ = x[2:].rstrip("S").split("M"); return float(m) + float(s_ or 0) / 60
box = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(L, "box", "*.csv.gz"))])
box["m"] = box.minutes.map(mins); act = box.groupby("player").m.apply(list)
def run(v, d):
    o = os.path.join(OUT, f"v{v or '0'}_{d}"); env = dict(os.environ, JIMBO_GAME_VARIANT=v, JIMBO_DUMP_TONIGHT="1")
    if not os.path.exists(os.path.join(o, "latest.json")):
        subprocess.run([sys.executable, os.path.join(ROOT, "pipeline", "nightly.py"), "--asof", d, "--sims", "200", "--no-history",
                        "--parsed", os.path.join(ROOT, "state", "season_parse"), "--out", o], env=env, check=True, capture_output=True)
    return v, d, o
with ThreadPoolExecutor(6) as ex: res = list(ex.map(lambda x: run(*x), [(v, d) for v in VAR for d in dates]))
rows, grows = [], []
for v, d, o in res:
    T = pd.read_csv(os.path.join(o, "tonight_min.csv"))
    g = meta[meta.date_et == d]
    import json; lt = json.load(open(os.path.join(o, "latest.json")))
    mg = {(x["home"], x["away"]): x["margin"] for x in lt["tonight"]}
    for _, gg in g.iterrows():
        b = box[box.game == gg.game].set_index("player").m
        for t in (gg.htm, gg.vtm):
            q = T[T.team == t]
            for _, r in q.iterrows():
                rows.append((v, d, t, r["name"], r.tn, float(b.get(r.pid, 0.0)) if r.pid >= 0 else 0.0))
        grows.append((v, d, gg.game, mg.get((gg.htm, gg.vtm), np.nan), gg.home_pts - gg.away_pts))
R = pd.DataFrame(rows, columns=["v", "date", "team", "name", "pred", "act"]); G = pd.DataFrame(grows, columns=["v", "date", "game", "model", "final"])
R.to_csv(os.path.join(OUT, "player_rows.csv"), index=False); G.to_csv(os.path.join(OUT, "game_rows.csv"), index=False)
R["e2"] = (R.pred - R.act) ** 2; G["e2"] = (G.model - G.final) ** 2
s = pd.DataFrame({"player_min_rmse": R.groupby("v").e2.mean() ** .5,
                  "phantom_min": R[(R.act == 0)].groupby("v").pred.sum() / R.groupby("v").date.nunique(),
                  "games": G.groupby("v").size(), "margin_rmse": G.groupby("v").e2.mean() ** .5})
print(s.round(3).to_string())
base = R[R.v == VAR[0]].reset_index(drop=True)
for v in VAR[1:]:
    x = R[R.v == v].reset_index(drop=True); dd = (x.e2 - base.e2).groupby(x.team + x.date).mean()
    gg = G[G.v == v].reset_index(drop=True).e2 - G[G.v == VAR[0]].reset_index(drop=True).e2
    print(f"{v:>4}: player-min dMSE {dd.mean():+.2f} (t {dd.mean() / dd.std() * len(dd) ** .5:+.1f}, team-games better {(dd < 0).mean():.0%}) | "
          f"margin dMSE {gg.mean():+.2f} (t {gg.mean() / gg.std() * len(gg) ** .5:+.1f})")
