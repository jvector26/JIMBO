"""Validate the age bins in player-rating units: does a player's next-season measured rating sit below/above the
3-season window rating (x slope), by age? Window = decayed (0.3/0.6/1.0) possession-weighted average of per-season box
contributions + 0.6 x window lineup (lineup_hist.parquet). 'Actual' next season = that season's box alone (+ optional
single-season lineup). Minutes-weighted means by age bin, season-clustered SE."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/code")
from features import COMPONENTS
D = "/home/claude/data"
F = [f for fs in COMPONENTS.values() for f in fs]
OFF = [f for f in F if f.startswith("o_")]
W = json.load(open(f"{D}/weights_prod.json")); Wa = {**W["Wo"], **W["Wd"]}
SLOPE = float(pd.read_csv(f"{D}/calib.csv", index_col=0)["0"]["slope"])
BINS = [(0, 21.99, "<=21"), (22, 23.99, "22-23"), (24, 25.99, "24-25"), (26, 27.99, "26-27"), (28, 29.99, "28-29"),
        (30, 31.99, "30-31"), (32, 33.99, "32-33"), (34, 35.99, "34-35"), (36, 99, "36+")]
fe = pd.read_parquet(f"{D}/features.parquet"); fe = fe[fe["min"] > 0].copy()
for x in F:
    wt = fe.poss_off if x in OFF else fe.poss_def
    fe[x] = fe[x] * 0.5 * Wa[x]
    fe[x] -= (fe[x] * wt).groupby(fe.season).transform("sum") / wt.groupby(fe.season).transform("sum")
fe["box_o"] = fe[OFF].sum(1); fe["box_d"] = fe[[f for f in F if f not in OFF]].sum(1)
LH = pd.read_parquet(f"{D}/lineup_hist.parquet")
LH["lw"] = LH.win_o + LH.win_d; LH["l1"] = LH.one_o + LH.one_d
fe = fe.merge(LH[["season", "player", "lw", "l1"]], on=["season", "player"], how="left")
idx = fe.set_index(["player", "season"])
rows = []
for N in range(1996, 2025):
    cur = fe[fe.season == N]
    bo = bd = wo = wd = 0
    for lag, dec in ((0, 1.0), (1, 0.6), (2, 0.3)):
        p = fe[fe.season == N - lag].set_index("player")
        bo = bo + (p.box_o * p.poss_off * dec).reindex(cur.player).fillna(0).to_numpy()
        bd = bd + (p.box_d * p.poss_def * dec).reindex(cur.player).fillna(0).to_numpy()
        wo = wo + (p.poss_off * dec).reindex(cur.player).fillna(0).to_numpy()
        wd = wd + (p.poss_def * dec).reindex(cur.player).fillna(0).to_numpy()
    win_box = bo / np.maximum(wo, 1) + bd / np.maximum(wd, 1)
    nxt = fe[fe.season == N + 1].set_index("player")
    rows.append(pd.DataFrame({"season": N, "player": cur.player.to_numpy(), "age_next": cur.age.to_numpy() + 1,
        "min_win": cur["min"].to_numpy(), "win": SLOPE * (win_box + 0.6 * cur.lw.fillna(0).to_numpy()),
        "nseasons": sum((fe[fe.season == N - l].set_index("player")["min"].reindex(cur.player).notna()).to_numpy() for l in (0, 1, 2)),
        "act_box": nxt.box_o.add(nxt.box_d).reindex(cur.player).to_numpy(),
        "act_l1": nxt.l1.reindex(cur.player).to_numpy(), "min_next": nxt["min"].reindex(cur.player).to_numpy()}))
A = pd.concat(rows).dropna(subset=["act_box", "min_next"])
A = A[(A.min_win >= 500) & (A.min_next >= 500)].copy()
A["act"] = A.act_box + 0.6 * A.act_l1.fillna(0)
A["bin"] = pd.cut(A.age_next, [b[0] - .001 for b in BINS] + [99], labels=[b[2] for b in BINS])
aging = pd.read_csv(f"{D}/aging.csv").set_index("bin").adj
def tab(df, col, lab):
    g = []
    for b, d in df.groupby("bin", observed=True):
        w = d.min_next; e = d[col] - d.win
        m = np.average(e, weights=w)
        se = np.sqrt(np.average([(np.average(x[col] - x.win, weights=x.min_next) - m) ** 2 for _, x in d.groupby("season") if len(x) > 3]) / max(d.season.nunique() - 1, 1))
        g.append((b, len(d), round(m, 3), round(se, 3), round(float(aging.get(b, np.nan)), 3)))
    print(f"\n{lab}: bin, n, measured next-season gap, SE, model bin")
    for r in g: print("  %-6s n=%-5d %+0.2f (%.2f)   model %+0.2f" % (r[0], r[1], r[2], r[3], r[4]))
tab(A, "act", "Box + single-season lineup")
tab(A, "act_box", "Box only")
y = A[A.age_next <= 21.99]
print("\n<=21 split by seasons of history:", {int(k): (len(d), round(np.average(d.act - d.win, weights=d.min_next), 2)) for k, d in y.groupby("nseasons")})
print("<=21 by half:", {h: round(np.average(d.act - d.win, weights=d.min_next), 2) for h, d in y.groupby(np.where(y.season < 2011, "96-10", "11-24"))})
print("<=21 by window rating:", {str(k): (len(d), round(np.average(d.act - d.win, weights=d.min_next), 2)) for k, d in y.groupby(pd.cut(y.win, [-9, -1, 0, 1, 9]), observed=True)})
print("<=21 count by exact age_next:", y.age_next.round().value_counts().sort_index().to_dict())

# --- how the next-season rating relates to the window rating, by age bin (model assumes act = 0.971*win + bin)
print("\nact = a + b*window (raw, pre-slope), minutes-weighted, by age bin; model says a = bin adj, b = %.3f" % SLOPE)
A["winraw"] = A.win / SLOPE
for b_, d in A.groupby("bin", observed=True):
    w = d.min_next.to_numpy(); X = np.c_[np.ones(len(d)), d.winraw.to_numpy()]; y = d.act.to_numpy()
    cf = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    bo = np.array([np.linalg.lstsq((lambda s: np.c_[np.ones(len(s)), s.winraw] * np.sqrt(s.min_next.to_numpy())[:, None])(s),
                                   (s.act * np.sqrt(s.min_next)).to_numpy(), rcond=None)[0]
                   for s in [d[d.season.isin(np.random.default_rng(i).choice(sorted(d.season.unique()), d.season.nunique()))] for i in range(100)]])
    print("  %-6s n=%-5d a %+0.2f (%.2f)  b %.2f (%.2f) | implied adj at rating +1.7: %+0.2f" %
          (b_, len(d), cf[0], bo[:, 0].std(), cf[1], bo[:, 1].std(), cf[0] + (cf[1] - SLOPE) * 1.7))
