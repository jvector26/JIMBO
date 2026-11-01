"""Home court: recency-weighted vs constant (session 40, 2026-11-01, user OK after vs-Vegas review).
Research-bar game backtest (ACTUAL minutes, base = box + .6 res_o + .8 res_d) + b2b_diff (production since session 37).
Only the intercept (home court) changes between variants. Season history of home court h_s = mean(margin - c*base - beta*b2b)
using the season-q LOSO slopes; seasons before 2015 (no backtest) use raw home margin minus the same b2b part (raw b2b counts)."""
import numpy as np, pandas as pd
D = "/home/claude/data"
B = pd.read_parquet(f"{D}/backtest_components.parquet").merge(pd.read_parquet(f"{D}/bt_split.parquet"), on=["game", "home_team", "away_team"])
G = pd.read_parquet(f"{D}/b2b_games.parquet")[["game", "b2b_h", "b2b_a"]]
B = B.merge(G, on="game", how="left").fillna({"b2b_h": 0, "b2b_a": 0})
F = [c for c in B.columns if c.startswith("f:")]
base = (B[F].sum(axis=1) + B.unrated + 0.6 * B.res_o + 0.8 * B.res_d).to_numpy()
bd = (B.b2b_h - B.b2b_a).to_numpy(); y = B.margin.to_numpy(); s = B.season.to_numpy()
seasons = np.unique(s); print("backtest seasons", seasons.min(), seasons.max(), "games", len(y))
# raw history (all seasons) for pre-2015 priors
A = pd.read_parquet(f"{D}/games_all.parquet"); A = A[A.score_ok != False]
A["m"] = A.home_pts - A.away_pts
Gb = pd.read_parquet(f"{D}/b2b_games.parquet")[["game", "season", "b2b_h", "b2b_a", "margin"]]
raw = A.groupby("season").m.mean()
bdraw = (Gb.b2b_h - Gb.b2b_a).groupby(Gb.season).mean()
X = np.c_[np.ones(len(y)), base, bd]
res = {}
H = {}   # per test season q: dict of season -> h_s, and the slopes
for q in seasons:
    tr = s != q; b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
    c, beta = b[1], b[2]
    h = {int(t): float(np.mean(y[s == t] - c * base[s == t] - beta * bd[s == t])) for t in seasons}
    for t in raw.index:
        if t not in h and t >= 1997: h[int(t)] = float(raw[t] - beta * bdraw.get(t, 0.0))
    H[q] = (b, h)
def pred(q, a):
    b, _ = H[q]; m = s == q
    return a + b[1] * base[m] + b[2] * bd[m]
def wmean(h, yrs, w):
    yrs = [t for t in yrs if t in h]; w = np.array([w(t) for t in yrs]); return float(np.dot(w, [h[t] for t in yrs]) / w.sum())
V = {
 "A LOSO const (all other)": lambda q, h: wmean(h, [t for t in seasons if t != q], lambda t: 1),
 "B prior since 2004":       lambda q, h: wmean(h, range(2004, q), lambda t: 1),
 "C prior 10 seasons":       lambda q, h: wmean(h, range(q-10, q), lambda t: 1),
 "D prior 5":                lambda q, h: wmean(h, range(q-5, q), lambda t: 1),
 "E prior 3":                lambda q, h: wmean(h, range(q-3, q), lambda t: 1),
 "F prior 2":                lambda q, h: wmean(h, range(q-2, q), lambda t: 1),
 "G prior 1":                lambda q, h: wmean(h, range(q-1, q), lambda t: 1),
 "H decay hl 2":             lambda q, h: wmean(h, range(2004, q), lambda t: 0.5 ** ((q-1-t)/2)),
 "I decay hl 3":             lambda q, h: wmean(h, range(2004, q), lambda t: 0.5 ** ((q-1-t)/3)),
 "J decay hl 5":             lambda q, h: wmean(h, range(2004, q), lambda t: 0.5 ** ((q-1-t)/5)),
 "K hl 3, bubble 2020 out":  lambda q, h: wmean(h, [t for t in range(2004, q) if t != 2020], lambda t: 0.5 ** ((q-1-t)/3)),
 "L prior 5, bubble out":    lambda q, h: wmean(h, [t for t in range(q-6, q) if t != 2020][-5:], lambda t: 1),
}
E = {}; HC = {}
for k, f in V.items():
    e = []; hc = []
    for q in seasons:
        a = f(q, H[q][1]); hc.append(a); e.append((y[s == q] - pred(q, a)) ** 2)
    E[k] = np.concatenate(e); HC[k] = hc
e0 = E["A LOSO const (all other)"]
print("\nhome court used per test season (" + " ".join(str(q)[2:] for q in seasons) + ")")
for k in V: print(f"{k:26s}", " ".join(f"{x:4.2f}" for x in HC[k]))
print("\nh_s by season (LOSO slopes of the last test season):", {t: round(v, 2) for t, v in sorted(H[seasons[-1]][1].items()) if t >= 2008})
print()
for k in V:
    d = pd.Series(E[k] - e0).groupby(s).mean(); r = np.sqrt(E[k].mean())
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if k[0] != "A" else 0
    h1, h2 = d[d.index <= 2019], d[d.index >= 2020]
    print(f"{k:26s} RMSE {r:.4f} ({r-np.sqrt(e0.mean()):+.4f}) better {(d<0).sum():2d}/{len(d)} t {t:5.2f} | 15-19 {h1.mean():+.3f} 20-25 {h2.mean():+.3f}")
# what each prior-only variant gives for 2026-27 (q=2026: all backtest seasons usable; slopes from full fit)
b = np.linalg.lstsq(X, y, rcond=None)[0]
h = {int(t): float(np.mean(y[s == t] - b[1]*base[s == t] - b[2]*bd[s == t])) for t in seasons}
for t in raw.index:
    if t not in h and t >= 1997: h[int(t)] = float(raw[t] - b[2] * bdraw.get(t, 0.0))
print("\nfull fit slopes", b.round(3), "\n2026-27 home court by variant:")
for k, f in V.items():
    if k[0] != "A": print(f"  {k:26s} {f(2026, h):.2f}")
print("  all 2015-25 equal (production-like)", round(np.mean([h[t] for t in seasons]), 2))
