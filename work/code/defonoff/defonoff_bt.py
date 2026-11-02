"""Do ratings that lean on the DEFENSIVE lineup residual (on/off) overstate players? (Clingan question, 2026-11-02)
Game backtest (research bar): base = box + .6 res_o + .8 res_d (lineup_port_bt harness, actual minutes, LOSO by season).
Extra regressors (game diff, minutes-weighted, pace-scaled like res_d):
  tail{k}:  max(res_d - k, 0)                -> negative coef = big defensive on/off is overstated
  lean:     res_d x [res_d >= 1.25 and box_est <= 3]  (Clingan-like: modest box, big def on/off)
  otail:    max(res_o - 1, 0)   (offense comparison)
box_est = pre_P rating/1.0501 - .6 res_o - .8 res_d (age/hustle ignored)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/code"); import lineup_port_bt as L
D = "/home/claude/data"; G, B, y, seas = L.G, L.B, L.y, L.seas
pre = pd.concat([pd.read_parquet(f"{D}/pre_P_{s}.parquet") for s in range(2014, 2026)])
pre = pre.groupby(["season", "player"]).rating.mean()
G["box_est"] = [pre.get((s, p), np.nan) for s, p in zip(G.season, G.player)]
G["box_est"] = G.box_est / 1.0501 - .6 * G.res_o - .8 * G.res_d
print("players with pre_P match:", round(G.box_est.notna().mean(), 3))

def gd(x):
    v = G.sgn * G.w48 * x
    return v.groupby(G.game).sum().reindex(B.index).fillna(0).values
base = L.diffs(np.ones(len(G))); P0 = B.box.values + .6 * base.o.values + .8 * base.d.values
b0, s0 = L.loso(P0[:, None])
print(f"base LOSO {b0:.4f}")
lean_mask = ((G.res_d >= 1.25) & (G.box_est <= 3)).astype(float)
X = {"tail0.5": np.maximum(G.res_d - .5, 0), "tail1": np.maximum(G.res_d - 1, 0), "tail1.5": np.maximum(G.res_d - 1.5, 0),
     "lean": G.res_d * lean_mask, "otail1": np.maximum(G.res_o - 1, 0)}
def clust(Xm, m):
    Xm = np.c_[np.ones(m.sum()), Xm[m]]; yy = y[m]; ss = seas[m]
    b = np.linalg.lstsq(Xm, yy, rcond=None)[0]; r = yy - Xm @ b; XtXi = np.linalg.inv(Xm.T @ Xm)
    meat = sum(np.outer(Xm[ss == s].T @ r[ss == s], Xm[ss == s].T @ r[ss == s]) for s in np.unique(ss))
    se = np.sqrt(np.diag(XtXi @ meat @ XtXi)); return b, se
rows = []
for k, x in X.items():
    e = gd(x); Xm = np.c_[P0, e]; l, s = L.loso(Xm)
    out = dict(term=k, loso=l, d=l - b0, better=int((s < s0).sum()))
    for nm, m in [("all", np.ones(len(y), bool)), ("15-19", seas <= 2019), ("20-25", seas >= 2020)]:
        b, se = clust(Xm, m); out[f"rel_{nm}"] = b[2] / b[1]; out[f"t_{nm}"] = b[2] / se[2]
    rows.append(out)
print(pd.DataFrame(rows).round(4).to_string())
# who is in the lean group, and how much
g = G[lean_mask > 0].groupby(["N", "player"]).agg(res_d=("res_d", "first"), box=("box_est", "first"), min=("min", "sum"))
print("lean group player-seasons:", len(g), " minutes share:", round(G["min"][lean_mask > 0].sum() / G["min"].sum(), 4))
ps = pd.read_parquet(f"{D}/player_seasons.parquet").drop_duplicates("player").set_index("player")
nm = "name" if "name" in ps else [c for c in ps.columns if "name" in c][0]
g["name"] = g.index.get_level_values(1).map(ps[nm]); print(g.sort_values("res_d", ascending=False).head(20).round(2).to_string())
