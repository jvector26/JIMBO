"""Back-to-back effect on game margin (session 37, user question 2026-10-29).
(1) TOTAL effect vs a team's own season level: margin ~ team-season FE (home +1 / away -1) + HCA + rest dummies.
    This is what the season sim / nightly (normal expected minutes) would need.
(2) FATIGUE-only: added to the game backtest that uses ACTUAL minutes (box + .6 res_o + .8 res_d, judgment-shrink base
    lives in bt files; here the bt_split base). Difference (1)-(2) ~ who-sits part.
"""
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.linalg import lsqr
D = "/home/claude/data"
G = pd.read_parquet(f"{D}/games_all.parquet")
G = G[G.season >= 2004].copy()
G["d"] = pd.to_datetime(G.date.astype(str))
G["margin"] = G.home_pts - G.away_pts
G = G[G.score_ok != False]
# team-game rest: days since previous game, and previous game location
L = pd.concat([G[["season", "game", "d", "home_team"]].rename(columns={"home_team": "t"}).assign(home=1),
               G[["season", "game", "d", "away_team"]].rename(columns={"away_team": "t"}).assign(home=0)]).sort_values(["t", "d", "game"])
L["prev"] = L.groupby(["season", "t"]).d.shift(1)
L["prev2"] = L.groupby(["season", "t"]).d.shift(2)
L["rest"] = (L.d - L.prev).dt.days - 1          # 0 = back-to-back
L["g34"] = ((L.d - L.prev2).dt.days <= 3).astype(int)   # 3rd game in 4 nights
L["prevhome"] = L.groupby(["season", "t"]).home.shift(1)
L["n"] = L.groupby(["season", "t"]).cumcount()
k = L.set_index(["game", "t"])
for side, col in [("h", "home_team"), ("a", "away_team")]:
    idx = pd.MultiIndex.from_arrays([G.game, G[col]])
    for c in ["rest", "g34", "prevhome", "n"]:
        G[f"{c}_{side}"] = k[c].reindex(idx).to_numpy()
G = G[(G.n_h >= 1) & (G.n_a >= 1)].copy()      # skip season openers (no rest defined)
G["b2b_h"] = (G.rest_h == 0).astype(int); G["b2b_a"] = (G.rest_a == 0).astype(int)
G["r1_h"] = (G.rest_h == 1).astype(int); G["r1_a"] = (G.rest_a == 1).astype(int)
G["r3_h"] = (G.rest_h >= 3).astype(int); G["r3_a"] = (G.rest_a >= 3).astype(int)
# road b2b where the previous game was ALSO away (travel); home b2b after away game
G["b2b_a_trav"] = ((G.b2b_a == 1) & (G.prevhome_a == 0)).astype(int)
print("games", len(G), "B2B share home %.3f away %.3f" % (G.b2b_h.mean(), G.b2b_a.mean()))
print(G.groupby("season")[["b2b_h", "b2b_a"]].sum().sum(axis=1).rename("team B2B games").to_string())

def fe_fit(df, extra):
    """margin ~ team-season FE + HCA + extra. Returns coef on extras, season-clustered SE."""
    ts = pd.Categorical(np.r_[df.season.astype(str) + df.htm, df.season.astype(str) + df.vtm])
    n = len(df); codes = ts.codes; nt = len(ts.categories)
    rows = np.r_[np.arange(n), np.arange(n)]; vals = np.r_[np.ones(n), -np.ones(n)]
    A = sp.csr_matrix((vals, (rows, codes)), shape=(n, nt))
    X = df[extra].to_numpy(float)
    # partial out FE: residualize y and X on A (+ constant via HCA column)
    Z = sp.hstack([A, sp.csr_matrix(np.ones((n, 1)))]).tocsr()
    def res(v): return v - Z @ lsqr(Z, v, atol=1e-10, btol=1e-10, iter_lim=5000)[0]
    y = res(df.margin.to_numpy(float)); Xr = np.column_stack([res(X[:, j]) for j in range(X.shape[1])])
    b = np.linalg.lstsq(Xr, y, rcond=None)[0]; e = y - Xr @ b
    XtXi = np.linalg.inv(Xr.T @ Xr); s = df.season.to_numpy()
    meat = sum(np.outer(Xr[s == q].T @ e[s == q], Xr[s == q].T @ e[s == q]) for q in np.unique(s))
    se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
    return b, se

spec = ["b2b_h", "b2b_a"]
spec2 = ["b2b_h", "b2b_a", "r3_h", "r3_a", "g34_h", "g34_a"]
for lab, m in [("2004-25", G.season >= 2004), ("2015-25", G.season >= 2015), ("2015-19", (G.season >= 2015) & (G.season <= 2019)),
               ("2020-25", G.season >= 2020), ("2004-14", G.season <= 2014)]:
    b, se = fe_fit(G[m], spec)
    print(f"TOTAL {lab}: home B2B {b[0]:+.2f} (t {b[0]/se[0]:.1f})  away B2B {b[1]:+.2f} (t {b[1]/se[1]:.1f})  [margin = home pts - away pts]")
b, se = fe_fit(G[G.season >= 2015], spec2)
print("TOTAL 2015-25 detailed:", " ".join(f"{c} {x:+.2f}(t {x/s:.1f})" for c, x, s in zip(spec2, b, se)))
b, se = fe_fit(G[G.season >= 2015], ["b2b_h", "b2b_a", "b2b_a_trav"])
print("TOTAL 2015-25 road B2B split by travel: home b2b %+.2f, away b2b (prev home) %+.2f, extra if prev also away %+.2f (t %.1f)" % (b[0], b[1], b[2], b[2]/se[2]))
G.to_parquet(f"{D}/b2b_games.parquet", index=False)
