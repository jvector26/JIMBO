"""B2B in the research-bar game backtest (ACTUAL minutes): fatigue part beyond who played. LOSO RMSE + season-clustered t."""
import numpy as np, pandas as pd
D = "/home/claude/data"
B = pd.read_parquet(f"{D}/backtest_components.parquet").merge(pd.read_parquet(f"{D}/bt_split.parquet"), on=["game", "home_team", "away_team"])
G = pd.read_parquet(f"{D}/b2b_games.parquet")[["game", "b2b_h", "b2b_a", "b2b_a_trav", "g34_h", "g34_a", "rest_h", "rest_a"]]
B = B.merge(G, on="game", how="left")
print("rows", len(B), "with rest info", B.b2b_h.notna().sum())
for c in ["b2b_h", "b2b_a", "b2b_a_trav", "g34_h", "g34_a"]: B[c] = B[c].fillna(0)
F = [c for c in B.columns if c.startswith("f:")]
base = B[F].sum(axis=1) + B.unrated + 0.6 * B.res_o + 0.8 * B.res_d
y = B.margin.to_numpy(); s = B.season.to_numpy()
def loso(X):
    X = np.c_[np.ones(len(y)), X]; e = []
    for q in np.unique(s):
        tr, te = s != q, s == q; b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; e.append((y[te] - X[te] @ b) ** 2)
    e = np.concatenate(e); return np.sqrt(e.mean()), e
def coefs(X, m):
    X = np.c_[np.ones(m.sum()), X[m]]; yy = y[m]; ss = s[m]
    b = np.linalg.lstsq(X, yy, rcond=None)[0]; r = yy - X @ b; XtXi = np.linalg.inv(X.T @ X)
    meat = sum(np.outer(X[ss == q].T @ r[ss == q], X[ss == q].T @ r[ss == q]) for q in np.unique(ss))
    return b, np.sqrt(np.diag(XtXi @ meat @ XtXi))
r0, e0 = loso(base.to_numpy()[:, None])
print(f"base LOSO {r0:.4f}")
B["b2b_diff"] = B.b2b_h - B.b2b_a
for lab, cols in [("b2b home/away", ["b2b_h", "b2b_a"]), ("b2b diff (one coef)", ["b2b_diff"]),
                  ("+ road travel", ["b2b_h", "b2b_a", "b2b_a_trav"]), ("+ 3in4", ["b2b_h", "b2b_a", "g34_h", "g34_a"])]:
    X = np.c_[base, B[cols]]
    r, e = loso(X)
    # season-clustered t of the per-season dMSE
    d = pd.Series(e - e0).groupby(s).mean(); t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    line = f"{lab:22s} LOSO {r:.4f} ({r-r0:+.4f}; better in {(d<0).sum()}/{len(d)} seasons, t {t:.1f})"
    for hl, m in [("all", np.ones(len(y), bool)), ("15-19", s <= 2019), ("20-25", s >= 2020)]:
        b, se = coefs(X, m)
        line += f" | {hl}: " + " ".join(f"{c} {x:+.2f}(t {x/q:.1f})" for c, x, q in zip(cols, b[2:], se[2:]))
    print(line)
