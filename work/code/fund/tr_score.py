import numpy as np, pandas as pd
D = "/home/claude/data"; FD = "/home/claude/fund"
B = pd.read_parquet(f"{D}/backtest_components.parquet").merge(pd.read_parquet(f"{D}/bt_split.parquet"), on=["game", "home_team", "away_team"])
B = B.merge(pd.read_parquet(f"{FD}/tr_bt.parquet"), on="game", how="left")
print("rows", len(B), "missing", B.filter(like="ev_").isna().any(axis=1).sum())
F = [c for c in B.columns if c.startswith("f:")]
base = (B[F].sum(axis=1) + B.unrated + .6 * B.res_o + .8 * B.res_d).to_numpy(); y = B.margin.values; seas = B.season.values
def err(X):
    X = np.c_[np.ones(len(y)), X]; e = np.full(len(y), np.nan)
    for s in np.unique(seas):
        tr, te = seas != s, seas == s; e[te] = (y[te] - X[te] @ np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]) ** 2
    return e
def coefs(X, m):
    X = np.c_[np.ones(len(y)), X][m]; yy = y[m]; ss = seas[m]; b = np.linalg.lstsq(X, yy, rcond=None)[0]; r = yy - X @ b; XtXi = np.linalg.inv(X.T @ X)
    meat = sum(np.outer(X[ss == s].T @ r[ss == s], X[ss == s].T @ r[ss == s]) for s in np.unique(ss)); return b, b / np.sqrt(np.diag(XtXi @ meat @ XtXi))
H1, H2, ALL = seas <= 2019, seas >= 2020, np.ones(len(y), bool)
e0 = err(base[:, None]); print(f"base LOSO {np.sqrt(e0.mean()):.4f}")
def line(name, cols):
    X = np.c_[base, B[cols].to_numpy()]; e1 = err(X); d = pd.Series(e1 - e0).groupby(seas).mean(); tp = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    b, t = coefs(X, ALL); _, t1 = coefs(X, H1); _, t2 = coefs(X, H2)
    print(f"{name:30s} dLOSO {np.sqrt(e1.mean())-np.sqrt(e0.mean()):+.4f} paired t {tp:+.2f} better {(d<0).sum()}/11 | " +
          "  ".join(f"{c[3:]}: coef {b[2+i]:+.2f} t {t[2+i]:+.1f} (halves {t1[2+i]:+.1f}/{t2[2+i]:+.1f})" for i, c in enumerate(cols)))
print("PRIMARY (k 1000), one at a time; expected signs o_trvol +, o_tron +, d_tron -")
for c in ["o_trvol", "o_tron", "d_tron"]: line(c, [f"ev_{c}_1000"])
line("all three jointly", [f"ev_{c}_1000" for c in ["o_trvol", "o_tron", "d_tron"]])
print("ROBUSTNESS")
for k in [300, 3000]:
    for c in ["o_trvol", "o_tron", "d_tron"]: line(f"{c} k{k}", [f"ev_{c}_{k}"])
for c in ["o_tronp", "d_tronp"]: line(c + " (frequency) k1000", [f"ev_{c}_1000"])
