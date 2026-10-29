import numpy as np, pandas as pd
B = pd.read_parquet("data/backtest_components.parquet").merge(pd.read_parquet("data/bt_split.parquet"), on=["game", "home_team", "away_team"])
E = pd.read_parquet("data/defpos_bt.parquet"); B = B.merge(E, on="game", how="left")
print("rows", len(B), "missing ev", B.filter(like="dp_").isna().any(axis=1).sum())
F = [c for c in B.columns if c.startswith("f:")]
box = B[F].sum(axis=1) + B.unrated
base = (box + .6 * B.res_o + .8 * B.res_d).to_numpy()
y = B.margin.values; seas = B.season.values

def design(X): return np.c_[np.ones(len(y)), X]
def loso(X):
    X = design(X); e = []
    for s in np.unique(seas):
        tr, te = seas != s, seas == s
        b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; e.append((y[te] - X[te] @ b) ** 2)
    return np.sqrt(np.concatenate(e).mean())
def coefs(X, m):
    X = design(X)[m]; yy = y[m]; ss = seas[m]
    b = np.linalg.lstsq(X, yy, rcond=None)[0]; r = yy - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    meat = sum(np.outer(X[ss == s].T @ r[ss == s], X[ss == s].T @ r[ss == s]) for s in np.unique(ss))
    return b, b / np.sqrt(np.diag(XtXi @ meat @ XtXi))
H1, H2, ALL = seas <= 2019, seas >= 2020, np.ones(len(y), bool)
b0 = loso(base[:, None]); print(f"base LOSO {b0:.4f}")
res = []
for c in [c for c in B.columns if c.startswith("dp_")]:
    X = np.c_[base, B[c]]
    b, t = coefs(X, ALL); b1, t1 = coefs(X, H1); b2, t2 = coefs(X, H2)
    res.append((c, loso(X) - b0, b[2], t[2], t1[2], t2[2]))
R = pd.DataFrame(res, columns=["feat", "dLOSO", "coef", "t", "t_15_19", "t_20_25"])
print(R.round(4).to_string(index=False))
