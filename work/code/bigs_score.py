"""Score bigs tests: base = box + 0.6 res_o + 0.8 res_d (production). Add term(s); LOSO RMSE change, coef, season-clustered t,
t by half (2015-19 / 2020-25). Coef scale: same units as base (1.0 = the term is worth as much as the base per unit)."""
import sys, numpy as np, pandas as pd
B = pd.read_parquet("data/backtest_components.parquet").merge(pd.read_parquet("data/bt_split.parquet"), on=["game", "home_team", "away_team"])
B = B.merge(pd.read_parquet("data/bigs_bt.parquet"), on="game")
F = [c for c in B.columns if c.startswith("f:")]
box = B[F].sum(axis=1) + B.unrated
base = (box + .6 * B.res_o + .8 * B.res_d).to_numpy()
y = B.margin.values; seas = B.season.values
H1, H2, ALL = seas <= 2019, seas >= 2020, np.ones(len(y), bool)
design = lambda X: np.c_[np.ones(len(y)), X]
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
b0 = loso(base[:, None])
def test(cols, label=None):
    X = np.c_[base, B[cols].to_numpy()]
    (b, t), (b1, t1), (b2, t2) = coefs(X, ALL), coefs(X, H1), coefs(X, H2)
    rel = b[2:] / b[1]
    s = " | ".join(f"{c[2:]} rel {r:+.2f} t {tt:+.1f} ({a:+.1f}/{c2:+.1f})" for c, r, tt, a, c2 in zip(cols, rel, t[2:], t1[2:], t2[2:]))
    print(f"{(label or '+'.join(cols)):28s} dLOSO {loso(X) - b0:+.4f} | {s}")
if __name__ == "__main__":
    print(f"base LOSO {b0:.4f}  (n={len(y)})")
    groups = [a.split(",") for a in sys.argv[1:]]
    for g in groups: test(["g_" + c for c in g])
