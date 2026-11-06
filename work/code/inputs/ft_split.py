import numpy as np, pandas as pd
B = pd.read_parquet("data/backtest_components.parquet").merge(pd.read_parquet("data/bt_split.parquet"), on=["game","home_team","away_team"])
F = [c for c in B.columns if c.startswith("f:")]
y = B.margin.values; seas = B.season.values; S = np.unique(seas)
base = B[F].sum(axis=1).values + B.unrated.values + .6*B.res_o.values + .8*B.res_d.values
rest = base - B["f:o_fteff"].values - B["f:o_ftvol"].values
def fit(X, m): return np.linalg.lstsq(X[m], y[m], rcond=None)[0]
def run(X, label):
    e = np.empty(len(y))
    for s in S:
        tr, te = seas != s, seas == s; e[te] = (y[te] - X[te] @ fit(X, tr))**2
    out = {"label": label, "loso": np.sqrt(e.mean())}
    for h, m in [("all", np.ones(len(y), bool)), ("15-19", seas <= 2019), ("20-25", seas >= 2020)]:
        b = fit(X, m); r = y[m] - X[m] @ b; XtXi = np.linalg.inv(X[m].T @ X[m])
        meat = sum(np.outer(X[m][seas[m]==s].T @ r[seas[m]==s], X[m][seas[m]==s].T @ r[seas[m]==s]) for s in np.unique(seas[m]))
        se = np.sqrt(np.diag(XtXi @ meat @ XtXi)); out[h] = (b, se)
    return e, out
one = np.ones(len(y))
e0, o0 = run(np.c_[one, base], "current")
e1, o1 = run(np.c_[one, rest, B["f:o_fteff"], B["f:o_ftvol"]], "split free")
print("current LOSO", round(o0["loso"],4), " split free", round(o1["loso"],4))
d = pd.Series(e1-e0).groupby(seas).mean(); print("paired t", round(d.mean()/(d.std(ddof=1)/np.sqrt(len(d))),2), "better seasons", (d<0).sum(), "of", len(d))
for h in ["all","15-19","20-25"]:
    b, se = o1[h]; rel = b[2:]/b[1]
    print(f"{h}: rest {b[1]:.3f} | fteff x{rel[0]:.2f} (t vs current {(b[2]-b[1])/se[2]:.1f}) | ftvol x{rel[1]:.2f} (t vs current {(b[3]-b[1])/se[3]:.1f})")
