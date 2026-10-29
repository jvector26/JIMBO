import numpy as np, pandas as pd
A = pd.read_parquet("/home/claude/data/young_prior_design.parquet")
AGE = ["<=21", "22-23", "24-25", "28-29", "30-31", "32-33", "34-35", "36+"]
BASE = ["base"] + AGE + ["other_unrated"]
def fit(d, cols):
    X = np.c_[np.ones(len(d)), d[cols].to_numpy()]; y = d.margin.to_numpy()
    b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X@b; XtXi = np.linalg.inv(X.T@X)
    M = sum(np.outer(X[i].T@e[i], X[i].T@e[i]) for i in d.groupby("season").indices.values())  # season-clustered
    return b, b/np.sqrt(np.diag(XtXi@M@XtXi))
def loso(cols):
    X = np.c_[np.ones(len(A)), A[cols].to_numpy()]; y = A.margin.to_numpy(); p = np.empty(len(A))
    for s in A.season.unique():
        tr = (A.season != s).to_numpy(); b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; p[~tr] = X[~tr]@b
    return np.sqrt(np.mean((y-p)**2))
def split(cols):
    tr, te = A[A.season <= 2020], A[A.season >= 2021]
    b, _ = fit(tr, cols); X = np.c_[np.ones(len(te)), te[cols].to_numpy()]
    return np.sqrt(np.mean((te.margin - X@b)**2))
MODELS = {
 "current (3 pick bins)": BASE + ["rk_1_3", "rk_4_10", "rk_11_30"],
 "rookie: log pick + draft age": BASE + ["rk", "rk_lp", "rk_age"],
 "  + height, intl": BASE + ["rk", "rk_lp", "rk_age", "rk_h", "rk_intl"],
 "young: 2nd/3rd-yr bumps": BASE + ["rk", "rk_lp", "rk_age", "y_yr2", "y_yr3"],
 "young: + pick, draft age": BASE + ["rk", "rk_lp", "rk_age", "y_yr2", "y_yr3", "y_lp", "y_dage"],
 "young: + box trend, late mpg": BASE + ["rk", "rk_lp", "rk_age", "y_yr2", "y_yr3", "y_lp", "y_dage", "y_trend", "y_late"],
 "  + trend for older too": BASE + ["rk", "rk_lp", "rk_age", "y_yr2", "y_yr3", "y_lp", "y_dage", "y_trend", "y_late", "all_trend"],
}
r0 = loso(MODELS["current (3 pick bins)"])
print(f"{'model':32s} {'LOSO':>7s} {'gain':>7s} {'test21-25':>9s}")
for k, c in MODELS.items(): r = loso(c); print(f"{k:32s} {r:7.4f} {r0-r:+7.4f} {split(c):9.4f}")
for k in ["current (3 pick bins)", "young: + box trend, late mpg", "  + trend for older too"]:
    c = MODELS[k]; b, t = fit(A, c); sl = b[1]
    print(f"\n{k}: slope {sl:.3f}; values in rating pts/100 (t):")
    print("  " + ", ".join(f"{n} {v/sl:+.2f} ({tt:.1f})" for n, v, tt in zip(c, b[1:], t[1:]) if n not in AGE and n != "base"))
    for nm, d in [("15-19", A[A.season <= 2019]), ("20-25", A[A.season >= 2020])]:
        b2, t2 = fit(d, c); print(f"  {nm}: " + ", ".join(f"{n} {v/b2[1]:+.2f}" for n, v in zip(c, b2[1:]) if n.startswith(("rk", "y_", "all"))))
