import numpy as np, pandas as pd
D = "/home/claude/data"
B = pd.read_parquet(f"{D}/backtest_components.parquet").merge(pd.read_parquet(f"{D}/bt_split.parquet"), on=["game","home_team","away_team"])
B = B.merge(pd.concat([pd.read_parquet(f"{D}/defsub_{N}.parquet") for N in range(2014, 2025)]), on=["game","home_team","away_team"])
F = [c for c in B.columns if c.startswith("f:")]
box = B[F].sum(axis=1) + B.unrated
base = (box + .6 * B.res_o + .8 * B.res_d).to_numpy()
y = B.margin.to_numpy(); seas = B.season.to_numpy()
DS = ["ds_r3", "ds_rrim", "ds_rmid", "ds_rfta", "ds_rtov", "ds_orb", "ds_p3", "ds_prim"]

def loso(X):
    X = np.c_[np.ones(len(y)), X]; e = np.empty(len(y))
    for s in np.unique(seas):
        tr, te = seas != s, seas == s
        b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; e[te] = y[te] - X[te] @ b
    return np.sqrt((e**2).mean())

def coefs(X, m):
    X = np.c_[np.ones(m.sum()), X[m]]; yy = y[m]; ss = seas[m]
    b = np.linalg.lstsq(X, yy, rcond=None)[0]; r = yy - X @ b
    Xi = np.linalg.inv(X.T @ X)
    meat = sum(np.outer(X[ss == s].T @ r[ss == s], X[ss == s].T @ r[ss == s]) for s in np.unique(ss))
    return b[1:], (b / np.sqrt(np.diag(Xi @ meat @ Xi)))[1:]

H1, H2, ALL = seas <= 2019, seas >= 2020, np.ones(len(y), bool)
print(f"base LOSO {loso(base[:, None]):.4f}")
print("\nsingle additions: LOSO | coef t (all) | coef halves")
for c in DS + ["ds_pts"]:
    X = np.c_[base, B[c]]
    b, t = coefs(X, ALL); b1, t1 = coefs(X, H1); b2, t2 = coefs(X, H2)
    print(f"{c:8s} {loso(X):.4f}  {b[1]:+.3f} (t {t[1]:+.1f})   {b1[1]:+.3f} (t {t1[1]:+.1f}) / {b2[1]:+.3f} (t {t2[1]:+.1f})")
X = np.c_[base, B[DS]]
print(f"\nall sub-outcomes: LOSO {loso(X):.4f}")
for nm, m in (("all", ALL), ("15-19", H1), ("20-25", H2)):
    b, t = coefs(X, m); print(nm, " ".join(f"{c[3:]} {bb:+.2f}({tt:+.1f})" for c, bb, tt in zip(DS, b[1:], t[1:])))
# replace raw on-court opp features
box2 = box - B["f:d_opprim"] - B["f:d_oppxq"]
base2 = (box2 + .6 * B.res_o + .8 * B.res_d).to_numpy()
print(f"\nbox without d_opprim/d_oppxq: LOSO {loso(base2[:, None]):.4f}")
print(f"  + all sub-outcomes: LOSO {loso(np.c_[base2, B[DS]]):.4f}")
print(f"  + rrim,prim,r3,rtov: LOSO {loso(np.c_[base2, B[['ds_rrim','ds_prim','ds_r3','ds_rtov']]]):.4f}")
B.to_parquet(f"{D}/defsub_games.parquet", index=False)
