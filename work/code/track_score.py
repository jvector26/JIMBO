import numpy as np, pandas as pd
B = pd.read_parquet("data/backtest_components.parquet").merge(pd.read_parquet("data/bt_split.parquet"), on=["game", "home_team", "away_team"])
B = B.merge(pd.read_parquet("data/track_bt.parquet"), on="game", how="left")
F = [c for c in B.columns if c.startswith("f:")]
base_all = (B[F].sum(axis=1) + B.unrated + .6 * B.res_o + .8 * B.res_d).to_numpy()
def run(mask, cols, h1, label):
    y = B.margin.values[mask]; seas = B.season.values[mask]; base = base_all[mask]
    D = lambda X: np.c_[np.ones(len(y)), X]
    def loso(X):
        X = D(X); e = []
        for s in np.unique(seas):
            tr, te = seas != s, seas == s; b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; e.append((y[te] - X[te] @ b) ** 2)
        return np.sqrt(np.concatenate(e).mean())
    def co(X, m):
        X = D(X)[m]; yy = y[m]; ss = seas[m]; b = np.linalg.lstsq(X, yy, rcond=None)[0]; r = yy - X @ b
        Xi = np.linalg.inv(X.T @ X); meat = sum(np.outer(X[ss == s].T @ r[ss == s], X[ss == s].T @ r[ss == s]) for s in np.unique(ss))
        return b, b / np.sqrt(np.diag(Xi @ meat @ Xi))
    b0 = loso(base[:, None]); print(f"\n{label}: n={len(y)} seasons {seas.min()}-{seas.max()} base LOSO {b0:.4f}")
    A1 = seas <= h1; A2 = ~A1; ALL = np.ones(len(y), bool); res = []
    for c in cols:
        X = np.c_[base, B[c].values[mask]]; b, t = co(X, ALL); _, t1 = co(X, A1); _, t2 = co(X, A2)
        res.append((c[3:], loso(X) - b0, b[2], t[2], t1[2], t2[2]))
    print(pd.DataFrame(res, columns=["feat", "dLOSO", "coef", "t", f"t_<= {h1}", "t_later"]).round(4).to_string(index=False))
hcols = [c for c in B.columns if c.startswith("tr_") and c.split("_")[1] in ("defl", "cont", "cont2", "cont3", "lbr", "dbox")]
rcols = [c for c in B.columns if c.startswith("tr_") and c.split("_")[1] in ("rimsave", "allsave", "rimdfga")]
run(B.N.values >= 2017, hcols, 2021, "HUSTLE (2017-18+ data)")
run(np.ones(len(B), bool), rcols, 2019, "DEFENDED FG (2013-14+ data)")

# joint models (hustle era)
m = B.N.values >= 2017; y = B.margin.values[m]; seas = B.season.values[m]; base = base_all[m]
def loso(X):
    X = np.c_[np.ones(len(y)), X]; e = []
    for s in np.unique(seas):
        tr, te = seas != s, seas == s; b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; e.append((y[te] - X[te] @ b) ** 2)
    return np.sqrt(np.concatenate(e).mean())
b0 = loso(base[:, None])
for combo in [["tr_lbr_1000"], ["tr_lbr_1000", "tr_cont3_3000"], ["tr_lbr_1000", "tr_cont3_3000", "tr_rimdfga_1000"],
              ["tr_lbr_1000", "tr_cont3_3000", "tr_defl_1000"], ["tr_lbr_1000", "tr_rimdfga_1000"]]:
    X = np.c_[base, B[combo].values[m]]; Xc = np.c_[np.ones(len(y)), X]; b = np.linalg.lstsq(Xc, y, rcond=None)[0]
    print(f"{'+'.join(c[3:] for c in combo):45s} dLOSO {loso(X) - b0:+.4f}  coefs {np.round(b[2:], 3)}")
x = pd.read_parquet("data/track_window_2025.parquet")
print("corr of window values (2023-25):"); print(x[["lbr_1000", "cont3_3000", "rimdfga_1000", "defl_1000"]].corr().round(2))
