# Per-input signal in the game backtest (2015-16..2025-26, LOSO), drop-one + free coefficients.
import numpy as np, pandas as pd
B = pd.read_parquet("data/backtest_components.parquet").merge(pd.read_parquet("data/bt_split.parquet"), on=["game","home_team","away_team"])
F = [c for c in B.columns if c.startswith("f:")]
B["lin_o"] = 0.6*B.res_o; B["lin_d"] = 0.8*B.res_d
INP = F + ["lin_o", "lin_d"]
y = B.margin.values; seas = B.season.values; S = np.unique(seas)
base = B[INP].sum(axis=1).values + B.unrated.values
def loso_err(pred):
    X = np.c_[np.ones(len(pred)), pred]; e = np.empty(len(y))
    for s in S:
        tr, te = seas != s, seas == s
        b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; e[te] = (y[te]-X[te]@b)**2
    return e
e0 = loso_err(base)
print("base LOSO", np.sqrt(e0.mean()).round(4))
rows = []
for c in INP:
    e1 = loso_err(base - B[c].values)
    d = pd.Series(e1 - e0).groupby(seas).mean()       # per-season MSE change from dropping
    t = d.mean()/(d.std(ddof=1)/np.sqrt(len(d)))
    rows.append(dict(input=c, drop_dRMSE=np.sqrt(e1.mean())-np.sqrt(e0.mean()), t=t, seasons_hurt=(d>0).sum(),
                     h1=d[d.index<=2019].mean(), h2=d[d.index>=2020].mean()))
# free coefficients: each input its own regressor (1.0 = weight right), season-clustered SE
X = np.c_[np.ones(len(y)), B[INP].values, B.unrated.values]
b = np.linalg.lstsq(X, y, rcond=None)[0]; r = y - X@b
XtXi = np.linalg.inv(X.T@X)
meat = sum(np.outer(X[seas==s].T@r[seas==s], X[seas==s].T@r[seas==s]) for s in S)
se = np.sqrt(np.diag(XtXi@meat@XtXi))
R = pd.DataFrame(rows); R["coef"] = b[1:1+len(INP)]; R["coef_t"] = R.coef/se[1:1+len(INP)]
R["sd_game"] = [B[c].std() for c in INP]
R.sort_values("drop_dRMSE").to_csv("data/input_signal.csv", index=False)
pd.set_option("display.width", 200); print(R.sort_values("drop_dRMSE").round(4).to_string(index=False))
