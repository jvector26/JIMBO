"""Component weight calibration check: free multipliers per component on the game backtest (actual minutes, 2015-25)."""
import numpy as np, pandas as pd
from features import COMPONENTS
B = pd.read_parquet("data/backtest_components.parquet").merge(pd.read_parquet("data/bt_split.parquet"), on=["game","home_team","away_team"])
y = B.margin.values; seas = B.season.values
H1, H2 = seas <= 2019, seas >= 2020
comp = {c: B[[f"f:{f}" for f in fs]].sum(axis=1).values for c, fs in COMPONENTS.items()}
res = .6*B.res_o.values + .8*B.res_d.values
base = sum(comp.values()) + B.unrated.values + res
def D(X): return np.c_[np.ones(len(y)), X]
def fit(X, m, ridge=0, prior=None):
    X = D(X)[m]; yy = y[m]
    if ridge:
        P = np.eye(X.shape[1])*ridge; P[0,0]=0
        pr = np.r_[0, prior]
        return np.linalg.solve(X.T@X + P, X.T@yy + P@pr)
    return np.linalg.lstsq(X, yy, rcond=None)[0]
def loso(X, ridge=0, prior=None):
    e=[]
    for s in np.unique(seas):
        tr, te = seas!=s, seas==s
        b = fit(X, tr, ridge, prior); e.append((y[te]-D(X)[te]@b)**2)
    return np.sqrt(np.concatenate(e).mean())
def cl_t(X, m):
    Xd = D(X)[m]; yy=y[m]; ss=seas[m]
    b = np.linalg.lstsq(Xd, yy, rcond=None)[0]; r = yy-Xd@b
    XtXi = np.linalg.inv(Xd.T@Xd)
    meat = sum(np.outer(Xd[ss==s].T@r[ss==s], Xd[ss==s].T@r[ss==s]) for s in np.unique(ss))
    se = np.sqrt(np.diag(XtXi@meat@XtXi)); return b, se
