import numpy as np, pandas as pd, sys
sys.path.insert(0, "code"); from features import COMPONENTS
B = pd.read_parquet("data/backtest_components.parquet").merge(pd.read_parquet("data/bt_split.parquet"), on=["game","home_team","away_team"])
B["lin_o"] = .6*B.res_o; B["lin_d"] = .8*B.res_d
G = {k: ["f:"+f for f in v] for k, v in COMPONENTS.items()}; G["lineup_off"] = ["lin_o"]; G["lineup_def"] = ["lin_d"]
y = B.margin.values; seas = B.season.values; S = np.unique(seas)
base = B[[c for g in G.values() for c in g]].sum(axis=1).values + B.unrated.values
def err(p):
    X = np.c_[np.ones(len(p)), p]; e = np.empty(len(y))
    for s in S:
        tr, te = seas != s, seas == s; b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; e[te] = (y[te]-X[te]@b)**2
    return e
e0 = err(base)
R = pd.read_parquet("data/ratings_2025.parquet"); m = 1-(1/3)*np.clip(1-R.poss_window/10000,0,1)
R["lin_o"] = .6*R.res_o*m; R["lin_d"] = .8*R.res_d*m; R = R[R.min_2025 >= 500]; w = R.min_2025
tot = R[[c for g in G.values() for c in g]].sum(axis=1)
def wsd(x): mu = np.average(x, weights=w); return np.sqrt(np.average((x-mu)**2, weights=w))
rows = []
for k, cols in G.items():
    gsum = B[cols].sum(axis=1).values
    e1 = err(base - gsum); d = pd.Series(e1-e0).groupby(seas).mean()
    p = R[cols].sum(axis=1)
    rows.append(dict(input=k, player_sd=wsd(p), var_share=np.cov(p, tot, aweights=w)[0,1]/np.cov(tot, tot, aweights=w)[0,1],
        drop_dRMSE=np.sqrt(e1.mean())-np.sqrt(e0.mean()), t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d))), hurt=(d>0).sum(),
        h1=np.sign(d[d.index<=2019].mean()), h2=np.sign(d[d.index>=2020].mean())))
out = pd.DataFrame(rows).sort_values("drop_dRMSE"); out.to_csv("data/component_signal.csv", index=False)
print(out.round(4).to_string(index=False))
