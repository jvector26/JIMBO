import numpy as np, pandas as pd
D = "/home/claude/data"
B = pd.read_parquet(f"{D}/backtest_components.parquet")
R = pd.concat([pd.read_parquet(f"{D}/ft_refit_{N}.parquet") for N in range(2014, 2025)])
B = B.merge(R, on=["game", "home_team", "away_team"])
F = [c for c in B.columns if c.startswith("f:")]
y = B.margin.values; seas = B.season.values; S = np.unique(seas)
rest = (B[F].sum(axis=1) - B["f:o_ftvol"] - B["f:o_fteff"] + B.unrated).to_numpy()
def pred(v, e, wo=.6, wd=.8):
    t = f"{v}_{e}"; return rest + v*B["f:o_ftvol"].to_numpy() + e*B["f:o_fteff"].to_numpy() + wo*B[f"res_o_{t}"].to_numpy() + wd*B[f"res_d_{t}"].to_numpy()
def err(p, msk=None):
    msk = np.ones(len(y), bool) if msk is None else msk
    X = np.c_[np.ones(len(y)), p]; e = np.full(len(y), np.nan)
    for s in np.unique(seas[msk]):
        tr = msk & (seas != s); te = seas == s
        b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; e[te] = (y[te] - X[te] @ b)**2
    return e
e0 = err(pred(1, 1)); print("base (1,1) LOSO", np.sqrt(e0.mean()).round(4))
rows = []
for v, e in [(1,1), (.5,1), (.5,2), (.5,2.5), (.75,1.75), (1,2.5)]:
    e1 = err(pred(v, e)); d = pd.Series(e1 - e0).groupby(seas).mean()
    t = d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if v != 1 or e != 1 else 0
    h = [d[d.index <= 2019].mean(), d[d.index >= 2020].mean()]
    th = [dd.mean()/(dd.std(ddof=1)/np.sqrt(len(dd))) if dd.std() > 0 else 0 for dd in (d[d.index <= 2019], d[d.index >= 2020])]
    rows.append(dict(vol=v, eff=e, LOSO=np.sqrt(e1.mean()), dLOSO=np.sqrt(e1.mean())-np.sqrt(e0.mean()), t=t, better=(d < 0).sum(),
                     dMSE_15_19=h[0], t_15_19=th[0], dMSE_20_25=h[1], t_20_25=th[1]))
out = pd.DataFrame(rows); print(out.round(4).to_string(index=False)); out.to_csv(f"{D}/ft_refit_score.csv", index=False)
best = out.iloc[out.LOSO.idxmin()]; v, e = best.vol, best.eff
print(f"\nlineup weights under best ({v},{e}):")
for wo, wd in [(.6,.8), (.4,.8), (.8,.8), (.6,1.0), (.6,.6)]:
    print(f"  wo {wo} wd {wd}: {np.sqrt(err(pred(v, e, wo, wd)).mean()):.4f}")
# how much of the dropped volume credit does res_o take back? regression of res change on ftvol change
dv = B["f:o_ftvol"]*(v-1); dr = .6*(B[f"res_o_{v}_{e}"]-B["res_o_1_1"])
print("res_o re-absorbs share of volume cut:", round(-np.polyfit(dv, dr, 1)[0], 2))
