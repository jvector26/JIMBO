import numpy as np, pandas as pd
B = pd.read_parquet("data/backtest_components.parquet")
D = pd.concat([pd.read_parquet(f"data/decay_bt_{N}.parquet") for N in range(2014, 2025)]).drop(columns=["home_team", "away_team"])
B = B.merge(D, on="game"); y = B.margin.values; seas = B.season.values
def loso(p):
    X = np.c_[np.ones(len(p)), p]; e = np.zeros(len(y))
    for s in np.unique(seas):
        tr, te = seas != s, seas == s; b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; e[te] = (y[te] - X[te] @ b) ** 2
    return e
def pred(tag, wo=.6, wd=.8, split=None):
    if not split:
        p = B[f"{tag}:box_o"] + B[f"{tag}:box_d"] + B["f:o_3eff"] + B.unrated + wo * B[f"{tag}:res_o"] + wd * B[f"{tag}:res_d"]
    else:  # box from one decay, residual from another
        bt, rt = split
        p = B[f"{bt}:box_o"] + B[f"{bt}:box_d"] + B["f:o_3eff"] + B.unrated + wo * B[f"{rt}:res_o"] + wd * B[f"{rt}:res_d"]
    return p.to_numpy()
e0 = loso(pred("d3")); r = lambda e, m=slice(None): np.sqrt(e[m].mean())
H1, H2 = seas <= 2019, seas >= 2020
print(f"d3 (.3,.6,1) production: {r(e0):.4f}  halves {r(e0,H1):.4f} / {r(e0,H2):.4f}")
def show(lbl, e):
    d = pd.Series(e - e0).groupby(seas).mean(); t = d.mean() / d.std() * np.sqrt(len(d))
    print(f"{lbl:34s} {r(e):.4f} (d {r(e)-r(e0):+.4f})  halves {r(e,H1)-r(e0,H1):+.4f} / {r(e,H2)-r(e0,H2):+.4f}  better {int((d<0).sum())}/11  t {t:+.1f}")
for tag, lab in [("d1", ".1,.4,1"), ("d4", ".4,.7,1"), ("d5", ".5,.8,1"), ("d7", ".7,.9,1")]:
    show(f"{tag} ({lab})", loso(pred(tag)))
print("-- box and residual decays separately (box, res):")
for bt, rt in [("d3", "d1"), ("d1", "d3"), ("d3", "d5"), ("d5", "d3"), ("d1", "d5"), ("d5", "d1")]:
    show(f"box {bt} / res {rt}", loso(pred(None, split=(bt, rt))))
print("-- lineup weights re-optimised per decay (grid w_off .4-.8, w_def .6-1.0):")
for tag in ["d1", "d3", "d5"]:
    best = min(((r(loso(pred(tag, wo, wd))), wo, wd) for wo in (.4, .5, .6, .7, .8) for wd in (.6, .7, .8, .9, 1.)))
    print(f"  {tag}: best {best[0]:.4f} at w_off {best[1]}, w_def {best[2]}")
