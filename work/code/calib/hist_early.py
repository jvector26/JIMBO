"""Session 51: historical calibration of JIMBO's PRESEASON team ratings (pre_T_{S}, projected minutes, no hindsight on games)
on each season's first N games (N=175 ~ where 2026-27 is now) and on the whole season. hca 2.0, sig 12.8 as production.
Approximation: live JIMBO also has ~3 weeks of in-season updates and the preseason Vegas gap term."""
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import norm
D="/home/claude/data"; HCA=2.0; SIG=12.8
GA=pd.read_parquet(f"{D}/games_all.parquet"); GA["y"]=GA.home_pts-GA.away_pts
R=[]
for S in range(2014,2026):
    T=pd.read_parquet(f"{D}/pre_T_{S}.parquet").set_index("team").margin
    g=GA[(GA.season==S)&GA.score_ok.fillna(True).astype(bool)].sort_values(["date","game"]).reset_index(drop=True)
    g["k"]=np.arange(len(g)); tri={"NOH":"NOP","NOK":"NOP","NJN":"BKN","CHH":"CHA","PHO":"PHX","GOS":"GSW","SAN":"SAS","UTH":"UTA","BRK":"BKN","CHO":"CHA"}; g["J"]=g.htm.replace(tri).map(T)-g.vtm.replace(tri).map(T)+HCA; R.append(g)
R=pd.concat(R).dropna(subset=["J"]); R=R[R.y!=0]
R["p"]=norm.cdf(R.J/SIG); R["w"]=(R.y>0).astype(int)
def report(nm,d):
    print(f"\n== {nm}: {len(d)} games, {d.season.nunique()} seasons")
    f=sm.OLS(d.y,sm.add_constant(d.J)).fit(); lg=np.log(d.p/(1-d.p)); fl=sm.Logit(d.w,sm.add_constant(lg)).fit(disp=0)
    print(f"margin slope {f.params.J:.2f} (se {f.bse.J:.2f}) | RMSE {np.sqrt(((d.y-d.J)**2).mean()):.2f} | logistic slope {fl.params.iloc[1]:.2f} (se {fl.bse.iloc[1]:.2f}) intercept {fl.params.iloc[0]:+.2f}")
    for lo in range(10):
        b=d[(d.p>=lo/10)&(d.p<(lo+1)/10)]
        if len(b): print(f"  {lo*10:2d}-{lo*10+10}%: n={len(b):5d} said {b.p.mean():.3f} home won {b.w.mean():.3f}  (margin said {b.J.mean():+.1f}, actual {b.y.mean():+.1f})")
    b=d[(d.p>=.3)&(d.p<.4)]
    print("  30-40% by season (won/n):", " ".join(f"{s%100:02d}:{int(x.w.sum())}/{len(x)}" for s,x in b.groupby("season")))
    return d
report("first 175 games of each season", R[R.k<175])
report("whole season", R)
# season-by-season early slope (is 2026-27's 1.23 unusual?)
print("\nearly-season (first 175) margin slope by season:")
for s,x in R[R.k<175].groupby("season"):
    f=sm.OLS(x.y,sm.add_constant(x.J)).fit(); print(f"  {s}: {f.params.J:.2f} (se {f.bse.J:.2f})  home dogs <40%: {int(x.w[x.p<.4].sum())}/{(x.p<.4).sum()} won, said {x.p[x.p<.4].mean():.2f}")
print("\nwhole-season margin slope by season (preseason ratings only):")
print("  "+" ".join(f"{s%100:02d}:{sm.OLS(x.y,sm.add_constant(x.J)).fit().params.J:.2f}" for s,x in R.groupby("season")))
# chance a season's first 175 games shows a 30-40% bucket this bad, if calibrated: simulate with each season's own p's
rng=np.random.default_rng(0); E=R[R.k<175]
for s,x in E.groupby("season"):
    pass
b=E[(E.p>=.3)&(E.p<.4)]; print(f"\npooled 12 seasons, 30-40% early: {b.w.sum()}/{len(b)} = {b.w.mean():.3f} vs said {b.p.mean():.3f}")
