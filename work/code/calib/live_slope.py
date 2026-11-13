"""Session 51: is JIMBO's live win chance too timid / too bold? 2026-27 games so far."""
import json, numpy as np, statsmodels.api as sm
from scipy.stats import norm, binom
v=json.load(open('/home/claude/gh/site/data/vegas.json'))
G=[r for r in v['games'] if r[5] not in (None,0) and r[4] is not None]
J=np.array([r[3] for r in G]); L=np.array([r[4] for r in G]); y=np.array([r[5] for r in G]); w=(y>0).astype(int)
pJ=np.array([r[9] if r[9] is not None else norm.cdf(r[3]/12.8) for r in G]); pL=norm.cdf(L/12.8)
print("games with a line", len(G))
for nm,x in [("JIMBO",J),("line",L)]:
    f=sm.OLS(y,sm.add_constant(x)).fit(); r=y-x
    print(f"{nm}: final = {f.params[0]:+.2f} + {f.params[1]:.2f} x pred (slope se {f.bse[1]:.2f}) | RMSE {np.sqrt((r**2).mean()):.2f} | mean pred {x.mean():+.2f} actual {y.mean():+.2f}")
for nm,p in [("JIMBO",pJ),("line",pL)]:
    lg=np.log(p/(1-p)); f=sm.Logit(w,sm.add_constant(lg)).fit(disp=0)
    print(f"{nm} logistic recalibration: intercept {f.params[0]:+.2f} (se {f.bse[0]:.2f}) slope {f.params[1]:.2f} (se {f.bse[1]:.2f})  [perfect = 0, 1]")
# the 29 games: line view
m=(pJ>=.3)&(pJ<.4); print(f"\n30-40% JIMBO bucket: n={m.sum()} won={w[m].sum()} JIMBO says {pJ[m].mean():.3f} line says {pL[m].mean():.3f}")
print(f"  mean margin: JIMBO {J[m].mean():+.1f} line {L[m].mean():+.1f} actual {y[m].mean():+.1f}")
print(f"  P(<= {w[m].sum()} wins | line's own chances) = {np.mean([ (np.random.rand(m.sum())<pL[m]).sum()<=w[m].sum() for _ in range(200000)]):.4f}")
lm=(pL>=.3)&(pL<.4); print(f"line's own 30-40% bucket: n={lm.sum()} won={w[lm].sum()} line says {pL[lm].mean():.3f}")
# underdog pooling: all home dogs (p<0.5), and all road dogs
for nm,mm in [("home teams JIMBO <40%",pJ<.4),("home teams JIMBO 40-60%",(pJ>=.4)&(pJ<.6)),("home teams JIMBO >60%",pJ>=.6)]:
    print(f"{nm}: n={mm.sum()} won {w[mm].mean():.3f} JIMBO {pJ[mm].mean():.3f} line {pL[mm].mean():.3f} | margin J {J[mm].mean():+.1f} line {L[mm].mean():+.1f} actual {y[mm].mean():+.1f}")
print("home win rate", w.mean().round(3), "JIMBO mean p", pJ.mean().round(3), "line mean p", pL.mean().round(3))
