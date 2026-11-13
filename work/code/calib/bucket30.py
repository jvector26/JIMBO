"""Session 51: the 30-40% home-win bucket on the live calibration chart (user question 2026-11-13)."""
import json, sys
from scipy.stats import norm
from collections import Counter
v=json.load(open('/home/claude/gh/site/data/vegas.json')); print(v['asof'], len(v['games']))
G=[r for r in v['games'] if r[5] not in (None,0)]
sig=12.8
pj=lambda r: r[9] if r[9] is not None else norm.cdf(r[3]/sig)
pv=lambda r: norm.cdf(r[4]/sig) if r[4] is not None else None
print("finals",len(G))
for lo in range(10):
    B=[r for r in G if lo/10<=pj(r)<(lo+1)/10]
    if B: print(f"{lo*10}-{lo*10+10}%: n={len(B)} won={sum(r[5]>0 for r in B)} J={sum(pj(r) for r in B)/len(B):.3f} line={sum(pv(r) for r in B if r[4] is not None)/max(1,sum(r[4] is not None for r in B)):.3f}")
B=[r for r in G if 0.3<=pj(r)<0.4]
print("home:",Counter(r[1] for r in B).most_common())
print("away:",Counter(r[2] for r in B).most_common())
for r in sorted(B): print(r[0],r[2],'@',r[1],'J %.2f'%pj(r),'Jmg %+.1f'%r[3],'line',r[4],'final',r[5])
