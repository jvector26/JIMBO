"""Expected-win effect of B2B adjustment on the REST of 2026-27 (current latest.json margins, game p = Phi((dm+HCA)/SIG))."""
import json, numpy as np, pandas as pd
from scipy.stats import norm
import sys
BETA = float(sys.argv[1]) if len(sys.argv) > 1 else 2.1
d = json.load(open('/home/claude/gh/site/data/latest.json')); asof = int(d['asof'][:10].replace('-', '')) if isinstance(d['asof'], str) else d['asof']
M = {t['t']: t['margin'] for t in d['teams']}; W = {t['t']: t['proj_w'] for t in d['teams']}
s = pd.read_csv('/home/claude/gh/live/2026/schedule.csv'); s = s[(s.kind == 'reg') & (s.home != s.away) & s.home.notna()]; s['d'] = pd.to_datetime(s.date_et)
L = pd.concat([s[['game', 'd', 'home']].rename(columns={'home': 't'}), s[['game', 'd', 'away']].rename(columns={'away': 't'})]).sort_values(['t', 'd'])
L['b2b'] = ((L.d - L.groupby('t').d.shift(1)).dt.days == 1).astype(int)
k = L.set_index(['game', 't']).b2b
s['bh'] = k.reindex(pd.MultiIndex.from_arrays([s.game, s.home])).values; s['ba'] = k.reindex(pd.MultiIndex.from_arrays([s.game, s.away])).values
r = s[s.d >= pd.Timestamp(str(asof)[:8])].copy()
HCA, SIG = 2.6, 12.8
dm = r.home.map(M) - r.away.map(M)
p0 = norm.cdf((dm + HCA) / SIG)
# refit HCA so the league-wide average home edge is unchanged (part of the old HCA was B2B)
hca1 = HCA - BETA * (r.ba - r.bh).mean() * 0 + 0  # placeholder, set below
adj = -BETA * (r.bh - r.ba)
hca1 = HCA - adj.mean()   # keep average home margin unchanged
p1 = norm.cdf((dm + hca1 + adj) / SIG)
out = []
for t in M:
    h, a = r.home == t, r.away == t
    out.append((t, int(r.bh[h].sum() + r.ba[a].sum()), int(r.ba[h].sum() + r.bh[a].sum()), (p1[h] - p0[h]).sum() - (p1[a] - p0[a]).sum()))
o = pd.DataFrame(out, columns=['team', 'own B2B left', 'opp B2B left', 'd wins']).sort_values('d wins')
print(f"beta {BETA}, HCA {HCA} -> {hca1:.2f} (avg adjustment {adj.mean():+.2f}), {len(r)} games left")
print(o.round(2).to_string(index=False)); print("sd of d wins %.2f" % o['d wins'].std())
