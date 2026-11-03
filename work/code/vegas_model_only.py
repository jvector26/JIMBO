import pandas as pd,numpy as np
from scipy import stats
g=pd.read_pickle('/tmp/claude-0/-home-claude/5627b446-f3cd-556f-ae2f-ccc31598218e/scratchpad/g.pkl')
TP=pd.read_csv('gh/state/2026/teams_pre.csv',index_col=0); gap0=TP.blend_margin_pre-TP.model_margin_pre
S=pd.read_csv('gh/live/2026/schedule.csv'); S=S[S.kind!='pre']; S['date']=S.date_et.str.replace('-','').astype(int)
def played(t,d): return int((((S.home==t)|(S.away==t))&(S.date<d)).sum())
adj=[]
for _,r in g.iterrows():
    dh=20/(20+played(r.home,r.date)); da=20/(20+played(r.away,r.date))
    adj.append(gap0[r.home]*dh-gap0[r.away]*da)
g['adj']=adj; g['M']=g.J-g.adj
print('preseason-gap part of the margin: mean|adj| %.2f, max %.2f'%(g.adj.abs().mean(),g.adj.abs().max()))
def rep(nm,P):
    e=g.F-P; eL=g.F-g.L; gap=P-g.L; n=len(g)
    cl=(e.abs()<eL.abs()).sum(); tie=(e.abs()==eL.abs()).sum()
    dm=e**2-eL**2
    side=np.sign(gap); cov=np.sign((g.F-g.L)*side)  # +1 covered, -1 not, 0 push / no side
    w=(cov>0).sum(); l=(cov<0).sum()
    big=gap.abs()>=3; wb=((cov>0)&big).sum(); lb=((cov<0)&big).sum()
    b=(eL*gap).sum()/(gap**2).sum(); se=np.sqrt(((eL-b*gap)**2).sum()/(n-1)/(gap**2).sum())
    straight=((np.sign(P)==np.sign(g.F))).sum()
    print(f'{nm:9s} closer {cl}-{n-cl-tie}-{tie} | RMSE {np.sqrt((e**2).mean()):.2f} (line {np.sqrt((eL**2).mean()):.2f}) MAE {e.abs().mean():.2f} (line {eL.abs().mean():.2f}) dMSE {dm.mean():+.1f} t {dm.mean()/dm.std()*np.sqrt(n):.2f} | ATS {w}-{l} ({w/(w+l):.0%}), gap3+ {wb}-{lb} | slope {b:+.2f}±{se:.2f} | winner picked {straight}/{n} | mean|gap| {gap.abs().mean():.2f}, home lean {gap.mean():+.2f}')
rep('blended',g.J); rep('model',g.M)
print('line picks winner', (np.sign(g.L)==np.sign(g.F)).sum())
g.to_pickle('/tmp/claude-0/-home-claude/5627b446-f3cd-556f-ae2f-ccc31598218e/scratchpad/g2.pkl')
