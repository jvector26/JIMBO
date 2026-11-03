import json,numpy as np,pandas as pd
from scipy import stats
d=json.load(open('gh/site/data/vegas.json'))
g=pd.DataFrame(d['games'],columns=['date','home','away','J','L','F','hp','ap','gid','p'])
g=g.dropna(subset=['L','F'])
g['gap']=g.J-g.L; g['eJ']=g.F-g.J; g['eL']=g.F-g.L
n=len(g); print('n',n)
cl=(g.eJ.abs()<g.eL.abs()); tie=(g.eJ.abs()==g.eL.abs())
print('closer',cl.sum(),'ties',tie.sum(),'line closer',(g.eJ.abs()>g.eL.abs()).sum())
print('binom one-sided',stats.binom.cdf(cl.sum(),n-tie.sum(),.5))
print('RMSE J %.2f L %.2f  MAE J %.2f L %.2f'%(np.sqrt((g.eJ**2).mean()),np.sqrt((g.eL**2).mean()),g.eJ.abs().mean(),g.eL.abs().mean()))
dm=g.eJ**2-g.eL**2; print('dMSE %.1f t %.2f'%(dm.mean(),dm.mean()/dm.std()*np.sqrt(n)))
da=g.eJ.abs()-g.eL.abs(); print('dMAE %.2f t %.2f'%(da.mean(),da.mean()/da.std()*np.sqrt(n)))
print('gap: mean %.2f sd %.2f  mean|gap| %.2f, |gap|>=3: %d'%(g.gap.mean(),g.gap.std(),g.gap.abs().mean(),(g.gap.abs()>=3).sum()))
print('mean final-line (home vs line) %.2f  t %.2f'%(g.eL.mean(),g.eL.mean()/g.eL.std()*np.sqrt(n)))
# home bias before/after 11-01 hca change
for nm,m in [('<=10-31',g.date<=20261031),('>=11-01',g.date>=20261101)]:
    s=g[m]; print(nm,len(s),'gap mean %.2f'%s.gap.mean(), 'closer %d/%d'%((s.eJ.abs()<s.eL.abs()).sum(),len(s)), 'RMSE J %.2f L %.2f'%(np.sqrt((s.eJ**2).mean()),np.sqrt((s.eL**2).mean())))
# info test: regress eL on gap through origin
b=(g.eL*g.gap).sum()/(g.gap**2).sum(); r=g.eL-b*g.gap; se=np.sqrt((r**2).sum()/(n-1)/(g.gap**2).sum())
print('slope (final-line) on gap %.2f se %.2f'%(b,se))
# gap demeaned (removes home bias)
gd=g.gap-g.gap.mean(); b2=(g.eL*gd).sum()/(gd**2).sum(); r=g.eL-g.eL.mean()-b2*gd; se2=np.sqrt((r**2).sum()/(n-2)/(gd**2).sum())
print('slope on demeaned gap %.2f se %.2f'%(b2,se2))
# expected closer rate under w with actual gaps, error sd = resid sd
sd=g.eL.std()
for w in [0,0.2,0.4,0.5,1]:
    pc=stats.norm.cdf((w-0.5)*g.gap.abs()/sd); pc[g.gap==0]=0.5
    print('w=%.1f expected closer %.1f%% (%.1f games)'%(w,100*pc.mean(),pc.sum()))
# bucket by |gap|
g['ag']=pd.cut(g.gap.abs(),[-.01,1,2,3,5,20])
print(g.groupby('ag',observed=True).apply(lambda s:pd.Series({'n':len(s),'closer':(s.eJ.abs()<s.eL.abs()).sum(),'side_right':((s.eL*np.sign(s.gap))>0).sum(),'mean_eL_toward_J':(s.eL*np.sign(s.gap)).mean()})))
# blowouts
g['dm']=dm; print(g.sort_values('dm',ascending=False).head(8)[['date','home','away','J','L','F','dm']])
print('dMSE without top5: %.1f'%g.sort_values('dm').dm.iloc[:-5].mean())
g.to_pickle('/tmp/claude-0/-home-claude/5627b446-f3cd-556f-ae2f-ccc31598218e/scratchpad/g.pkl')
