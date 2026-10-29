# Does a thin-sample lineup residual carry to the next season, and does changing teams matter?
import pandas as pd, numpy as np
lh=pd.read_parquet('data/lineup_hist.parquet')
ps=pd.read_parquet('data/player_seasons.parquet')[['season','player','teams','min','on_poss','age']]
ps['t_first']=ps.teams.str.split('/').str[0]; ps['t_last']=ps.teams.str.split('/').str[-1]
# window possessions N-2..N (unweighted) as sample size
pw=ps.pivot_table(index='player',columns='season',values='on_poss',aggfunc='sum').fillna(0)
rows=[]
for N in range(1998,2025):
    cols=[c for c in (N-2,N-1,N) if c in pw.columns]
    w=pw[cols].sum(1).rename('poss_win')
    a=lh[lh.season==N][['player','win_o','win_d']]
    b=lh[lh.season==N+1][['player','one_o','one_d']]
    x=a.merge(b,on='player').merge(w,left_on='player',right_index=True)
    x=x.merge(ps[ps.season==N][['player','t_last','age','min']],on='player')
    x=x.merge(ps[ps.season==N+1][['player','t_first','min']].rename(columns={'min':'min1'}),on='player')
    x['N']=N; rows.append(x)
d=pd.concat(rows); d=d[(d.min1>=500)&(d['min']>=300)]
d['moved']=d.t_last!=d.t_first
d['x']=0.6*d.win_o+0.8*d.win_d; d['y']=0.6*d.one_o+0.8*d.one_d
d['thin']=pd.cut(d.poss_win,[0,2500,4000,7000,1e9],labels=['<2.5k','2.5-4k','4-7k','7k+'])
def slope(g):
    w=g.min1; xm=np.average(g.x,weights=w); ym=np.average(g.y,weights=w)
    b=np.sum(w*(g.x-xm)*(g.y-ym))/np.sum(w*(g.x-xm)**2)
    # season-block bootstrap se
    bs=[]; rng=np.random.default_rng(0); Ns=g.N.unique()
    for _ in range(300):
        s=pd.concat([g[g.N==n] for n in rng.choice(Ns,len(Ns))]); ww=s.min1
        xm2=np.average(s.x,weights=ww); ym2=np.average(s.y,weights=ww)
        bs.append(np.sum(ww*(s.x-xm2)*(s.y-ym2))/np.sum(ww*(s.x-xm2)**2))
    return pd.Series({'n':len(g),'slope':b,'se':np.std(bs),'sd_x':np.sqrt(np.cov(g.x,aweights=w))})
print('ALL'); print(slope(d).round(3).to_dict())
print(d.groupby('thin',observed=True).apply(slope).round(3))
print(d.groupby('moved').apply(slope).round(3))
print(d.groupby(['thin','moved'],observed=True).apply(slope).round(3))
young=d[(d.age<=22)]
print('age<=22'); print(young.groupby('thin',observed=True).apply(slope).round(3))
d.to_parquet('data/lineup_persist.parquet')
