import pickle, numpy as np, pandas as pd
D="/home/claude/data"; import os; TAG=os.environ.get("TAG","")
rows=[]
for S in range(2014,2026):
    for r in pickle.load(open(f"{D}/inseason{TAG}_{S}.pkl","rb")):
        rows.append(dict(season=S,cp=r['cp'],key=str(r['key']),n=r['n'],sse=r['sse'],tsse=r['team_sse'],tn=r['team_n']))
d=pd.DataFrame(rows)
agg=d.groupby(['key','cp'])[['n','sse','tsse','tn']].sum()
agg['g']=np.sqrt(agg.sse/agg.n); agg['t']=np.sqrt(agg.tsse/agg.tn)
G=agg.g.unstack(); T=agg.t.unstack()
cps=[c for c in G.columns if c>0]
base='(\'M0\', None, None, False, None, None)'
print("GAME RMSE minus M0 (x1000)"); print(((G[cps]-G.loc[base,cps])*1000).round(0).astype(int).sort_values(cps[3]).to_string())
print("\nTEAM rest-of-season RMSE"); print(T[cps].round(3).loc[((T[cps]).mean(axis=1)).sort_values().index].to_string())
d.to_parquet(f"{D}/inseason_summary.parquet")
