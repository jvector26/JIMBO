"""JIMBO production-style in-season prediction (M4 ratings refit through the season + ridge team offsets lam 80, ACTUAL minutes)
vs historical lines. Research only."""
import numpy as np, pandas as pd
D="/home/claude/data"; SEAS=np.arange(2014,2026)
GA=pd.read_parquet(f"{D}/games_all.parquet"); GA["margin"]=GA.home_pts-GA.away_pts
G=GA[GA.season.isin(SEAS)].sort_values(["date","game"]).reset_index(drop=True)
y=G.margin.to_numpy(float); seas=G.season.to_numpy(); dates=G.date.to_numpy()
teams=np.unique(np.r_[G.home_team,G.away_team]); ti={t:i for i,t in enumerate(teams)}
hi=G.home_team.map(ti).to_numpy(); ai=G.away_team.map(ti).to_numpy(); nT=len(teams)
P={}; pre_raw=np.empty(len(G))
for S in SEAS:
    t=pd.read_parquet(f"{D}/elo_m4_preds_{S}.parquet"); piv=t.pivot(index="cp_date",columns="game",values="pred").sort_index()
    idx=np.flatnonzero(seas==S); gm=G.game.to_numpy()[idx]; cpd=piv.index.to_numpy()
    act=cpd[np.searchsorted(cpd,dates[idx],side="right")-1]; M=piv[gm].to_numpy(); row={d:i for i,d in enumerate(cpd)}
    pre_raw[idx]=M[[row[a] for a in act],np.arange(len(idx))]; P[S]=(idx,M,row,act)
X=np.c_[np.ones(len(y)),pre_raw]; cal={S:np.linalg.lstsq(X[seas!=S],y[seas!=S],rcond=None)[0] for S in SEAS}
base=np.array([cal[s][0]+cal[s][1]*p for s,p in zip(seas,pre_raw)])
off=np.zeros(len(y))
for S in SEAS:
    idx,M,row,act=P[S]; a0,b0=cal[S]; days=np.unique(dates[idx]); dg=[np.flatnonzero(dates[idx]==d) for d in days]
    for di,d in enumerate(days):
        today=dg[di]; past=np.concatenate(dg[:di]) if di else np.array([],int)
        if len(past)<15: continue
        res=y[idx[past]]-(a0+b0*M[row[act[today[0]]],past]); hp,ap=hi[idx[past]],ai[idx[past]]
        C=np.zeros((nT,nT)); np.add.at(C,(hp,hp),1); np.add.at(C,(ap,ap),1); np.add.at(C,(hp,ap),-1); np.add.at(C,(ap,hp),-1)
        bv=np.zeros(nT); np.add.at(bv,hp,res); np.add.at(bv,ap,-res)
        o=np.linalg.solve(C+80*np.eye(nT),bv); off[idx[today]]=o[hi[idx[today]]]-o[ai[idx[today]]]
G["J"]=base+off; G["Jbase"]=base
# team games played before this game (for early/late split)
tg=pd.concat([G[["game","season","date","home_team"]].rename(columns={"home_team":"t"}),G[["game","season","date","away_team"]].rename(columns={"away_team":"t"})]).sort_values(["date","game"])
tg["n"]=tg.groupby(["season","t"]).cumcount(); G["gp"]=G.game.map(tg.groupby("game").n.mean())
rm=lambda e:np.sqrt((e**2).mean())
print(f"M4 LOSO {rm(y-base):.4f}  + ridge80 {rm(y-G.J):.4f}  (log: 13.0383 / 13.0088)")
L=pd.read_csv("hist_game_lines.csv"); D2=G.merge(L[["game","open","close","line","src"]],on="game",how="left"); D2.to_pickle("m4_with_lines.pkl")
def clus(x,yv,g):
    b=(x*yv).sum()/(x*x).sum(); r=yv-b*x; se=np.sqrt(sum(((x*r)[g==s].sum())**2 for s in np.unique(g)))/(x*x).sum(); return b,se
def rep(d,col,name,J="J"):
    d=d[d[col].notna()]; g=d.season.values; ej=(d.margin-d[J]).values; el=(d.margin-d[col]).values; gap=(d[J]-d[col]).values
    b,se=clus(gap,el,g); dm=pd.Series(ej**2-el**2).groupby(g).mean(); t=dm.mean()/(dm.std()/np.sqrt(len(dm))) if len(dm)>2 else np.nan
    m=np.abs(gap)>=3; w=(np.sign(gap[m])==np.sign(el[m])).sum(); l=(np.sign(gap[m])==-np.sign(el[m])).sum()
    cl=(np.abs(ej)<np.abs(el)).sum(); fa=(np.abs(ej)>np.abs(el)).sum()
    print(f"{name:24s} n {len(d):5d} | RMSE J {rm(ej):.3f} line {rm(el):.3f} (dMSE t {t:+.1f}) | weight on J {b:+.3f} se {se:.3f} t {b/se:+.1f} | sd gap {gap.std():.2f} | closer {cl/(cl+fa):.3f} | ATS gap3+ {w}-{l} ({w/(w+l):.3f})")
print("\n== vs best available line (close 2014-21 where known else open; SBR single line 2022+) ==")
rep(D2,"line","all 2014-25"); rep(D2[D2.season<=2019],"line","2014-19"); rep(D2[D2.season>=2020],"line","2020-25")
rep(D2[D2.season<=2021],"close","true close 2014-21"); rep(D2[D2.season<=2021],"open","open 2014-21"); rep(D2[D2.season>=2022],"line","SBR line 2022-25")
print("-- by season --")
for s in SEAS: rep(D2[D2.season==s],"line",str(s))
print("-- by team games played --")
for lo,hi_ in [(0,10),(10,25),(25,50),(50,83)]: rep(D2[(D2.gp>=lo)&(D2.gp<hi_)],"line",f"games {lo}-{hi_}")
print("-- M4 without team offsets --"); rep(D2,"line","all, no offsets","Jbase")
