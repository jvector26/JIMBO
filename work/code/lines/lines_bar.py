"""LINES BAR (session 47): second research-bar score = does a change help once the betting line is in the forecast?
For a candidate, in-season production-style predictions (M4 refit through the season + ridge team offsets lam 80) are scored by
  raw   LOSO RMSE of margin ~ 1 + [pred, offsets(, extra)]
  blend LOSO RMSE of margin ~ 1 + [pred, offsets(, extra)] + line
  wt    weight on JIMBO: slope of (final - line) on (J - line), J = the raw LOSO fit
each against the base on the SAME games (season-paired t on dMSE, halves 2015-19 / 2020-25).
Minutes: 'act' = actual minutes (who played is known), 'tr' = each player's mean over his team's previous 5 games (pregame-knowable).
Usage: python3 lines_bar.py [tag ...]   tags = variant prefixes written by m4_var_preds.py (PRE_TAG), e.g. woff4_ shr_ neff_"""
import sys, numpy as np, pandas as pd
D="/home/claude/data"; LD="/home/claude/lines"; SEAS=np.arange(2015,2026)
GA=pd.read_parquet(f"{D}/games_all.parquet"); GA["margin"]=GA.home_pts-GA.away_pts
G=GA[GA.season.isin(SEAS)].sort_values(["date","game"]).reset_index(drop=True)
L=pd.read_csv(f"{LD}/hist_game_lines.csv"); G=G.merge(L[["game","open","close","line"]],on="game",how="left")
y=G.margin.to_numpy(float); seas=G.season.to_numpy(); dates=G.date.to_numpy()
teams=np.unique(np.r_[G.home_team,G.away_team]); ti={t:i for i,t in enumerate(teams)}
hi=G.home_team.map(ti).to_numpy(); ai=G.away_team.map(ti).to_numpy(); nT=len(teams)

def points_elo(k=.04,c=.6,hca=2.5):
    A=GA.sort_values(["date","game"]); tt=np.unique(np.r_[A.home_team,A.away_team]); r=dict.fromkeys(tt,0.0); out={}; cur=None
    for s,g,h,a,m in zip(A.season,A.game,A.home_team,A.away_team,A.margin):
        if s!=cur: r={t:c*v for t,v in r.items()}; cur=s
        d=r[h]-r[a]; out[g]=d; e=m-(d+hca); r[h]+=k*e; r[a]-=k*e
    return G.game.map(out).to_numpy()

def load(tag=""):
    pa=np.full(len(G),np.nan); pt=np.full(len(G),np.nan); off=np.zeros(len(G))
    for S in SEAS:
        t=pd.read_parquet(f"{LD}/m4tr_{tag}{S}.parquet"); idx=np.flatnonzero(seas==S); gm=G.game.to_numpy()[idx]
        pv=t.pivot(index="cp_date",columns="game",values="pred").sort_index(); pv2=t.pivot(index="cp_date",columns="game",values="pred_tr").sort_index()
        cpd=pv.index.to_numpy(); act=np.searchsorted(cpd,dates[idx],side="right")-1
        M=pv[gm].to_numpy(); M2=pv2[gm].to_numpy(); ar=np.arange(len(idx)); pa[idx]=M[act,ar]; pt[idx]=M2[act,ar]
        P[S]=(idx,M,act)
    # calibration of the actual-minutes base (LOSO) for the offset residuals, as production
    X=np.c_[np.ones(len(y)),pa]
    for S in SEAS:
        a0,b0=np.linalg.lstsq(X[seas!=S],y[seas!=S],rcond=None)[0]; idx,M,act=P[S]; days=np.unique(dates[idx]); dg=[np.flatnonzero(dates[idx]==d) for d in days]
        for di in range(len(days)):
            today=dg[di]; past=np.concatenate(dg[:di]) if di else np.array([],int)
            if len(past)<15: continue
            res=y[idx[past]]-(a0+b0*M[act[today[0]],past]); hp,ap=hi[idx[past]],ai[idx[past]]
            C=np.zeros((nT,nT)); np.add.at(C,(hp,hp),1); np.add.at(C,(ap,ap),1); np.add.at(C,(hp,ap),-1); np.add.at(C,(ap,hp),-1)
            bv=np.zeros(nT); np.add.at(bv,hp,res); np.add.at(bv,ap,-res); o=np.linalg.solve(C+80*np.eye(nT),bv); off[idx[today]]=o[hi[idx[today]]]-o[ai[idx[today]]]
    return {"act":pa,"tr":pt,"off":off}
P={}
def loso(X,m):
    X=np.c_[np.ones(len(y)),X]; p=np.full(len(y),np.nan)
    for S in SEAS:
        tr=m&(seas!=S); te=m&(seas==S); p[te]=X[te]@np.linalg.lstsq(X[tr],y[tr],rcond=None)[0]
    return p
def pair(pv,pb,m):
    d=pd.Series(((y-pv)**2-(y-pb)**2)[m]).groupby(seas[m]).mean(); t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d)))
    r=lambda p,mm:np.sqrt(((y-p)[mm]**2).mean()); e=m&(seas<=2019); l=m&(seas>=2020)
    return r(pv,m)-r(pb,m),t,int((d<0).sum()),len(d),r(pv,e)-r(pb,e),r(pv,l)-r(pb,l)
def wt(J,m):
    x=(J-G.line.to_numpy())[m]; e=(y-G.line.to_numpy())[m]; g=seas[m]; b=(x*e).sum()/(x*x).sum(); rr=e-b*x
    return b,np.sqrt(sum(((x*rr)[g==s].sum())**2 for s in np.unique(g)))/(x*x).sum()
if __name__=="__main__":
    base=load(""); elo=points_elo(); line=G.line.to_numpy()
    m=np.isfinite(line)&np.isfinite(base["tr"]); print("games",m.sum(),"seasons 2015-25 (2025-26 to Jan 7)")
    cands=[("Elo as third input",base,True)]+[(t,load(t),False) for t in sys.argv[1:]]
    for mins in ["tr","act"]:
        Xb=np.c_[base[mins],base["off"]]; rb=loso(Xb,m); bb=loso(np.c_[Xb,line],m); lo=loso(line,m)
        r=lambda p:np.sqrt(((y-p)[m]**2).mean()); w=wt(rb,m)
        print(f"\n=== minutes: {mins} ===  line alone {r(lo):.4f} | BASE raw {r(rb):.4f}  blend {r(bb):.4f} (vs line alone {r(bb)-r(lo):+.4f})  weight {w[0]:.3f} se {w[1]:.3f}")
        for nm,v,el in cands:
            Xv=np.c_[v[mins],v["off"]]
            if el: Xv=np.c_[Xv,elo]
            rv=loso(Xv,m); bv=loso(np.c_[Xv,line],m); a=pair(rv,rb,m); b=pair(bv,bb,m); w=wt(rv,m)
            print(f"{nm:20s} RAW {a[0]:+.4f} t {a[1]:+.1f} {a[2]}/{a[3]} halves {a[4]:+.4f}/{a[5]:+.4f} || BLEND {b[0]:+.4f} t {b[1]:+.1f} {b[2]}/{b[3]} halves {b[4]:+.4f}/{b[5]:+.4f} || weight {w[0]:.3f}")
