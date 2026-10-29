# Predictive test of shift-sign ratings vs plus-minus vs JIMBO. Ratings from seasons T-3..T-1 (decay .3,.6,1) predict T games w/ actual minutes.
import numpy as np, pandas as pd, scipy.sparse as sp
D="/home/claude/data"; DEC={1:1.0,2:0.6,3:0.3}
def exposures(y):
    st=pd.read_parquet(f"{D}/stints_{y}.parquet").sort_values(["game","period","t_start"]).reset_index(drop=True)
    st["idx"]=st.groupby("game").cumcount(); st["sec"]=(st.t_end-st.t_start).clip(lower=0)
    g=pd.read_parquet(f"{D}/games_{y}.parquet").set_index("game").home_team
    rows=[]
    for me in "AB":
        x=pd.DataFrame({"game":st.game,"idx":st.idx,"team":st[f"team{me}"],"p":st[f"lineup{me}"].str.split("-"),"sec":st.sec})
        rows.append(x.explode("p"))
    sp_=pd.concat(rows,ignore_index=True); sp_["p"]=sp_.p.astype(np.int64)
    sh=pd.read_parquet(f"{D}/shifts_{y}.parquet").reset_index(names="sid")
    # end idx: start_idx + number of stints; recompute from exploded table
    s2=sp_.sort_values(["game","team","p","idx"])
    new=(s2.game.ne(s2.game.shift())|s2.p.ne(s2.p.shift())|s2.idx.ne(s2.idx.shift()+1))
    s2["sid"]=new.cumsum()-1   # same ordering as shifts.py -> same ids
    rng=s2.groupby("sid").agg(game=("game","first"),team=("team","first"),a=("idx","min"),b=("idx","max"))
    assert len(rng)==len(sh)
    # stint -> shift membership
    mem=s2[["sid","game","idx"]]
    j=mem.merge(sp_,on=["game","idx"])      # all 10 players in each stint of each shift
    j["own"]=np.where(j.team.values==rng.team.values[j.sid.values],1.0,-1.0)
    j=j.groupby(["sid","p"]).agg(sec=("sec","sum"),own=("own","first")).reset_index()
    tot=rng.join(sh.set_index("sid")[["sec"]]).sec
    j["w"]=j.own*j.sec/tot.values[j.sid.values].clip(min=1)
    sh["home"]=(sh.team.values==g.reindex(sh.game).values).astype(float)
    return sh,j
cache={}
def exp_(y):
    if y not in cache: cache[y]=exposures(y)
    return cache[y]
def ridge(T,lam,ycol):
    shs,js,off=[],[],0
    for k in (1,2,3):
        sh,j=exp_(T-k); sh=sh.copy(); j=j.copy()
        sh["wt"]=DEC[k]*(1.0 if ycol=="sign" else sh.poss.clip(lower=0.5))
        sh["y"]=sh.sign if ycol=="sign" else 100*sh.m/sh.poss.clip(lower=0.5)
        j["sid"]+=off; sh["sid"]+=off; off+=len(sh); shs.append(sh); js.append(j)
    sh=pd.concat(shs); j=pd.concat(js)
    pl=np.unique(j.p); pi=pd.Series(np.arange(len(pl)),pl)
    X=sp.csr_matrix((j.w.values,(j.sid.values,pi[j.p].values)),shape=(off,len(pl)))
    X=sp.hstack([X,sp.csr_matrix(sh.home.values.reshape(-1,1)*2-1)]).tocsr()
    wt=sh.wt.values; A=(X.T.multiply(wt)@X).toarray(); A[np.arange(len(pl)),np.arange(len(pl))]+=lam
    b=np.linalg.solve(A,X.T@(wt*sh.y.values))
    return pd.Series(b[:-1],pl)
def raw(T,k,ycol):
    a=[]
    for d in (1,2,3):
        sh=pd.read_parquet(f"{D}/shifts_{T-d}.parquet"); w=DEC[d]
        a.append(pd.DataFrame({"p":sh.player,"num":w*(sh.sign if ycol=="sign" else sh.m),"den":w*(1.0 if ycol=="sign" else sh.poss)}))
    a=pd.concat(a).groupby("p").sum()
    return a.num/(a.den+k)
if __name__=="__main__":
    out=[]
    for T in range(2015,2026):
        pg=pd.read_parquet(f"{D}/player_games_{T}.parquet")[["game","player","team","min"]]
        gm=pd.read_parquet(f"{D}/games_{T}.parquet"); gm["margin"]=gm.home_pts-gm.away_pts
        R={}
        for k in (0,10,30,100): R[f"sign_raw_k{k}"]=raw(T,k,"sign")
        for k in (0,300,1000,3000): R[f"pm_raw_k{k}"]=raw(T,k,"pm")
        for lam in (30,100,300,1000): R[f"sign_adj_l{lam}"]=ridge(T,lam,"sign")
        for lam in (1000,3000,10000): R[f"pm_adj_l{lam}"]=ridge(T,lam,"pm")
        g=gm[["game","home_team","away_team","margin"]].copy(); g["season"]=T
        hk=pd.MultiIndex.from_arrays([g.game,g.home_team]); ak=pd.MultiIndex.from_arrays([g.game,g.away_team])
        rated=pg.player.isin(R["sign_raw_k0"].index)
        for c,r in list(R.items())+[("unrated",None)]:
            v=(pg.player.map(r).fillna(0) if r is not None else (~rated).astype(float))*pg["min"]/48
            t=v.groupby([pg.game,pg.team]).sum(); g[c]=t.reindex(hk).to_numpy()-t.reindex(ak).to_numpy()
        out.append(g); print(T,flush=True)
    pd.concat(out).to_parquet(f"{D}/shift_bt.parquet",index=False)
