"""Is the low-season-minutes effect a garbage-time artifact? Rebuild the minute shares from stints:
 ng = non-garbage stints only (same rule as the RAPM fits), h1 = first half only (rotations set before the game is decided).
Target stays the final margin; base harness unchanged (M4 + offsets + B2B, actual minutes)."""
import numpy as np, pandas as pd, lines_bar as LB
from lines_bar import G,y,seas,SEAS,loso
D="/home/claude/data"; T=pd.read_parquet("thin_table.parquet"); rows=[]
for S in range(2015,2026):
    st=pd.read_parquet(f"{D}/stints_{S}.parquet",columns=["game","t_start","t_end","margin_start","teamA","lineupA","teamB","lineupB"])
    st["dur"]=st.t_end-st.t_start; m_=st.margin_start.abs(); st["ng"]=~(((st.t_start>=2160)&(m_>=20))|((st.t_start>=2520)&(m_>=15))); st["h1"]=st.t_start<1440
    for side in "AB":
        e=st[["game","dur","ng","h1","team"+side,"lineup"+side]].rename(columns={"team"+side:"team","lineup"+side:"lu"})
        e=e.assign(player=e.lu.str.split("-")).explode("player"); e=e[e.player.str.len()>0]; e["player"]=e.player.astype(np.int64)
        e["s_ng"]=e.dur*e.ng; e["s_h1"]=e.dur*e.h1; rows.append(e.groupby(["game","team","player"])[["dur","s_ng","s_h1"]].sum().reset_index())
    print(S,flush=True)
E=pd.concat(rows).groupby(["game","team","player"],as_index=False).sum()
for c in ["dur","s_ng","s_h1"]: E["w_"+c]=5*E[c]/E.groupby(["game","team"])[c].transform("sum")
T=T.merge(E[["game","team","player","w_dur","w_s_ng","w_s_h1"]],on=["game","team","player"],how="left").fillna({"w_dur":0,"w_s_ng":0,"w_s_h1":0})
sg=np.sign(T.sw); gi=pd.Series(np.arange(len(G)),index=G.game.to_numpy()); T=T[T.game.isin(gi.index)]; T["gi"]=T.game.map(gi).to_numpy(); sg=np.sign(T.sw.to_numpy())
print("check: stint-based share vs box share corr",np.corrcoef(T.w_dur,T.sw.abs())[0,1].round(4))
b=LB.load(""); line=G.line.to_numpy(); F=pd.read_parquet("gap_features.parquet"); H=F[F.home].set_index("game"); A=F[~F.home].set_index("game")
b2b=(G.game.map(H.rest)==1).to_numpy(float)-(G.game.map(A.rest)==1).to_numpy(float)
m=np.isfinite(b["act"])&np.isfinite(b["tr"]); base=np.c_[b["act"],b["off"],b2b]; p0=loso(base,m); ml=m&np.isfinite(line); r=lambda p,mk:np.sqrt(((y-p)**2)[mk].mean())
M=600; f=np.clip(1-T.cum.to_numpy()/M,0,1); o=(T.grp=="out").to_numpy(); st_=None
def reg(mask,w):
    v=np.zeros(len(G)); np.add.at(v,T.gi.to_numpy()[mask],w[mask]); return v
print(f"base {r(p0,m):.4f}; fade length 600 min, 2 rates (not on opening roster / everyone else)")
for lab,col in [("all minutes (box)",T.sw.abs().to_numpy()),("all minutes (stints)",T.w_dur.to_numpy()),("non-garbage minutes only",T.w_s_ng.to_numpy()),("first-half minutes only",T.w_s_h1.to_numpy())]:
    w=sg*col*f; X=np.column_stack([base,reg((f>0)&o,w),reg((f>0)&~o,w)]); p=loso(X,m); a=LB.pair(p,p0,m)
    bt=np.linalg.lstsq(np.c_[np.ones(m.sum()),X[m]],y[m],rcond=None)[0]; rr=y[m]-np.c_[np.ones(m.sum()),X[m]]@bt; Xm=np.c_[np.ones(m.sum()),X[m]]; g=seas[m]; XtXi=np.linalg.inv(Xm.T@Xm)
    se=np.sqrt(np.diag(XtXi@sum(np.outer(Xm[g==s].T@rr[g==s],Xm[g==s].T@rr[g==s]) for s in np.unique(g))@XtXi))
    print(f"  {lab:26s} dRMSE {a[0]:+.4f} t {a[1]:+.1f} {a[2]}/{a[3]} halves {a[4]:+.4f}/{a[5]:+.4f} | rates {bt[4]:+.2f} (t {bt[4]/se[4]:+.1f}) / {bt[5]:+.2f} (t {bt[5]/se[5]:+.1f}) | gap to line {r(p0,ml)-np.sqrt(((y-line)**2)[ml].mean()):+.3f} -> {r(p,ml)-np.sqrt(((y-line)**2)[ml].mean()):+.3f}")
# starters only: low-season-minute players who START tonight
pgs=pd.concat([pd.read_parquet(f"{D}/player_games_{S}.parquet",columns=["game","player","started"]) for S in range(2015,2026)])
T=T.merge(pgs,on=["game","player"],how="left"); stt=(T.started==1).to_numpy(); w=T.sw.to_numpy()*f
X=np.column_stack([base,reg((f>0)&stt,w),reg((f>0)&~stt,w)]); p=loso(X,m); a=LB.pair(p,p0,m); bt=np.linalg.lstsq(np.c_[np.ones(m.sum()),X[m]],y[m],rcond=None)[0]
print(f"  split by STARTED tonight (box minutes): dRMSE {a[0]:+.4f} t {a[1]:+.1f} | rate starters {bt[4]:+.2f}, non-starters {bt[5]:+.2f}; starters' share of the low-minute exposure {np.abs(w[(f>0)&stt]).sum()/np.abs(w[f>0]).sum():.0%}")
