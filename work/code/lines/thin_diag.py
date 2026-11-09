"""THIN-HISTORY DIAGNOSIS (session 47). For each game: minute shares by player category x season-to-date exposure, signed home - away.
Regress JIMBO's residual (actual minutes + back-to-back, LOSO) on them: coef = how much JIMBO under(+)/over(-)rates that group,
in points per 100 poss for a full-time player, relative to established veterans. Also excess squared error vs the line per unit share."""
import numpy as np, pandas as pd, lines_bar as LB
from lines_bar import G,y,seas,SEAS,loso
D="/home/claude/data"; ps=pd.read_parquet(f"{D}/player_seasons.parquet")
b=LB.load(""); line=G.line.to_numpy(); F=pd.read_parquet("gap_features.parquet")
H=F[F.home].set_index("game"); A=F[~F.home].set_index("game")
b2b=(G.game.map(H.rest)==1).to_numpy(float)-(G.game.map(A.rest)==1).to_numpy(float)
m=np.isfinite(line)&np.isfinite(b["tr"])
p4=loso(np.c_[b["act"],b["off"],b2b],m); res=y-p4; e4=res**2; el=(y-line)**2
CATS=["rook_r1","newcomer","callup_nohist","late_add_hist","vet_thin","vet_mid"]; EXP=["<200 min","200-600","600+"]
rows=[]
for S in SEAS:
    pg=pd.read_parquet(f"{D}/player_games_{S}.parquet"); pg=pg[pg["min"]>0].sort_values(["date","game"])
    pre=pd.read_parquet(f"{D}/pre_P_{S}.parquet").drop_duplicates("player").set_index("player")
    hist=ps[ps.season.between(S-3,S-1)].groupby("player").on_poss.sum()
    def cat(p):
        k=pre.kind.get(p)
        if k=="rookie": return "rook_r1"
        if k=="newcomer": return "newcomer"
        h=hist.get(p,0)
        if k is None: return "callup_nohist" if h<=0 else ("late_add_hist" if h<2000 else None)
        return "vet_thin" if h<2000 else ("vet_mid" if h<6000 else None)
    cm={p:cat(p) for p in pg.player.unique()}; pg["cat"]=pg.player.map(cm)
    pg["cum"]=pg.groupby("player")["min"].cumsum()-pg["min"]; pg["ex"]=np.where(pg.cum<200,0,np.where(pg.cum<600,1,2))
    hm=G[G.season==S].set_index("game").home_team; pg=pg[pg.game.isin(hm.index)]
    pg["ws"]=5*pg["min"]/pg.groupby(["game","team"])["min"].transform("sum"); pg["sg"]=np.where(pg.team==pg.game.map(hm),1.0,-1.0)
    x=pg[pg.cat.notna()]; x=x.assign(key=x.cat+"|"+x.ex.astype(str))
    rows.append(x.assign(v=x.ws*x.sg).pivot_table(index="game",columns="key",values="v",aggfunc="sum").assign(_k="s"))
    rows.append(x.pivot_table(index="game",columns="key",values="ws",aggfunc="sum").assign(_k="u"))
R=pd.concat(rows).fillna(0); keys=[f"{c}|{e}" for c in CATS for e in range(3)]
Sg=R[R._k=="s"].drop(columns="_k").reindex(G.game).fillna(0)[keys].to_numpy(); U=R[R._k=="u"].drop(columns="_k").reindex(G.game).fillna(0)[keys].to_numpy()
def ols(X,yv,g):
    bta=np.linalg.lstsq(X,yv,rcond=None)[0]; rr=yv-X@bta; XtXi=np.linalg.inv(X.T@X)
    meat=sum(np.outer(X[g==s].T@rr[g==s],X[g==s].T@rr[g==s]) for s in np.unique(g)); return bta,np.sqrt(np.diag(XtXi@meat@XtXi))
bt,se=ols(Sg[m],res[m],seas[m]); X2=np.c_[np.ones(m.sum()),U[m]]; b2,se2=ols(X2,(e4-el)[m],seas[m])
print(f"games {m.sum()}; residual = final - JIMBO (actual minutes + back-to-back), error {np.sqrt(e4[m].mean()):.3f}, line {np.sqrt(el[m].mean()):.3f}")
print(f"{'group':16s}{'played so far':>14s}{'% of all min':>13s}{'JIMBO bias':>12s}{'t':>6s}   {'excess sq.err vs line per full-time player':>44s}{'t':>6s}")
tot=U[m].sum()/ (10*m.sum())
for i,k in enumerate(keys):
    c,e=k.split("|"); print(f"{c:16s}{EXP[int(e)]:>14s}{U[m][:,i].sum()/(10*m.sum()):13.1%}{bt[i]:+12.2f}{bt[i]/se[i]:+6.1f}   {b2[i+1]:+44.1f}{b2[i+1]/se2[i+1]:+6.1f}")
print("established vets (reference): share",f"{1-tot:.1%}")
# collapsed by category, and by exposure
for nm,grp in [("by group",[[i for i,k in enumerate(keys) if k.startswith(c)] for c in CATS]),("by exposure",[[i for i,k in enumerate(keys) if k.endswith(str(e))] for e in range(3)])]:
    Xs=np.column_stack([Sg[:,g].sum(1) for g in grp]); bt_,se_=ols(Xs[m],res[m],seas[m]); Xu=np.c_[np.ones(m.sum()),np.column_stack([U[:,g].sum(1) for g in grp])[m]]; bu,su=ols(Xu,(e4-el)[m],seas[m])
    print("--",nm)
    for j,lb in enumerate(CATS if nm=="by group" else EXP):
        hb_=np.array([ols(Xs[m&(seas<=2019)],res[m&(seas<=2019)],seas[m&(seas<=2019)])[0][j],ols(Xs[m&(seas>=2020)],res[m&(seas>=2020)],seas[m&(seas>=2020)])[0][j]])
        print(f"  {lb:16s} share {U[:,grp[j]].sum(1)[m].sum()/(10*m.sum()):6.1%}  bias {bt_[j]:+.2f} (t {bt_[j]/se_[j]:+.1f}; halves {hb_[0]:+.2f}/{hb_[1]:+.2f})   excess sq.err {bu[j+1]:+.1f} (t {bu[j+1]/su[j+1]:+.1f})")
# how much would fixing the biases close? LOSO with the 18 signed shares as regressors
pv=loso(np.c_[b["act"],b["off"],b2b,Sg],m); a=LB.pair(pv,p4,m); print(f"\nLOSO with group x exposure bias corrections: {a[0]:+.4f} t {a[1]:+.1f} {a[2]}/{a[3]} halves {a[4]:+.4f}/{a[5]:+.4f}  (gap to line was {np.sqrt(e4[m].mean())-np.sqrt(el[m].mean()):+.3f})")
np.save("thin_Sg.npy",Sg)
