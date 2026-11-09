"""LOW-SEASON-MINUTES PENALTY, proper version (session 47). penalty_p = -a_grp x clip(1 - season_min/M, 0, 1) for team game >= G0,
plus a separate 'back from a layoff' term (5+ team games missed, 600+ season minutes). a_grp fit LOSO; M picked on the training
seasons only (nested). Harness 1: in-season JIMBO (M4 + offsets + back-to-back, actual minutes). Harness 2: the research bar
(preseason ratings, actual minutes, backtest_components + bt_split)."""
import sys, numpy as np, pandas as pd, lines_bar as LB
from lines_bar import G,y,seas,SEAS,loso
D="/home/claude/data"; T=pd.read_parquet("thin_table.parquet"); gi=pd.Series(np.arange(len(G)),index=G.game.to_numpy())
T=T[T.game.isin(gi.index)].copy(); T["gi"]=T.game.map(gi).to_numpy()
b=LB.load(""); line=G.line.to_numpy(); F=pd.read_parquet("gap_features.parquet"); H=F[F.home].set_index("game"); A=F[~F.home].set_index("game")
b2b=(G.game.map(H.rest)==1).to_numpy(float)-(G.game.map(A.rest)==1).to_numpy(float)
gp=((G.game.map(H.gp)+G.game.map(A.gp))/2).to_numpy()
B=pd.read_parquet(f"{D}/backtest_components.parquet").merge(pd.read_parquet(f"{D}/bt_split.parquet"),on=["game","home_team","away_team"])
B["x"]=B[[c for c in B.columns if c.startswith("f:")]].sum(axis=1)+B.unrated+.6*B.res_o+.8*B.res_d; xbar=G.game.map(B.assign(game=B.game.astype(int)).set_index("game").x).to_numpy()
def reg(mask,w):  # game-level regressor: sum of signed share x weight over the masked player-games
    v=np.zeros(len(G)); np.add.at(v,T.gi.to_numpy()[mask],(T.sw.to_numpy()*w)[mask]); return v
def terms(M,G0,struct):
    f=np.clip(1-T.cum.to_numpy()/M,0,1); on=(T.tgp.to_numpy()>=G0)&(f>0); grp=T.grp.to_numpy()
    if struct=="one": return [reg(on,f)]
    if struct=="two": return [reg(on&(grp=="out"),f),reg(on&(grp!="out"),f)]
    return [reg(on&(grp==k),f) for k in ["out","reg","bench","fringe"]]
lay=reg((T.missed.to_numpy()>=5)&(T.cum.to_numpy()>=600),np.ones(len(T)))
def nested(base,m,struct,Ms,G0=15,extra=None):
    """LOSO; M picked on training seasons by in-sample RMSE."""
    p=np.full(len(y),np.nan); pick=[]; Xs={M:np.column_stack([np.ones(len(y)),base]+terms(M,G0,struct)+([extra] if extra is not None else [])) for M in Ms}
    for S in SEAS:
        tr=m&(seas!=S); te=m&(seas==S); best=None
        for M in Ms:
            bt=np.linalg.lstsq(Xs[M][tr],y[tr],rcond=None)[0]; e=((y[tr]-Xs[M][tr]@bt)**2).mean()
            if best is None or e<best[0]: best=(e,M,bt)
        p[te]=Xs[best[1]][te]@best[2]; pick.append(best[1])
    return p,pick
def full(base,m,struct,M,G0=15,extra=None):
    X=np.column_stack([np.ones(len(y)),base]+terms(M,G0,struct)+([extra] if extra is not None else [])); bt=np.linalg.lstsq(X[m],y[m],rcond=None)[0]; rr=y[m]-X[m]@bt; g=seas[m]
    XtXi=np.linalg.inv(X[m].T@X[m]); meat=sum(np.outer(X[m][g==s].T@rr[g==s],X[m][g==s].T@rr[g==s]) for s in np.unique(g)); se=np.sqrt(np.diag(XtXi@meat@XtXi)); k=base.shape[1]+1
    return bt[k:],bt[k:]/se[k:]
Ms=[200,300,450,600,900]
for hname,base,m in [("IN-SEASON harness (M4 + offsets + B2B, actual minutes)",np.c_[b["act"],b["off"],b2b],np.isfinite(b["act"])&np.isfinite(b["tr"])),
                     ("RESEARCH BAR harness (preseason ratings, actual minutes; + B2B)",np.c_[xbar,b2b],np.isfinite(xbar))]:
    p0=loso(base,m); r=lambda p,mk=None:np.sqrt(((y-p)**2)[m if mk is None else mk].mean()); ml=m&np.isfinite(line)
    print(f"\n=== {hname}: games {m.sum()}, base LOSO {r(p0):.4f}" + (f" | on games with a line: JIMBO {r(p0,ml):.3f} line {np.sqrt(((y-line)**2)[ml].mean()):.3f}" if ml.sum() else ""))
    for struct,lab in [("one","1 rate for everyone"),("two","2 rates: not on opening roster / everyone else"),("four","4 rates: outsider / last-yr regular / bench / fringe")]:
        p,pick=nested(base,m,struct,Ms); a=LB.pair(p,p0,m); M=max(set(pick),key=pick.count); co,t=full(base,m,struct,M)
        print(f"{lab:52s} dRMSE {a[0]:+.4f} t {a[1]:+.1f} {a[2]}/{a[3]} halves {a[4]:+.4f}/{a[5]:+.4f} | M picked {sorted(set(pick))} | at M={M}: rates {np.round(co,2)} t {np.round(t,1)}" + (f" | gap to line {r(p0,ml)-np.sqrt(((y-line)**2)[ml].mean()):+.3f} -> {r(p,ml)-np.sqrt(((y-line)**2)[ml].mean()):+.3f}" if ml.sum() else ""))
        if struct=="two": p2=p; M2=M
    # sensitivity + layoff
    for G0 in [0,10,20]:
        p,pick=nested(base,m,"two",Ms,G0=G0); a=LB.pair(p,p0,m); print(f"   2 rates, start at team game {G0:2d}: dRMSE {a[0]:+.4f} t {a[1]:+.1f} {a[2]}/{a[3]} halves {a[4]:+.4f}/{a[5]:+.4f}")
    for M in Ms:
        X=np.column_stack([base]+terms(M,15,"two")); p=loso(X,m); a=LB.pair(p,p0,m); print(f"   2 rates, fixed fade length M={M:4d} min: dRMSE {a[0]:+.4f} t {a[1]:+.1f} {a[2]}/{a[3]}")
    p,pick=nested(base,m,"two",Ms,extra=lay); a=LB.pair(p,p2,m); co,t=full(base,m,"two",M2,extra=lay)
    print(f"   + back-from-layoff term (5+ games missed, 600+ season min): step dRMSE {a[0]:+.4f} t {a[1]:+.1f} {a[2]}/{a[3]} | coef {co[-1]:+.2f} t {t[-1]:+.1f}")
    d=pd.Series(((y-p2)**2-(y-p0)**2)[m]).groupby(seas[m]).mean(); print("   2-rate dMSE by season:",d.round(2).to_dict())
    for lb,mk in [("team games <15",gp<15),("15-40",(gp>=15)&(gp<40)),("40-70",(gp>=40)&(gp<70)),("70+",gp>=70)]:
        mk=m&mk&np.isfinite(gp); print(f"   {lb:14s} n {mk.sum():5d}: error {r(p0,mk):.3f} -> {r(p2,mk):.3f}")
