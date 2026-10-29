import numpy as np, pandas as pd
D="/home/claude/data"
S=pd.read_parquet(f"{D}/shift_bt.parquet")
B=pd.read_parquet(f"{D}/backtest_components.parquet")
F=[c for c in B.columns if c.startswith("f:")]
B["jimbo"]=B[F].sum(axis=1)+0.6*B.lineup_impact; B["j_unr"]=B.unrated
M=S.merge(B[["game","jimbo","j_unr"]],on="game")
def loso(cols):
    err=np.zeros(len(M)); 
    for s in M.season.unique():
        tr=M.season!=s; X=np.c_[np.ones(len(M)),M[cols]]
        b=np.linalg.lstsq(X[tr],M.margin[tr],rcond=None)[0]; err[~tr]=M.margin[~tr]-X[~tr]@b
    return np.sqrt((err**2).mean()), (np.sign(M.margin-err)==np.sign(M.margin)).mean(), err
res={}
cands=[c for c in S.columns if c.startswith(("sign_","pm_"))]
print("games",len(M))
print(f"{'home only':22s} {loso([])[0]:.3f}")
for c in cands:
    r,w,_=loso([c,"unrated"]); res[c]=r; print(f"{c:22s} {r:.3f}  winners {w:.3f}")
r,w,ej=loso(["jimbo","j_unr"]); print(f"{'JIMBO':22s} {r:.3f}  winners {w:.3f}")
best_s=min([c for c in cands if c.startswith('sign')],key=res.get); best_p=min([c for c in cands if c.startswith('pm')],key=res.get)
for extra in (best_s,best_p):
    r,w,_=loso(["jimbo","j_unr",extra]); X=np.c_[np.ones(len(M)),M[["jimbo","j_unr",extra]]]
    b=np.linalg.lstsq(X,M.margin,rcond=None)[0]; e=M.margin-X@b
    # season-clustered t for extra
    XtXi=np.linalg.inv(X.T@X); meat=sum(np.outer(X[M.season==s].T@e[M.season==s],X[M.season==s].T@e[M.season==s]) for s in M.season.unique())
    se=np.sqrt(np.diag(XtXi@meat@XtXi)); print(f"JIMBO + {extra}: {r:.3f}  coef t {b[3]/se[3]:.1f}")
for h,(a,z) in {"2015-19":(2015,2019),"2020-25":(2020,2025)}.items():
    k=M.season.between(a,z)
    print(h, {c:round(np.sqrt((loso([c]+(['unrated'] if c!='jimbo' else ['j_unr']))[2][k]**2).mean()),3) for c in (best_s,best_p,'jimbo')})
