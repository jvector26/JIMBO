"""Fit the calibration (slope, unrated value, rookie mapping) on each share kind, and by 2SLS (full-game shares instrumented with
first-half shares: removes score-driven minutes without the attenuation of using a half-game measure directly)."""
import numpy as np, pandas as pd, json
A=pd.read_parquet("recal_design.parquet"); RV=json.load(open("/home/claude/data/rookie_value.json")); cal=pd.read_csv("/home/claude/data/calib.csv",index_col=0).iloc[:,0]
C=["base","rk_col","rk_col_cs","rk_non_val","other_unrated"]; AG=[c for c in A.columns if c.startswith("A:") and c!="A:26-27"]
K={k:A[A.kind==k].sort_values("game").reset_index(drop=True) for k in ["full","ng","h1"]}
seas=K["full"].season.to_numpy(); SS=np.unique(seas)
def ols(X,y): return np.linalg.lstsq(X,y,rcond=None)[0]
def est(kind,cols,idx,age_in_y=True):
    d=K[kind].iloc[idx]; y=d.margin.to_numpy()-(d.age_off.to_numpy() if age_in_y else 0); return ols(np.c_[np.ones(len(d)),d[cols].to_numpy()],y)
def iv(cols,idx,age_in_y=True):
    f=K["full"].iloc[idx]; h=K["h1"].iloc[idx]; Z=np.c_[np.ones(len(h)),h[cols].to_numpy()]; Xf=f[cols].to_numpy(); Xhat=Z@ols(Z,Xf)
    y=f.margin.to_numpy()-(f.age_off.to_numpy() if age_in_y else 0); return ols(np.c_[np.ones(len(f)),Xhat],y)
def boot(fn,n=200):
    rng=np.random.default_rng(0); out=[]
    for _ in range(n):
        idx=np.concatenate([np.flatnonzero(seas==s) for s in rng.choice(SS,len(SS))]); out.append(fn(idx))
    return np.array(out)
def derived(b): return np.array([b[1],b[5]/b[1],b[2]/b[1],b[3]/b[1],b[4]/b[1]])
all_=np.arange(len(seas)); names=["slope","unrated value","college rookie level (col_a)","college score slope (col_b)","non-college rookie scale"]
print("current production: slope %.3f  unrated %.2f  col_a %.2f  col_b %.2f  (non-college scale 1 by construction)"%(cal["slope"],cal["unrated"],RV["col_a"],RV["col_b"]))
res={}
for lab,fn in [("all minutes (current method)",lambda i:est("full",C,i)),("non-garbage minutes",lambda i:est("ng",C,i)),("first-half minutes, direct",lambda i:est("h1",C,i)),("2SLS: full minutes, first-half instrument",lambda i:iv(C,i))]:
    d=derived(fn(all_)); bs=np.array([derived(b) for b in boot(fn)]); res[lab]=(d,bs)
    print(f"\n{lab}"); [print(f"   {n:30s} {v:+.3f}  (se {s:.3f})") for n,v,s in zip(names,d,bs.std(0))]
d0,b0=res["all minutes (current method)"]; d1,b1=res["2SLS: full minutes, first-half instrument"]; dd=b1-b0
print("\n2SLS minus current method (paired bootstrap):"); [print(f"   {n:30s} {a-b:+.3f}  (se {s:.3f}, t {(a-b)/s:+.1f})") for n,a,b,s in zip(names,d1,d0,dd.std(0))]
# halves
for lab,mk in [("2015-19",seas<=2019),("2020-25",seas>=2020)]:
    i=np.flatnonzero(mk); print(f"{lab}: current method slope {derived(est('full',C,i))[0]:.3f} unrated {derived(est('full',C,i))[1]:+.2f} col_a {derived(est('full',C,i))[2]:+.2f} | 2SLS slope {derived(iv(C,i))[0]:.3f} unrated {derived(iv(C,i))[1]:+.2f} col_a {derived(iv(C,i))[2]:+.2f}")
# game-level age residuals on top of the player-level curve (should be ~0 if the curve is right): full vs 2SLS
CA=C+AG; f_=est("full",CA,all_); v_=iv(CA,all_); bf=boot(lambda i:est("full",CA,i),100); bv=boot(lambda i:iv(CA,i),100)
print("\nage-bin residuals after the production age curve (pts/100, vs 26-27):  current method | 2SLS")
for j,c in enumerate(AG): print(f"   {c[2:]:6s} {f_[6+j]/f_[1]:+.2f} (se {(bf[:,6+j]/bf[:,1]).std():.2f}) | {v_[6+j]/v_[1]:+.2f} (se {(bv[:,6+j]/bv[:,1]).std():.2f})")
json.dump({"names":names,"full":d0.tolist(),"iv":d1.tolist()},open("recal_result.json","w"))
