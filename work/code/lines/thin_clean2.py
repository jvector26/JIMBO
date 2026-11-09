import numpy as np, pandas as pd
src=open("thin_clean.py").read().split("M=600;")[0]; exec(src)
M=600; f=np.clip(1-T.cum.to_numpy()/M,0,1); o=(T.grp=="out").to_numpy(); h1=sg*T.w_s_h1.to_numpy()*f
def reg(mask,w):
    v=np.zeros(len(G)); np.add.at(v,T.gi.to_numpy()[mask],w[mask]); return v
def run(lab,cols):
    X=np.column_stack([base]+cols); p=loso(X,m); a=LB.pair(p,p0,m); Xm=np.c_[np.ones(m.sum()),X[m]]; bt=np.linalg.lstsq(Xm,y[m],rcond=None)[0]; rr=y[m]-Xm@bt; g=seas[m]; XtXi=np.linalg.inv(Xm.T@Xm)
    se=np.sqrt(np.diag(XtXi@sum(np.outer(Xm[g==s].T@rr[g==s],Xm[g==s].T@rr[g==s]) for s in np.unique(g))@XtXi)); k=base.shape[1]+1
    print(f"  {lab:58s} dRMSE {a[0]:+.4f} t {a[1]:+.1f} {a[2]}/{a[3]} halves {a[4]:+.4f}/{a[5]:+.4f} | rates "+" ".join(f"{c:+.2f} (t {c/s:+.1f})" for c,s in zip(bt[k:],se[k:])))
print("first-half minute shares, fade 600:")
run("not on opening roster only",[reg((f>0)&o,h1)])
# fill-in nights: his team is missing 20%+ of its recent minutes tonight
ab=pd.concat([H.absent,A.absent]); key=pd.MultiIndex.from_frame(pd.concat([H.reset_index()[["game","team"]],A.reset_index()[["game","team"]]]))
absent=pd.Series(ab.to_numpy(),index=key); ta=absent.reindex(pd.MultiIndex.from_frame(T[["game","team"]])).to_numpy()
fill=ta>=.20
run("everyone else: fill-in nights (20%+ regulars out) vs other",[reg((f>0)&o,h1),reg((f>0)&~o&fill,h1),reg((f>0)&~o&~fill&np.isfinite(ta),h1)])
# role expansion in general (any player): tonight's first-half share minus his trailing share, on fill-in nights -- is the stand-in over-rated regardless of season minutes?
W=pd.read_parquet("thin_pregame.parquet").set_index(["game","player"]).tw; tw=W.reindex(pd.MultiIndex.from_frame(T[["game","player"]])).fillna(0).to_numpy()
up=np.clip(T.w_s_h1.to_numpy()-tw,0,None)*sg; est=(T.cum.to_numpy()>=600)
run("role expansion (1st-half share above recent share): <600 season min vs 600+",[reg(~est,up),reg(est,up)])
