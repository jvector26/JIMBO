import numpy as np, pandas as pd, lines_bar as LB
from lines_bar import G,y,seas,SEAS,loso
W=pd.read_parquet("thin_pregame.parquet"); gi=pd.Series(np.arange(len(G)),index=G.game.to_numpy()); W=W[W.game.isin(gi.index)]; W["gi"]=W.game.map(gi).to_numpy()
b=LB.load(""); line=G.line.to_numpy(); F=pd.read_parquet("gap_features.parquet"); H=F[F.home].set_index("game"); A=F[~F.home].set_index("game")
b2b=(G.game.map(H.rest)==1).to_numpy(float)-(G.game.map(A.rest)==1).to_numpy(float)
m=np.isfinite(b["tr"]); base=np.c_[b["tr"],b["off"],b2b]; p0=loso(base,m); ml=m&np.isfinite(line)
r=lambda p,mk:np.sqrt(((y-p)**2)[mk].mean())
print(f"PREGAME harness (trailing minutes + offsets + B2B): games {m.sum()} base {r(p0,m):.4f} | with a line: JIMBO {r(p0,ml):.3f} line {np.sqrt(((y-line)**2)[ml].mean()):.3f}")
def reg(mask,w):
    v=np.zeros(len(G)); np.add.at(v,W.gi.to_numpy()[mask],(W.tsw.to_numpy()*w)[mask]); return v
for M in [300,450,600,900]:
    f=np.clip(1-W.cum.to_numpy()/M,0,1); o=W.out.to_numpy()
    for lab,cols in [("1 rate",[reg(f>0,f)]),("2 rates",[reg((f>0)&o,f),reg((f>0)&~o,f)])]:
        X=np.column_stack([base]+cols); p=loso(X,m); a=LB.pair(p,p0,m); co=np.linalg.lstsq(np.c_[np.ones(m.sum()),X[m]],y[m],rcond=None)[0][4:]
        print(f"  fade {M:4d} min, {lab:8s} dRMSE {a[0]:+.4f} t {a[1]:+.1f} {a[2]}/{a[3]} halves {a[4]:+.4f}/{a[5]:+.4f} | rates {np.round(co,2)} | gap to line {r(p0,ml)-np.sqrt(((y-line)**2)[ml].mean()):+.3f} -> {r(p,ml)-np.sqrt(((y-line)**2)[ml].mean()):+.3f}")
