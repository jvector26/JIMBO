"""Where does JIMBO's information beyond the line come from? Blend LOSO RMSE (margin ~ line + pieces), trailing and actual minutes."""
import numpy as np, pandas as pd, lines_bar as LB
from lines_bar import *
base=LB.load(""); line=G.line.to_numpy(); elo=points_elo()
pre={"act":np.full(len(G),np.nan),"tr":np.full(len(G),np.nan)}
for S in SEAS:
    t=pd.read_parquet(f"{LD}/m4tr_{S}.parquet"); t=t[t.cp==0].set_index("game"); idx=np.flatnonzero(seas==S); gm=G.game.to_numpy()[idx]
    pre["act"][idx]=t.pred.reindex(gm).to_numpy(); pre["tr"][idx]=t.pred_tr.reindex(gm).to_numpy()
m=np.isfinite(line)&np.isfinite(base["tr"])&np.isfinite(pre["tr"]); r=lambda p:np.sqrt(((y-p)[m]**2).mean())
def tt(pv,pb):
    a=LB.pair(pv,pb,m); return f"{a[0]:+.4f} (t {a[1]:+.1f}, {a[2]}/{a[3]}, halves {a[4]:+.4f}/{a[5]:+.4f})"
lo=loso(line,m); print("games",m.sum(),"line alone",round(r(lo),4))
for mins in ["tr","act"]:
    a=loso(np.c_[line,pre[mins]],m); b=loso(np.c_[line,base[mins]],m); c=loso(np.c_[line,base[mins],base["off"]],m); o=loso(np.c_[line,base["off"]],m)
    print(f"-- minutes {mins}")
    print("  line + preseason ratings      ",round(r(a),4),"vs line",tt(a,lo))
    print("  line + in-season ratings (M4) ",round(r(b),4),"vs preseason step",tt(b,a))
    print("  line + M4 + team offsets      ",round(r(c),4),"vs M4 step",tt(c,b))
    print("  line + team offsets only      ",round(r(o),4),"vs line",tt(o,lo))
    print("  standalone (no line): preseason",round(r(loso(pre[mins],m)),4),"M4",round(r(loso(base[mins],m)),4),"M4+off",round(r(loso(np.c_[base[mins],base['off']],m)),4))
ta=loso(np.c_[line,base["tr"],base["off"]],m); aa=loso(np.c_[line,base["act"],base["off"]],m); print("knowing who played, with the line in:",tt(aa,ta))
ta=loso(np.c_[base["tr"],base["off"]],m); aa=loso(np.c_[base["act"],base["off"]],m); print("knowing who played, no line:",tt(aa,ta))
# by line source
for nm,mm in [("true close 2015-21",m&(seas<=2021)&np.isfinite(G.close.to_numpy())),("SBR 2022-25",m&(seas>=2022))]:
    lo2=loso(line,mm); c2=loso(np.c_[line,base["tr"],base["off"]],mm); rr=lambda p:np.sqrt(((y-p)[mm]**2).mean()); print(nm,"line",round(rr(lo2),4),"blend tr",round(rr(c2),4),round(rr(c2)-rr(lo2),4))
