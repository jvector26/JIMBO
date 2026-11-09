import numpy as np, pandas as pd, lines_bar as LB
from lines_bar import G,y,seas,SEAS,loso
exec(open("thin_diag.py").read().split("def ols")[0].split("CATS=")[0])   # base, p4, res, m
Sg=np.load("thin_Sg.npy"); CATS=["rook_r1","newcomer","callup_nohist","late_add_hist","vet_thin","vet_mid"]
ix=lambda cats,exps:[CATS.index(c)*3+e for c in cats for e in exps]
s=lambda cats,exps:Sg[:,ix(cats,exps)].sum(1)
ALL=CATS; FR=["callup_nohist","late_add_hist"]; TH=["rook_r1","vet_thin","vet_mid"]
forms={"A  1 term: all thin groups, <200 min":[s(ALL,[0])],
 "B  2 terms: all thin <200, 200-600":[s(ALL,[0]),s(ALL,[1])],
 "C  3 terms: call-ups/late adds <600; others <200; others 200-600":[s(FR,[0,1]),s(TH,[0]),s(TH,[1])],
 "D  2 terms: call-ups/late adds (any); vet_thin (any)":[s(FR,[0,1,2]),s(["vet_thin"],[0,1,2])],
 "E  call-ups/late adds only (any exposure)":[s(FR,[0,1,2])],
 "F  E + vet_thin<600 + rookies<600":[s(FR,[0,1,2]),s(["vet_thin"],[0,1]),s(["rook_r1"],[0,1])]}
base=np.c_[b["act"],b["off"],b2b]; gap=np.sqrt(((y-p4)**2)[m].mean())-np.sqrt(((y-line)**2)[m].mean())
print(f"base error {np.sqrt(((y-p4)**2)[m].mean()):.4f}, gap to line {gap:+.3f}")
for nm,cols in forms.items():
    X=np.column_stack(cols); pv=loso(np.c_[base,X],m); a=LB.pair(pv,p4,m); co=np.linalg.lstsq(np.c_[np.ones(m.sum()),base[m],X[m]],y[m],rcond=None)[0][4:]
    print(f"{nm:66s} dRMSE {a[0]:+.4f} t {a[1]:+.1f} {a[2]}/{a[3]} halves {a[4]:+.4f}/{a[5]:+.4f} | coefs {np.round(co,2)} | closes {-a[0]/gap:.0%}")
# per-season dMSE for form C
X=np.column_stack(forms["C  3 terms: call-ups/late adds <600; others <200; others 200-600"]); pv=loso(np.c_[base,X],m)
d=pd.Series(((y-pv)**2-(y-p4)**2)[m]).groupby(seas[m]).mean(); print("form C dMSE by season:",d.round(2).to_dict())
# yardstick after form C, and early-season vs mid-season split of the exposure effect
pC=pv; F=pd.read_parquet("gap_features.parquet"); H=F[F.home].set_index("game"); A=F[~F.home].set_index("game")
g_=lambda c:(G.game.map(H[c]).to_numpy(),G.game.map(A[c]).to_numpy())
nw=np.fmax(*g_("newp")); aq=np.fmax(*g_("acq")); gp=(g_("gp")[0]+g_("gp")[1])/2; lowk=(nw>=.25)|(aq>=.10)
r=lambda p,mk:np.sqrt(((y-p)**2)[mk].mean())-np.sqrt(((y-line)**2)[mk].mean())
print("\ngap to line before -> after form C:  all %+.3f -> %+.3f | thin-history games %+.3f -> %+.3f | other games %+.3f -> %+.3f | games 70+ %+.3f -> %+.3f"%(r(p4,m),r(pC,m),r(p4,m&lowk),r(pC,m&lowk),r(p4,m&~lowk),r(pC,m&~lowk),r(p4,m&(gp>=70)),r(pC,m&(gp>=70))))
early=gp<15
for lb,cats in [("rookies r1",["rook_r1"]),("vet_thin",["vet_thin"]),("vet_mid",["vet_mid"]),("call-ups/late adds",FR)]:
    x=s(cats,[0]); out=[]
    for nm,mk in [("team games <15",m&early),("team games 15+",m&~early)]:
        xx=x[mk]; e=(y-p4)[mk]; bb=(xx*e).sum()/(xx*xx).sum(); rr=e-bb*xx; g=seas[mk]; se=np.sqrt(sum(((xx*rr)[g==q].sum())**2 for q in np.unique(g)))/(xx*xx).sum(); out.append(f"{nm}: {bb:+.2f} (t {bb/se:+.1f})")
    print(f"  first 200 min, {lb:20s}",*out)
