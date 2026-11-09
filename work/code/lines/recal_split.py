import numpy as np, pandas as pd
exec(open("recal_fit.py").read().split("all_=np.arange")[0])
all_=np.arange(len(seas))
C2=["base","rk_col","rk_col_cs","rk_non_val","unr_open","unr_late","thin_open","rated_late"]
lab=["slope","rookie level col_a","college slope col_b","non-college scale","unrated on an opening roster","unrated, joined later (call-ups)","EXTRA for rated thin players (<2000 poss) on opening roster","EXTRA for rated players who joined later"]
def der(b): return np.r_[b[1],b[2:]/b[1]]
for nm,fn in [("all minutes (current method)",lambda i:est("full",C2,i)),("2SLS (first-half instrument)",lambda i:iv(C2,i))]:
    d=der(fn(all_)); bs=np.array([der(b) for b in boot(fn,150)]); print("\n"+nm)
    for n,v,s_ in zip(lab,d,bs.std(0)): print(f"   {n:60s} {v:+.2f} (se {s_:.2f})")
    for hl,mk in [("2015-19",seas<=2019),("2020-25",seas>=2020)]:
        dh=der(fn(np.flatnonzero(mk))); print(f"   {hl}: slope {dh[0]:.3f} | unrated open {dh[4]:+.2f} | late {dh[5]:+.2f} | thin extra {dh[6]:+.2f} | rated-late extra {dh[7]:+.2f}")
f=K["full"]; print("\nshare of minutes: unrated open %.2f%%, unrated late %.2f%%, thin open %.2f%%, rated late %.2f%%"%tuple(100*np.abs(f[c]).mean()/ (2*np.abs(f[['unr_open']]).mean().iloc[0]/np.abs(f['unr_open']).mean()) if False else 0 for c in ["unr_open","unr_late","thin_open","rated_late"]))
