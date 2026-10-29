import numpy as np, pandas as pd
exec(open("defsub_score.py").read().split("H1, H2")[0])
H1, H2, ALL = seas <= 2019, seas >= 2020, np.ones(len(y), bool)
dr = B["f:d_dreb"].to_numpy(); rest = base - dr
print("post-hoc dreb multiplier (res_d fixed):")
for m in [0, .25, .5, .75, 1, 1.25]:
    print(f"  x{m:<5} LOSO {loso((rest + m*dr)[:, None]):.4f}")
for nm, msk in (("all", ALL), ("15-19", H1), ("20-25", H2)):
    b, t = coefs(np.c_[rest, dr], msk)
    print(f"free {nm}: rest {b[0]:.3f}  dreb {b[1]:.3f} (t {t[1]:.1f})  relative {b[1]/b[0]:.2f}")
# relative-weight test: t of (dreb coef - rest coef)
X = np.c_[rest, dr]
for nm, msk in (("all", ALL), ("15-19", H1), ("20-25", H2)):
    b, t = coefs(np.c_[base, dr], msk)   # coef on extra dreb beyond 1x
    print(f"extra-dreb {nm}: {b[1]:+.3f} (t {t[1]:+.1f})")
print("sd of game diffs: dreb", dr.std().round(3), " rest", rest.std().round(3))
