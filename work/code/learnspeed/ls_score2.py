import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/code/learnspeed"); from ls_lib import *
from ls_preds import HF, TAUS, DLAMS, H0S
pd.set_option("display.width", 250)
N = next_frame(); y = N.margin.to_numpy(float); seas = N.season.to_numpy(); tgn = N.tgn.to_numpy()
STAGES = [(3, 10), (10, 25), (25, 50), (50, 70), (70, 99)]
GR = ["nohist", "thin", "young", "prime", "old"]
def show(nm, pv, pb, m):
    r = pair(pv, pb, y, seas, m); st = " ".join(f"{rmse(pv,y,m&(tgn>=lo)&(tgn<hi))-rmse(pb,y,m&(tgn>=lo)&(tgn<hi)):+.4f}" for lo, hi in STAGES)
    print(f"{nm:34s} {r[0]:+.4f} t {r[1]:+.1f} {r[2]}/{r[3]} halves {r[4]:+.4f}/{r[5]:+.4f} | stages {st}")
for mn in ["a", "t"]:
    c = lambda k: N[f"{mn}_{k}"].to_numpy(float)
    m = np.isfinite(c("pre")) & (tgn >= 3)
    base = c("pre") + c("bx_1.0") + c("rp_1.0_6000"); pb = loso(base, y, seas, m)
    print(f"\n===== minutes {mn} ===== base {rmse(pb,y,m):.4f}   (stages: 3-10, 10-25, 25-50, 50-70, 70+)")
    print("-- by player group: margin ~ pre + each group's update (slope > 1: that group is learned too slowly; < 1 too fast)")
    X = np.c_[c("pre"), np.column_stack([c(f"g_{k}") for k in GR])]
    r = creg(X, y, seas, m, ["pre"] + GR); r["t_vs_1"] = (r.coef - 1) / r.se; r["sd"] = [X[m][:, i].std() for i in range(X.shape[1])]; print(r.round(3).T.to_string())
    for lo, hi in [(3, 25), (25, 99)]:
        mm = m & (tgn >= lo) & (tgn < hi); r = creg(X, y, seas, mm, ["pre"] + GR); print(f"  games {lo}-{hi}: ", "  ".join(f"{k} {v.coef:.2f} ({v.se:.2f})" for k, v in r.iterrows()))
    show("free group slopes (LOSO)", loso(X, y, seas, m), pb, m)
    print("-- variants vs base")
    for f in HF: show(f"thin-history penalty x{f} (rh)", loso(c("pre") + c("bx_1.0") + c(f"rh_{f}"), y, seas, m), pb, m)
    for h in H0S: show(f"no-history box prior H0 {h}", loso(c("pre") + c(f"bh_{h}") + c("rp_1.0_6000"), y, seas, m), pb, m)
    for t in TAUS:
        for l in DLAMS: show(f"recency decay tau {t}d lam {l}", loso(c("pre") + c("bx_1.0") + c(f"rd_{t}_{l}"), y, seas, m), pb, m)
    show("no box update (rapm only, dS1 prior)", loso(c("pre") + c("rp_1.0_6000"), y, seas, m), pb, m)
    show("lam 3000 after game 50, else 6000", loso(c("pre") + c("bx_1.0") + np.where(tgn >= 50, c("rp_1.0_3000"), c("rp_1.0_6000")), y, seas, m), pb, m)
    show("lam 12000 to game 10, 6000, 3000 50+", loso(c("pre") + c("bx_1.0") + np.where(tgn >= 50, c("rp_1.0_3000"), np.where(tgn < 10, c("rp_1.0_12000"), c("rp_1.0_6000"))), y, seas, m), pb, m)
    show("free slopes pre/box/rapm (LOSO)", loso(np.c_[c("pre"), c("bx_1.0"), c("rp_1.0_6000")], y, seas, m), pb, m)
    show("avg of lam 3000 and 12000", loso(c("pre") + c("bx_1.0") + .5 * c("rp_1.0_3000") + .5 * c("rp_1.0_12000"), y, seas, m), pb, m)
