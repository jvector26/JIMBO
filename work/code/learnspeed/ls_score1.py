import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/code/learnspeed"); from ls_lib import *
pd.set_option("display.width", 250)
N = next_frame(refresh=True); y = N.margin.to_numpy(float); seas = N.season.to_numpy(); tgn = N.tgn.to_numpy()
print("games", len(N), "with trailing", N.t_pre.notna().sum(), "with line", N.line.notna().sum())
STAGES = [(0, 5), (5, 10), (10, 25), (25, 50), (50, 70), (70, 99)]
for mn in ["a", "t"]:
    c = lambda k: N[f"{mn}_{k}"].to_numpy(float)
    m = np.isfinite(c("pre")) & (tgn >= 3)
    base = c("pre") + c("bx_1.0") + c("rp_1.0_6000")
    pb = loso(base, y, seas, m); p0 = loso(c("pre"), y, seas, m)
    print(f"\n===== minutes {mn} ===== games {m.sum()}  preseason-only {rmse(p0,y,m):.4f}  base M4 {rmse(pb,y,m):.4f}")
    print("-- calibration: margin ~ pre + box delta + rapm delta (a slope above the 'pre' slope = learning too slowly)")
    X = np.c_[c("pre"), c("bx_1.0"), c("rp_1.0_6000")]
    r = creg(X, y, seas, m, ["pre", "box", "rapm"]); print("all      ", "  ".join(f"{k} {v.coef:.3f} ({v.se:.3f})" for k, v in r.iterrows()))
    for lo, hi in STAGES:
        mm = m & (tgn >= lo) & (tgn < hi); r = creg(X, y, seas, mm, ["pre", "box", "rapm"])
        tot = creg(np.c_[c("pre"), c("bx_1.0") + c("rp_1.0_6000")], y, seas, mm, ["pre", "delta"])
        print(f"g{lo:2d}-{hi:2d} n {mm.sum():5d}", "  ".join(f"{k} {v.coef:.3f} ({v.se:.3f})" for k, v in r.iterrows()), f" | total delta {tot.coef['delta']:.3f} ({tot.se['delta']:.3f})  sd(delta) {(c('bx_1.0')+c('rp_1.0_6000'))[mm].std():.2f}")
    print("-- grid: LOSO RMSE change vs base (dS 1, lam 6000); rows dS, cols lam")
    from ls_preds import DSS, LAMS
    tab = pd.DataFrame(index=DSS, columns=LAMS, dtype=float); tt = tab.copy()
    for a in DSS:
        for l in LAMS:
            pv = loso(c("pre") + c(f"bx_{a}") + c(f"rp_{a}_{l}"), y, seas, m); r = pair(pv, pb, y, seas, m); tab.loc[a, l] = r[0]; tt.loc[a, l] = r[1]
    print(tab.round(4).to_string()); print("t:"); print(tt.round(1).to_string())
    print("-- by stage, dS 1: RMSE change vs base by lam (calibration fit on all games LOSO, error measured in the stage)")
    rows = []
    for l in LAMS:
        pv = loso(c("pre") + c("bx_1.0") + c(f"rp_1.0_{l}"), y, seas, m)
        rows.append([rmse(pv, y, m & (tgn >= lo) & (tgn < hi)) - rmse(pb, y, m & (tgn >= lo) & (tgn < hi)) for lo, hi in STAGES])
    print(pd.DataFrame(rows, index=LAMS, columns=[f"{a}-{b}" for a, b in STAGES]).round(4).to_string())
    print("-- by stage, lam 6000: by dS")
    rows = []
    for a in DSS:
        pv = loso(c("pre") + c(f"bx_{a}") + c(f"rp_{a}_6000"), y, seas, m)
        rows.append([rmse(pv, y, m & (tgn >= lo) & (tgn < hi)) - rmse(pb, y, m & (tgn >= lo) & (tgn < hi)) for lo, hi in STAGES])
    print(pd.DataFrame(rows, index=DSS, columns=[f"{a}-{b}" for a, b in STAGES]).round(4).to_string())
