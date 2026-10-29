"""Production coefficients for the depth-tier minutes model: act82 = a_tier + b_tier * m0 (m0 = current recal'd
min_model incl age mult), fit on 2015-25 excl 2019-20/2020-21, pooled over early (~4 wk) and late (last pre-opener) snapshots."""
import json, numpy as np, pandas as pd, subprocess, sys
D = "/home/claude/data"
rows = []
for snap in ("early", "late"):
    P = pd.read_parquet(f"{D}/depth_test_{snap}.parquet"); P = P[(P.variant == "baseline") & ~P.season.isin([2019, 2020])]
    rows.append(P.assign(snap=snap))
P = pd.concat(rows)
coef = {}
for t in "SRLN":
    g = P[P.tier == t]; X = np.c_[np.ones(len(g)), g.m0]; b = np.linalg.lstsq(X, g.act82, rcond=None)[0]
    coef[t] = [round(float(b[0]), 2), round(float(b[1]), 4)]
    print(t, len(g) // 2, "a %.0f b %.3f" % tuple(b), "| e.g. m0 500/1000/1500/2000 ->", [round(b[0] + b[1] * x) for x in (500, 1000, 1500, 2000)])
json.dump({"coef": coef, "note": "want = a[tier] + b[tier]*m0, clip 0..3000; tier S/R/L from RealGM, N = not on the team's chart; "
           "m0 = recal'd min_model incl age mult. Fit code/depth_fit.py (2026-09-23)."}, open(f"{D}/depth_tier.json", "w"), indent=1)
