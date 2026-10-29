import sys, numpy as np, pandas as pd
P = pd.read_parquet(sys.argv[1])
res = []
for c in [c for c in P.columns if c.startswith("pred_")]:
    X = np.c_[np.ones(len(P)), P[c]]; b = np.linalg.lstsq(X, P.margin, rcond=None)[0]; e = P.margin - X @ b
    # also uncalibrated (slope forced to 1)
    e1 = P.margin - (P[c] + (P.margin - P[c]).mean())
    res.append((c[5:], np.sqrt((e**2).mean()), np.sqrt((e1**2).mean()), ((X @ b > 0) == (P.margin > 0)).mean(), b[1]))
print(pd.DataFrame(res, columns=["model", "RMSE_cal", "RMSE_raw", "winner_acc", "slope"]).sort_values("RMSE_cal").round(3).to_string(index=False))
