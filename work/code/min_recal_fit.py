"""Season-minutes recalibration (session 13): projected minutes (incl. age multiplier, pre-allocation) -> expected actual,
piecewise linear with knots at 1000 and 2000, fit on minutes_cal.parquet (2014-25 excl. 2019-20, full-season absentees in).
LOSO: player minutes RMSE 597.7 -> 583.3, top-30 projected 2559 vs actual 2224 -> 2356 vs 2232, team RMSE 8.642 -> 8.639.
-> data/min_recal.json. Applied in project.py as a games (availability) factor; mpg when playing unchanged."""
import json, numpy as np, pandas as pd
M = pd.read_parquet("data/minutes_cal.parquet"); M = M[~M.season.isin([2019, 2020])]
y = M.mins * 82 / M.ng; x = M.min_model_age.values
X = np.c_[np.ones(len(x)), x, np.clip(x - 1000, 0, None), np.clip(x - 2000, 0, None)]
b = np.linalg.lstsq(X, y.values, rcond=None)[0]
json.dump({"coef": b.tolist(), "knots": [1000, 2000], "note": "min_new = c0 + c1*m + c2*max(m-1000,0) + c3*max(m-2000,0); m = min_model incl. age mult"}, open("data/min_recal.json", "w"), indent=1)
print(b.round(4))
