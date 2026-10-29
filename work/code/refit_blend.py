"""Refit forecast blend (model + Vegas + roster age) on the preseason backtest; update blend.json.
Run after: preseason_backtest.py -> score_preseason.py -> minutes_age_fix.py. tau scaled from the 2026-09-18 fit."""
import numpy as np, pandas as pd, json
D = "/home/claude/data"
A = pd.read_parquet(f"{D}/preseason_backtest.parquet").merge(pd.read_parquet(f"{D}/direction_features.parquet")[["season", "team", "mw_age"]], on=["season", "team"])
A = A[A.season != 2019].copy(); A["y"] = 82*A.wpct-41; A["mod"] = 82*A.model_wpct_agefix-41; A["mkt"] = 82*A.mkt_wpct-41
A["age_c"] = A.mw_age - A.groupby("season").mw_age.transform("mean")
# model + Vegas weights constrained to sum to 1 (2026-09-19): y - mkt = a*(mod - mkt) + c*age. Free fit summed to 1.11
# (se 0.065, not significant) and pushed extreme teams beyond both inputs; constrained LOSO 8.161 vs free 8.141.
X = A[["mod", "mkt", "age_c"]].to_numpy(); y = A.y.to_numpy(); p = np.empty(len(A))
Z = np.c_[A["mod"] - A.mkt, A.age_c]; yz = (A.y - A.mkt).to_numpy()
for s in A.season.unique():
    tr = (A.season != s).to_numpy(); p[~tr] = A.mkt.to_numpy()[~tr] + Z[~tr] @ np.linalg.lstsq(Z[tr], yz[tr], rcond=None)[0]
a, c = np.linalg.lstsq(Z, yz, rcond=None)[0]; b = np.array([a, 1 - a, c]); r = lambda c: np.sqrt(np.mean((y-c)**2))
sd = np.sqrt(np.mean((y-p)**2) - 18.9); tau = 2.4439402242600163 * sd / 6.95
print(f"model {r(A['mod']):.3f} Vegas {r(A.mkt):.3f} blend LOSO {r(p):.3f} | w {b.round(3)} tau_blend {tau:.3f}")
B = json.load(open(f"{D}/blend.json")); tau *= B.get("tau_cal", 1.0)   # replay calibration (session 21): ranges cover ~80%
B.update(constrained=True, w_model=float(b[0]), w_vegas=float(b[1]), w_age=float(b[2]), tau_blend=float(tau))
json.dump(B, open(f"{D}/blend.json", "w"), indent=1)
