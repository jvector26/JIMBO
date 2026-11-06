# How much each input moves 2026 player ratings: minutes-weighted SD of each contribution (players with 500+ min in 2024-25).
import numpy as np, pandas as pd, sys
sys.path.insert(0, "code")
R = pd.read_parquet("data/ratings_2025.parquet")
H = pd.read_csv("data/hustle_2026.csv").set_index("player")
m = 1 - (1/3)*np.clip(1 - R.poss_window/10000, 0, 1)
R["lin_o"] = 0.6*R.res_o*m; R["lin_d"] = 0.8*R.res_d*m
R["hu_lbr"] = 1.74*R.player.map(H.lbr).fillna(0); R["hu_c3"] = 0.253*R.player.map(H.c3).fillna(0)
R = R[R.min_2025 >= 500]; w = R.min_2025
F = [c for c in R.columns if c.startswith("f:")] + ["lin_o","lin_d","hu_lbr","hu_c3"]
def wsd(x): mu = np.average(x, weights=w); return np.sqrt(np.average((x-mu)**2, weights=w))
tot = R[F].sum(axis=1)
out = pd.DataFrame({"input": F, "sd": [wsd(R[c]) for c in F],
                    "corr_total": [np.corrcoef(R[c], tot)[0,1] for c in F]})
out["share_var"] = [np.cov(R[c], tot, aweights=w)[0,1]/np.cov(tot, tot, aweights=w)[0,1] for c in F]
print(len(R), "players; total sd", round(wsd(tot),2)); print(out.sort_values("sd").round(3).to_string(index=False))
out.to_csv("data/input_spread.csv", index=False)
