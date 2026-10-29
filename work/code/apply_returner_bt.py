"""Apply the returner adjustment (returner.json) to existing pre_P_{S} and re-derive pre_T_{S} without rerunning
preseason_backtest.py (rating is additive; minutes unchanged, so this equals a full rerun). Idempotent."""
import numpy as np, pandas as pd, json
from scipy.stats import norm
D = "/home/claude/data"
RET = json.load(open(f"{D}/returner.json"))
ps = pd.read_parquet(f"{D}/player_seasons.parquet"); gm = pd.read_parquet(f"{D}/games_all.parquet")
ng = gm.groupby("season").size() * 2 / 30
GP = ps.groupby(["season", "player"]).gp.sum(); MN = ps.groupby(["season", "player"])["min"].sum()
for S in range(2014, 2026):
    P = pd.read_parquet(f"{D}/pre_P_{S}.parquet"); T = pd.read_parquet(f"{D}/pre_T_{S}.parquet")
    if "returner" in P: P["rating"] -= P.returner
    N = S - 1
    g1 = np.array([GP.get((N - 1, p), 0) for p in P.player]); m1 = np.array([MN.get((N - 1, p), 0) for p in P.player])
    g0 = np.array([GP.get((N, p), 0) for p in P.player])
    flag = (P.kind == "vet") & (g1 >= RET["min_gp"]) & (m1 / np.maximum(g1, 1) >= RET["min_mpg"]) & (g0 / ng[N] < RET["max_avail"])
    P["returner"] = np.where(flag, RET["adj"], 0.0); P["rating"] += P.returner
    pace = (T.margin / T.net).mean() * 100
    net = (P.rating * P.min_proj / 3936).groupby(P.team).sum(); net -= net.mean()
    T["net"] = T.team.map(net); T["margin"] = T.net * pace / 100; T["model_wpct"] = norm.cdf(T.margin / 12.8)
    P.to_parquet(f"{D}/pre_P_{S}.parquet", index=False); T.to_parquet(f"{D}/pre_T_{S}.parquet", index=False)
    print(S, "returners", int(flag.sum()), end=" | ")
