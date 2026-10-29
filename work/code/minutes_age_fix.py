"""Age multiplier on projected minutes (actual/projected by age), estimated leave-one-season-out, then re-run the
team allocation and re-score. Target is min_model (pre-allocation), so allocation re-balances as in production."""
import numpy as np, pandas as pd
from scipy.stats import norm
D = "/home/claude/data"
ps = pd.read_parquet(f"{D}/player_seasons.parquet")
EDGES = [0, 22, 25, 28, 31, 34, 50]; LABS = ["<=22", "23-25", "26-28", "29-31", "32-34", "35+"]
rows = []
for S in range(2014, 2026):
    P = pd.read_parquet(f"{D}/pre_P_{S}.parquet")
    last = ps[ps.season < S].sort_values("season").groupby("player").last()
    P["age"] = (P.player.map(last.age) + (S - P.player.map(last.season))).fillna(21)
    ng = {2019: 1056 * 2 / 30, 2020: 72}.get(S, 82)
    P["act_min"] = P.player.map(ps[ps.season == S].groupby("player")["min"].sum()).fillna(0) * 82 / ng
    P["bin"] = pd.cut(P.age, EDGES, labels=LABS).astype(str)
    rows.append(P)
P = pd.concat(rows, ignore_index=True)

def allocate(g, col):
    m = g[col].clip(upper=3000).copy(); excess = m.sum() - 19680
    if excess > 0:
        for i in m.sort_values().index:
            cut = min(excess, m[i]); m[i] -= cut; excess -= cut
            if excess <= 0: break
    else:
        for _ in range(10):
            room = m < 3000; m[room] *= (19680 - m[~room].sum()) / m[room].sum(); m = m.clip(upper=3000)
    return m

def mults(train):
    t = train[train.season != 2019].groupby("bin")[["act_min", "min_model"]].sum()
    return (t.act_min / t.min_model)

out = []
for S in range(2014, 2026):
    k = mults(P[P.season != S])                         # leave-one-season-out multipliers
    d = P[P.season == S].copy()
    d["min_adj"] = d.min_model * d.bin.map(k)
    d["min_proj2"] = d.groupby("team", group_keys=False).apply(lambda g: allocate(g, "min_adj"))
    T = pd.read_parquet(f"{D}/pre_T_{S}.parquet").set_index("team")
    pace = (T.margin / T.net).mean() * 100
    net = (d.rating * d.min_proj2 / 3936).groupby(d.team).sum(); net -= net.mean()
    T["model_wpct_agefix"] = norm.cdf(net * pace / 100 / 12.8)
    T.reset_index()[["season", "team", "model_wpct_agefix"]].pipe(out.append)
M = pd.concat(out)
print("minutes multipliers (all seasons):", mults(P).round(3).to_dict())
A = pd.read_parquet(f"{D}/preseason_backtest.parquet").merge(M, on=["season", "team"])
A.to_parquet(f"{D}/preseason_backtest.parquet", index=False)
