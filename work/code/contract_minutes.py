"""Pending test from contract_year.py: do scheduled-expiry (contract-year) vets play fewer minutes than JIMBO's
preseason minutes model predicts? Uses pre_P_{S} (player-level preseason projections)."""
import numpy as np, pandas as pd
D = "/home/claude/data"
P = pd.read_parquet(f"{D}/injury_return.parquet")
c = pd.read_parquet(f"{D}/contract_flags.parquet")[["player", "season", "cy", "rookie4"]]
P = P.merge(c, on=["player", "season"], how="inner"); P = P[P.season != 2019]
ps = pd.read_parquet(f"{D}/player_seasons.parquet"); last = ps.groupby(["season", "player"]).age.first()
P["age"] = [last.get((s-1, p), np.nan) + 1 for s, p in zip(P.season, P.player)]
P["vet_cy"] = P.cy & ~P.rookie4
P["bench"] = P.min_model < 1500
P["cy_bench"] = P.vet_cy & P.bench; P["cy_start"] = P.vet_cy & ~P.bench
def ols(d, y, xs):
    X = np.c_[np.ones(len(d)), d[xs].astype(float).to_numpy()]; yy = d[y].to_numpy()
    b = np.linalg.lstsq(X, yy, rcond=None)[0]; e = yy - X@b; XtXi = np.linalg.inv(X.T@X)
    M = sum(np.outer(X[i].T@e[i], X[i].T@e[i]) for i in d.groupby("player").indices.values())
    return {k: (round(v, 1), round(v/s, 1)) for k, v, s in zip(["c"]+xs, b, np.sqrt(np.diag(XtXi@M@XtXi)))}
print("n", len(P), "seasons", P.season.min(), "-", P.season.max(), "| vet CY", P.vet_cy.sum(), "bench CY", P.cy_bench.sum())
print("all:  ", ols(P, "act_min82", ["min_model", "vet_cy", "rookie4"]))
print("split:", ols(P, "act_min82", ["min_model", "cy_bench", "cy_start", "rookie4"]))
for nm, d in [("2014-18", P[P.season < 2019]), ("2020-22", P[P.season > 2019])]:
    print(nm, ols(d, "act_min82", ["min_model", "cy_bench", "cy_start", "rookie4"]))
for g in ["cy_bench", "cy_start"]:
    d = P[P[g]]; print(f"{g}: proj {d.min_model.mean():.0f} actual {d.act_min82.mean():.0f} ratio {d.act_min82.sum()/d.min_model.sum():.2f}")
d = P[P.bench & ~P.vet_cy]; print(f"other bench: ratio {d.act_min82.sum()/d.min_model.sum():.2f}")
