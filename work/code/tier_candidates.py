"""Players whose depth-chart tier disagrees with JIMBO's own minutes ranking, ranked by model-win swing.
Model-implied tier: m0 (min_model_nodepth) rank within team: 1-5 S, 6-9 R, 10+ L. Swing = team model wins if the
player's tier were switched to the model-implied one (rest of team re-allocated as in project.py)."""
import json, numpy as np, pandas as pd
from math import erf, sqrt
D = "/home/claude/data"
P = pd.read_csv(f"{D}/players_proj_2026.csv"); DT = json.load(open(f"{D}/depth_tier.json"))["coef"]
pace = json.load(open(f"{D}/pace_2025.json"))["pace"]
def allocate(m):
    m = m.clip(upper=3000).copy(); ex = m.sum() - 19680
    if ex > 0:
        for i in m.sort_values().index:
            c = min(ex, m[i]); m[i] -= c; ex -= c
            if ex <= 0: break
    else:
        for _ in range(10):
            room = m < 3000; m[room] *= (19680 - m[~room].sum()) / m[room].sum(); m = m.clip(upper=3000)
    return m
def team_net(g): return (g.rating * allocate(g.min_model)).sum() / 3936
base_net = P.groupby("team").apply(team_net); mu = base_net.mean()
W = lambda net: 82 * 0.5 * (1 + erf(net * pace / 100 / 12.8 / sqrt(2)))
P["rank_m0"] = P.groupby("team").min_model_nodepth.rank(ascending=False, method="first")
P["model_tier"] = np.select([P.rank_m0 <= 5, P.rank_m0 <= 9], ["S", "R"], "L")
rows = []
for i, r in P[P.depth_tier.isin(list(DT))].iterrows():
    g = P[P.team == r.team].copy()
    out = {"name": r["name"], "team": r.team, "age": r.age, "rating": round(r.rating, 2), "chart": r.depth_tier,
           "model_tier": r.model_tier, "m0": round(r.min_model_nodepth), "min_now": round(r.min_proj)}
    for t in ["S", "R", "L", "N"]:
        g2 = g.copy(); g2.loc[i, "min_model"] = float(np.clip(DT[t][0] + DT[t][1] * r.min_model_nodepth, 0, 3000))
        n = team_net(g2); new_mu = mu + (n - base_net[r.team]) / 30
        out[f"w_{t}"] = round(W(n - new_mu) - W(base_net[r.team] - mu), 2)
        out[f"min_{t}"] = round(allocate(g2.min_model)[i])
    rows.append(out)
C = pd.DataFrame(rows)
C["swing_model"] = [r[f"w_{r.model_tier}"] for _, r in C.iterrows()]
C.to_csv(f"{D}/tier_candidates.csv", index=False)
if __name__ == "__main__":
    pd.set_option("display.width", 250)
    M = C[C.chart != C.model_tier].copy(); M["abs"] = M.swing_model.abs()
    print(M.sort_values("abs", ascending=False).head(30).drop(columns="abs").to_string())
    print(C[C.name.str.contains("Jakuc")].to_string())
