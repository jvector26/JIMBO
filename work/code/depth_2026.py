"""Apply the depth-tier minutes model to the 2026-27 projections (today's RealGM chart) and report changes."""
import gzip, glob, json, sys, re, numpy as np, pandas as pd
sys.argv = ["x"]; exec(open("/home/claude/code/depth_parse.py").read().split("ps = pd.read_parquet")[0])  # parse(), norm(), TEAMS
D = "/home/claude/data"
f = sorted(glob.glob("/home/claude/gh/live/2026/depth/realgm_*.html.gz"))[-1]
R = pd.DataFrame(parse(gzip.open(f).read().decode("latin-1"))); R["team"] = R.team_name.map(TEAMS)
R = R.sort_values(["team", "row", "col"]).drop_duplicates("slug"); R["key"] = R.slug.str.replace("-", " ").map(norm)
P = pd.read_csv(f"{D}/players_proj_2026.csv"); P["key"] = P.name.map(norm)
m = P.merge(R[["key", "team", "tier"]].rename(columns={"team": "dteam"}), on="key", how="left")
m["tier"] = np.where(m.tier.notna() & (m.dteam == m.team), m.tier, "N")
print(f, "| chart players", len(R), "| our rostered players on their team's chart:", (m.tier != "N").sum(), "of", len(m))
print("not on their team's chart:", m[m.tier == "N"].assign(other=m.dteam)[["team", "name", "other", "min_proj"]].round(0).to_string())
C = json.load(open(f"{D}/depth_tier.json"))["coef"]
m["want"] = [np.clip(C[t][0] + C[t][1] * x, 0, 3000) for t, x in zip(m.tier, m.min_model)]
def allocate(g, col):
    x = g[col].clip(0, 3000).copy(); ex = x.sum() - 19680
    if ex > 0:
        for i in x.sort_values().index:
            c = min(ex, x[i]); x[i] -= c; ex -= c
            if ex <= 0: break
    else:
        for _ in range(10):
            room = x < 3000; x[room] *= (19680 - x[~room].sum()) / x[room].sum(); x = x.clip(upper=3000)
    return x
m["min_new"] = m.groupby("team", group_keys=False).apply(lambda g: allocate(g, "want"))
m["d"] = m.min_new - m.min_proj
pace = json.load(open(f"{D}/pace_2025.json"))["pace"]
from math import erf, sqrt
def wins(col):
    T = (m.rating * m[col] / 3936).groupby(m.team).sum(); T -= T.mean()
    return 82 * (T * pace / 100).map(lambda x: 0.5 * (1 + erf(x / 12.8 / sqrt(2))))
W = pd.DataFrame({"old": wins("min_proj"), "new": wins("min_new")}); W["d"] = W.new - W.old
print(W.sort_values("d").round(1).T.to_string())
show = ["Harrison Barnes", "Tobias Harris", "Ty Jerome", "Bruce Brown", "James Wiseman", "Cameron Boozer", "Mikel Brown Jr.", "Brayden Burries"]
print(m[m.name.isin(show)][["team", "name", "tier", "rating", "min_proj", "min_new"]].round(0).to_string())
print("biggest moves:"); print(pd.concat([m.nsmallest(12, "d"), m.nlargest(8, "d")])[["team", "name", "tier", "rating", "min_proj", "min_new"]].round(0).to_string())
m.to_csv(f"{D}/depth_2026_preview.csv", index=False)
