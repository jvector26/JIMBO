"""Does a tank season mislead player ratings? Player-level (age_check.py framework, production weights: dreb halved,
lineup 0.6 off / 0.8 def, slope). For each player-season N (500+ min in N and N+1):
  win   = decayed 3-season window rating (box + lineup window), raw (pre-slope)
  one   = season-N single-season rating (box_N + lineup_N)
  act   = next season measured rating (box + single-season lineup)
  T     = share of season-N minutes played for a tank team (from player_games per team)
Regress act on win, (one - win), age bins, season FE; add T and T*(one - win) [box and lineup parts separately].
T coef > 0: players coming off tank teams beat their projection (tank season drags ratings down).
T*(one-win) < 0: tank-season deviations are over-weighted (tank season should get less window weight).
Cluster SE by team-season of origin."""
import sys, json, numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, "/home/claude/code")
from features import COMPONENTS
D = "/home/claude/data"
F = [f for fs in COMPONENTS.values() for f in fs]; OFF = [f for f in F if f.startswith("o_")]
W = json.load(open(f"{D}/weights_prod.json")); Wa = {**W["Wo"], **W["Wd"]}
LWO, LWD = 0.6, 0.8
fe = pd.read_parquet(f"{D}/features.parquet"); fe = fe[fe["min"] > 0].copy()
for x in F:
    wt = fe.poss_off if x in OFF else fe.poss_def
    fe[x] = fe[x] * 0.5 * Wa[x]
    fe[x] -= (fe[x] * wt).groupby(fe.season).transform("sum") / wt.groupby(fe.season).transform("sum")
fe["box_o"] = fe[OFF].sum(axis=1); fe["box_d"] = fe[[f for f in F if f not in OFF]].sum(axis=1)
LH = pd.read_parquet(f"{D}/lineup_hist.parquet")
LH["lw"] = LWO * LH.win_o + LWD * LH.win_d; LH["l1"] = LWO * LH.one_o + LWD * LH.one_d
fe = fe.merge(LH[["season", "player", "lw", "l1"]], on=["season", "player"], how="left")
fe[["lw", "l1"]] = fe[["lw", "l1"]].fillna(0)

# team labels
TT = pd.read_parquet(f"{D}/tank_team_seasons.parquet")
TT["bad"] = TT.wpct <= 0.33
TT["late"] = (TT.m_late <= -9) & (TT["drop"] <= -4)
TT["tank"] = TT.bad | TT.late
g = pd.read_parquet(f"{D}/games_all.parquet")
idmap = pd.concat([g[["season", "home_team", "htm"]].set_axis(["season", "tid", "team"], axis=1),
                   g[["season", "away_team", "vtm"]].set_axis(["season", "tid", "team"], axis=1)]).drop_duplicates()
pg = pd.concat([pd.read_parquet(f"{D}/player_games_{y}.parquet", columns=["player", "team", "min", "season"]) for y in range(1996, 2026)])
pg = pg.rename(columns={"team": "tid"}).merge(idmap, on=["season", "tid"], how="left")
pt = pg.groupby(["season", "player", "team"])["min"].sum().reset_index()
pt = pt.merge(TT[["season", "team", "tank", "bad", "late", "wpct"]], on=["season", "team"], how="left")
for c in ["tank", "bad", "late"]: pt[c] = pt[c].fillna(False).astype(float) * pt["min"]
agg = pt.groupby(["season", "player"]).agg(pmin=("min", "sum"), tank=("tank", "sum"), bad=("bad", "sum"), late=("late", "sum"))
for c in ["tank", "bad", "late"]: agg[c] /= agg.pmin
prim = pt.sort_values("min").groupby(["season", "player"]).tail(1).set_index(["season", "player"])[["team", "wpct"]]
agg = agg.join(prim).reset_index()
lastteam = pd.concat([pd.read_parquet(f"{D}/player_games_{y}.parquet", columns=["player", "team", "season", "date"]) for y in range(1996, 2026)])
lastteam = lastteam.sort_values("date").groupby(["season", "player"]).tail(1).rename(columns={"team": "tid"}).merge(idmap, on=["season", "tid"])
lastteam = lastteam.set_index(["season", "player"]).team
firstteam = pd.concat([pd.read_parquet(f"{D}/player_games_{y}.parquet", columns=["player", "team", "season", "date"]) for y in range(1996, 2026)])
firstteam = firstteam.sort_values("date").groupby(["season", "player"]).head(1).rename(columns={"team": "tid"}).merge(idmap, on=["season", "tid"]).set_index(["season", "player"]).team

rows = []
for N in range(1996, 2025):
    cur = fe[fe.season == N]
    acc = {k: 0 for k in ["bo", "bd", "wo", "wd", "lw"]}
    bo = bd = wo = wd = 0
    for lag, dec in ((0, 1.0), (1, 0.6), (2, 0.3)):
        p = fe[fe.season == N - lag].set_index("player")
        bo = bo + (p.box_o * p.poss_off * dec).reindex(cur.player).fillna(0).to_numpy()
        bd = bd + (p.box_d * p.poss_def * dec).reindex(cur.player).fillna(0).to_numpy()
        wo = wo + (p.poss_off * dec).reindex(cur.player).fillna(0).to_numpy()
        wd = wd + (p.poss_def * dec).reindex(cur.player).fillna(0).to_numpy()
    nxt = fe[fe.season == N + 1].set_index("player")
    rows.append(pd.DataFrame({"season": N, "player": cur.player.to_numpy(), "name": cur.name.to_numpy(), "age_next": cur.age.to_numpy() + 1,
        "min_N": cur["min"].to_numpy(), "win_box": bo / np.maximum(wo, 1) + bd / np.maximum(wd, 1), "win_lin": cur.lw.to_numpy(),
        "one_box": (cur.box_o + cur.box_d).to_numpy(), "one_lin": cur.l1.to_numpy(),
        "act_box": nxt.box_o.add(nxt.box_d).reindex(cur.player).to_numpy(), "act_lin": nxt.l1.reindex(cur.player).to_numpy(),
        "min_next": nxt["min"].reindex(cur.player).to_numpy()}))
A = pd.concat(rows).dropna(subset=["act_box", "min_next"])
A = A[(A.min_N >= 500) & (A.min_next >= 500)].copy()
A = A.merge(agg, on=["season", "player"], how="left")
A["team_next"] = [firstteam.get((s + 1, p), None) for s, p in zip(A.season, A.player)]
A["team_lastN"] = [lastteam.get((s, p), None) for s, p in zip(A.season, A.player)]
A["moved"] = (A.team_next != A.team_lastN).astype(float)
A["win"] = A.win_box + A.win_lin; A["one"] = A.one_box + A.one_lin; A["act"] = A.act_box + A.act_lin
A["dbox"] = A.one_box - A.win_box; A["dlin"] = A.one_lin - A.win_lin
A["bin"] = pd.cut(A.age_next, [0, 21.99, 23.99, 25.99, 27.99, 29.99, 31.99, 33.99, 35.99, 99]).astype(str)
A["clu"] = A.season.astype(str) + A.team.astype(str)
A.to_parquet(f"{D}/tank_player.parquet")

def fit(d, extra, y="act", lab=""):
    X = pd.concat([d[["win", "dbox", "dlin"] + extra], pd.get_dummies(d.bin, drop_first=True, dtype=float),
                   pd.get_dummies(d.season, prefix="s", drop_first=True, dtype=float)], axis=1)
    X = sm.add_constant(X)
    m = sm.WLS(d[y], X, weights=d.min_next).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.clu)[0]})
    out = "  ".join(f"{e} {m.params[e]:+.3f} (t {m.tvalues[e]:+.1f})" for e in extra)
    print(f"{lab:28s} n={len(d):5d}  {out}")
    return m

if __name__ == "__main__":
    print("players with T>0.5:", (A.tank > .5).sum(), " team-seasons tank:", int(TT.tank.sum()), " bad:", int(TT.bad.sum()), " late:", int(TT.late.sum()))
    for lab_def in ["tank", "bad", "late"]:
        A["TK"] = A[lab_def]; A["Tdbox"] = A.TK * A.dbox; A["Tdlin"] = A.TK * A.dlin
        print(f"\n=== definition: {lab_def} ===")
        for per, d in [("1996-2024", A), ("1996-2010", A[A.season <= 2010]), ("2011-2024", A[A.season >= 2011])]:
            fit(d, ["TK", "Tdbox", "Tdlin"], lab=f"{per} act")
        fit(A, ["TK", "Tdbox", "Tdlin"], y="act_box", lab="all, target box only")
        fit(A, ["TK", "Tdbox", "Tdlin"], y="act_lin", lab="all, target lineup only")
        A["Tmv"] = A.TK * A.moved; A["mv"] = A.moved
        fit(A, ["TK", "Tmv", "mv", "Tdbox", "Tdlin"], lab="all, moved interaction")
