"""Return from (nearly) full-season absence: do returners beat/miss their preseason projection (rating, minutes)?"""
import numpy as np, pandas as pd, json
D = "/home/claude/data"
ps = pd.read_parquet(f"{D}/player_seasons.parquet"); gm = pd.read_parquet(f"{D}/games_all.parquet")
ng = gm.groupby("season").size() * 2 / 30
W = json.load(open(f"{D}/weights_bt.json")); f = pd.read_parquet(f"{D}/features.parquet")
f["box"] = sum(f[c]*v for c, v in {**W["Wo"], **W["Wd"]}.items()) * 0.5
P = pd.concat([pd.read_parquet(f"{D}/pre_P_{S}.parquet") for S in range(2014, 2026)])
P = P[P.kind == "vet"].copy()
g = lambda s, col: ps.set_index(["season", "player"])[col].groupby(level=[0, 1]).sum()
GP, MIN = g(0, "gp"), g(0, "min")
look = lambda S, p, s: s.get((S, p), 0.0)
P["gpN"] = [look(S-1, p, GP) for S, p in zip(P.season, P.player)]
P["gpN1"] = [look(S-2, p, GP) for S, p in zip(P.season, P.player)]
P["mpgN1"] = [look(S-2, p, MIN) / max(look(S-2, p, GP), 1) for S, p in zip(P.season, P.player)]
P["availN"] = P.gpN / (P.season - 1).map(ng)
P["act_min"] = [look(S, p, MIN) for S, p in zip(P.season, P.player)]
P["act_min82"] = P.act_min * 82 / P.season.map(ng)
P["act_gp82"] = np.array([look(S, p, GP) for S, p in zip(P.season, P.player)]) * 82 / P.season.map(ng)
P["act_box"] = P.set_index(["season", "player"]).index.map(f.set_index(["season", "player"]).box.groupby(level=[0, 1]).first())
P["reg_before"] = (P.mpgN1 >= 20) & (P.gpN1 >= 30)
P["missed"] = P.reg_before & (P.availN < 0.2)       # (nearly) full season out
P["partial"] = P.reg_before & P.availN.between(0.2, 0.5)
P.to_parquet(f"{D}/injury_return.parquet", index=False)
print("returners (missed ~full season):", P.missed.sum(), "| partial-season group:", P.partial.sum())
print(P[P.missed].assign(name=lambda d: d.player.map(ps.drop_duplicates('player').set_index('player').name))
      [["season", "name", "team", "gpN", "mpgN1", "min_model", "act_min82", "rating", "act_box"]].round(1).to_string())
