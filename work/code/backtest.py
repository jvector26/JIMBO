import sys, json, numpy as np, pandas as pd
import impact as I

TRAIN = range(1996, 2014)          # component weights learned only from these seasons
TEST = range(2014, 2025)           # ratings as of end of N predict season N+1 (2014-15 .. 2025-26)
UNRATED = -2.0                     # rookies / players with no data in window

Wo, Wd, hca = I.fit_weights(TRAIN, lam=4000)
json.dump({"Wo": Wo.to_dict(), "Wd": Wd.to_dict(), "hca": hca}, open("/home/claude/data/weights_bt.json", "w"))
print("weights fit; HCA per 100:", round(hca, 2), flush=True)

pg = pd.read_parquet(f"{I.D}/player_games_all.parquet")[["season", "game", "player", "team", "min"]]
gm = pd.read_parquet(f"{I.D}/games_all.parquet")
gm["margin"] = gm.home_pts - gm.away_pts

configs = {"full_l2000": (True, 2000), "full_l5000": (True, 5000), "full_l12000": (True, 12000),
           "rapm_l2000": (False, 2000), "rapm_l5000": (False, 5000), "rapm_l12000": (False, 12000)}
preds = []
for N in TEST:
    rows = I.load_rows([N - 2, N - 1, N])
    pace = (rows.groupby("game").poss.sum() / 2).mean()  # possessions per team per game (garbage-filtered)
    ratings = {}
    for name, (prior, lam) in configs.items():
        o = I.fit_window(N, Wo, Wd, lam=lam, use_prior=prior, rows=rows)
        ratings[name] = o.set_index("player")["total"]
        if name == "full_l5000":
            ratings["box_only"] = o.set_index("player")[list(I.COMPONENTS)].sum(axis=1)
    nxt = pg[pg.season == N + 1]
    g = gm[gm.season == N + 1][["game", "home_team", "away_team", "margin"]].copy()
    prev_mov = pd.concat([gm[gm.season == N].assign(t=lambda d: d.home_team, m=lambda d: d.margin),
                          gm[gm.season == N].assign(t=lambda d: d.away_team, m=lambda d: -d.margin)]).groupby("t").m.mean()
    g["pred_prev_mov"] = g.home_team.map(prev_mov).fillna(0) - g.away_team.map(prev_mov).fillna(0)
    for name, rt in ratings.items():
        v = nxt.player.map(rt).fillna(UNRATED) * nxt["min"] / 48
        team = v.groupby([nxt.game, nxt.team]).sum()
        h = team.reindex(pd.MultiIndex.from_arrays([g.game, g.home_team])).to_numpy()
        a = team.reindex(pd.MultiIndex.from_arrays([g.game, g.away_team])).to_numpy()
        g[f"pred_{name}"] = (h - a) * pace / 100
    g["season"] = N + 1
    preds.append(g)
    print("done", N + 1, flush=True)

P = pd.concat(preds, ignore_index=True)
P.to_parquet(f"{I.D}/backtest_preds.parquet", index=False)
