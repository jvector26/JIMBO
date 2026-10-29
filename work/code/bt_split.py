"""Game-level backtest diffs for the lineup residual split into offense (res_o) and defense (res_d)."""
import json, numpy as np, pandas as pd, impact as I
TEST = range(2014, 2025)
W = json.load(open(f"{I.D}/weights_bt.json")); Wo, Wd = pd.Series(W["Wo"]) * .5, pd.Series(W["Wd"]) * .5
pg = pd.concat([pd.read_parquet(f"{I.D}/player_games_{y}.parquet").assign(season=y) for y in range(2015, 2026)])
pg = pg[["season", "game", "player", "team", "min"]]
gm = pd.read_parquet(f"{I.D}/games_all.parquet")
out, rat = [], []
for N in TEST:
    rows = I.load_rows([N - 2, N - 1, N]); pace = (rows.groupby("game").poss.sum() / 2).mean()
    o = I.fit_window(N, Wo, Wd, lam=2000, rows=rows).set_index("player")
    rat.append(o[["res_o", "res_d", "def_rebounding", "rim_protection", "perimeter_def"]].assign(season=N).reset_index())
    nxt = pg[pg.season == N + 1]
    g = gm[gm.season == N + 1][["game", "home_team", "away_team"]].copy()
    hk = pd.MultiIndex.from_arrays([g.game, g.home_team]); ak = pd.MultiIndex.from_arrays([g.game, g.away_team])
    for c in ["res_o", "res_d"]:
        v = nxt.player.map(o[c]).fillna(0) * nxt["min"] / 48
        t = v.groupby([nxt.game, nxt.team]).sum()
        g[c] = (t.reindex(hk).to_numpy() - t.reindex(ak).to_numpy()) * pace / 100
    out.append(g); print("done", N + 1, flush=True)
pd.concat(out, ignore_index=True).to_parquet(f"{I.D}/bt_split.parquet", index=False)
pd.concat(rat, ignore_index=True).to_parquet(f"{I.D}/bt_split_ratings.parquet", index=False)
