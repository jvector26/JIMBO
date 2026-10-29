"""Production-like game minutes backtest: every (team game, rostered player) incl. absences. Roster = appeared for the team
on or before and on or after that game. Absences are classified as a stand-in for the injury report (not in the data):
'out' if the absence run is 3+ straight games or the player is a regular (played-only mpg 20+), else 'dnp' (active,
coach's decision). -> data/game_minutes_active.parquet"""
import numpy as np, pandas as pd
D = "/home/claude/data"
MC = pd.read_parquet(f"{D}/minutes_cal.parquet")
rows = []
for S in range(2014, 2026):
    g = pd.read_parquet(f"{D}/games_{S}.parquet")[["game", "date", "home_team", "away_team"]]
    pg = pd.read_parquet(f"{D}/player_games_{S}.parquet")[["game", "player", "team", "min", "started"]]
    pg = pg[pg["min"] > 0]
    mc = MC[MC.season == S].set_index("player")
    pre_tg = mc.min_proj / 82; pre_wp = (mc.mpg_wp * mc.amult).where(mc.mpg_wp.notna(), mc.min_proj / 70)
    tg = pd.concat([g.assign(team=g.home_team), g.assign(team=g.away_team)]).sort_values(["date", "game"])
    tg["k"] = tg.groupby("team").cumcount()
    pg = pg.merge(tg[["game", "team", "k"]], on=["game", "team"])
    for t, x in pg.groupby("team"):
        games = tg[tg.team == t].game.values; K = len(games)
        for p, y in x.groupby("player"):
            k0, k1 = y.k.min(), y.k.max()
            mins = np.full(K, np.nan); mins[y.k.values] = y["min"].values
            st = np.zeros(K); st[y.k.values] = y.started.values
            ks = np.arange(k0, k1 + 1); m = mins[k0:k1 + 1]; played = ~np.isnan(m); m0 = np.nan_to_num(m)
            # run length of absence runs
            run = np.zeros(len(ks), int); i = 0
            while i < len(ks):
                if not played[i]:
                    j = i
                    while j < len(ks) and not played[j]: j += 1
                    run[i:j] = j - i; i = j
                else: i += 1
            c = np.concatenate([[0], np.cumsum(m0)]); cp = np.concatenate([[0], np.cumsum(played)])
            n = ks - k0
            std_tg = np.where(n > 0, c[n] / np.maximum(n, 1), np.nan)
            lo = np.maximum(0, n - 10); l10_tg = np.where(n > 0, (c[n] - c[lo]) / np.maximum(n - lo, 1), np.nan)
            npl = cp[n]; std_p = np.where(npl > 0, c[n] / np.maximum(npl, 1), np.nan)
            pm = m[played]; pidx = np.cumsum(played) - played   # index into played list of games before
            cpm = np.concatenate([[0], np.cumsum(pm)])
            l5_p = np.where(npl > 0, (cpm[npl] - cpm[np.maximum(npl - 5, 0)]) / np.maximum(np.minimum(npl, 5), 1), np.nan)
            rows.append(pd.DataFrame({"season": S, "team": t, "game": games[ks], "k": ks, "player": p, "min": m0, "played": played,
                                      "started": st[k0:k1 + 1], "run": run, "n_tm": n, "std_tg": std_tg, "l10_tg": l10_tg, "np": npl,
                                      "std_p": std_p, "l5_p": l5_p}))
    print(S, flush=True)
A = pd.concat(rows, ignore_index=True)
MCi = MC.set_index(["season", "player"])
key = pd.MultiIndex.from_arrays([A.season, A.player])
A["pre_tg"] = (MCi.min_proj / 82).reindex(key).values
A["pre_wp"] = (MCi.mpg_wp * MCi.amult).where(MCi.mpg_wp.notna(), MCi.min_proj / 70).reindex(key).values
A["rating"] = MCi.rating.reindex(key).values
A.to_parquet(f"{D}/game_minutes_active.parquet", index=False); print(len(A))
