"""Game-level minutes backtest (session 13). For every player-game that happened 2014-2025, features known BEFORE the
game: current nightly rule inputs (per TEAM game since joining, DNPs = 0) and played-only inputs (minutes when he played).
-> data/game_minutes.parquet"""
import numpy as np, pandas as pd
D = "/home/claude/data"
MC = pd.read_parquet(f"{D}/minutes_cal.parquet")
rows = []
for S in range(2014, 2026):
    g = pd.read_parquet(f"{D}/games_{S}.parquet")[["game", "date", "home_team", "away_team", "home_pts", "away_pts"]]
    pg = pd.read_parquet(f"{D}/player_games_{S}.parquet")[["game", "player", "team", "min", "started", "date"]]
    pg = pg[pg["min"] > 0]
    mc = MC[MC.season == S].set_index("player")
    # preseason priors: current = allocated season minutes / 82 (per team game); conditional = mpg when playing x age mult
    pre_tg = mc.min_proj / 82
    pre_wp = (mc.mpg_wp * mc.amult).where(mc.mpg_wp.notna(), mc.min_proj / 70)
    tg = pd.concat([g.assign(team=g.home_team), g.assign(team=g.away_team)]).sort_values(["date", "game"])
    tg["k"] = tg.groupby("team").cumcount()
    pg = pg.merge(tg[["game", "team", "k"]], on=["game", "team"])
    # played-only sequence across the season (all teams), in date order
    pg = pg.sort_values(["player", "date", "game"])
    gp = pg.groupby("player")
    pg["np"] = gp.cumcount()                                        # games played before this one
    cs = gp["min"].cumsum() - pg["min"]
    pg["std_p"] = cs / pg.np.replace(0, np.nan)
    pg["l5_p"] = gp["min"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    pg["l1_p"] = gp["min"].shift(1)
    pg["st5"] = gp["started"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    pg["days_rest"] = (pd.to_datetime(pg.date.astype(str)) - pd.to_datetime(gp.date.shift(1).astype("Int64").astype(str), errors="coerce")).dt.days
    # current nightly rule: per team game since joining the current team, DNPs count 0
    pg = pg.sort_values(["team", "player", "k"])
    gt = pg.groupby(["team", "player"])
    pg["join"] = gt.k.transform("min"); pg["n_tm"] = pg.k - pg["join"]
    out = []
    for (t, p), x in gt:
        ks = x.k.values; mins = x["min"].values; j = ks[0]
        arr = np.zeros(ks.max() + 1); arr[ks] = mins
        c = np.concatenate([[0], np.cumsum(arr)])
        n = ks - j
        std = np.where(n > 0, (c[ks] - c[j]) / np.maximum(n, 1), np.nan)
        lo = np.maximum(j, ks - 10); l10 = np.where(n > 0, (c[ks] - c[lo]) / np.maximum(ks - lo, 1), np.nan)
        out.append(pd.DataFrame({"idx": x.index, "std_tg": std, "l10_tg": l10}))
    o = pd.concat(out).set_index("idx"); pg = pg.join(o)
    pg["pre_tg"] = pg.player.map(pre_tg); pg["pre_wp"] = pg.player.map(pre_wp)
    pg["rating"] = pg.player.map(mc.rating)
    pg["season"] = S
    rows.append(pg); print(S, len(pg), "pre coverage %.3f" % pg.pre_tg.notna().mean(), flush=True)
G = pd.concat(rows, ignore_index=True)
G.to_parquet(f"{D}/game_minutes.parquet", index=False)
