"""T3 hole in the lineup: coverage of lead handlers / rim protectors per team-game (see PREREG.md). Output fund/hole_bt.parquet."""
import numpy as np, pandas as pd
D = "/home/claude/data"; FD = "/home/claude/fund"
gm = pd.read_parquet(f"{D}/games_all.parquet"); out = []
for N in range(2014, 2025):
    r = pd.read_parquet(f"{FD}/ratings_all_{N}.parquet").set_index("player"); el = r[r.min_N >= 500]
    pg = pd.read_parquet(f"{D}/player_games_{N+1}.parquet")[["game", "player", "team", "min", "date"]]; pg = pg[pg["min"] > 0].sort_values(["date", "game"])
    prev = pd.read_parquet(f"{D}/player_games_{N}.parquet"); prev = prev[prev["min"] > 0].groupby("player")["min"].mean()
    cs = pg.groupby("player")["min"].cumsum() - pg["min"]; cn = pg.groupby("player").cumcount()
    pg["trail"] = np.where(cn > 0, cs / cn.replace(0, np.nan), pg.player.map(prev)); pg["trail"] = pg.trail.fillna(0)
    tg = pg.drop_duplicates(["game", "team"])[["game", "team", "date"]]; tg["tgn"] = tg.groupby("team").cumcount()
    res = tg.set_index(["game", "team"])[["tgn"]]
    for comp, tag in (("playmaking", "h"), ("rim_protection", "r")):
        for q in (70, 80, 90):
            thr = np.percentile(el[comp], q); grp = set(r.index[r[comp] >= thr])
            m = pg[pg.player.isin(grp)]
            for mn, col in (("a", "min"), ("t", "trail")):
                s = m.groupby(["game", "team"])[col].sum().reindex(res.index).fillna(0)
                res[f"cov_{tag}{q}_{mn}"] = np.minimum(s, 48) / 48
                if q == 80: res[f"sum_{tag}{q}_{mn}"] = s / 48
    g = gm[gm.season == N + 1][["game", "home_team", "away_team"]]
    hk = pd.MultiIndex.from_arrays([g.game, g.home_team]); ak = pd.MultiIndex.from_arrays([g.game, g.away_team])
    H, A = res.reindex(hk), res.reindex(ak)
    dd = pd.DataFrame(H.to_numpy() - A.to_numpy(), columns=res.columns); dd["game"] = g.game.values
    dd["tgn_min"] = np.minimum(H.tgn.to_numpy(), A.tgn.to_numpy()); dd["lowh"] = (H["cov_h80_t"].to_numpy() < .5).astype(int) + (A["cov_h80_t"].to_numpy() < .5)
    dd["h_home"] = H["cov_h80_t"].to_numpy(); dd["r_home"] = H["cov_r80_t"].to_numpy(); out.append(dd.drop(columns="tgn"))
pd.concat(out).to_parquet(f"{FD}/hole_bt.parquet", index=False); print("ok")
