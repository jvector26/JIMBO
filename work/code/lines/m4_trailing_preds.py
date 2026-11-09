"""Pass 1 for the Elo test against the production in-season setup (session 44).
For each season S 2014..2025 and checkpoints every league-median team games, update the preseason ratings with
production M4 (box update dS 1, RAPM delta lam 6000, H0 2000; inseason_bt.py functions) on games before the
checkpoint date, then predict EVERY game of the season with actual minutes. Output data/elo_m4_preds_{S}.parquet:
cp, cp_date, game, pred (raw margin incl. HCA 2.5, as inseason_bt)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/code")
import inseason_bt as IB

D = IB.D
LAM, DS, H0 = 6000.0, 1.0, 2000.0


def run(S, step=1):
    N = S - 1
    pre = pd.read_parquet(f"{D}/pre_P_{S}.parquet").drop_duplicates("player").set_index("player")
    g = pd.read_parquet(f"{D}/games_{S}.parquet")[["game", "date", "home_team", "away_team", "home_pts", "away_pts"]]
    g["margin"] = g.home_pts - g.away_pts
    pg = pd.read_parquet(f"{D}/player_games_{S}.parquet"); pg = pg[pg["min"] > 0]
    st = pd.read_parquet(f"{D}/stints_{S}.parquet")
    st_date = st.game.map(g.set_index("game").date)
    home_of = g.set_index("game").home_team
    O_h, D_h, wo_h, wd_h = IB.history_box(N)
    allp = pd.Index(pg.player.unique())
    hb = pd.Series(IB.box_of(pd.concat([O_h.div(wo_h, axis=0), D_h.div(wd_h, axis=0)], axis=1).fillna(0)), index=O_h.index)
    pre_r = pre.rating.reindex(allp).fillna(IB.SLOPE * hb.reindex(allp)).fillna(IB.SLOPE * IB.UNRATED)
    tg = pd.concat([g[["game", "date", "home_team"]].rename(columns={"home_team": "t"}),
                    g[["game", "date", "away_team"]].rename(columns={"away_team": "t"})]).sort_values(["date", "game"])
    dates = np.sort(g.date.unique())
    med = pd.Series({d: tg[tg.date < d].groupby("t").size().reindex(tg.t.unique()).fillna(0).median() for d in dates})
    pg = pg.merge(g[["game", "home_team"]], on="game")
    pg["side"] = np.where(pg.team == pg.home_team, 1.0, -1.0)
    pg["wshare"] = 5 * pg["min"] / pg.groupby(["game", "team"])["min"].transform("sum") * pg.side

    # --- trailing minutes (research: lower bound, no knowledge of tonight's absences) ---
    gd = g.set_index("game").date
    pgt = pg.assign(date=pg.game.map(gd))
    rows = []
    for t_, dft in pgt.groupby("team"):
        gl = dft[["game", "date"]].drop_duplicates().sort_values(["date", "game"]).game.to_numpy()
        mm = dft.pivot_table(index="game", columns="player", values="min", aggfunc="sum").reindex(gl).fillna(0)
        tr = mm.rolling(5, min_periods=3).mean().shift(1)
        sd = dft.drop_duplicates("game").set_index("game").side.reindex(gl)
        for gi, gid in enumerate(gl):
            v = tr.iloc[gi]
            if v.isna().all() or v.sum() <= 0: continue
            v = v[v > 0]; rows.append(pd.DataFrame({"game": gid, "player": v.index, "ws": 5 * v.values / v.sum() * sd.iloc[gi]}))
    TR = pd.concat(rows, ignore_index=True)
    full = TR.groupby("game").ws.apply(lambda x: (x > 0).any() and (x < 0).any())
    TR = TR[TR.game.map(full)]
    s0 = pd.read_parquet(f"{D}/stints_{N}.parquet")
    pa0 = sum(s0[f"fg2a{z}"] + s0[f"fg3a{z}"] - s0[f"oreb{z}"] + s0[f"tov{z}"] + 0.44 * s0[f"fta{z}"] for z in "AB")
    pace0 = pa0.sum() / 2 / s0.game.nunique()
    cps = []
    for cp in range(0, 90, step):
        d = dates[0] if cp == 0 else (med.index[med >= cp][0] if (med >= cp).any() else None)
        if d is None or (cps and cps[-1][1] == d): continue
        cps.append((cp, d))
    out = []
    for cp, d in cps:
        past_g = g[g.date < d]; pst = st[st_date < d]
        if len(past_g) >= 60:
            pa = sum(pst[f"fg2a{z}"] + pst[f"fg3a{z}"] - pst[f"oreb{z}"] + pst[f"tov{z}"] + 0.44 * pst[f"fta{z}"] for z in "AB")
            pace = pa.sum() / 2 / pst.game.nunique()
        else:
            pace = pace0
        r = pre_r.copy()
        if len(past_g) >= 15:
            ppg = pg[pg.game.isin(past_g.game)]
            fnow = IB.partial_features(ppg, pst, None)
            ids = fnow.index
            Oh = O_h.reindex(ids).fillna(0); Dh = D_h.reindex(ids).fillna(0)
            woh = wo_h.reindex(ids).fillna(0); wdh = wd_h.reindex(ids).fillna(0)
            hist_box = pd.Series(IB.box_of(pd.concat([Oh.div(woh.replace(0, np.nan), axis=0),
                                                     Dh.div(wdh.replace(0, np.nan), axis=0)], axis=1).fillna(0)), index=ids)
            nohist = woh < 1
            hist_box[nohist] = (pre_r.reindex(ids) / IB.SLOPE)[nohist]
            woh = woh.where(~nohist, H0); wdh = wdh.where(~nohist, H0)
            cur_box = pd.Series(IB.box_of(fnow), index=ids)
            wc = DS * (fnow.poss_off + fnow.poss_def) / 2; wh = (woh + wdh) / 2
            r = r.add(IB.SLOPE * ((hist_box * wh + cur_box * wc) / (wh + wc) - hist_box), fill_value=0)
            A, B, y, w, hA, _ = IB.stint_rows(pst, home_of, g.set_index("game").date)
            prior = r.reindex(np.unique(np.r_[A.ravel(), B.ravel()])).fillna(IB.SLOPE * IB.UNRATED)
            dlt, _ = IB.rapm_delta(A, B, y, w, hA, prior, LAM)
            r = r.add(dlt, fill_value=0)
        s = (pg.wshare * pg.player.map(r).fillna(IB.SLOPE * IB.UNRATED)).groupby(pg.game).sum()
        s2 = (TR.ws * TR.player.map(r).fillna(IB.SLOPE * IB.UNRATED)).groupby(TR.game).sum()
        out.append(pd.DataFrame({"cp": cp, "cp_date": d, "game": g.game.to_numpy(),
                                 "pred": g.game.map(s).fillna(0).to_numpy() * pace / 100 + IB.HCA,
                                 "pred_tr": g.game.map(s2).to_numpy() * pace / 100 + IB.HCA}))
    res = pd.concat(out, ignore_index=True)
    res.to_parquet(f"/home/claude/lines/m4tr_{S}.parquet", index=False)
    return len(cps)


if __name__ == "__main__":
    import time
    for S in map(int, sys.argv[1:] or range(2014, 2026)):
        t = time.time(); n = run(S); print(S, n, "checkpoints", round(time.time() - t, 1), "s", flush=True)
