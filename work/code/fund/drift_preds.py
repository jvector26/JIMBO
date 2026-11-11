"""T1 WITHIN-SEASON DRIFT (2026-11-11): slim copy of learnspeed/ls_preds.py (production cell only) plus exposure
columns e_r1 / e_r2 / e_y3 / e_old = minute-share exposure (x5 x pace/100) of player groups, see fund/PREREG.md.
Output /home/claude/fund/ls/ls_{S}.parquet.
--- original docstring: LEARNING-SPEED TEST, pass 1 (session 48, 2026-11-10). For each season S and each league-median team-game checkpoint, rebuild the
in-season rating update (M4: box update + RAPM delta) on games before the checkpoint for a GRID of learning speeds, and store the
game-level pieces for every later game, with actual and trailing-5-game (pregame-knowable) minute shares.
Pieces (points of home margin, before HCA): pre | bx_{dS} box delta | bh_{H0} box delta with another no-history prior weight |
rp_{dS}_{lam} RAPM delta on the dS prior | rh_{f} RAPM delta with a per-player penalty lam*(f+(1-f)*min(hist poss/5000,1)) |
rd_{tau}_{lam} RAPM delta with stint weights decayed exp(-days/tau) | g_{group} base (dS 1, lam 6000) total delta by player group.
Output /home/claude/ls/ls_{S}.parquet. Usage: python3 ls_preds.py 2014 2015 ..."""
import sys, time, numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, "/home/claude/code")
import inseason_bt as IB
D = IB.D; OUT = "/home/claude/fund/ls"; import os; os.makedirs(OUT, exist_ok=True)
FIRST = pd.read_parquet(f"{D}/features.parquet").groupby("player").season.min()
DSS = [1.0]; LAMS = [6000]; H0S = []
HF = []; TAUS = []; DLAMS = []
EG = ["r1", "r2", "y3", "old"]
BD = pd.read_parquet(f"{D}/birthdates.parquet").set_index("player").birthdate


def run(S):
    N = S - 1
    pre = pd.read_parquet(f"{D}/pre_P_{S}.parquet").drop_duplicates("player").set_index("player")
    g = pd.read_parquet(f"{D}/games_{S}.parquet")[["game", "date", "home_team", "away_team", "home_pts", "away_pts"]].sort_values(["date", "game"]).reset_index(drop=True)
    pg = pd.read_parquet(f"{D}/player_games_{S}.parquet"); pg = pg[pg["min"] > 0]
    pg["player"] = pg.player.astype(np.int64); pg["game"] = pg.game.astype(np.int64); pg["team"] = pg.team.astype(np.int64)
    st = pd.read_parquet(f"{D}/stints_{S}.parquet")
    gdate = g.set_index("game").date; st_date = st.game.map(gdate).to_numpy(); home_of = g.set_index("game").home_team
    O_h, D_h, wo_h, wd_h = IB.history_box(N)
    allp = pd.Index(np.sort(pg.player.unique()))
    hb = pd.Series(IB.box_of(pd.concat([O_h.div(wo_h, axis=0), D_h.div(wd_h, axis=0)], axis=1).fillna(0)), index=O_h.index)
    pre_r = pre.rating.reindex(allp).fillna(IB.SLOPE * hb.reindex(allp)).fillna(IB.SLOPE * IB.UNRATED)
    wh_all = ((wo_h + wd_h) / 2).reindex(allp).fillna(0)
    d0 = pd.Timestamp(f"{S}-10-20"); age = ((d0 - BD.reindex(allp)).dt.days / 365.25)
    grp = np.where(wh_all < 1, "nohist", np.where(wh_all < 3000, "thin", np.where(age < 24, "young", np.where(age < 30, "prime", "old"))))
    grp = pd.Series(grp, index=allp)
    first = FIRST.reindex(allp); a_ = age.to_numpy(); nh = (wh_all < 1).to_numpy()
    EX = {"r1": nh & (a_ < 25), "r2": (~nh) & (first.to_numpy() == N) & (a_ < 25), "y3": (~nh) & (first.to_numpy() < N) & (a_ < 23), "old": (~nh) & (a_ >= 32)}
    tg = pd.concat([g[["game", "date", "home_team"]].rename(columns={"home_team": "t"}),
                    g[["game", "date", "away_team"]].rename(columns={"away_team": "t"})]).sort_values(["date", "game"])
    dates = np.sort(g.date.unique())
    med = pd.Series({d: tg[tg.date < d].groupby("t").size().reindex(tg.t.unique()).fillna(0).median() for d in dates})
    pg = pg.merge(g[["game", "home_team"]], on="game")
    pg["side"] = np.where(pg.team == pg.home_team, 1.0, -1.0)
    pg["wshare"] = 5 * pg["min"] / pg.groupby(["game", "team"])["min"].transform("sum") * pg.side
    gi = pd.Index(g.game); pi = allp
    Gact = sp.csr_matrix((pg.wshare.to_numpy(), (gi.get_indexer(pg.game), pi.get_indexer(pg.player))), shape=(len(gi), len(pi)))
    pgt = pg.assign(date=pg.game.map(gdate)); rows = []
    for t_, dft in pgt.groupby("team"):
        gl = dft[["game", "date"]].drop_duplicates().sort_values(["date", "game"]).game.to_numpy()
        mm = dft.pivot_table(index="game", columns="player", values="min", aggfunc="sum").reindex(gl).fillna(0)
        tr = mm.rolling(5, min_periods=3).mean().shift(1)
        sd = dft.drop_duplicates("game").set_index("game").side.reindex(gl)
        for k, gid in enumerate(gl):
            v = tr.iloc[k]
            if v.isna().all() or v.sum() <= 0: continue
            v = v[v > 0]; rows.append(pd.DataFrame({"game": gid, "player": v.index, "ws": 5 * v.values / v.sum() * sd.iloc[k]}))
    TR = pd.concat(rows, ignore_index=True)
    full = TR.groupby("game").ws.apply(lambda x: (x > 0).any() and (x < 0).any()); TR = TR[TR.game.map(full)]
    Gtr = sp.csr_matrix((TR.ws.to_numpy(), (gi.get_indexer(TR.game), pi.get_indexer(TR.player))), shape=(len(gi), len(pi)))
    has_tr = np.zeros(len(gi), bool); has_tr[gi.get_indexer(TR.game.unique())] = True
    s0 = pd.read_parquet(f"{D}/stints_{N}.parquet")
    pace0 = sum(s0[f"fg2a{z}"] + s0[f"fg3a{z}"] - s0[f"oreb{z}"] + s0[f"tov{z}"] + 0.44 * s0[f"fta{z}"] for z in "AB").sum() / 2 / s0.game.nunique()
    cps = []
    for cp in range(0, 90):
        d = dates[0] if cp == 0 else (med.index[med >= cp][0] if (med >= cp).any() else None)
        if d is None or (cps and cps[-1][1] == d): continue
        cps.append((cp, d))
    gd = g.date.to_numpy(); out = []
    for ci, (cp, d) in enumerate(cps):
        past_g = g[g.date < d]; pst = st[st_date < d]
        if len(past_g) >= 60:
            pace = sum(pst[f"fg2a{z}"] + pst[f"fg3a{z}"] - pst[f"oreb{z}"] + pst[f"tov{z}"] + 0.44 * pst[f"fta{z}"] for z in "AB").sum() / 2 / pst.game.nunique()
        else: pace = pace0
        V = {"pre": pre_r.to_numpy()}
        zero = np.zeros(len(pi))
        keys = [f"bx_{a}" for a in DSS] + [f"bh_{h}" for h in H0S] + [f"rp_{a}_{l}" for a in DSS for l in LAMS] + [f"rh_{f}" for f in HF] + \
               [f"rd_{t}_{l}" for t in TAUS for l in DLAMS] + [f"g_{k}" for k in ["nohist", "thin", "young", "prime", "old"]]
        keys = keys + [f"e_{k}" for k in EG]
        for k in keys: V[k] = zero
        for k in EG: V[f"e_{k}"] = EX[k].astype(float)
        if len(past_g) >= 15:
            ppg = pg[pg.game.isin(past_g.game)]
            fnow = IB.partial_features(ppg, pst, None); ids = fnow.index
            Oh = O_h.reindex(ids).fillna(0); Dh = D_h.reindex(ids).fillna(0); woh = wo_h.reindex(ids).fillna(0); wdh = wd_h.reindex(ids).fillna(0)
            hist_box = pd.Series(IB.box_of(pd.concat([Oh.div(woh.replace(0, np.nan), axis=0), Dh.div(wdh.replace(0, np.nan), axis=0)], axis=1).fillna(0)), index=ids)
            nohist = woh < 1; hist_box[nohist] = (pre_r.reindex(ids) / IB.SLOPE)[nohist]
            cur_box = pd.Series(IB.box_of(fnow), index=ids); cposs = (fnow.poss_off + fnow.poss_def) / 2
            def boxd(dS, H0):
                wo2 = woh.where(~nohist, H0); wd2 = wdh.where(~nohist, H0); wc = dS * cposs; wh = (wo2 + wd2) / 2
                return (IB.SLOPE * ((hist_box * wh + cur_box * wc) / (wh + wc) - hist_box)).reindex(pi).fillna(0).to_numpy()
            for a in DSS: V[f"bx_{a}"] = boxd(a, 2000.0)
            for h in H0S: V[f"bh_{h}"] = boxd(1.0, float(h))
            A, B, y, w, hA, dt = IB.stint_rows(pst, home_of, gdate)
            n = len(y); rws = np.repeat(np.arange(n), 5)
            X = sp.csr_matrix((np.r_[np.ones(5 * n), -np.ones(5 * n)], (np.r_[rws, rws], np.r_[pi.get_indexer(A.ravel()), pi.get_indexer(B.ravel())])), shape=(n, len(pi)))
            X = sp.hstack([X, sp.csr_matrix(hA[:, None])]).tocsr(); P = len(pi)
            def solver(wts):
                Xw = X.multiply(wts[:, None]).tocsr(); M = (X.T @ Xw).toarray(); return M, Xw
            def solve(M, Xw, prior, pen):
                r = y - X[:, :P] @ prior; b = Xw.T @ r
                return np.linalg.solve(M + np.diag(np.r_[pen, 1e-6]), b)[:P]
            M, Xw = solver(w)
            for a in DSS:
                prior = V["pre"] + V[f"bx_{a}"]
                for l in LAMS: V[f"rp_{a}_{l}"] = solve(M, Xw, prior, np.full(P, float(l)))
            prior1 = V["pre"] + V["bx_1.0"]
            for f in HF: V[f"rh_{f}"] = solve(M, Xw, prior1, 6000.0 * (f + (1 - f) * np.minimum(wh_all.to_numpy() / 5000, 1)))
            days = (pd.Timestamp(str(d)) - pd.DatetimeIndex(dt)).days.to_numpy().astype(float)
            for t in TAUS:
                M2, Xw2 = solver(w * np.exp(-days / t))
                for l in DLAMS: V[f"rd_{t}_{l}"] = solve(M2, Xw2, prior1, np.full(P, float(l)))
            tot = V["bx_1.0"] + V["rp_1.0_6000"]
            for k in ["nohist", "thin", "young", "prime", "old"]: V[f"g_{k}"] = np.where(grp.to_numpy() == k, tot, 0.0)
        fut = np.flatnonzero(gd >= d)
        Rm = np.column_stack([V[k] for k in ["pre"] + keys]).astype(np.float64)
        a_ = (Gact[fut] @ Rm) * pace / 100; t_ = (Gtr[fut] @ Rm) * pace / 100; t_[~has_tr[fut]] = np.nan
        df = pd.DataFrame(np.c_[a_, t_].astype(np.float32), columns=[f"a_{k}" for k in ["pre"] + keys] + [f"t_{k}" for k in ["pre"] + keys])
        df.insert(0, "game", gi[fut].to_numpy()); df.insert(0, "cp_date", d); df.insert(0, "cp", cp)
        out.append(df)
    pd.concat(out, ignore_index=True).to_parquet(f"{OUT}/ls_{S}.parquet", index=False)
    return len(cps)


if __name__ == "__main__":
    for S in map(int, sys.argv[1:]):
        t = time.time(); n = run(S); print(S, n, "checkpoints", round(time.time() - t, 1), "s", flush=True)
