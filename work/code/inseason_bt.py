"""In-season update backtest (daily-pipeline design).

For each season S (2014-15..2025-26) and checkpoint g (league-median team games played), use only games before the
checkpoint date to update the preseason player ratings (pre_P_S), then predict every later game using the ACTUAL
minutes of that game (isolates rating quality from minutes). Scored on game margins and on each team's
rest-of-season average margin.

Methods
  M0  preseason ratings only
  M1  + team offsets: ridge on past-game residuals (team-level learning, lam_t)
  M2  player "RAPM toward prior": ridge on season-to-date stint net ratings, shrinking each player's delta toward 0 (lam)
  M3  box update of the prior: box rating recomputed with the current season's possessions added to the player's
      decayed history window at weight dS (rookies/newcomers: prior counts as H0 possessions)
  M4  M3 prior + M2 delta; optionally + M1 team offsets
"""
import json, sys, numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.linalg import spsolve
sys.path.insert(0, "/home/claude/code")
from features import OFF_FEATS, DEF_FEATS
D = "/home/claude/data"
cal = pd.read_csv(f"{D}/calib.csv", index_col=0).iloc[:, 0]
SLOPE, UNRATED = cal["slope"], cal["unrated"]
W = json.load(open(f"{D}/weights_bt.json"))
Wo, Wd = pd.Series(W["Wo"]) * .5, pd.Series(W["Wd"]) * .5
FEATS = pd.read_parquet(f"{D}/features.parquet")
HCA = 2.5
TAG = __import__("os").environ.get("TAG", "")
CHECK = [0, 5, 10, 15, 20, 30, 41, 55, 70]


def box_of(df, drop3=False):
    o = [c for c in OFF_FEATS if not (drop3 and c == "o_3eff")]
    return df[o].to_numpy() @ Wo[o].values + df[DEF_FEATS].to_numpy() @ Wd[DEF_FEATS].values


def partial_features(pg, st, lg_ps):
    """features.py recipe on season-to-date totals. pg: player-games so far; st: stints so far."""
    tot = pg.groupby("player")[["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf",
                                "rim_fgm", "rim_fga", "mid_fgm", "mid_fga", "astd_fgm"]].sum()
    out = []
    for me, op in (("A", "B"), ("B", "A")):
        x = pd.DataFrame({
            "lineup": st[f"lineup{me}"].values,
            "poss_off": (st[f"fg2a{me}"] + st[f"fg3a{me}"] - st[f"oreb{me}"] + st[f"tov{me}"] + 0.44 * st[f"fta{me}"]).values,
            "poss_def": (st[f"fg2a{op}"] + st[f"fg3a{op}"] - st[f"oreb{op}"] + st[f"tov{op}"] + 0.44 * st[f"fta{op}"]).values,
            "opp_rim_a": st[f"za1{op}"].values, "opp_rim_m": st[f"rim_m{op}"].values,
            "opp_mid_a": (st[f"za2{op}"] + st[f"za3{op}"]).values, "opp_3_a": (st[f"za4{op}"] + st[f"za5{op}"] + st[f"za6{op}"]).values})
        x["player"] = x.pop("lineup").str.split("-")
        x = x.explode("player"); x = x[x.player != ""]; x["player"] = x.player.astype(np.int64)
        out.append(x.groupby("player").sum())
    oc = pd.concat(out).groupby(level=0).sum()
    f = tot.join(oc, how="inner"); f = f[(f.poss_off > 0) & (f.poss_def > 0)].copy()
    L = f[["fg3m", "fg3a", "rim_fgm", "rim_fga", "mid_fgm", "mid_fga", "ftm", "fta"]].sum()
    p3, prim, pmid, pft = L.fg3m / L.fg3a, L.rim_fgm / L.rim_fga, L.mid_fgm / L.mid_fga, L.ftm / L.fta
    sh = lambda m, a, p, k: (m + k * p) / (a + k)
    po, pd_ = f.poss_off / 100, f.poss_def / 100
    f["o_3vol"] = f.fg3a / po; f["o_3eff"] = f.o_3vol * 3 * (sh(f.fg3m, f.fg3a, p3, 250) - p3)
    f["o_rimvol"] = f.rim_fga / po; f["o_rimeff"] = f.o_rimvol * 2 * (sh(f.rim_fgm, f.rim_fga, prim, 100) - prim)
    f["o_midvol"] = f.mid_fga / po; f["o_mideff"] = f.o_midvol * 2 * (sh(f.mid_fgm, f.mid_fga, pmid, 150) - pmid)
    f["o_ftvol"] = f.fta / po; f["o_fteff"] = f.o_ftvol * (sh(f.ftm, f.fta, pft, 75) - pft)
    f["o_unast"] = (f.fgm - f.astd_fgm) / po; f["o_ast"] = f.ast / po; f["o_tov"] = f.tov / po; f["o_oreb"] = f.oreb / po
    f["d_dreb"] = f.dreb / pd_; f["d_blk"] = f.blk / pd_; f["d_stl"] = f.stl / pd_; f["d_pf"] = f.pf / pd_
    f["d_opprim"] = (f.opp_rim_a / pd_) * 2 * (sh(f.opp_rim_m, f.opp_rim_a, prim, 400) - prim)
    ofga = f.opp_rim_a + f.opp_mid_a + f.opp_3_a
    f["d_oppxq"] = (f.opp_rim_a * 2 * prim + f.opp_mid_a * 2 * pmid + f.opp_3_a * 3 * p3) / ofga * ofga / pd_
    for c in OFF_FEATS + DEF_FEATS:
        w = f.poss_off if c.startswith("o_") else f.poss_def
        f[c] = ((f[c] - (f[c] * w).sum() / w.sum()) * (w / (w + 300))).fillna(0.0)
    return f


def stint_rows(st, home_of, date_of):
    m = st.margin_start.abs()
    st = st[~(((st.t_start >= 2160) & (m >= 20)) | ((st.t_start >= 2520) & (m >= 15)))]
    st = st[(st.lineupA.str.count("-") == 4) & (st.lineupB.str.count("-") == 4)]
    pa = st.fg2aA + st.fg3aA - st.orebA + st.tovA + 0.44 * st.ftaA
    pb = st.fg2aB + st.fg3aB - st.orebB + st.tovB + 0.44 * st.ftaB
    poss = ((pa + pb) / 2).to_numpy()
    net = (2 * st.fg2mA + 3 * st.x3A + st.xftA - 2 * st.fg2mB - 3 * st.x3B - st.xftB).to_numpy()
    k = poss > 0.5
    A = st.lineupA.str.split("-", expand=True).astype(np.int64).to_numpy()[k]
    B = st.lineupB.str.split("-", expand=True).astype(np.int64).to_numpy()[k]
    homeA = np.where(st.teamA.to_numpy() == st.game.map(home_of).to_numpy(), 1.0, -1.0)[k]
    dt = pd.to_datetime(st.game.map(date_of).astype(str)).to_numpy()[k]
    return A, B, 100 * net[k] / poss[k], poss[k], homeA, dt


def rapm_delta(A, B, y, w, homeA, prior, lam):
    """Ridge: y = sum_A (prior+d) - sum_B (prior+d) + h*homeA ; d ~ 0 with penalty lam."""
    players = np.unique(np.concatenate([A.ravel(), B.ravel()]))
    idx = pd.Index(players)
    n = len(y); P = len(players)
    rows = np.repeat(np.arange(n), 5)
    X = sp.csr_matrix((np.r_[np.ones(5 * n), -np.ones(5 * n)], (np.r_[rows, rows], np.r_[idx.get_indexer(A.ravel()), idx.get_indexer(B.ravel())])), shape=(n, P))
    pr = prior.reindex(players).to_numpy()
    r = y - X @ pr
    X = sp.hstack([X, sp.csr_matrix(homeA[:, None])]).tocsr()
    Xw = X.multiply(w[:, None]).tocsr()
    Amat = (X.T @ Xw) + sp.diags(np.r_[np.full(P, lam), 1e-6])
    b = Xw.T @ r
    beta = spsolve(sp.csc_matrix(Amat), b)
    return pd.Series(beta[:P], index=players), beta[P]


def team_offsets(past, lam_t):
    """Ridge team offsets on past-game residuals (actual - predicted margin)."""
    teams = pd.Index(np.unique(np.r_[past.home_team, past.away_team]))
    n = len(past)
    X = sp.csr_matrix((np.r_[np.ones(n), -np.ones(n)], (np.r_[np.arange(n), np.arange(n)],
                       np.r_[teams.get_indexer(past.home_team), teams.get_indexer(past.away_team)])), shape=(n, len(teams)))
    A = X.T @ X + lam_t * sp.eye(len(teams))
    return pd.Series(spsolve(sp.csc_matrix(A), X.T @ (past.margin - past.pred).to_numpy()), index=teams)


def history_box(N):
    """Decayed box rating and weight (poss) from seasons N-2..N, as fit_window."""
    fw = FEATS[FEATS.season.between(N - 2, N)].copy()
    dec = fw.season.map({N - 2: .3, N - 1: .6, N: 1.0})
    fw["wo"], fw["wd"] = fw.poss_off * dec, fw.poss_def * dec
    O = (fw[OFF_FEATS].mul(fw.wo, axis=0)).groupby(fw.player).sum()
    Dd = (fw[DEF_FEATS].mul(fw.wd, axis=0)).groupby(fw.player).sum()
    wo, wd = fw.groupby("player").wo.sum(), fw.groupby("player").wd.sum()
    return O, Dd, wo, wd


def run_season(S, grid):
    N = S - 1
    pre = pd.read_parquet(f"{D}/pre_P_{S}.parquet").drop_duplicates("player").set_index("player")
    g = pd.read_parquet(f"{D}/games_{S}.parquet")[["game", "date", "home_team", "away_team", "home_pts", "away_pts"]]
    g["margin"] = g.home_pts - g.away_pts
    pg = pd.read_parquet(f"{D}/player_games_{S}.parquet")
    pg = pg[pg["min"] > 0]
    st = pd.read_parquet(f"{D}/stints_{S}.parquet")
    st_date = st.game.map(g.set_index("game").date)
    home_of = g.set_index("game").home_team
    O_h, D_h, wo_h, wd_h = history_box(N)
    # preseason rating for every player who could appear
    allp = pd.Index(pg.player.unique())
    hb = pd.Series(box_of(pd.concat([O_h.div(wo_h, axis=0), D_h.div(wd_h, axis=0)], axis=1).fillna(0)), index=O_h.index)
    pre_r = pre.rating.reindex(allp)
    fill = SLOPE * hb.reindex(allp)          # rated player not on an opening roster: box-only history rating
    pre_r = pre_r.fillna(fill).fillna(SLOPE * UNRATED)
    # team game-count checkpoints
    tg = pd.concat([g[["game", "date", "home_team"]].rename(columns={"home_team": "t"}), g[["game", "date", "away_team"]].rename(columns={"away_team": "t"})]).sort_values(["date", "game"])
    tg["n"] = tg.groupby("t").cumcount()
    dates = np.sort(g.date.unique())
    med = pd.Series({d: tg[tg.date < d].groupby("t").size().reindex(tg.t.unique()).fillna(0).median() for d in dates})
    # per-game minutes shares for evaluation
    pg = pg.merge(g[["game", "home_team"]], on="game")
    pg["side"] = np.where(pg.team == pg.home_team, 1.0, -1.0)
    tm = pg.groupby(["game", "team"])["min"].transform("sum")
    pg["wshare"] = 5 * pg["min"] / tm * pg.side
    res = []
    for cp in CHECK:
        if cp >= g.groupby("home_team").size().mean() * 2 - 5: continue
        d = med.index[med >= cp][0] if cp > 0 else dates[0]
        past_g = g[g.date < d]; fut = g[g.date >= d]
        if len(fut) < 30: continue
        pst = st[st_date < d]
        # pace: past games of S if enough, else stints of N
        if len(past_g) >= 60:
            pa = sum(pst[f"fg2a{z}"] + pst[f"fg3a{z}"] - pst[f"oreb{z}"] + pst[f"tov{z}"] + 0.44 * pst[f"fta{z}"] for z in "AB")
            pace = pa.sum() / 2 / pst.game.nunique()
        else:
            s0 = pd.read_parquet(f"{D}/stints_{N}.parquet")
            pa = sum(s0[f"fg2a{z}"] + s0[f"fg3a{z}"] - s0[f"oreb{z}"] + s0[f"tov{z}"] + 0.44 * s0[f"fta{z}"] for z in "AB")
            pace = pa.sum() / 2 / s0.game.nunique()
        have = len(past_g) >= 15
        if have:
            ppg = pg[pg.game.isin(past_g.game)]
            fnow = partial_features(ppg, pst, None)
            A, B, y, w, hA, sdt = stint_rows(pst, home_of, g.set_index('game').date)
            age_d = (pd.Timestamp(str(d)) - pd.to_datetime(sdt)).days.to_numpy() if hasattr(pd.to_datetime(sdt), 'days') else (pd.Timestamp(str(d)) - pd.to_datetime(sdt)) / np.timedelta64(1, 'D')

        def predict(r, games_df):
            x = pg[pg.game.isin(games_df.game)]
            s = (x.wshare * x.player.map(r).fillna(SLOPE * UNRATED)).groupby(x.game).sum()
            return games_df.game.map(s).fillna(0).to_numpy() * pace / 100 + HCA

        for key in grid:
            meth, lam, dS, drop3, lam_t, H0 = key[:6]; hl = key[6] if len(key) > 6 else None
            r = pre_r.copy()
            if have and meth in ("M3", "M4"):
                # box update: add current-season features at weight dS to history window
                ids = fnow.index
                Oh = O_h.reindex(ids).fillna(0); Dh = D_h.reindex(ids).fillna(0)
                woh = wo_h.reindex(ids).fillna(0); wdh = wd_h.reindex(ids).fillna(0)
                hist_box = pd.Series(box_of(pd.concat([Oh.div(woh.replace(0, np.nan), axis=0), Dh.div(wdh.replace(0, np.nan), axis=0)], axis=1).fillna(0), drop3), index=ids)
                nohist = woh < 1
                hist_box[nohist] = (pre_r.reindex(ids) / SLOPE)[nohist]
                woh = woh.where(~nohist, H0); wdh = wdh.where(~nohist, H0)
                cur_box = pd.Series(box_of(fnow, drop3), index=ids)
                wc = dS * (fnow.poss_off + fnow.poss_def) / 2
                wh = (woh + wdh) / 2
                new_box = (hist_box * wh + cur_box * wc) / (wh + wc)
                r = r.add(SLOPE * (new_box - hist_box), fill_value=0)
            if have and meth in ("M2", "M4"):
                prior = r.reindex(np.unique(np.r_[A.ravel(), B.ravel()]))
                prior = prior.fillna(SLOPE * UNRATED)
                ww = w * 0.5 ** (np.asarray(age_d, float) / hl) if hl else w
                dlt, _ = rapm_delta(A, B, y, ww, hA, prior, lam)
                r = r.add(dlt, fill_value=0)
            pf = predict(r, fut)
            if have and lam_t is not None:
                pp = past_g.assign(pred=predict(r, past_g))
                off = team_offsets(pp, lam_t)
                pf = pf + fut.home_team.map(off).fillna(0).to_numpy() - fut.away_team.map(off).fillna(0).to_numpy()
            e = fut.margin.to_numpy() - pf
            # team rest-of-season average margin
            tt = pd.concat([pd.DataFrame({"t": fut.home_team, "a": fut.margin, "p": pf}),
                            pd.DataFrame({"t": fut.away_team, "a": -fut.margin, "p": -pf})]).groupby("t").agg(a=("a", "mean"), p=("p", "mean"), n=("a", "size"))
            res.append(dict(season=S, cp=cp, key=key, n=len(fut), sse=float((e ** 2).sum()),
                            team_sse=float(((tt.a - tt.p) ** 2 * tt.n).sum()), team_n=int(tt.n.sum()),
                            team_err=tt.a - tt.p))
    return res


def default_grid():
    G = [("M0", None, None, False, None, None)]
    G += [("M1", None, None, False, lt, None) for lt in (5, 10, 20, 40)]
    G += [("M2", lam, None, False, None, None) for lam in (1000, 2000, 4000, 8000, 16000)]
    G += [("M3", None, dS, d3, None, 2000) for dS in (0.5, 1, 2, 4) for d3 in (False, True)]
    G += [("M4", lam, dS, False, None, 2000) for lam in (2000, 4000, 8000) for dS in (1, 2)]
    G += [("M4", 4000, 1, False, lt, 2000) for lt in (20, 40)]
    return G


def grid2():
    G = [("M0", None, None, False, None, None)]
    for lam in (3000, 4000, 6000):
        for hl in (None, 30, 60, 120):
            for lt in (None, 40, 80):
                G.append(("M4", lam, 1, False, lt, 2000, hl))
    return G


if __name__ == "__main__":
    import pickle
    seasons = list(map(int, sys.argv[1:])) or list(range(2014, 2026))
    grid = grid2() if TAG else default_grid()
    for S in seasons:
        out = run_season(S, grid)
        pickle.dump(out, open(f"{D}/inseason{TAG}_{S}.pkl", "wb"))
        m0 = [r for r in out if r["key"][0] == "M0"]
        print(S, " ".join(f"{r['cp']}:{np.sqrt(r['sse']/r['n']):.2f}" for r in m0), flush=True)
