"""JIMBO in-season update (method M4 from code/inseason_bt.py, chosen 2026-09-19):
  1) box update: season-to-date features (features.py recipe) added to each player's decayed N-2..N window at
     weight 1.0 per possession; rating += SLOPE * (new_box - hist_box). No-history players: prior = 2000 poss.
  2) RAPM toward prior: ridge on season-to-date stint net ratings (luck-adjusted, garbage time dropped), player
     deltas shrunk to 0 with LAM 6000.
  3) team offsets: ridge on past-game residuals, lam_t 80 games.
Backtest (2014-25): game RMSE -0.10 @10 games, -0.25 @30, -0.38 @55 vs preseason-only."""
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.linalg import spsolve

OFF_FEATS = ["o_3eff", "o_3vol", "o_rimeff", "o_rimvol", "o_mideff", "o_midvol", "o_fteff", "o_ftvol", "o_unast",
             "o_ast", "o_tov", "o_oreb"]
DEF_FEATS = ["d_dreb", "d_blk", "d_opprim", "d_stl", "d_pf", "d_oppxq"]
TOT = ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf",
       "rim_fgm", "rim_fga", "mid_fgm", "mid_fga", "astd_fgm"]


def box_of(df, Wo, Wd):
    return df[OFF_FEATS].to_numpy() @ Wo[OFF_FEATS].values + df[DEF_FEATS].to_numpy() @ Wd[DEF_FEATS].values


def partial_features(pg, st):
    """features.py recipe on season-to-date totals (pg: player-games, st: stints)."""
    tot = pg.groupby("player")[TOT].sum()
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


def stint_rows(st, home_team_of_game):
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
    homeA = np.where(st.teamA.to_numpy() == st.game.map(home_team_of_game).to_numpy(), 1.0, -1.0)[k]
    return A, B, 100 * net[k] / poss[k], poss[k], homeA


def rapm_delta(A, B, y, w, homeA, prior, lam):
    players = np.unique(np.concatenate([A.ravel(), B.ravel()]))
    idx = pd.Index(players); n = len(y); P = len(players)
    rows = np.repeat(np.arange(n), 5)
    X = sp.csr_matrix((np.r_[np.ones(5 * n), -np.ones(5 * n)], (np.r_[rows, rows], np.r_[idx.get_indexer(A.ravel()), idx.get_indexer(B.ravel())])), shape=(n, P))
    r = y - X @ prior.reindex(players).to_numpy()
    X = sp.hstack([X, sp.csr_matrix(homeA[:, None])]).tocsr()
    Xw = X.multiply(w[:, None]).tocsr()
    beta = spsolve(sp.csc_matrix((X.T @ Xw) + sp.diags(np.r_[np.full(P, lam), 1e-6])), Xw.T @ r)
    return pd.Series(beta[:P], index=players), beta[P]


def team_offsets(past, lam_t):
    teams = pd.Index(np.unique(np.r_[past.home, past.away])); n = len(past)
    X = sp.csr_matrix((np.r_[np.ones(n), -np.ones(n)], (np.r_[np.arange(n), np.arange(n)],
                       np.r_[teams.get_indexer(past.home), teams.get_indexer(past.away)])), shape=(n, len(teams)))
    return pd.Series(spsolve(sp.csc_matrix(X.T @ X + lam_t * sp.eye(len(teams))), X.T @ (past.margin - past.pred).to_numpy()), index=teams)


def update_ratings(prior, hist, pg, st, games, cfg):
    """prior: Series player -> preseason rating (post-slope). hist: DataFrame (index player) with OFF/DEF window
    sums and wo, wd (decayed possessions). pg/st/games: season to date (games: game, home, away, margin).
    Returns (ratings Series, detail DataFrame)."""
    SL, Wo, Wd = cfg["slope"], pd.Series(cfg["Wo"]), pd.Series(cfg["Wd"])
    r = prior.copy()
    det = pd.DataFrame(index=prior.index); det["prior"] = prior; det["box_delta"] = 0.0; det["rapm_delta"] = 0.0
    if len(games) < cfg.get("min_games", 15):
        return r, det
    fnow = partial_features(pg[pg["min"] > 0], st)
    ids = fnow.index
    missing = ids.difference(r.index)
    if len(missing):  # appeared but not in the state: newcomer value
        r = pd.concat([r, pd.Series(SL * cfg["unrated"], index=missing)])
        det = det.reindex(r.index); det.loc[missing, "prior"] = SL * cfg["unrated"]; det = det.fillna(0.0)
    H = hist.reindex(ids)
    woh, wdh = H.wo.fillna(0), H.wd.fillna(0)
    avg = pd.concat([H[OFF_FEATS].div(woh.replace(0, np.nan), axis=0), H[DEF_FEATS].div(wdh.replace(0, np.nan), axis=0)], axis=1).fillna(0)
    hist_box = pd.Series(box_of(avg, Wo, Wd), index=ids)
    nohist = woh < 1
    hist_box[nohist] = (r.reindex(ids) / SL)[nohist]
    woh = woh.where(~nohist, cfg["H0"]); wdh = wdh.where(~nohist, cfg["H0"])
    cur_box = pd.Series(box_of(fnow, Wo, Wd), index=ids)
    wc = cfg["dS"] * (fnow.poss_off + fnow.poss_def) / 2; wh = (woh + wdh) / 2
    bd = SL * ((hist_box * wh + cur_box * wc) / (wh + wc) - hist_box)
    r = r.add(bd, fill_value=0); det.loc[ids, "box_delta"] = bd
    A, B, y, w, hA = stint_rows(st, games.set_index("game").home_id)
    pr = r.reindex(np.unique(np.r_[A.ravel(), B.ravel()])).fillna(SL * cfg["unrated"])
    dl, _ = rapm_delta(A, B, y, w, hA, pr, cfg["lam"])
    r = r.add(dl, fill_value=0); det = det.reindex(r.index).fillna(0.0); det.loc[dl.index, "rapm_delta"] = dl
    det["poss"] = ((fnow.poss_off + fnow.poss_def) / 2).reindex(det.index).fillna(0)
    return r, det
