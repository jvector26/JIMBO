"""Player impact model.

Step 1 (fit_weights): learn component weights W from lineup outcomes over many seasons.
   y = luck-adjusted points per 100 possessions for the offense in a stint (weighted by possessions)
   y ~ season intercept + home + sum_offense(W_o . x_o) + sum_defense(W_d . x_d) + player-season residuals (ridge)
Step 2 (fit_window): ratings as of the end of season N from seasons N-2..N (decayed), holding W fixed.
   rating = W . x (box-score components) + residual (lineup impact the box score misses, shrunk to 0)
Defense sign convention in outputs: positive = good (fewer points allowed).
"""
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from features import OFF_FEATS, DEF_FEATS, COMPONENTS

D = "/home/claude/data"
_games = None
_feats = None


def games():
    global _games
    if _games is None:
        _games = pd.read_parquet(f"{D}/games_all.parquet")[["game", "home_team", "season", "date", "home_pts", "away_pts", "away_team"]]
    return _games


def feats():
    global _feats
    if _feats is None:
        _feats = pd.read_parquet(f"{D}/features.parquet")
    return _feats


def load_rows(seasons, drop_garbage=True):
    """One row per stint side (offense vs defense)."""
    out = []
    home = games().set_index("game")["home_team"]
    for s in seasons:
        st = pd.read_parquet(f"{D}/stints_{s}.parquet")
        if drop_garbage:
            m = st.margin_start.abs()
            g = ((st.t_start >= 2160) & (m >= 20)) | ((st.t_start >= 2520) & (m >= 15))
            st = st[~g]
        for me, op in (("A", "B"), ("B", "A")):
            poss = st[f"fg2a{me}"] + st[f"fg3a{me}"] - st[f"oreb{me}"] + st[f"tov{me}"] + 0.44 * st[f"fta{me}"]
            adj = 2 * st[f"fg2m{me}"] + 3 * st[f"x3{me}"] + st[f"xft{me}"]
            x = pd.DataFrame({"season": s, "game": st.game.values, "off": st[f"lineup{me}"].values,
                              "dfn": st[f"lineup{op}"].values, "poss": poss.values, "pts": adj.values,
                              "home_off": (st[f"team{me}"].values == st.game.map(home).values).astype(np.int8)})
            out.append(x[x.poss > 0.5])
    r = pd.concat(out, ignore_index=True)
    r = r[(r.off.str.count("-") == 4) & (r.dfn.str.count("-") == 4)]
    r["y"] = 100 * r.pts / r.poss
    return r.reset_index(drop=True)


def lineup_matrix(lineups, index):
    """Sparse n x p indicator matrix: 1 for each of the five players (by index dict)."""
    ids = lineup_ids(pd.Series(lineups))
    rows = np.repeat(np.arange(len(ids)), 5)
    cols = np.array([index.get(p, -1) for p in ids.ravel()])
    ok = cols >= 0
    return sp.csr_matrix((np.ones(ok.sum()), (rows[ok], cols[ok])), shape=(len(ids), len(index))), ids


def feature_sum(ids, seasons, cols, f):
    """Sum of each lineup's player features (for that season). ids: n x 5 int player ids; seasons: length-n ints."""
    keys = f.season.to_numpy(np.int64) * 10**8 + f.player.to_numpy(np.int64)
    M = np.vstack([f[cols].to_numpy(), np.zeros((1, len(cols)))])  # last row = zeros for unknown players
    lut = pd.Index(keys)
    q = (np.repeat(np.asarray(seasons, np.int64), 5) * 10**8 + np.asarray(ids, np.int64).ravel())
    pos = lut.get_indexer(q)
    pos[pos < 0] = len(M) - 1
    out = np.zeros((len(ids), len(cols)))
    for j in range(5):
        out += M[pos[j::5]]
    return out


def lineup_ids(series):
    """Vectorized '1-2-3-4-5' -> n x 5 int64 array."""
    return series.str.split("-", expand=True).astype(np.int64).to_numpy()


def solve_blocks(Xs, F, y, w, pen_s, pen_f, chunk=150_000):
    """Weighted ridge with a sparse block Xs and dense block F; normal equations accumulated in row chunks."""
    Xs = Xs.tocsr()
    p, q = Xs.shape[1], F.shape[1]
    A11 = sp.csc_matrix((p, p)); A12 = np.zeros((p, q)); A22 = np.zeros((q, q)); b1 = np.zeros(p); b2 = np.zeros(q)
    for i in range(0, Xs.shape[0], chunk):
        xs = Xs[i:i + chunk].astype(np.float64); f = F[i:i + chunk].astype(np.float64); ww = w[i:i + chunk]; yy = y[i:i + chunk]
        xsw = xs.multiply(ww[:, None]).tocsr(); fw = f * ww[:, None]
        A11 = A11 + (xs.T @ xsw).tocsc()
        A12 += np.asarray(xs.T @ fw); A22 += f.T @ fw
        b1 += xsw.T @ yy; b2 += fw.T @ yy
    A11 = A11 + sp.diags(pen_s); A22 += np.diag(pen_f)
    A = sp.bmat([[A11, sp.csc_matrix(A12)], [sp.csc_matrix(A12.T), sp.csc_matrix(A22)]]).tocsr()
    del A11
    bb = np.concatenate([b1, b2])
    if A.shape[0] <= 20000:
        return spsolve(A.tocsc(), bb)
    # large systems: preconditioned conjugate gradient (no factorization, low memory)
    from scipy.sparse.linalg import cg, LinearOperator
    d = A.diagonal(); M = LinearOperator(A.shape, matvec=lambda v: v / d)
    x, info = cg(A, bb, M=M, rtol=1e-10, maxiter=20000)
    print("cg info", info, "resid", float(np.linalg.norm(A @ x - bb) / np.linalg.norm(bb)), flush=True)
    return x


def solve(X, y, w, pen):
    Xw = X.multiply(w[:, None]).tocsr() if sp.issparse(X) else X * w[:, None]
    A = (X.T @ Xw) + sp.diags(pen)
    b = Xw.T @ y
    return spsolve(sp.csc_matrix(A), b)


def load_rows_int(seasons, drop_garbage=True):
    """Like load_rows, but lineups parsed per season into int arrays (no text kept). Returns dict of arrays."""
    home = games().set_index("game")["home_team"]
    parts = {k: [] for k in ("season", "poss", "y", "home_off", "ids_o", "ids_d")}
    for s in seasons:
        st = pd.read_parquet(f"{D}/stints_{s}.parquet")
        if drop_garbage:
            m = st.margin_start.abs()
            st = st[~(((st.t_start >= 2160) & (m >= 20)) | ((st.t_start >= 2520) & (m >= 15)))]
        st = st[(st.lineupA.str.count("-") == 4) & (st.lineupB.str.count("-") == 4)]
        LA, LB = lineup_ids(st.lineupA), lineup_ids(st.lineupB)
        hm = st.game.map(home).to_numpy()
        for me, op, Lm, Lo in (("A", "B", LA, LB), ("B", "A", LB, LA)):
            poss = (st[f"fg2a{me}"] + st[f"fg3a{me}"] - st[f"oreb{me}"] + st[f"tov{me}"] + 0.44 * st[f"fta{me}"]).to_numpy()
            adj = (2 * st[f"fg2m{me}"] + 3 * st[f"x3{me}"] + st[f"xft{me}"]).to_numpy()
            k = poss > 0.5
            parts["season"].append(np.full(k.sum(), s, np.int64)); parts["poss"].append(poss[k])
            parts["y"].append(100 * adj[k] / poss[k]); parts["home_off"].append((st[f"team{me}"].to_numpy() == hm)[k].astype(np.float32))
            parts["ids_o"].append(Lm[k]); parts["ids_d"].append(Lo[k])
        del st
    return {k: np.concatenate(v) for k, v in parts.items()}


def fit_weights(seasons, lam=4000.0, lag=0):
    """Joint fit: component weights (tiny ridge) + player-season residuals (ridge lam)."""
    R = load_rows_int(seasons)
    f = feats()
    n = len(R["y"])
    ids_o, ids_d, sea = R["ids_o"], R["ids_d"], R["season"]
    ko = (np.repeat(sea, 5) * 10**8 + ids_o.ravel())
    kd = (np.repeat(sea, 5) * 10**8 + ids_d.ravel())
    codes, uniq = pd.factorize(np.concatenate([ko, kd]))
    P = len(uniq)
    rows5 = np.repeat(np.arange(n), 5)
    Xo = sp.csr_matrix((np.ones(5 * n, dtype=np.float32), (rows5, codes[:5 * n])), shape=(n, P))
    Xd = sp.csr_matrix((np.ones(5 * n, dtype=np.float32), (rows5, codes[5 * n:])), shape=(n, P))
    del ko, kd, codes, rows5
    Fo = feature_sum(ids_o, sea - lag, OFF_FEATS, f)
    Fd = feature_sum(ids_d, sea - lag, DEF_FEATS, f)
    seas = pd.get_dummies(sea).to_numpy(np.float32)
    Xfix = np.hstack([seas, R["home_off"][:, None], Fo, Fd]).astype(np.float32)
    del Fo, Fd
    Xs = sp.hstack([Xo, Xd]).tocsr()
    del Xo, Xd
    pen_f = np.concatenate([np.zeros(seas.shape[1] + 1), np.full(len(OFF_FEATS) + len(DEF_FEATS), 1.0)])
    beta = solve_blocks(Xs, Xfix, R["y"], R["poss"], np.full(2 * P, lam), pen_f)
    nfix0 = 2 * P + seas.shape[1]
    hca = beta[nfix0]
    Wo = pd.Series(beta[nfix0 + 1: nfix0 + 1 + len(OFF_FEATS)], index=OFF_FEATS)
    Wd = pd.Series(-beta[nfix0 + 1 + len(OFF_FEATS):], index=DEF_FEATS)  # flip: positive = good defense
    return Wo, Wd, hca


def fit_window(N, Wo, Wd, lam=4000.0, decay=(0.3, 0.6, 1.0), use_prior=True, rows=None, lam_o=None, box_decay=None):
    """Ratings as of end of season N using seasons N-2..N. Returns per-player O, D, components."""
    seasons = [N - 2, N - 1, N]
    seasons = [s for s in seasons if s >= 1996]
    r = rows if rows is not None else load_rows(seasons)
    f = feats()
    wdec = dict(zip([N - 2, N - 1, N], decay))
    w = r.poss.to_numpy() * r.season.map(wdec).to_numpy()
    players = pd.unique(np.concatenate([np.concatenate(r.off.str.split("-").values),
                                        np.concatenate(r.dfn.str.split("-").values)]).astype(np.int64))
    idx = {p: i for i, p in enumerate(players)}
    Xo, ids_o = lineup_matrix(r.off.values, idx)
    Xd, ids_d = lineup_matrix(r.dfn.values, idx)
    y = r.y.to_numpy().copy()
    if use_prior:
        y = y - feature_sum(ids_o, r.season.values, OFF_FEATS, f) @ Wo.values \
              + feature_sum(ids_d, r.season.values, DEF_FEATS, f) @ Wd.values
    seas = pd.get_dummies(r.season).to_numpy(float)
    X = sp.hstack([Xo, Xd, sp.csr_matrix(np.hstack([seas, r.home_off.to_numpy(float)[:, None]]))]).tocsr()
    P = len(players)
    pen = np.concatenate([np.full(P, lam if lam_o is None else lam_o), np.full(P, lam), np.zeros(seas.shape[1] + 1)])
    beta = solve(X, y, w, pen)
    res = pd.DataFrame({"player": players, "res_o": beta[:P], "res_d": -beta[P:2 * P]})

    # box-score part: decay- and possession-weighted average of the player's features over the window
    fw = f[f.season.isin(seasons)].copy()
    bdec = wdec if box_decay is None else dict(zip([N - 2, N - 1, N], box_decay))  # box-part season weights (session 12)
    fw["wo"] = fw.poss_off * fw.season.map(bdec)
    fw["wd"] = fw.poss_def * fw.season.map(bdec)
    comp = {}
    for c, cols in COMPONENTS.items():
        oc = [x for x in cols if x.startswith("o_")]
        dc = [x for x in cols if x.startswith("d_")]
        v = 0
        if oc:
            v = v + (fw[oc].to_numpy() @ Wo[oc].values) * fw.wo
        if dc:
            v = v + (fw[dc].to_numpy() @ Wd[dc].values) * fw.wd
        fw[c] = v
    featcols = []
    for x in OFF_FEATS + DEF_FEATS:
        wt = fw.wo if x.startswith("o_") else fw.wd
        W_ = Wo[x] if x.startswith("o_") else Wd[x]
        fw["f:" + x] = fw[x] * W_ * wt
        featcols.append("f:" + x)
    agg_o = fw.groupby("player")["wo"].sum()
    agg_d = fw.groupby("player")["wd"].sum()
    out = fw.groupby("player")[list(COMPONENTS) + featcols].sum()
    for c, cols in COMPONENTS.items():
        denom = agg_o if cols[0].startswith("o_") else agg_d
        out[c] = out[c] / denom
        for x in cols:
            out["f:" + x] = out["f:" + x] / denom
    out = out.reset_index().merge(res, on="player", how="outer").fillna(0.0)
    if not use_prior:
        out[list(COMPONENTS) + featcols] = 0.0
    ocomp = [c for c, cols in COMPONENTS.items() if cols[0].startswith("o_")]
    dcomp = [c for c, cols in COMPONENTS.items() if cols[0].startswith("d_")]
    out["off"] = out[ocomp].sum(axis=1) + out.res_o
    out["def"] = out[dcomp].sum(axis=1) + out.res_d
    out["total"] = out["off"] + out["def"]
    out["lineup_impact"] = out.res_o + out.res_d
    poss = r.assign(n=1)
    last = f[f.season == N].set_index("player")
    out["poss_window"] = out.player.map((fw.groupby("player").poss_off.sum()))
    out["min_N"] = out.player.map(last["min"]).fillna(0)
    out["season"] = N
    return out
