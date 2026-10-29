"""Teammate-adjusted defensive sub-outcomes (defense roadmap #2).
Stint-side ridge of each defensive sub-outcome on the 5 defenders + 5 offensive players (controls),
season FE + home, decay-weighted window N-2..N. Defender coefficient = player's effect on the opponent's rate
(units below; NEGATIVE = fewer allowed, except tov where positive = more forced).
"""
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import impact as I

D = I.D
# name: (numerator fn, denominator fn, scale) ; outcome = scale * num/den, weight = den
OUTS = {
    "r3":    ("fg3a", "poss", 100),     # opp 3PA per 100 poss
    "rrim":  ("za1", "poss", 100),      # opp rim FGA per 100 poss
    "rmid":  ("mid", "poss", 100),      # opp midrange FGA per 100 poss
    "rfta":  ("fta", "poss", 100),      # opp FTA per 100 poss
    "rtov":  ("tov", "poss", 100),      # opp TOV per 100 poss (forced)
    "orb":   ("oreb", "rebch", 100),    # opp OREB% of available rebounds
    "p3":    ("l3", "fg3a", 100),       # opp 3P% minus shooter-expected (luck-adjusted), pct pts
    "prim":  ("rim_m", "za1", 100),     # opp rim FG%, pct pts
    "pts":   ("adj", "poss", 100),      # luck-adjusted opp pts / 100 (reference: plain RAPM-D)
}


def rows(seasons):
    home = I.games().set_index("game")["home_team"]
    parts = []
    for s in seasons:
        st = pd.read_parquet(f"{D}/stints_{s}.parquet")
        m = st.margin_start.abs()
        st = st[~(((st.t_start >= 2160) & (m >= 20)) | ((st.t_start >= 2520) & (m >= 15)))]
        st = st[(st.lineupA.str.count("-") == 4) & (st.lineupB.str.count("-") == 4)]
        LA, LB = I.lineup_ids(st.lineupA), I.lineup_ids(st.lineupB)
        hm = st.game.map(home).to_numpy()
        for me, op, Lm, Lo in (("A", "B", LA, LB), ("B", "A", LB, LA)):
            g = lambda c: st[f"{c}{me}"].to_numpy(float)
            d = dict(
                poss=g("fg2a") + g("fg3a") - g("oreb") + g("tov") + 0.44 * g("fta"),
                fg3a=g("fg3a"), za1=g("za1"), mid=g("za2") + g("za3"), fta=g("fta"), tov=g("tov"),
                oreb=g("oreb"), rebch=g("oreb") + st[f"dreb{op}"].to_numpy(float),
                l3=g("fg3m") - g("x3"), rim_m=g("rim_m"), adj=2 * g("fg2m") + 3 * g("x3") + g("xft"))
            k = d["poss"] > 0.5
            df = pd.DataFrame({c: v[k] for c, v in d.items()})
            df["season"] = s; df["game"] = st.game.to_numpy()[k]
            df["home_off"] = (st[f"team{me}"].to_numpy() == hm)[k].astype(float)
            df["ids_o"] = list(Lm[k]); df["ids_d"] = list(Lo[k])
            parts.append(df)
    return pd.concat(parts, ignore_index=True)


def design(r, players=None):
    io = np.vstack(r.ids_o.to_numpy()); idd = np.vstack(r.ids_d.to_numpy())
    if players is None:
        players = pd.unique(np.concatenate([io.ravel(), idd.ravel()]))
    idx = pd.Index(players); P = len(players); n = len(r)
    ro = np.repeat(np.arange(n), 5)
    co = idx.get_indexer(io.ravel()); cd = idx.get_indexer(idd.ravel())
    ok_o, ok_d = co >= 0, cd >= 0
    Xo = sp.csr_matrix((np.ones(ok_o.sum()), (ro[ok_o], co[ok_o])), shape=(n, P))
    Xd = sp.csr_matrix((np.ones(ok_d.sum()), (ro[ok_d], cd[ok_d])), shape=(n, P))
    seas = pd.get_dummies(r.season).to_numpy(float)
    X = sp.hstack([Xo, Xd, sp.csr_matrix(np.c_[seas, r.home_off.to_numpy()])]).tocsr()
    return X, players, seas.shape[1] + 1


def fit(X, P, nfe, y, w, lam):
    Xw = X.multiply(w[:, None]).tocsr()
    pen = np.r_[np.full(2 * P, lam), np.full(nfe, 1e-6)]
    A = (X.T @ Xw) + sp.diags(pen)
    return spsolve(sp.csc_matrix(A), Xw.T @ y)


def outcome(r, name, decay_w):
    num, den, sc = OUTS[name]
    dn = r[den].to_numpy(); ok = dn > 0
    y = np.zeros(len(r)); y[ok] = sc * r[num].to_numpy()[ok] / dn[ok]
    return y, dn * decay_w * ok


def fit_window(N, lams, r=None):
    """Defender coefficients for every outcome, window N-2..N. Returns DataFrame player x outcome (+ def poss)."""
    seasons = [N - 2, N - 1, N]
    r = rows(seasons) if r is None else r
    dec = r.season.map(dict(zip(seasons, (0.3, 0.6, 1.0)))).to_numpy()
    X, players, nfe = design(r)
    P = len(players)
    out = pd.DataFrame({"player": players})
    for name, lam in lams.items():
        y, w = outcome(r, name, dec)
        b = fit(X, P, nfe, y, w, lam)
        out[name] = b[P:2 * P]
    Xd = X[:, P:2 * P]
    out["dposs"] = np.asarray(Xd.T @ (r.poss.to_numpy() * dec)).ravel()
    return out
