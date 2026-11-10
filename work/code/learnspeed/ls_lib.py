"""LEARNING-SPEED TEST helpers: next-game frame (each game predicted from the latest checkpoint on or before its date) + scoring."""
import numpy as np, pandas as pd, os
D = "/home/claude/data"; LS = "/home/claude/ls"; SEAS = list(range(2014, 2026)); HCA = 2.5


def next_frame(refresh=False):
    f = f"{LS}/next.parquet"
    if os.path.exists(f) and not refresh: return pd.read_parquet(f)
    out = []
    for S in SEAS:
        t = pd.read_parquet(f"{LS}/ls_{S}.parquet")
        g = pd.read_parquet(f"{D}/games_{S}.parquet")[["game", "date", "home_team", "away_team", "home_pts", "away_pts"]].sort_values(["date", "game"])
        cpd = np.sort(t.cp_date.unique()); g["cp_date"] = cpd[np.searchsorted(cpd, g.date.to_numpy(), side="right") - 1]
        x = g.merge(t, on=["game", "cp_date"], how="left")
        tg = pd.concat([g[["game", "date", "home_team"]].rename(columns={"home_team": "t"}), g[["game", "date", "away_team"]].rename(columns={"away_team": "t"})]).sort_values(["date", "game"])
        tg["n"] = tg.groupby("t").cumcount(); nn = tg.groupby("game").n.agg(["min", "max"])
        x["tgn"] = x.game.map(nn["min"]).to_numpy(); x["tgx"] = x.game.map(nn["max"]).to_numpy()
        tg["prev"] = tg.groupby("t").date.shift(1)
        dd = (pd.to_datetime(tg.date.astype(str)) - pd.to_datetime(tg.prev.astype("Int64").astype(str), errors="coerce")).dt.days
        tg["b2b"] = (dd == 1).astype(float); hb = tg.set_index(["game", "t"]).b2b
        x["b2b"] = [hb.get((gm, h), 0) - hb.get((gm, a), 0) for gm, h, a in zip(x.game, x.home_team, x.away_team)]
        x["season"] = S; x["margin"] = x.home_pts - x.away_pts; out.append(x)
    N = pd.concat(out, ignore_index=True)
    try:
        L = pd.read_csv("/home/claude/lines/hist_game_lines.csv"); N = N.merge(L[["game", "line"]], on="game", how="left")
    except Exception: N["line"] = np.nan
    N.to_parquet(f, index=False); return N


def loso(X, y, seas, m):
    X = np.c_[np.ones(len(y)), X]; p = np.full(len(y), np.nan)
    for S in np.unique(seas[m]):
        tr = m & (seas != S); te = m & (seas == S); p[te] = X[te] @ np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
    return p


def rmse(p, y, m): return np.sqrt(((y - p)[m] ** 2).mean())


def pair(pv, pb, y, seas, m):
    d = pd.Series(((y - pv) ** 2 - (y - pb) ** 2)[m]).groupby(seas[m]).mean(); t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    e = m & (seas <= 2019); l = m & (seas >= 2020)
    return rmse(pv, y, m) - rmse(pb, y, m), t, int((d < 0).sum()), len(d), rmse(pv, y, e) - rmse(pb, y, e), rmse(pv, y, l) - rmse(pb, y, l)


def creg(X, y, seas, m, names):
    """OLS with season-clustered se."""
    X = np.c_[np.ones(m.sum()), X[m]]; yy = y[m]; g = seas[m]; XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ yy; e = yy - X @ b
    meat = sum(np.outer(X[g == s].T @ e[g == s], X[g == s].T @ e[g == s]) for s in np.unique(g)); G = len(np.unique(g))
    se = np.sqrt(np.diag(XtXi @ meat @ XtXi) * G / (G - 1))
    return pd.DataFrame({"coef": b[1:], "se": se[1:]}, index=names)
