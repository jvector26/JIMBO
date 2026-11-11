"""T3 follow-up (post hoc, after seeing the 70th-percentile result): is handler coverage special, or a proxy for a depleted team?
Placebos: the same coverage built on other components; control: trailing minutes of active established players (500+ min in N) / 240."""
import numpy as np, pandas as pd
D = "/home/claude/data"; FD = "/home/claude/fund"
gm = pd.read_parquet(f"{D}/games_all.parquet"); out = []
COMPS = ["playmaking", "rim_protection", "three_pt", "ball_security", "def_rebounding", "perimeter_def", "self_creation", "total"]
for N in range(2014, 2025):
    r = pd.read_parquet(f"{FD}/ratings_all_{N}.parquet").set_index("player"); el = r[r.min_N >= 500]
    pg = pd.read_parquet(f"{D}/player_games_{N+1}.parquet")[["game", "player", "team", "min", "date"]]; pg = pg[pg["min"] > 0].sort_values(["date", "game"])
    prev = pd.read_parquet(f"{D}/player_games_{N}.parquet"); prev = prev[prev["min"] > 0].groupby("player")["min"].mean()
    cs = pg.groupby("player")["min"].cumsum() - pg["min"]; cn = pg.groupby("player").cumcount()
    pg["trail"] = pd.Series(np.where(cn > 0, cs / cn.replace(0, np.nan), pg.player.map(prev)), index=pg.index).fillna(0)
    tg = pg.drop_duplicates(["game", "team"])[["game", "team"]]; res = pd.DataFrame(index=pd.MultiIndex.from_frame(tg))
    res["est"] = np.minimum(pg[pg.player.isin(el.index)].groupby(["game", "team"]).trail.sum().reindex(res.index).fillna(0) / 240, 1)
    res["n_el"] = pg[pg.player.isin(el.index)].groupby(["game", "team"]).size().reindex(res.index).fillna(0)
    for c in COMPS:
        for q in (70, 80):
            grp = set(r.index[r[c] >= np.percentile(el[c], q)])
            res[f"{c}{q}"] = np.minimum(pg[pg.player.isin(grp)].groupby(["game", "team"]).trail.sum().reindex(res.index).fillna(0), 48) / 48
    g = gm[gm.season == N + 1][["game", "home_team", "away_team"]]
    hk = pd.MultiIndex.from_arrays([g.game, g.home_team]); ak = pd.MultiIndex.from_arrays([g.game, g.away_team])
    dd = pd.DataFrame(res.reindex(hk).to_numpy() - res.reindex(ak).to_numpy(), columns=["p_" + c for c in res.columns]); dd["game"] = g.game.values; out.append(dd)
P = pd.concat(out)
B = pd.read_parquet(f"{D}/backtest_components.parquet").merge(pd.read_parquet(f"{D}/bt_split.parquet"), on=["game", "home_team", "away_team"])
B = B.merge(pd.read_parquet(f"{FD}/hole_bt.parquet")[["game", "tgn_min"]], on="game").merge(P, on="game")
F = [c for c in B.columns if c.startswith("f:")]
base = (B[F].sum(axis=1) + B.unrated + .6 * B.res_o + .8 * B.res_d).to_numpy(); y = B.margin.values; seas = B.season.values; M = (B.tgn_min >= 5).to_numpy()
def err(X):
    X = np.c_[np.ones(len(y)), X]; e = np.full(len(y), np.nan)
    for s in np.unique(seas):
        tr, te = M & (seas != s), M & (seas == s); e[te] = (y[te] - X[te] @ np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]) ** 2
    return e
def coefs(X, m):
    X = np.c_[np.ones(len(y)), X][m]; yy = y[m]; ss = seas[m]; b = np.linalg.lstsq(X, yy, rcond=None)[0]; r = yy - X @ b; XtXi = np.linalg.inv(X.T @ X)
    meat = sum(np.outer(X[ss == s].T @ r[ss == s], X[ss == s].T @ r[ss == s]) for s in np.unique(ss)); return b, b / np.sqrt(np.diag(XtXi @ meat @ XtXi))
def line(name, cols, e0, extra=None):
    X = np.c_[base, B[cols].to_numpy()] if extra is None else np.c_[base, B[extra].to_numpy(), B[cols].to_numpy()]; k0 = 2 + (0 if extra is None else len(extra))
    e1 = err(X); d = pd.Series((e1 - e0)[M]).groupby(seas[M]).mean(); tp = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    b, t = coefs(X, M); _, t1 = coefs(X, M & (seas <= 2019)); _, t2 = coefs(X, M & (seas >= 2020))
    print(f"{name:34s} dLOSO {np.sqrt(np.nanmean(e1))-np.sqrt(np.nanmean(e0)):+.4f} paired t {tp:+.2f} {(d<0).sum()}/11 | " + "  ".join(f"coef {b[k0+i]:+.2f} t {t[k0+i]:+.1f} ({t1[k0+i]:+.1f}/{t2[k0+i]:+.1f})" for i in range(len(cols))))
e0 = err(base[:, None]); print("A. coverage built on each component, one at a time, no control")
for c in COMPS:
    for q in (70, 80): line(f"{c} {q}", [f"p_{c}{q}"], e0)
print("B. control alone: established-player trailing minutes / 240, and count of established players active")
line("established minutes share", ["p_est"], e0); line("established players active", ["p_n_el"], e0); line("both", ["p_est", "p_n_el"], e0)
ec = err(np.c_[base, B[["p_est", "p_n_el"]].to_numpy()]); print("C. each coverage ON TOP of the two controls (dLOSO vs base + controls)")
for c in COMPS:
    for q in (70, 80): line(f"{c} {q}", [f"p_{c}{q}"], ec, extra=["p_est", "p_n_el"])
print("corr of handler70 coverage diff with controls:", np.corrcoef(B.p_playmaking70[M], B.p_est[M])[0, 1].round(2), np.corrcoef(B.p_playmaking70[M], B.p_n_el[M])[0, 1].round(2))
