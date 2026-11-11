import numpy as np, pandas as pd
D = "/home/claude/data"; FD = "/home/claude/fund"
B = pd.read_parquet(f"{D}/backtest_components.parquet").merge(pd.read_parquet(f"{D}/bt_split.parquet"), on=["game", "home_team", "away_team"])
B = B.merge(pd.read_parquet(f"{FD}/hole_bt.parquet"), on="game", how="left")
F = [c for c in B.columns if c.startswith("f:")]
base = (B[F].sum(axis=1) + B.unrated + .6 * B.res_o + .8 * B.res_d).to_numpy(); y = B.margin.values; seas = B.season.values
def err(X, m):
    X = np.c_[np.ones(len(y)), X]; e = np.full(len(y), np.nan)
    for s in np.unique(seas):
        tr, te = m & (seas != s), m & (seas == s); e[te] = (y[te] - X[te] @ np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]) ** 2
    return e
def coefs(X, m):
    X = np.c_[np.ones(len(y)), X][m]; yy = y[m]; ss = seas[m]; b = np.linalg.lstsq(X, yy, rcond=None)[0]; r = yy - X @ b; XtXi = np.linalg.inv(X.T @ X)
    meat = sum(np.outer(X[ss == s].T @ r[ss == s], X[ss == s].T @ r[ss == s]) for s in np.unique(ss)); return b, b / np.sqrt(np.diag(XtXi @ meat @ XtXi))
M = (B.tgn_min >= 5).to_numpy(); print("games (both teams at game 6+):", M.sum(), "of", len(B))
print("team-games: home lead-handler coverage (active x trailing): mean", B.h_home[M].mean().round(3), "share below 1:", (B.h_home[M] < .999).mean().round(3), "below .5:", (B.h_home[M] < .5).mean().round(3),
      "| rim protector: mean", B.r_home[M].mean().round(3), "below 1:", (B.r_home[M] < .999).mean().round(3), "below .5:", (B.r_home[M] < .5).mean().round(3))
e0 = err(base[:, None], M); print(f"base LOSO on these games {np.sqrt(np.nanmean(e0)):.4f}")
def line(name, cols):
    X = np.c_[base, B[cols].to_numpy()]; e1 = err(X, M); ok = M
    d = pd.Series((e1 - e0)[ok]).groupby(seas[ok]).mean(); tp = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    b, t = coefs(X, M); _, t1 = coefs(X, M & (seas <= 2019)); _, t2 = coefs(X, M & (seas >= 2020))
    print(f"{name:28s} dLOSO {np.sqrt(np.nanmean(e1))-np.sqrt(np.nanmean(e0)):+.4f} paired t {tp:+.2f} better {(d<0).sum()}/11 | " +
          "  ".join(f"{c}: coef {b[2+i]:+.2f} t {t[2+i]:+.1f} (halves {t1[2+i]:+.1f}/{t2[2+i]:+.1f})" for i, c in enumerate(cols)))
print("PRIMARY: 80th percentile, active x trailing minutes; expected sign + for both (coef = points for going from no coverage to full)")
line("handler", ["cov_h80_t"]); line("rim protector", ["cov_r80_t"]); line("both jointly", ["cov_h80_t", "cov_r80_t"])
print("ROBUSTNESS")
line("both, actual minutes", ["cov_h80_a", "cov_r80_a"])
for q in (70, 90): line(f"both, {q}th pct", [f"cov_h{q}_t", f"cov_r{q}_t"])
line("uncapped sums (linear check)", ["sum_h80_t", "sum_r80_t"]); line("capped + uncapped", ["cov_h80_t", "sum_h80_t", "cov_r80_t", "sum_r80_t"])
