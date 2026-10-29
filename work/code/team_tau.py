"""Team-specific projection uncertainty. Target: LOSO blend residual (actual wins - blend forecast).
Model: e ~ N(0, luck + exp(a + b.x)), luck = 82 p(1-p). Compare held-out log-likelihood vs constant variance."""
import numpy as np, pandas as pd
from scipy.optimize import minimize
D = "/home/claude/data"
ps = pd.read_parquet(f"{D}/player_seasons.parquet")
A = pd.read_parquet(f"{D}/preseason_backtest.parquet").merge(pd.read_parquet(f"{D}/direction_features.parquet")[["season", "team", "mw_age"]], on=["season", "team"])
A = A[A.season != 2019].copy(); A["y"] = 82*A.wpct-41; A["mod"] = 82*A.model_wpct_agefix-41; A["mkt"] = 82*A.mkt_wpct-41
A["age_c"] = A.mw_age - A.groupby("season").mw_age.transform("mean")
X = A[["mod", "mkt", "age_c"]].to_numpy(); y = A.y.to_numpy(); p = np.empty(len(A))
for s in A.season.unique():
    tr = (A.season != s).to_numpy(); p[~tr] = X[~tr] @ np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
A["pred"] = p; A["e"] = y - p; A["luck"] = 82 * (0.5 + A.pred/82) * (0.5 - A.pred/82)
# team features from player-level preseason projections
rows = []
for S in range(2014, 2026):
    P = pd.read_parquet(f"{D}/pre_P_{S}.parquet")
    prev = ps[ps.season.between(S-3, S-1)].groupby("player").agg(mn=("min", "sum"), gp=("gp", "sum"))
    mpg = (prev.mn / prev.gp.clip(lower=1)).clip(10, 38)
    P["mpg"] = P.player.map(mpg).fillna(20)
    P["avail"] = (P.min_proj / (P.mpg * 82)).clip(0.05, 1)
    P["vfull"] = (P.rating + 2) * P.mpg * 82 / 3936                     # season value above replacement if fully healthy
    P["v"] = (P.rating + 2) * P.min_proj / 3936
    P["thin"] = (P.kind != "vet") | (P.player.map(prev.mn).fillna(0) < 1500)
    g = P.groupby("team")
    rows.append(pd.DataFrame({"season": S,
        "star": g.v.max(),                                                               # pts/100 riding on top player
        "star_share": g.v.max() / g.v.apply(lambda s: s.clip(lower=0).sum()),
        "inj_sd": g.apply(lambda d: np.sqrt(np.sum(d.vfull.clip(lower=0)**2 * d.avail*(1-d.avail)))),
        "thin_share": g.apply(lambda d: d.min_proj[d.thin].sum() / d.min_proj.sum()),
        "top2": g.v.apply(lambda s: s.nlargest(2).sum())}).reset_index())
F = pd.concat(rows); A = A.merge(F, on=["season", "team"])
A["bad"] = (A.pred < -8).astype(float)                                  # projected <33 wins (tank/seller risk)
A["absdev"] = A.pred.abs()
FEATS = ["star", "star_share", "inj_sd", "thin_share", "age_c", "bad", "absdev"]
for c in FEATS: A[c + "_z"] = (A[c] - A[c].mean()) / A[c].std()

def nll(theta, d, cols):
    v = d.luck.to_numpy() + np.exp(theta[0] + d[cols].to_numpy() @ theta[1:])
    return 0.5 * np.sum(np.log(v) + d.e.to_numpy()**2 / v)
def fit(d, cols): return minimize(nll, np.r_[np.log(48), np.zeros(len(cols))], args=(d, cols), method="Nelder-Mead" if not cols else "BFGS").x
def loso(cols):
    ll = 0
    for s in A.season.unique():
        th = fit(A[A.season != s], cols); ll -= nll(th, A[A.season == s], cols)
    return ll
base = loso([])
print(f"n {len(A)} team-seasons; residual sd {A.e.std():.2f} wins; luck part {np.sqrt(A.luck.mean()):.2f}")
print(f"{'feature':12s} {'LOSO dLL':>8s} {'coef (log-var per sd)':>22s} {'t~':>5s} | halves")
for c in FEATS:
    cols = [c + "_z"]; th = fit(A, cols)
    # rough se via numerical Hessian
    h = 1e-3; f0 = nll(th, A, cols); tp = th.copy(); tp[1] += h; tm = th.copy(); tm[1] -= h
    se = 1/np.sqrt(max((nll(tp, A, cols) - 2*f0 + nll(tm, A, cols)) / h**2, 1e-9))
    e1 = fit(A[A.season <= 2018], cols)[1]; e2 = fit(A[A.season >= 2020], cols)[1]
    print(f"{c:12s} {loso(cols)-base:+8.2f} {th[1]:+22.3f} {th[1]/se:5.1f} | {e1:+.2f} {e2:+.2f}")
A.to_parquet(f"{D}/team_tau.parquet", index=False)
