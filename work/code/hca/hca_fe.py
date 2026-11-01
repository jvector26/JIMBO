"""Home court by season in the TOTAL view (team-season fixed effects + b2b diff), i.e. what nightly margins need,
plus production constants 2.54/2.60 scored in the research-bar backtest (actual minutes, b2b fatigue term fit LOSO)."""
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.linalg import lsqr
D = "/home/claude/data"
G = pd.read_parquet(f"{D}/b2b_games.parquet")
out = {}
for S, g in G[G.season >= 2004].groupby("season"):
    ts = sorted(set(g.home_team) | set(g.away_team)); ix = {t: i for i, t in enumerate(ts)}; n = len(g)
    r = np.r_[np.arange(n), np.arange(n)]; c = np.r_[g.home_team.map(ix), g.away_team.map(ix)]
    v = np.r_[np.ones(n), -np.ones(n)]
    X = sp.csr_matrix((v, (r, c)), shape=(n, len(ts))).toarray()
    X = np.c_[X, np.ones(n), g.b2b_h - g.b2b_a]
    X = np.r_[X, np.r_[np.ones(len(ts)), 0, 0][None, :]]; yy = np.r_[g.margin, 0]   # sum-to-zero
    b = np.linalg.lstsq(X, yy, rcond=None)[0]
    out[S] = (b[-2], b[-1], (g.margin - b[-2]).shape[0])
T = pd.DataFrame(out, index=["hca", "b2b", "n"]).T
print(T.round(2).to_string())
for a, z in [(2004, 2014), (2015, 2019), (2021, 2025), (2022, 2025), (2015, 2025)]:
    m = T.loc[a:z]; print(f"{a}-{z} mean hca {m.hca.mean():.2f} (games-weighted {np.average(m.hca, weights=m.n):.2f})")
# backtest: constant home court levels
B = pd.read_parquet(f"{D}/backtest_components.parquet").merge(pd.read_parquet(f"{D}/bt_split.parquet"), on=["game", "home_team", "away_team"])
B = B.merge(G[["game", "b2b_h", "b2b_a"]], on="game", how="left").fillna({"b2b_h": 0, "b2b_a": 0})
F = [c for c in B.columns if c.startswith("f:")]
base = (B[F].sum(axis=1) + B.unrated + 0.6 * B.res_o + 0.8 * B.res_d).to_numpy(); bd = (B.b2b_h - B.b2b_a).to_numpy()
y = B.margin.to_numpy(); s = B.season.to_numpy()
def run(fixed):
    e = []
    for q in np.unique(s):
        tr = s != q
        if fixed is None:
            X = np.c_[np.ones(tr.sum()), base[tr], bd[tr]]; b = np.linalg.lstsq(X, y[tr], rcond=None)[0]; a = b[0]
        else:
            X = np.c_[base[tr], bd[tr]]; b = np.r_[0, np.linalg.lstsq(X, y[tr] - fixed, rcond=None)[0]]; a = fixed
        m = s == q; e.append((y[m] - a - b[1]*base[m] - b[2]*bd[m]) ** 2)
    return np.concatenate(e)
e0 = run(None); print(f"\nLOSO fitted const RMSE {np.sqrt(e0.mean()):.4f}")
for h in [2.6, 2.54, 2.3, 2.0, 1.8, 1.6]:
    e = run(h); d = pd.Series(e - e0).groupby(s).mean(); t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    print(f"fixed {h:.2f}: RMSE {np.sqrt(e.mean()):.4f} ({np.sqrt(e.mean())-np.sqrt(e0.mean()):+.4f}) worse in {(d>0).sum()}/11 t {t:.2f}"
          f" | 15-19 {d[d.index<=2019].mean():+.3f} 20-25 {d[d.index>=2020].mean():+.3f}")
