import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import defsub as S, impact as I
N = 2025; seasons = [2023, 2024, 2025]; r = S.rows(seasons)
dec = r.season.map(dict(zip(seasons, (.3, .6, 1.)))).to_numpy()
X, players, nfe = S.design(r); P = len(players)
rat = pd.DataFrame({"player": players})
for name, lo in dict(r3=500, rrim=1000, rtov=1000, orb=1000, rfta=1000, prim=1000, pts=1000).items():
    y, w = S.outcome(r, name, dec)
    Xw = X.multiply(w[:, None]).tocsr(); pen = np.r_[np.full(P, lo), np.full(P, 16000 if name!="pts" else 2000), np.full(nfe, 1e-6)]
    rat[name] = spsolve(sp.csc_matrix(X.T @ Xw) + sp.diags(pen, format="csc"), Xw.T @ y)[P:2*P]
R = pd.read_parquet(f"{I.D}/ratings_2025.parquet")
f = pd.read_parquet(f"{I.D}/features.parquet"); f = f[f.season == 2025][["player", "name", "min", "d_opprim", "d_oppxq"]]
M = f.merge(rat, on="player").merge(R[["player", "f:d_opprim", "f:d_oppxq", "perimeter_def", "rim_protection", "res_d", "def"]], on="player")
M = M[M["min"] > 800].copy()
for c in ["r3", "rrim", "rtov", "rfta", "prim", "pts", "f:d_opprim", "f:d_oppxq", "def"]:
    asc = c in ["r3", "rrim", "rfta", "prim", "pts"]   # lower opp rate = better
    M["rk_" + c] = M[c].rank(ascending=asc).astype(int)
print("players (800+ min):", len(M))
who = ["Amen Thompson", "Ausar Thompson", "Luguentz Dort", "Dyson Daniels", "Herbert Jones", "Toumani Camara", "Alex Caruso",
       "Jrue Holiday", "OG Anunoby", "Derrick White", "Jaden McDaniels", "Victor Wembanyama", "Rudy Gobert", "Evan Mobley", "Jalen Suggs"]
cols = ["name", "def", "rk_def", "rk_f:d_opprim", "rk_f:d_oppxq", "rk_r3", "rk_rrim", "rk_rtov", "rk_rfta", "rk_prim", "rk_pts"]
print(M[M.name.isin(who)][cols].sort_values("rk_def").to_string(index=False))
print("\ntop 12 by forced TOV (adjusted):"); print(M.sort_values("rtov", ascending=False).head(12)[["name", "rtov"]].round(2).to_string(index=False))
# spread: sd across players, in opp events per 100
print("\nsd across 800+min players:", M[["r3", "rrim", "rtov", "rfta", "prim", "pts"]].std().round(2).to_dict())
M.to_parquet(f"{I.D}/defsub_2025_players.parquet", index=False)
