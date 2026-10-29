"""Variant: shrink only players with <= 3 NBA seasons through N (career seasons in player_seasons)."""
import sys, subprocess, shutil, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/code"); sys.path.insert(0, "/home/claude/code/shrink")
from lineup_port_bt import G, diffs, fixed, ps
D = "/home/claude/data"; A, P1 = 1/3, 10000
car = {N: ps[ps.season <= N].groupby("player").season.nunique() for N in range(2013, 2025)}
Pw = {N: ps[ps.season.between(N - 2, N)].groupby("player").on_poss.sum() for N in range(2013, 2025)}
G["Praw"] = [Pw[n].get(p, 0.0) for n, p in zip(G.N, G.player)]; G["yrs"] = [car[n].get(p, 0) for n, p in zip(G.N, G.player)]
b0, s0 = fixed(diffs(np.ones(len(G))))
for lab, msk in [("all", np.ones(len(G), bool)), ("young<=3", G.yrs.values <= 3)]:
    m = 1 - A * np.clip(1 - G.Praw / P1, 0, 1) * msk
    l, s = fixed(diffs(m)); print(f"game {lab}: LOSO {l:.4f} (d {l-b0:+.4f}) halves {np.sqrt(np.mean(s[:5]**2))-np.sqrt(np.mean(s0[:5]**2)):+.4f} {np.sqrt(np.mean(s[5:]**2))-np.sqrt(np.mean(s0[5:]**2)):+.4f}")
R = pd.read_parquet(f"{D}/bt_split_ratings.parquet"); SLOPE = pd.read_csv(f"{D}/calib.csv", index_col=0).iloc[:, 0]["slope"]
tag = "shky_"
for S in range(2014, 2026):
    N = S - 1; r = R[R.season == N].set_index("player"); d = pd.read_parquet(f"{D}/pre_P_{S}.parquet")
    lin = (0.6 * d.player.map(r.res_o) + 0.8 * d.player.map(r.res_d)).fillna(0)
    ramp = np.clip(1 - d.player.map(Pw[N]).fillna(0) / P1, 0, 1) * (d.player.map(car[N]).fillna(0) <= 3)
    has = d.player.isin(r.index) & d.kind.eq("vet")
    d.loc[has, "rating"] = d.rating[has] - SLOPE * A * ramp[has] * lin[has]
    d.to_parquet(f"{D}/{tag}pre_P_{S}.parquet", index=False); shutil.copy(f"{D}/pre_T_{S}.parquet", f"{D}/{tag}pre_T_{S}.parquet")
print("preseason young-only:", subprocess.run(["python3", "-W", "ignore", "/home/claude/code/age_eval.py", tag], capture_output=True, text=True).stdout.strip())
