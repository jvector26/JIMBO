"""Preseason team backtest for the judgment shrink: rating -= SLOPE * a*ramp * (0.6 res_o + 0.8 res_d), window ending S-1
(bt_split_ratings), P = raw window on_poss. Writes tagged pre_P/pre_T and scores with age_eval.py."""
import sys, shutil, subprocess, numpy as np, pandas as pd
D = "/home/claude/data"; SLOPE = pd.read_csv(f"{D}/calib.csv", index_col=0).iloc[:, 0]["slope"]
R = pd.read_parquet(f"{D}/bt_split_ratings.parquet"); ps = pd.read_parquet(f"{D}/player_seasons.parquet")
def run(a, P1, tag):
    for S in range(2014, 2026):
        N = S - 1; r = R[R.season == N].set_index("player")
        Pw = ps[ps.season.between(N - 2, N)].groupby("player").on_poss.sum()
        d = pd.read_parquet(f"{D}/pre_P_{S}.parquet")
        lin = (0.6 * d.player.map(r.res_o) + 0.8 * d.player.map(r.res_d)).fillna(0)
        ramp = np.clip(1 - d.player.map(Pw).fillna(0) / P1, 0, 1)
        has = d.player.isin(r.index) & d.kind.ne("newcomer") & d.kind.ne("rookie")
        d.loc[has, "rating"] = d.rating[has] - SLOPE * a * ramp[has] * lin[has]
        d.to_parquet(f"{D}/{tag}pre_P_{S}.parquet", index=False); shutil.copy(f"{D}/pre_T_{S}.parquet", f"{D}/{tag}pre_T_{S}.parquet")
    return subprocess.run(["python3", "-W", "ignore", "/home/claude/code/age_eval.py", tag], capture_output=True, text=True).stdout.strip()
print(pd.read_parquet(f"{D}/pre_P_2020.parquet").kind.value_counts().to_dict())
print(run(0, 10000, "shk0_"))
for P1 in [8000, 10000]:
    for a in [.25, .33, .5, .75, 1.0]:
        print(f"P1 {P1} a {a}:", run(a, P1, f"shk_{P1}_{int(a*100)}_"))
