"""Preseason team backtest (2014-25, age_eval scoring) for the separated-possessions shrink (session 24).
Same construction as code/shrink/shrink_pre.py: rating -= SLOPE * [0.6 res_o (1-m_o) + 0.8 res_d (1-m_d)],
m = 1 - a*clip(1 - P*c/k / P1, 0, 1), window N = S-1 residuals from bt_split_ratings, P = raw window on_poss."""
import sys, json, shutil, subprocess, numpy as np, pandas as pd
D = "/home/claude/data"; SLOPE = pd.read_csv(f"{D}/calib.csv", index_col=0).iloc[:, 0]["slope"]
R = pd.read_parquet(f"{D}/bt_split_ratings.parquet"); ps = pd.read_parquet(f"{D}/player_seasons.parquet")
S = pd.read_parquet(f"{D}/sep_poss.parquet"); K = json.load(open(f"{D}/sep_k.json"))
S["c_neff_o"] = (S.neff_o / S.nnaive_o).clip(0, 1); S["c_neff_d"] = (S.neff_d / S.nnaive_d).clip(0, 1)
S["c_sh1"] = 1 - S.sh1; S["c_sep2"] = S.sep2
for c in K: S[c] = S[c] / K[c]
VAR = {"total": (None, None), "neff": ("c_neff_o", "c_neff_d"), "top1": ("c_sh1", "c_sh1"), "sep2": ("c_sep2", "c_sep2")}

def run(var, a, P1, tag):
    co, cd = VAR[var]
    for Sn in range(2014, 2026):
        N = Sn - 1; r = R[R.season == N].set_index("player"); s = S[S.N == N].set_index("player")
        Pw = ps[ps.season.between(N - 2, N)].groupby("player").on_poss.sum()
        d = pd.read_parquet(f"{D}/pre_P_{Sn}.parquet")
        P = d.player.map(Pw).fillna(0)
        fo = d.player.map(s[co]).fillna(1.0) if co else 1.0; fd = d.player.map(s[cd]).fillna(1.0) if cd else 1.0
        so = a * np.clip(1 - P * fo / P1, 0, 1); sd = a * np.clip(1 - P * fd / P1, 0, 1)
        cut = (0.6 * d.player.map(r.res_o) * so + 0.8 * d.player.map(r.res_d) * sd).fillna(0)
        has = d.player.isin(r.index) & d.kind.ne("newcomer") & d.kind.ne("rookie")
        d.loc[has, "rating"] = d.rating[has] - SLOPE * cut[has]
        d.to_parquet(f"{D}/{tag}pre_P_{Sn}.parquet", index=False); shutil.copy(f"{D}/pre_T_{Sn}.parquet", f"{D}/{tag}pre_T_{Sn}.parquet")
    return subprocess.run(["python3", "-W", "ignore", "/home/claude/code/age_eval.py", tag], capture_output=True, text=True).stdout.strip()

if __name__ == "__main__":
    print("no shrink:", run("total", 0, 10000, "sp0_"))
    for P1, a in [(10000, 1 / 3), (12000, 1 / 3), (15000, 1 / 3), (10000, .5)]:
        for var in VAR:
            print(f"{var:6s} P1 {P1} a {a:.2f}:", run(var, a, P1, f"sp_{var}_{P1}_{int(a*100)}_"), flush=True)
