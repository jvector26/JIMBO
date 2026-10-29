"""Defense #7 applied: hustle term from NBA.com hustle stats (gabriel1200/site_Data hustle.csv, 2017-18+).
value = C_LBR * lbr + C_C3 * cont3 (post-slope rating pts/100), lbr = defensive loose balls recovered, cont3 = contested
3PT shots; per 100 def poss, season-centred, shrunk poss/(poss+k) (k 1000 / 3000), window N-2..N decay .3/.6/1 over
seasons with data. Coefs from the joint game-level fit (code/track_score.py, seasons 2018-2025). -> data/hustle_{N+1}.csv"""
import json, sys, numpy as np, pandas as pd
D = "/home/claude/data"
CFG = {"c_lbr": 1.74, "k_lbr": 1000, "c_c3": 0.253, "k_c3": 3000, "first_season": 2017,
       "source": "github.com/gabriel1200/site_Data hustle.csv (year = season end)"}
def window(N):
    f = pd.read_parquet(f"{D}/features.parquet")[["season", "player", "poss_def"]]
    h = pd.read_csv(f"{D}/tracking/hustle.csv"); h["season"] = h.year - 1
    h = h.groupby(["season", "PLAYER_ID"]).sum(numeric_only=True).reset_index().rename(columns={"PLAYER_ID": "player"})
    x = f[f.season.between(max(N - 2, CFG["first_season"]), N)].merge(
        h[["season", "player", "DEF_LOOSE_BALLS_RECOVERED", "CONTESTED_SHOTS_3PT"]], on=["season", "player"], how="left")
    have = x.season.isin(h.season.unique())
    out = {}
    for c, src, k in [("lbr", "DEF_LOOSE_BALLS_RECOVERED", CFG["k_lbr"]), ("c3", "CONTESTED_SHOTS_3PT", CFG["k_c3"])]:
        v = x[src].fillna(0)
        lg = v[have].groupby(x.season[have]).sum() / x[have].groupby("season").poss_def.sum()
        x[c] = ((v - x.season.map(lg) * x.poss_def) / (x.poss_def + k) * 100).where(have)
    x["wt"] = x.poss_def * x.season.map({N - 2: .3, N - 1: .6, N: 1.})
    for c in ("lbr", "c3"):
        ww = x.wt.where(x[c].notna(), 0)
        out[c] = (x[c].fillna(0) * ww).groupby(x.player).sum() / ww.groupby(x.player).sum().replace(0, np.nan)
    o = pd.DataFrame(out).fillna(0)
    o["hustle"] = CFG["c_lbr"] * o.lbr + CFG["c_c3"] * o.c3
    return o
if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    o = window(N); o.to_csv(f"{D}/hustle_{N+1}.csv"); json.dump(CFG, open(f"{D}/hustle.json", "w"), indent=1)
    print(f"hustle window {N-2}-{N}: {len(o)} players, sd {o.hustle.std():.3f}, range {o.hustle.min():.2f}..{o.hustle.max():.2f}")
