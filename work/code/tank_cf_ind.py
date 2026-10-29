"""Approximate: how much would each team's 2026 projection move if 2025-26 got window weight 0.5 (or 0) instead of 1.0?
Box window exact (poss-weighted decayed average); lineup window approximated by the same average of single-season values."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/code")
from features import COMPONENTS
D = "/home/claude/data"
F = [f for fs in COMPONENTS.values() for f in fs]; OFF = [f for f in F if f.startswith("o_")]
W = json.load(open(f"{D}/weights_prod.json")); Wa = {**W["Wo"], **W["Wd"]}
fe = pd.read_parquet(f"{D}/features.parquet"); fe = fe[fe["min"] > 0].copy()
for x in F:
    wt = fe.poss_off if x in OFF else fe.poss_def
    fe[x] = fe[x] * 0.5 * Wa[x]; fe[x] -= (fe[x] * wt).groupby(fe.season).transform("sum") / wt.groupby(fe.season).transform("sum")
fe["box"] = fe[OFF].sum(axis=1) + fe[[f for f in F if f not in OFF]].sum(axis=1)
LH = pd.read_parquet(f"{D}/lineup_hist.parquet"); LH["l1"] = 0.6 * LH.one_o + 0.8 * LH.one_d
fe = fe.merge(LH[["season", "player", "l1"]], on=["season", "player"], how="left").fillna({"l1": 0})
fe = fe[fe.season >= 2023]; fe["poss"] = (fe.poss_off + fe.poss_def) / 2
def window(w25):
    dec = {2023: 0.3, 2024: 0.6, 2025: w25}
    f = fe.assign(w=fe.poss * fe.season.map(dec))
    g = f.groupby("player").apply(lambda x: pd.Series({"box": np.average(x.box, weights=x.w) if x.w.sum() > 0 else np.nan,
                                                        "lin": np.average(x.l1, weights=x.w) if x.w.sum() > 0 else np.nan}), include_groups=False)
    return g
base, half, zero = window(1.0), window(0.5), window(1e-6)
P = pd.read_csv(f"{D}/players_proj_2026.csv")
P = P[P.player.notna()].copy(); P["player"] = P.player.astype(int)
SLOPE = 1.07
for lab, cf in [("half", half), ("zero", zero)]:
    d = (cf.box + cf.lin - base.box - base.lin) * SLOPE
    P["d_" + lab] = P.player.map(d).fillna(0)
P["c_half"] = P.d_half * P.min_proj / 3936; P["c_zero"] = P.d_zero * P.min_proj / 3936
T = P.groupby("team")[["c_half", "c_zero"]].sum()
T -= T.mean()                      # zero-sum
T["wins_half"] = T.c_half * 2.55; T["wins_zero"] = T.c_zero * 2.55
print(T.sort_values("wins_half").round(2).to_string())
print(P[P.team == "IND"].sort_values("min_proj", ascending=False).head(11)[["name", "rating", "min_proj", "d_half", "d_zero"]].round(2).to_string(index=False))
