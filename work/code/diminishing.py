"""Diminishing returns / lineup fit on stints. Player profiles from the PRIOR season (no mechanical overlap).
y (luck-adj pts/100 for the offense) ~ season FE + home + per-component lineup sums (off side: offense comps of the 5
offensive players; def side: defense comps of the 5 defenders) + squared sums (+ selected interactions).
Squared coef < 0 => diminishing returns (5 great rebounders worth less than 5x one)."""
import numpy as np, pandas as pd, json, sys
sys.path.insert(0, "/home/claude/code")
from impact import load_rows
from features import COMPONENTS
D = "/home/claude/data"
W = json.load(open(f"{D}/weights_prod.json")); Wall = {**W["Wo"], **W["Wd"]}
f = pd.read_parquet(f"{D}/features.parquet"); ps = pd.read_parquet(f"{D}/player_seasons.parquet")
f = f[f["min"] >= 300]
for comp, fs in COMPONENTS.items(): f[comp] = sum(f[c]*Wall[c] for c in fs)
OFFC = [c for c in COMPONENTS if COMPONENTS[c][0].startswith("o_")]; DEFC = [c for c in COMPONENTS if COMPONENTS[c][0].startswith("d_")]
f = f.merge(ps.groupby(["season", "player"]).height_in.first().reset_index(), on=["season", "player"], how="left")
f["big"] = (f.height_in >= 82).astype(float)
rows = []
for S in range(2012, 2026):
    r = load_rows([S])
    prof = f[f.season.isin([S-1, S-2])].sort_values("season").drop_duplicates("player", keep="last").set_index("player")
    for side, comps in (("off", OFFC + ["big"]), ("dfn", DEFC + ["big"])):
        ids = r[side].str.split("-", expand=True).astype(np.int64).to_numpy()
        known = np.isin(ids, prof.index.to_numpy())
        r[side + "_unk"] = (~known).sum(1)
        for c in comps:
            v = prof[c].reindex(ids.ravel()).fillna(0 if c != "big" else prof[c].mean()).to_numpy().reshape(ids.shape)
            r[f"{side}_{c}"] = v.sum(1)
    rows.append(r.drop(columns=["off", "dfn", "pts"]))
    print(S, len(r), flush=True)
R = pd.concat(rows, ignore_index=True)
R.to_parquet(f"{D}/dr_rows.parquet", index=False)
