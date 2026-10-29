import json, numpy as np, pandas as pd, impact as I
ALPHA, LAM = 0.5, 2000
import os
if os.environ.get("REFIT_W"):  # full weight refit (slow). NOTE: then re-halve d_dreb (defense #5, 2026-09-20)
    Wo, Wd, hca = I.fit_weights(range(1996, 2026), lam=4000); Wd["d_dreb"] *= 0.5
    json.dump({"Wo": Wo.to_dict(), "Wd": Wd.to_dict(), "hca": hca, "alpha": ALPHA, "lam": LAM}, open(f"{I.D}/weights_prod.json", "w"))
else:
    _W = json.load(open(f"{I.D}/weights_prod.json")); Wo, Wd, hca = pd.Series(_W["Wo"]), pd.Series(_W["Wd"]), _W["hca"]
print("prod W", Wo.round(2).to_dict(), Wd.round(2).to_dict(), "hca", round(hca, 2), flush=True)
BOX_DECAY = (0.1, 0.4, 1.0)  # box-part season weights (session 12 decay test); lineup residual keeps (.3,.6,1)
r = I.fit_window(2025, Wo * ALPHA, Wd * ALPHA, lam=LAM, box_decay=BOX_DECAY)
f = I.feats(); last = f[f.season == 2025].set_index("player")
r["name"] = r.player.map(pd.read_parquet(f"{I.D}/player_seasons.parquet").drop_duplicates("player", keep="last").set_index("player")["name"])
r["team_2025"] = r.player.map(last["teams"]); r["age_2025"] = r.player.map(last["age"]); r["min_2025"] = r.player.map(last["min"]).fillna(0)
# re-center so the minutes-weighted average 2025-26 player is 0 (off and def separately)
w = r.min_2025
for side, comps in (("off", [c for c, v in I.COMPONENTS.items() if v[0].startswith("o_")] + ["res_o"]),
                    ("def", [c for c, v in I.COMPONENTS.items() if v[0].startswith("d_")] + ["res_d"])):
    shift = (r[side] * w).sum() / w.sum()
    r["res_o" if side == "off" else "res_d"] -= shift
    r[side] = r[comps].sum(axis=1)
r["total"] = r["off"] + r["def"]; r["lineup_impact"] = r.res_o + r.res_d
r.to_parquet(f"{I.D}/ratings_2025.parquet", index=False)
print("saved", len(r))
