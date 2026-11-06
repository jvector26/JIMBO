"""2026 preview of the foul-drawing reweight: production.py recipe with Wo scaled, written to ratings_2025_ft{tag}.parquet (production file untouched)."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/code"); import impact as I
_W = json.load(open(f"{I.D}/weights_prod.json")); Wo0, Wd = pd.Series(_W["Wo"]), pd.Series(_W["Wd"])
def run(v, e):
    Wo = Wo0.copy(); Wo["o_ftvol"] *= v; Wo["o_fteff"] *= e
    r = I.fit_window(2025, Wo * .5, Wd * .5, lam=2000, box_decay=(0.1, 0.4, 1.0))
    f = I.feats(); last = f[f.season == 2025].set_index("player"); r["min_2025"] = r.player.map(last["min"]).fillna(0); w = r.min_2025
    for side, comps in (("off", [c for c, vv in I.COMPONENTS.items() if vv[0].startswith("o_")] + ["res_o"]),
                        ("def", [c for c, vv in I.COMPONENTS.items() if vv[0].startswith("d_")] + ["res_d"])):
        sh = (r[side] * w).sum() / w.sum(); r["res_o" if side == "off" else "res_d"] -= sh; r[side] = r[comps].sum(axis=1)
    m = 1 - (1/3) * np.clip(1 - r.poss_window / 10000, 0, 1)
    box = r[list(I.COMPONENTS)].sum(axis=1)
    r["rating"] = 1.0501 * (box + m * (.6 * r.res_o + .8 * r.res_d))
    r.to_parquet(f"{I.D}/ratings_2025_ft{v}_{e}.parquet", index=False); return r.set_index("player")
a = run(1, 1); b = run(.5, 2.5)
prod = pd.read_parquet(f"{I.D}/ratings_2025.parquet").set_index("player")
print("reproduce production 'total' max diff:", (a.total - prod.total.reindex(a.index)).abs().max().round(4))
d = pd.DataFrame({"name": prod.name.reindex(a.index), "min": a.min_2025, "before": a.rating, "after": b.rating,
                  "fd_before": 1.0501*a.foul_drawing, "fd_after": 1.0501*b.foul_drawing})
d["change"] = d.after - d.before; d["box_part"] = d.fd_after - d.fd_before; d["lineup_part"] = d.change - d.box_part
q = d[d["min"] >= 1000]
print("players 1000+ min:", len(q), " sd change", q.change.std().round(2), " |change|>0.5:", (q.change.abs() > .5).sum())
cols = ["name", "before", "after", "change", "box_part", "lineup_part"]
print("\nBiggest drops:\n", q.sort_values("change").head(12)[cols].round(2).to_string(index=False))
print("\nBiggest gains:\n", q.sort_values("change").tail(10)[cols].round(2).to_string(index=False))
q["rk_b"] = q.before.rank(ascending=False); q["rk_a"] = q.after.rank(ascending=False)
print("\nTop 15 after:\n", q.sort_values("after", ascending=False).head(15)[["name","before","after","rk_b","rk_a"]].round(2).to_string(index=False))
d.to_csv(f"{I.D}/ft_preview_2026.csv")
