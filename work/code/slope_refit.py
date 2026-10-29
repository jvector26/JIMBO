"""Backlog #1: calibration slope refit under split lineup weights (0.6 off / 0.8 def) with current caches
(dreb 0.5x, shooting prior). Same design as rookie_refit.py; compares base = comps + 0.6*lineup_impact (old)
with base = comps + 0.6*res_o + 0.8*res_d (production)."""
import numpy as np, pandas as pd, json
D = "data"
A = pd.read_parquet(f"{D}/age_design2.parquet").merge(
    pd.read_parquet(f"{D}/backtest_components.parquet").drop(columns=["season"]), on="game").merge(
    pd.read_parquet(f"{D}/bt_split.parquet")[["game", "res_o", "res_d"]], on="game")
comp = [c for c in A.columns if c.startswith("f:")]
AGES = [int(c[1:]) for c in A.columns if c[0] == "A" and c[1:].isdigit()]
BINS = [(0, 21, "<=21"), (22, 23, "22-23"), (24, 25, "24-25"), (26, 27, "26-27"), (28, 29, "28-29"),
        (30, 31, "30-31"), (32, 33, "32-33"), (34, 35, "34-35"), (36, 99, "36+")]
CURVE = pd.read_csv(f"{D}/aging_player1.csv").set_index("bin").adj.to_dict()
binof = lambda a: next(l for lo, hi, l in BINS if lo <= a <= hi)
off_age = A[[f"A{a}" for a in AGES]].to_numpy() @ np.array([CURVE[binof(a)] for a in AGES])
y = A.margin.to_numpy() - off_age; seas = A.season.to_numpy()
cols = ["base", "rk_col", "rk_col_cs", "rk_non_val", "other_unrated"]
def fit(base):
    A["base"] = base; X = np.c_[np.ones(len(A)), A[cols].to_numpy()]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    rng = np.random.default_rng(0); S = sorted(set(seas)); bs = []
    for _ in range(100):
        m = np.concatenate([np.where(seas == s)[0] for s in rng.choice(S, len(S))])
        bs.append(np.linalg.lstsq(X[m], y[m], rcond=None)[0])
    e = []
    for s in S:  # LOSO
        tr, te = seas != s, seas == s
        bb = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; e.append((y[te] - X[te] @ bb) ** 2)
    return b, np.array(bs).std(0), np.sqrt(np.concatenate(e).mean())
old = fit(A[comp].sum(axis=1) + 0.6 * A.lineup_impact)
new = fit(A[comp].sum(axis=1) + 0.6 * A.res_o + 0.8 * A.res_d)
for nm, (b, se, l) in [("old base (0.6 single)", old), ("new base (0.6/0.8)", new)]:
    print(f"{nm}: LOSO {l:.4f} | " + "  ".join(f"{c} {b[i]:+.3f}({se[i]:.3f})" for i, c in enumerate(["const"] + cols)))
pre = lambda b: {"slope": b[1], "col_a": b[2] / b[1], "col_b": b[3] / b[1], "non": b[4] / b[1], "unrated": b[5] / b[1]}
po, pn = pre(old[0]), pre(new[0])
ratio = {k: pn[k] / po[k] for k in po}
print("pre-slope old -> new:", {k: (round(po[k], 3), round(pn[k], 3)) for k in po}); print("ratios:", {k: round(v, 3) for k, v in ratio.items()})
json.dump({"slope_new": new[0][1], "slope_old_design": old[0][1], "ratio": ratio}, open(f"{D}/slope_refit.json", "w"), indent=1)
