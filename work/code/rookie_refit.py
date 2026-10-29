"""Refit slope, rookie college mapping (col_a, col_b), non-college rookie scale and the unrated value at game level
with the NEW age curve (player-derived shape) held fixed. Design: age_design2.parquet."""
import numpy as np, pandas as pd, json
D = "data"
A = pd.read_parquet(f"{D}/age_design2.parquet").merge(
    pd.read_parquet(f"{D}/backtest_components.parquet").drop(columns=["season"]), on="game")
comp = [c for c in A.columns if c.startswith("f:")]
A["base"] = A[comp].sum(axis=1) + 0.6 * A["lineup_impact"]
AGES = [int(c[1:]) for c in A.columns if c[0] == "A" and c[1:].isdigit()]
X_age = A[[f"A{a}" for a in AGES]].to_numpy()
BINS = [(0, 21, "<=21"), (22, 23, "22-23"), (24, 25, "24-25"), (26, 27, "26-27"), (28, 29, "28-29"),
        (30, 31, "30-31"), (32, 33, "32-33"), (34, 35, "34-35"), (36, 99, "36+")]
CURVE = pd.read_csv(f"{D}/aging_player1.csv").set_index("bin").adj.to_dict()
binof = lambda a: next(l for lo, hi, l in BINS if lo <= a <= hi)
off_age = X_age @ np.array([CURVE[binof(a)] for a in AGES])
cols = ["base", "rk_col", "rk_col_cs", "rk_non_val", "other_unrated"]
X = np.c_[np.ones(len(A)), A[cols].to_numpy()]; y = A.margin.to_numpy() - off_age
b = np.linalg.lstsq(X, y, rcond=None)[0]
seasons = sorted(A.season.unique()); rng = np.random.default_rng(0)
bs = []
for _ in range(100):
    m = np.concatenate([np.where(A.season.to_numpy() == s)[0] for s in rng.choice(seasons, len(seasons))])
    bs.append(np.linalg.lstsq(X[m], y[m], rcond=None)[0])
bs = np.array(bs)
for i, c in enumerate(["const"] + cols):
    print("  %-12s %+0.3f  se %.3f" % (c, b[i], bs[:, i].std()))
RV = json.load(open(f"{D}/rookie_value.json"))
print("\nnew slope %.3f | college rookies: a %+.2f b %.2f (old %+.2f %.2f) | non-college scale %.2f | unrated %+.2f (old %+.2f)"
      % (b[1], b[2], b[3], RV["col_a"], RV["col_b"], b[4], b[5], -4.685))
