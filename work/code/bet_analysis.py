"""Hypothetical win-total betting: stake proportional to |P(over) - 0.5|, $1000 total, side = more likely one."""
import numpy as np, pandas as pd
from scipy.stats import norm
import simulate as SIM
D = "data"
S = pd.read_csv(f"{D}/sim_2026.csv").set_index("team")
L = pd.read_csv(f"{D}/win_totals_2026.csv").set_index("team")
dec = lambda o: 1 + (o / 100 if o > 0 else 100 / -o)       # decimal payout per $1
def plan(pcol, W):
    p = S[pcol]; side_over = p > 0.5
    edge = (p - 0.5).abs(); stake = 1000 * edge / edge.sum()
    pw = np.where(side_over, p, 1 - p)
    odds = np.where(side_over, L.over.reindex(S.index), L.under.reindex(S.index))
    d = np.array([dec(o) for o in odds])
    ev = stake * (pw * (d - 1) - (1 - pw))
    T = pd.DataFrame({"side": np.where(side_over, "over", "under"), "line": L.line.reindex(S.index), "odds": odds,
                      "p_win": pw, "stake": stake, "ev": ev}).sort_values("stake", ascending=False)
    # profit distribution from simulated seasons (columns in SIM.TEAMS order)
    idx = [SIM.IDX[t] for t in S.index]; w = W[:, idx]
    won = np.where(side_over, w > T.line.reindex(S.index).values, w < T.line.reindex(S.index).values)
    st = stake.values; prof = (won * st * (d - 1) - (~won) * st).sum(1)
    return T, prof
for name, col, f in [("FORECAST (blend: model + market + roster age)", "blend_p_over", "sim_wins_2026_blend.npy"),
                     ("MODEL ALONE", "p_over", "sim_wins_2026_model.npy")]:
    T, prof = plan(col, np.load(f"{D}/{f}").astype(int))
    print(f"\n== {name}\n", T.round(3).head(12).to_string())
    print(f"expected profit ${T.ev.sum():.0f} (ROI {T.ev.sum()/10:.1f}%); sim profit mean ${prof.mean():.0f}, "
          f"p10 ${np.percentile(prof,10):.0f}, p90 ${np.percentile(prof,90):.0f}, P(lose money) {np.mean(prof<0):.2f}")

# Historical check, 2014-15..2025-26 (excl. 2019-20): leave-one-season-out blend -> P(over) -> same staking at -110
A = pd.read_parquet(f"{D}/preseason_backtest.parquet").merge(pd.read_parquet(f"{D}/direction_features.parquet"), on=["season", "team"])
A = A[A.season != 2019].copy()
A["y"] = 82 * A.wpct - 41; A["mod"] = 82 * A.model_wpct_agefix - 41; A["mkt"] = 82 * A.mkt_wpct - 41
A["age_c"] = A.mw_age - A.groupby("season").mw_age.transform("mean")
out = []
for s in sorted(A.season.unique()):
    tr, te = A.season != s, A.season == s
    for nm, cols in [("blend", ["mod", "mkt", "age_c"]), ("model", ["mod"])]:
        X = A[cols].to_numpy(); b = np.linalg.lstsq(X[tr], A.y[tr], rcond=None)[0]
        sd = np.sqrt(np.mean((A.y[tr] - X[tr] @ b) ** 2))
        pred = X[te] @ b if nm == "blend" else A.loc[te, "mod"].to_numpy()   # model: raw projection
        if nm == "model": sd = np.sqrt(np.mean((A.y[tr] - A["mod"][tr]) ** 2))
        p = norm.cdf((pred - A.mkt[te]) / sd)
        edge = np.abs(p - 0.5); stake = 1000 * edge / edge.sum()
        over = p > 0.5; y = A.y[te].to_numpy(); line = A.mkt[te].to_numpy()
        win = np.where(over, y > line, y < line); push = y == line
        prof = np.where(push, 0, np.where(win, stake * 100 / 110, -stake))
        out.append({"season": s, "kind": nm, "profit": prof.sum(), "hit": win[~push].mean()})
H = pd.DataFrame(out).pivot(index="season", columns="kind", values="profit")
print("\n== Historical, $1000 per season at -110 (leave-one-season-out)\n", H.round(0).to_string())
print("total:", H.sum().round(0).to_dict(), "| avg/season:", H.mean().round(0).to_dict(), "| losing seasons:", (H < 0).sum().to_dict())
hit = pd.DataFrame(out).groupby("kind").hit.mean(); print("avg hit rate:", hit.round(3).to_dict())
