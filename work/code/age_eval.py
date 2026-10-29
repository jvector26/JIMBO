"""Score one preseason-backtest variant (written with BT_TAG) the same way as score_preseason.py + minutes_age_fix.py
+ refit_blend.py: team RMSE in wins, and leave-one-season-out blend (model + Vegas + roster age) RMSE."""
import sys, numpy as np, pandas as pd
from scipy.stats import norm
D = "/home/claude/data"; TAG = sys.argv[1] if len(sys.argv) > 1 else ""
gm = pd.read_parquet(f"{D}/games_all.parquet")
id2abbr = pd.read_parquet(f"{D}/games_2025.parquet").drop_duplicates("home_team").set_index("home_team").htm.to_dict()
res = pd.concat([gm.assign(t=gm.home_team, w=gm.home_pts > gm.away_pts), gm.assign(t=gm.away_team, w=gm.away_pts > gm.home_pts)])
act = res.groupby(["season", "t"]).agg(wpct=("w", "mean")).reset_index(); act["team"] = act.t.map(id2abbr)
ps = pd.read_parquet(f"{D}/player_seasons.parquet")
EDGES = [0, 22, 25, 28, 31, 34, 50]; LABS = ["<=22", "23-25", "26-28", "29-31", "32-34", "35+"]
rows = []
for S in range(2014, 2026):
    P = pd.read_parquet(f"{D}/{TAG}pre_P_{S}.parquet")
    last = ps[ps.season < S].sort_values("season").groupby("player").last()
    P["age"] = (P.player.map(last.age) + (S - P.player.map(last.season))).fillna(21)
    ng = {2019: 1056 * 2 / 30, 2020: 72}.get(S, 82)
    P["act_min"] = P.player.map(ps[ps.season == S].groupby("player")["min"].sum()).fillna(0) * 82 / ng
    P["bin"] = pd.cut(P.age, EDGES, labels=LABS).astype(str); rows.append(P)
P = pd.concat(rows, ignore_index=True)
def allocate(g, col):
    m = g[col].clip(upper=3000).copy(); excess = m.sum() - 19680
    if excess > 0:
        for i in m.sort_values().index:
            cut = min(excess, m[i]); m[i] -= cut; excess -= cut
            if excess <= 0: break
    else:
        for _ in range(10):
            room = m < 3000; m[room] *= (19680 - m[~room].sum()) / m[room].sum(); m = m.clip(upper=3000)
    return m
def mults(train):
    t = train[train.season != 2019].groupby("bin")[["act_min", "min_model"]].sum(); return t.act_min / t.min_model
out = []
for S in range(2014, 2026):
    k = mults(P[P.season != S]); d = P[P.season == S].copy()
    d["min_adj"] = d.min_model * d.bin.map(k)
    d["min_proj2"] = d.groupby("team", group_keys=False).apply(lambda g: allocate(g, "min_adj"))
    T = pd.read_parquet(f"{D}/{TAG}pre_T_{S}.parquet").set_index("team")
    pace = (T.margin / T.net).mean() * 100
    net = (d.rating * d.min_proj2 / 3936).groupby(d.team).sum(); net -= net.mean()
    T["model_wpct_agefix"] = norm.cdf(net * pace / 100 / 12.8)
    out.append(T.reset_index()[["season", "team", "model_wpct_agefix"]])
M = pd.concat(out)
L = pd.read_csv(f"{D}/hist_lines.csv"); L["mkt_wpct"] = L.line / np.where(L.season == 2020, 72, 82)
T = pd.concat([pd.read_parquet(f"{D}/{TAG}pre_T_{s}.parquet") for s in range(2014, 2026)])
A = T.merge(act[["season", "team", "wpct"]], on=["season", "team"]).merge(L[["season", "team", "mkt_wpct"]], on=["season", "team"]).merge(M, on=["season", "team"])
A = A.merge(pd.read_parquet(f"{D}/direction_features.parquet")[["season", "team", "mw_age"]], on=["season", "team"])
A = A[A.season != 2019].copy()
A["y"] = 82 * A.wpct - 41; A["mod"] = 82 * A.model_wpct_agefix - 41; A["raw"] = 82 * A.model_wpct - 41
A["mkt"] = 82 * A.mkt_wpct - 41; A["age_c"] = A.mw_age - A.groupby("season").mw_age.transform("mean")
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
Z = np.c_[A["mod"] - A.mkt, A.age_c]; yz = (A.y - A.mkt).to_numpy(); p = np.empty(len(A))
for s in A.season.unique():
    tr = (A.season != s).to_numpy()
    p[~tr] = A.mkt.to_numpy()[~tr] + Z[~tr] @ np.linalg.lstsq(Z[tr], yz[tr], rcond=None)[0]
a, c = np.linalg.lstsq(Z, yz, rcond=None)[0]
sl = np.polyfit(A["mod"], A.y, 1)[0]
print("%-10s model(raw) %.3f  model(minfix) %.3f  slope %.2f  Vegas %.3f  blend LOSO %.3f  w_model %.2f w_age %.2f"
      % (TAG or "current", r(A.y - A["raw"]), r(A.y - A["mod"]), sl, r(A.y - A.mkt), r(A.y - p), a, c))
