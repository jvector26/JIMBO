"""Does the RealGM preseason depth-chart tier improve JIMBO's preseason minutes? (LOSO by season, 2015-25; scored excl
2019-20 and 2020-21 for players, excl 2019-20 for teams, as in minutes_recal.py). Baseline = current production
(recal of min_model incl age mult, smallest-first trim)."""
import json, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy.stats import norm
D = "/home/claude/data"
SNAP = sys.argv[1] if len(sys.argv) > 1 else "late"
M = pd.read_parquet(f"{D}/minutes_cal.parquet"); M = M[M.season >= 2015].copy()
M["act82"] = M.mins * 82 / M.ng
c = json.load(open(f"{D}/min_recal.json"))["coef"]
m = M.min_model_age
M["m0"] = (c[0] + c[1] * m + c[2] * (m - 1000).clip(lower=0) + c[3] * (m - 2000).clip(lower=0)).clip(0, 3000)
OPEN = {2015: "20151027", 2016: "20161025", 2017: "20171017", 2018: "20181016", 2019: "20191022", 2020: "20201222",
        2021: "20211019", 2022: "20221018", 2023: "20231024", 2024: "20241022", 2025: "20251021"}
Dp = pd.read_parquet(f"{D}/depth_realgm.parquet")
Dp = Dp[Dp.snap.map(lambda s: True) & (Dp.groupby(["season", "snap"]).team.transform("nunique") == 30)]
snaps = Dp[["season", "snap"]].drop_duplicates()
def pick(g):
    S = g.season.iloc[0]; o = pd.Timestamp(OPEN[S]); d = pd.to_datetime(g.snap)
    if SNAP == "late": return g.snap.iloc[d.argmax()]
    return g.snap.iloc[(d - (o - pd.Timedelta(days=28))).abs().argmin()]
use = pd.DataFrame([{"season": S, "snap": pick(g.assign(season=S))} for S, g in snaps.groupby("season")])
print("snapshots:", dict(zip(use.season, use.snap)))
Dp = Dp.merge(use, on=["season", "snap"])[["season", "team", "player", "tier", "depth_rank"]]
M = M.merge(Dp.rename(columns={"team": "dteam"}), on=["season", "player"], how="left")
M["tier"] = np.where(M.tier.notna() & (M.dteam == M.team), M.tier, "N")  # N = not on this team's chart
M["depth_rank"] = M.depth_rank.where(M.tier != "N", 20)
FIT = ~M.season.isin([2019, 2020]); SC = ~M.season.isin([2019, 2020])
print(M[SC].groupby("tier").agg(n=("act82", "size"), m0=("m0", "mean"), act=("act82", "mean"),
      rmse0=("act82", lambda s: 0)).drop(columns="rmse0").round(0).T.to_string())
A = pd.read_parquet(f"{D}/preseason_backtest.parquet")
pace = {S: (lambda T: (T.margin / T.net).mean() * 100)(pd.read_parquet(f"{D}/pre_T_{S}.parquet")) for S in range(2015, 2026)}
ORD = {"N": 0, "L": 1, "R": 2, "S": 3}
def allocate(g, col, bytier):
    m = g[col].clip(0, 3000).copy(); excess = m.sum() - 19680
    if excess > 0:
        order = (g.assign(_o=g.tier.map(ORD), _m=m).sort_values(["_o", "_m"]).index if bytier else m.sort_values().index)
        for i in order:
            cut = min(excess, m[i]); m[i] -= cut; excess -= cut
            if excess <= 0: break
    else:
        for _ in range(10):
            room = m < 3000; m[room] *= (19680 - m[~room].sum()) / m[room].sum(); m = m.clip(upper=3000)
    return m
def ls(X, y): return np.linalg.lstsq(X, y, rcond=None)[0]
def feats(d, kind):
    T = [(d.tier == t).astype(float).values for t in "SRLN"]
    if kind == "tier": return np.column_stack(T + [t * d.m0.values for t in T])
    if kind == "tier+rank":
        r = np.minimum(d.depth_rank.values, 15)
        return np.column_stack(T + [t * d.m0.values for t in T] + [r, r * d.m0.values / 1000])
def run(name, kind=None, bytier=False, shrink=1.0):
    out = []
    for S in sorted(M.season.unique()):
        tr = M[FIT & (M.season != S)]; te = M[M.season == S].copy()
        if kind:
            b = ls(feats(tr, kind), tr.act82.values)
            te["want"] = (1 - shrink) * te.m0 + shrink * (feats(te, kind) @ b).clip(0, 3000)
        else: te["want"] = te.m0
        te["proj"] = te.groupby("team", group_keys=False).apply(lambda g: allocate(g, "want", bytier))
        q = te[te.rating.notna()].copy()
        q["mt"] = q.groupby("team", group_keys=False).apply(lambda g: allocate(g, "want", bytier))
        te["mt"] = q.mt
        out.append(te)
    P = pd.concat(out); s = P[SC.reindex(P.index).values] if False else P[~P.season.isin([2019, 2020])]
    pr = s.assign(se=(s.act82 - s.proj) ** 2).groupby("season").se.mean()
    q = P[P.mt.notna()]
    net = (q.rating * q.mt / 3936).groupby([q.season, q.team]).sum(); net = net - net.groupby(level=0).transform("mean")
    T = net.rename("net").reset_index(); T["wp"] = norm.cdf(T.net * T.season.map(pace) / 100 / 12.8)
    T = T.merge(A[["season", "team", "wpct"]], on=["season", "team"]); T = T[T.season != 2019]
    tt = T.assign(se=(82 * (T.wp - T.wpct)) ** 2).groupby("season").se.mean()
    return name, P, pr, tt
res = [run("baseline"), run("trim by tier", None, True), run("tier model", "tier"), run("tier model + trim", "tier", True),
       run("tier+rank + trim", "tier+rank", True)]
_, _, pr0, tt0 = res[0]
print(f"\nsnapshot={SNAP}   player-min RMSE | team-win RMSE  (t = season-clustered vs baseline, negative = better)")
for name, P, pr, tt in res:
    dp, dt = pr - pr0, tt - tt0
    tp = dp.mean() / dp.std() * np.sqrt(len(dp)) if dp.std() > 0 else 0; tq = dt.mean() / dt.std() * np.sqrt(len(dt)) if dt.std() > 0 else 0
    h1 = [S for S in pr.index if S <= 2018]; h2 = [S for S in pr.index if S >= 2021]
    print(f"{name:20s} {np.sqrt(pr.mean()):6.1f} (t {tp:5.1f}; halves {np.sqrt(pr[h1].mean()):.0f}/{np.sqrt(pr[h2].mean()):.0f}) | "
          f"{np.sqrt(tt.mean()):.3f} (t {tq:5.1f}; halves {np.sqrt(tt[[S for S in tt.index if S<=2018]].mean()):.2f}/{np.sqrt(tt[[S for S in tt.index if S>=2020]].mean()):.2f})")
pd.concat([r[1].assign(variant=r[0]) for r in res]).to_parquet(f"{D}/depth_test_{SNAP}.parquet", index=False)
